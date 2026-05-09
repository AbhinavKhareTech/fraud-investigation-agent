"""
Investigation Orchestrator - Multi-turn fraud investigation agent.

Custom lightweight state machine (~250 LOC). No LangGraph/LangChain dependency.
PS3 brief says "minimal agent, maximum reliability" - a framework adds 3 deps
for a 2-tool agent. This gives us full control over retry logic, failure handling,
and investigation state transitions.
"""

import json
import logging
from enum import Enum

from agent.llm import LLMClient
from agent.memory import InvestigationMemory, Finding
from agent.tools import ToolRegistry
from agent.prompts import SYSTEM_PROMPT, format_investigation_context

logger = logging.getLogger(__name__)


class AgentAction(Enum):
    CALL_TOOL = "call_tool"
    RESPOND = "respond"


class InvestigationOrchestrator:
    """
    State-machine orchestrator for multi-turn fraud investigations.

    Design rationale (see ARCHITECTURE.md):
    - Explicit phases (TRIAGE -> DEEP_DIVE -> SYNTHESIS) vs. free-form chat
    - Structured memory (evidence ledger) vs. raw chat history
    - Tool results validated before memory insertion
    - Graceful degradation: tool failures logged, investigation continues
      with partial evidence and a gap flag
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        max_tool_calls_per_turn: int = 3,
        max_turns: int = 20,
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.max_turns = max_turns
        self.sessions: dict[str, InvestigationMemory] = {}

    def get_or_create_session(self, session_id: str) -> InvestigationMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = InvestigationMemory(session_id=session_id)
        return self.sessions[session_id]

    async def investigate(self, session_id: str, user_message: str) -> dict:
        """
        Process one turn of an investigation.

        Returns structured response envelope with response text,
        session metadata, tool calls made, and evidence gaps.
        """
        memory = self.get_or_create_session(session_id)
        memory.add_message("user", user_message)

        if memory.turn_count >= self.max_turns:
            return self._build_response(
                memory,
                "Investigation has reached the maximum turn limit. "
                "Producing final summary with available evidence.",
                tool_calls=[],
            )

        context = format_investigation_context(memory)
        tools_schema = self.tools.get_schemas()

        # Agent loop: LLM decides whether to call tools or respond
        tool_calls_this_turn = []
        for _ in range(self.max_tool_calls_per_turn):
            decision = await self._get_llm_decision(context, tools_schema, memory)

            if decision["action"] == AgentAction.CALL_TOOL:
                tool_result = await self._execute_tool(
                    decision["tool_name"], decision["tool_args"], memory,
                )
                tool_calls_this_turn.append({
                    "tool": decision["tool_name"],
                    "args": decision["tool_args"],
                    "success": tool_result["success"],
                })
                context = format_investigation_context(memory)
            else:
                break

        # Generate final response
        response_text = await self._generate_response(context, memory)
        memory.add_message("assistant", response_text)
        memory.increment_turn()

        return self._build_response(memory, response_text, tool_calls_this_turn)

    async def _get_llm_decision(
        self, context: str, tools_schema: list[dict], memory: InvestigationMemory,
    ) -> dict:
        """Ask the LLM: should we call a tool or respond?"""
        decision_prompt = f"""{context}

Based on the investigation state above, decide your next action.
Respond with a JSON object:

If you need more data, call a tool:
{{"action": "call_tool", "tool_name": "<name>", "tool_args": {{...}}, "reasoning": "<why>"}}

If you have enough evidence to respond to the analyst:
{{"action": "respond", "reasoning": "<why>"}}

Available tools:
{json.dumps(tools_schema, indent=2)}

Current analyst request: {memory.get_messages()[-1]['content'] if memory.get_messages() else 'N/A'}

Rules:
- Do NOT re-call a tool with the same arguments already in the investigation state
- If a tool previously failed, note the gap rather than retrying with identical args
- Prefer depth (follow the money trail) over breadth (scatter-shot tool calls)
- If the investigation has sufficient evidence, move to respond
"""
        try:
            raw = await self.llm.complete(
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": decision_prompt}],
                temperature=0.1,
            )
            return self._parse_decision(raw)
        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return {"action": AgentAction.RESPOND, "reasoning": f"LLM error: {e}"}

    def _parse_decision(self, raw_response: str) -> dict:
        """Parse LLM decision JSON. Fallback to RESPOND on parse failure."""
        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]

            data = json.loads(cleaned)
            action_str = data.get("action", "respond")

            if action_str == "call_tool":
                return {
                    "action": AgentAction.CALL_TOOL,
                    "tool_name": data["tool_name"],
                    "tool_args": data.get("tool_args", {}),
                    "reasoning": data.get("reasoning", ""),
                }
            return {"action": AgentAction.RESPOND, "reasoning": data.get("reasoning", "")}
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM decision, defaulting to respond: {e}")
            return {"action": AgentAction.RESPOND, "reasoning": "Parse failure fallback"}

    async def _execute_tool(
        self, tool_name: str, tool_args: dict, memory: InvestigationMemory,
    ) -> dict:
        """
        Execute a tool call with error handling and memory update.

        On success: result stored as Finding in memory.
        On failure: gap flagged in memory, investigation continues.
        """
        logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
        memory.record_tool_call(tool_name, tool_args)

        try:
            result = await self.tools.call(tool_name, tool_args)

            if not self._validate_tool_result(tool_name, result):
                memory.add_evidence_gap(
                    tool_name=tool_name,
                    reason="Tool returned invalid/empty result",
                    args=tool_args,
                )
                return {"success": False, "error": "Invalid result"}

            finding = Finding(
                source_tool=tool_name,
                entity_id=tool_args.get("merchant_id") or tool_args.get("payment_id", "unknown"),
                data=result,
                confidence=self._assess_confidence(result),
            )
            memory.add_finding(finding)

            for key in ["merchant_id", "payer_id", "payment_id"]:
                if key in tool_args:
                    memory.mark_entity_examined(tool_args[key])

            return {"success": True, "result": result}

        except TimeoutError:
            logger.error(f"Tool {tool_name} timed out")
            memory.add_evidence_gap(tool_name=tool_name, reason="Tool call timed out", args=tool_args)
            return {"success": False, "error": "timeout"}

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            memory.add_evidence_gap(tool_name=tool_name, reason=str(e), args=tool_args)
            return {"success": False, "error": str(e)}

    def _validate_tool_result(self, tool_name: str, result: dict) -> bool:
        """Basic validation that tool result contains expected fields."""
        if not result or not isinstance(result, dict):
            return False
        required_fields = {
            "assess_payment_risk": ["decision", "ensemble_score"],
            "detect_merchant_ring": ["rings_detected"],
        }
        fields = required_fields.get(tool_name, [])
        return all(f in result for f in fields)

    def _assess_confidence(self, result: dict) -> float:
        """Map tool result quality to a confidence score."""
        score = result.get("ensemble_score", 0.5)
        signals = result.get("graph_signals", [])
        signal_boost = min(len(signals) * 0.05, 0.2)

        # Conflict detection (Failure Mode 1 fix): if graph shows high-risk
        # structural signals but ensemble score is low, boost confidence
        # to reflect graph signal severity
        has_structural_risk = any(
            keyword in str(signals).upper()
            for keyword in ["SHARED_BANK", "RING_HIGH", "DEVICE_MULE"]
        )
        if has_structural_risk and score < 0.5:
            score = max(score, 0.55)  # Floor at REVIEW threshold

        return min(score + signal_boost, 1.0)

    async def _generate_response(self, context: str, memory: InvestigationMemory) -> str:
        """Generate the agent's response to the analyst."""
        response_prompt = f"""{context}

Generate a clear, structured response for the fraud analyst.

Guidelines:
- Lead with the most critical finding
- Cite specific evidence (transaction IDs, risk scores, graph signals)
- If there are evidence gaps from tool failures, acknowledge them explicitly
- Suggest next investigation steps if the case is not concluded
- Do NOT fabricate data not present in tool results
- Keep the response concise and actionable
"""
        try:
            response = await self.llm.complete(
                system=SYSTEM_PROMPT,
                messages=memory.get_messages() + [
                    {"role": "user", "content": response_prompt}
                ],
                temperature=0.3,
            )
            return response
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return (
                "I encountered an error generating the analysis. "
                f"Current investigation state: {json.dumps(memory.get_summary())}. "
                "Please retry or ask a specific question about the findings so far."
            )

    def _build_response(
        self, memory: InvestigationMemory, response_text: str, tool_calls: list,
    ) -> dict:
        return {
            "response": response_text,
            "session_id": memory.session_id,
            "phase": memory.phase.value,
            "entities_examined": sorted(memory.entities_examined),
            "findings_count": len(memory.findings),
            "tool_calls_made": tool_calls,
            "evidence_gaps": [g.to_dict() for g in memory.evidence_gaps],
            "turn": memory.turn_count,
        }
