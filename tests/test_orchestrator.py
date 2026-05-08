"""Tests for InvestigationOrchestrator (unit tests with mocked LLM)."""

import pytest
import json

from agent.orchestrator import InvestigationOrchestrator
from agent.tools import ToolRegistry


class MockLLMClient:
    """Mock LLM that returns scripted tool-calling decisions."""

    def __init__(self, decisions: list[str]):
        self._decisions = iter(decisions)
        self.model = "mock"

    async def complete(self, system, messages, temperature=0.3):
        try:
            return next(self._decisions)
        except StopIteration:
            return json.dumps({"action": "respond", "reasoning": "done"})


class TestOrchestrator:

    @staticmethod
    def _tool_call(tool_name, args):
        return json.dumps({
            "action": "call_tool", "tool_name": tool_name,
            "tool_args": args, "reasoning": "test",
        })

    @staticmethod
    def _respond():
        return json.dumps({"action": "respond", "reasoning": "sufficient evidence"})

    @pytest.mark.asyncio
    async def test_single_turn_investigation(self):
        decisions = [
            self._tool_call("assess_payment_risk", {"merchant_id": "mrc_00005"}),
            self._respond(),
            "Risk assessment complete: mrc_00005 is BLOCK with score 0.588.",
        ]
        llm = MockLLMClient(decisions)
        tools = ToolRegistry.from_mock()
        orch = InvestigationOrchestrator(llm_client=llm, tool_registry=tools)

        result = await orch.investigate("test-001", "Investigate mrc_00005")

        assert result["session_id"] == "test-001"
        assert result["findings_count"] >= 1
        assert len(result["tool_calls_made"]) >= 1
        assert result["tool_calls_made"][0]["tool"] == "assess_payment_risk"
        assert result["tool_calls_made"][0]["success"] is True

    @pytest.mark.asyncio
    async def test_multi_tool_investigation(self):
        decisions = [
            self._tool_call("assess_payment_risk", {"merchant_id": "mrc_00005"}),
            self._tool_call("detect_merchant_ring", {"merchant_id": "mrc_00005"}),
            self._respond(),
            "Investigation complete. mrc_00005 is part of ring_b.",
        ]
        llm = MockLLMClient(decisions)
        tools = ToolRegistry.from_mock()
        orch = InvestigationOrchestrator(llm_client=llm, tool_registry=tools)

        result = await orch.investigate("test-002", "Investigate mrc_00005")
        assert result["findings_count"] >= 2
        assert len(result["tool_calls_made"]) >= 2

    @pytest.mark.asyncio
    async def test_graceful_tool_failure(self):
        tools = ToolRegistry.from_mock()

        def _fail(**kwargs):
            raise TimeoutError("Graph service timed out")
        tools._tools["detect_merchant_ring"] = _fail

        decisions = [
            self._tool_call("assess_payment_risk", {"merchant_id": "mrc_00005"}),
            self._tool_call("detect_merchant_ring", {"merchant_id": "mrc_00005"}),
            self._respond(),
            "Investigation found risk signals but ring analysis timed out.",
        ]
        llm = MockLLMClient(decisions)
        orch = InvestigationOrchestrator(llm_client=llm, tool_registry=tools)

        result = await orch.investigate("test-003", "Investigate mrc_00005")
        assert result["findings_count"] >= 1
        assert len(result["evidence_gaps"]) >= 1
        assert result["evidence_gaps"][0]["tool_name"] == "detect_merchant_ring"

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        decisions = [self._respond(), "Turn response."] * 25
        llm = MockLLMClient(decisions)
        tools = ToolRegistry.from_mock()
        orch = InvestigationOrchestrator(llm_client=llm, tool_registry=tools, max_turns=3)

        for i in range(4):
            result = await orch.investigate("test-004", f"Turn {i}")

        assert "maximum turn limit" in result["response"]

    @pytest.mark.asyncio
    async def test_parse_decision_fallback(self):
        decisions = ["this is not json at all", "Responding with analysis."]
        llm = MockLLMClient(decisions)
        tools = ToolRegistry.from_mock()
        orch = InvestigationOrchestrator(llm_client=llm, tool_registry=tools)

        result = await orch.investigate("test-005", "Check mrc_00001")
        assert "response" in result

    @pytest.mark.asyncio
    async def test_clean_merchant_investigation(self):
        decisions = [
            self._tool_call("assess_payment_risk", {"merchant_id": "mrc_00001"}),
            self._respond(),
            "Merchant mrc_00001 shows low risk with score 0.12. No fraud signals detected. ALLOW.",
        ]
        llm = MockLLMClient(decisions)
        tools = ToolRegistry.from_mock()
        orch = InvestigationOrchestrator(llm_client=llm, tool_registry=tools)

        result = await orch.investigate("test-006", "Check mrc_00001")
        assert result["findings_count"] >= 1
        assert len(result["evidence_gaps"]) == 0

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        decisions = [self._respond(), "Response A.", self._respond(), "Response B."]
        llm = MockLLMClient(decisions)
        tools = ToolRegistry.from_mock()
        orch = InvestigationOrchestrator(llm_client=llm, tool_registry=tools)

        await orch.investigate("session-a", "Message A")
        await orch.investigate("session-b", "Message B")

        assert "session-a" in orch.sessions
        assert "session-b" in orch.sessions
        assert orch.sessions["session-a"].session_id != orch.sessions["session-b"].session_id
