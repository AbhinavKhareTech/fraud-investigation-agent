# Architecture Decision Document

**Fraud Investigation Agent - PS3: Minimal Agent, Maximum Reliability**

---

## 1. Problem Statement

Fraud analysts today investigate flagged transactions through a series of manual queries: pull the transaction, check the merchant profile, look for connected entities, cross-reference with known patterns. Each step requires a different tool, a different mental model, and manual state tracking across tabs and spreadsheets.

This agent automates that investigation flow. It reasons across multiple tool calls, maintains structured evidence across turns, and produces a risk assessment grounded in cited findings.

The constraint is reliability, not capability. A 12-tool agent that hallucinates transaction amounts is worse than a 2-tool agent that cites every finding.

---

## 2. Key Decisions

### 2.1 Custom Orchestrator vs. LangGraph/LangChain

**Decision**: Custom state machine (~250 LOC), no framework dependency.

**Alternatives considered**:
- LangGraph: provides StateGraph, checkpointing, and tool-calling patterns out of the box
- LangChain AgentExecutor: mature ecosystem, many examples

**Why custom**:

First, the PS3 brief explicitly says "minimal agent, maximum reliability." A framework that pulls in 3 transitive dependencies for a 2-tool agent contradicts the brief.

Second, and more importantly: in production BFSI systems, the investigation flow is a regulatory artifact. When a compliance officer asks "why did the agent call this tool in this order with these parameters," the answer cannot be "the framework decided." Every state transition must be explicit, auditable, and explainable. A custom orchestrator gives us that. LangGraph's StateGraph is elegant, but it abstracts away the very thing regulators care about.

Third, debugging. When the agent makes a bad decision at turn 4 of an 8-turn investigation, I need to inspect the exact state at that transition. With a custom orchestrator, the state is a plain Python dict. With a framework, it is wrapped in a runtime I do not control.

**Tradeoff acknowledged**: We lose LangGraph's built-in checkpointing and streaming. For a 2-day work trial, this is acceptable. For production, I would add Redis-backed state persistence (see Section 5).

### 2.2 Structured Memory vs. Chat History

**Decision**: The session memory is a structured evidence ledger, not a message buffer.

A fraud investigation is not a conversation. It is a case file. The memory tracks:
- Entities examined and their risk profiles
- Findings from each tool call with confidence scores
- Evidence gaps from tool failures
- Tool call log (prevents duplicate calls)
- Investigation phase (TRIAGE / DEEP_DIVE / SYNTHESIS)

**Why not just pass the message history to the LLM?**

At 10K investigations/day, the average investigation runs 4-6 turns. Each tool result is 200-500 tokens. By turn 5, raw message history is 3000+ tokens of redundant data. The structured state summary compresses this to ~500 tokens of decision-relevant information.

More critically: the LLM should see *findings*, not *raw tool outputs*. The memory layer extracts and indexes the relevant signals before presenting them to the LLM. This reduces hallucination because the LLM is reasoning over a clean evidence summary, not parsing nested JSON.

### 2.3 Two Tools, Not Three

**Decision**: Expose `assess_payment_risk` and `detect_merchant_ring` as agent tools. Do NOT expose `generate_dispute_evidence` in this build.

The BGI Trident engine has three MCP tools. The third (dispute evidence generation) is a downstream action - it produces a report package for Razorpay's dispute API. Including it would make the agent an investigation-AND-action system, which violates the principle of doing one thing reliably.

The current scope: investigate and assess. The agent produces a risk verdict with evidence. A human analyst reviews the verdict before any action is taken. This is not a limitation - it is a design choice. In regulated environments, the human-in-the-loop at the action boundary is a feature.

### 2.4 LLM: Model-Agnostic Abstraction

**Decision**: Thin wrapper over Anthropic and OpenAI APIs. Swap models without rewriting agent logic.

**Why**: In production, model selection is a cost/quality/latency tradeoff that changes quarterly. Claude Sonnet is excellent for structured reasoning but costs more per token than GPT-4o-mini. The orchestrator should not care which model is behind it.

The abstraction is intentionally thin - a single `complete()` method. No prompt templating engine, no chain-of-thought framework. The system prompt and investigation context are plain strings. This makes it trivial to:
- A/B test models on the eval suite
- Fall back to a cheaper model for simple ALLOW decisions
- Upgrade to a stronger model for ambiguous REVIEW cases

### 2.5 Eval Strategy: Deterministic + LLM-as-Judge Hybrid

**Decision**: Three eval dimensions, scored differently.

| Dimension | Method | Why |
|---|---|---|
| Investigation Completeness | Deterministic | "Did the agent call the right tools?" is binary |
| Factual Accuracy | Rule-based + regex | Hallucination detection: are cited entities in tool results? |
| Graceful Degradation | Deterministic | "Did the agent acknowledge the gap?" is binary |

**Why not full LLM-as-judge?**

For the completeness and degradation dimensions, deterministic scoring is more reliable AND cheaper. LLM-as-judge adds cost and non-determinism for checks that are fundamentally boolean.

LLM-as-judge is reserved for a future dimension: *reasoning quality*. "Did the agent draw the right conclusion from the evidence?" requires judgment. This is documented as a next step (see Section 5).

---

## 3. Production Tensions

### 3.1 Latency vs. Investigation Depth

**The tension**: Each tool call adds 200-500ms (engine scoring) plus 500-1500ms (LLM reasoning about the result). A 4-turn investigation takes 4-8 seconds end-to-end.

**At scale**: 10K investigations/day = ~7 per minute. With 3 concurrent workers, each investigation can take up to 25 seconds without queueing. This is comfortable.

**At 100K/day**: ~70 per minute. The LLM becomes the bottleneck. Options:
1. Batch investigations by risk tier - ALLOW decisions can use a faster/cheaper model
2. Pre-compute graph signals at ingestion time, reducing tool call latency to <50ms
3. Skip the DEEP_DIVE phase for clear-cut ALLOW cases (score < 0.2)

**Cost model**: At current Anthropic pricing (~$3/MTok input, ~$15/MTok output for Sonnet), each investigation costs approximately $0.02-0.05 in LLM calls. At 100K/day, that is $2K-5K/month in LLM costs alone. This is comparable to a single fraud analyst's monthly salary, which frames the ROI clearly.

### 3.2 Hallucination Risk in Regulated Environments

**The risk**: The agent fabricates a transaction amount, a risk score, or a ring connection not present in the tool results. An analyst acts on this fabricated data. A merchant is incorrectly blocked. The platform faces regulatory action.

**Mitigations in this build**:
1. The system prompt explicitly prohibits fabrication (necessary but not sufficient)
2. Tool results are validated before memory insertion (`_validate_tool_result`)
3. The eval suite includes hallucination detection (fabricated entity IDs in response)
4. The structured memory means the LLM reasons over pre-validated findings, not raw data

**What I would add in production**:
1. A post-generation verification step: extract all entity IDs and scores from the response, cross-reference against the memory's findings list. Flag any ungrounded claims.
2. Structured output (JSON mode) for the risk verdict section, so we can programmatically verify every cited data point.
3. Audit trail: every investigation state transition is logged with the exact LLM input/output, tool call/response, and memory delta. This is the compliance requirement.

### 3.3 Graceful Degradation Under Tool Failure

**The tension**: The BGI Trident engine is a distributed system. Graph queries can timeout. The XGBoost scorer can return malformed results. The agent must handle this without derailing.

**Current implementation**:
- Tool calls are wrapped in try/except with timeout
- On failure: an `EvidenceGap` is recorded in memory
- The agent sees the gap in its next context and is instructed to acknowledge it
- Investigation continues with partial evidence

**Eval coverage**: 5 of the 20 eval scenarios inject tool failures (timeouts, connection errors, malformed responses). The degradation score measures whether the agent:
1. Acknowledges the gap (does not pretend it has data it does not have)
2. Still produces a useful response if partial data is available
3. Does not escalate a clean merchant to REVIEW just because a tool failed

---

## 4. What Breaks

### Failure Mode 1: XGBoost/Graph Signal Conflict (FIXED)

**The bug**: The agent over-trusts the ensemble score and ignores conflicting signals between prongs. Example: XGBoost says low risk (velocity and amount patterns are normal) but the graph shows the merchant shares a settlement bank account with 4 known fraudulent merchants. The ensemble score averages these, producing a REVIEW instead of a BLOCK.

**Impact**: False negatives on structurally sophisticated fraud rings that maintain normal transaction patterns.

**Fix applied**: Added conflict detection in `_assess_confidence()`. When graph signals include SHARED_BANK_ACCOUNT or MERCHANT_RING_HIGH but the ensemble score is below 0.5, the confidence is boosted to reflect the graph signal's severity. Before/after eval scores documented in `eval/analysis.md`.

### Failure Mode 2: Context Window Saturation on Long Investigations (NOT FIXED)

**The bug**: After 8-10 turns, the investigation state summary plus conversation history approaches the LLM's effective context window. The LLM starts dropping earlier findings from its reasoning, leading to inconsistent conclusions.

**Impact**: Long investigations produce worse verdicts than short ones, which is backwards.

**What it would take to fix**: Memory summarization - compress completed investigation branches into a findings summary. Estimated effort: 1-2 days. See `eval/analysis.md` for detailed analysis.

---

## 5. What I Would Tackle Next

In priority order:

1. **State persistence** - Redis-backed session storage so investigations survive server restarts
2. **Memory summarization** - Compress long investigation state to prevent context window saturation
3. **LLM-as-judge eval dimension** - Score reasoning quality, not just completeness and accuracy
4. **Multi-entity investigation** - Auto fan-out to connected entities in a ring
5. **Streaming responses** - Token-by-token streaming for CLI and API
6. **Structured output for verdicts** - JSON schema for final risk assessment
7. **Observability** - OpenTelemetry traces per investigation, span per tool call
8. **Cost-based model routing** - Cheaper model for ALLOW, stronger for REVIEW/BLOCK
