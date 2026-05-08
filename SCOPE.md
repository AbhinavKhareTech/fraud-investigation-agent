# Scope Document

**What I built, what I cut, and why.**

---

## Time Budget

2 focused days, as specified by the work trial.

---

## What is Pre-existing (NOT Part of the 2-Day Scope)

The [trident-payment-fraud](https://github.com/AbhinavKhareTech/trident-payment-fraud) repository contains:

- BGI Trident PaymentRiskEngine: three-prong fraud scoring (XGBoost + graph-native + ensemble)
- Synthetic dataset: 2,175 transactions with 3 planted fraud rings (refund cycling, shared settlement bank, card testing burst)
- 3 MCP tools: `assess_payment_risk`, `detect_merchant_ring`, `generate_dispute_evidence`
- 19 passing tests
- Razorpay MCP integration layer

This engine is the "backend intelligence" the agent calls. I built it as part of a separate project. It is included here as a pip dependency.

---

## What I Built in 2 Days

### Day 1: Agent Core

| Component | What | Lines |
|---|---|---|
| `agent/orchestrator.py` | Custom state-machine agent loop with tool calling, retry logic, and graceful failure handling | ~250 |
| `agent/memory.py` | Structured investigation memory: evidence ledger with phases, findings, gaps, entity tracking | ~200 |
| `agent/tools.py` | Tool registry wrapping BGI Trident engine. Live mode + mock mode for eval. | ~200 |
| `agent/prompts.py` | System prompt with 3 critical guardrails (no fabrication, cite sources, acknowledge gaps) | ~80 |
| `agent/llm.py` | Model-agnostic LLM client (Anthropic/OpenAI) | ~80 |
| `agent/cli.py` | Interactive CLI for live demo | ~90 |

### Day 2: Eval + API + Docs

| Component | What | Lines |
|---|---|---|
| `eval/scenarios.json` | 20 test cases across 4 categories (true positive, true negative, ambiguous, degraded) | ~250 |
| `eval/run_eval.py` | Automated eval pipeline scoring completeness, accuracy, and graceful degradation | ~200 |
| `api/server.py` | FastAPI server with /investigate, /sessions endpoints | ~80 |
| `ARCHITECTURE.md` | Decision doc covering framework choice, memory design, production tensions, failure modes | ~200 |
| `SCOPE.md` | This document | ~100 |
| `tests/` | Unit tests for memory, tools, and orchestrator | ~200 |

**Total new code**: ~1,900 lines across 16 files.

---

## What I Cut and Why

### Cut: `generate_dispute_evidence` as an Agent Tool

The BGI Trident engine has a third tool for generating dispute evidence packages. I excluded it because:
- It is a downstream *action*, not an investigation step
- Including it would make the agent both investigator and actor, violating separation of concerns
- In regulated environments, the investigation-to-action boundary should have human review

### Cut: Frontend / Dashboard

A React dashboard showing investigation progress would be compelling in a demo but is not what PS3 asks for. The CLI and FastAPI endpoints are sufficient to demonstrate the agent's capabilities. The 20-30 minute presentation will use the CLI.

### Cut: LangGraph / LangChain

Considered and rejected. A 2-tool agent does not need a framework. See ARCHITECTURE.md Section 2.1 for the full rationale.

### Cut: Persistent State (Redis/Postgres)

Investigation sessions are in-memory only. For a work trial demo, this is fine. For production, you would need Redis-backed sessions so investigations survive restarts. Estimated effort: half a day.

### Cut: Streaming Responses

The agent blocks until the full response is generated. Streaming would improve UX in the CLI and API but adds complexity (async generators, partial response handling) that is not core to the PS3 evaluation criteria.

### Cut: Multi-Model Routing

Using a cheaper model for clear ALLOW decisions and a stronger model for REVIEW/BLOCK would reduce costs at scale. Documented in ARCHITECTURE.md as a next step but not implemented.

---

## What I Would Build Next

See ARCHITECTURE.md Section 5 for the prioritized list. The top three:

1. State persistence (Redis)
2. Memory summarization for long investigations
3. LLM-as-judge eval dimension for reasoning quality
