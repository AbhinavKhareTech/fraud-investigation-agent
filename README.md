# Fraud Investigation Agent

# Fraud Investigation Agent

**An autonomous, auditable AI agent for merchant fraud investigations.**

This agent helps fraud analysts by autonomously investigating suspicious merchants and transactions using a custom lightweight orchestrator. It performs risk assessment, detects coordinated merchant rings, maintains a structured evidence ledger, and delivers clear, cited verdicts with confidence scores.

Built as a focused, production-minded solution: **no heavy frameworks** (LangGraph/LangChain avoided for maximum transparency and auditability).

### Key Features

- **Multi-turn autonomous investigation** with controlled tool usage
- **Structured Evidence Memory** — tracks findings, gaps, and confidence (not just chat history)
- **Merchant Ring Detection** powered by graph signals
- **Risk Scoring** using XGBoost + structural features
- **Graceful degradation** and explicit uncertainty handling
- **Model agnostic** (Claude, GPT-4o, etc.)
- **Full evaluation suite** with 20+ test scenarios

### Tech Stack

- Python 3.11+
- Custom async orchestrator (~250 LOC)
- FastAPI + CLI interface
- Pydantic v2 structured outputs
- Trident Fraud Engine (graph + ML backend)

---

**Perfect for fintech, payment companies, and fraud teams** looking for reliable agentic systems that regulators and analysts can actually trust.

**A multi-turn conversational agent for payment fraud investigation, built on the [BGI Trident](https://github.com/AbhinavKhareTech/trident-payment-fraud) fraud detection engine.**

> **Work Trial Context**: This repo is a 2-day deliverable for Meraki Labs PS3 (Minimal Agent, Maximum Reliability). The BGI Trident engine (graph-native fraud scoring, synthetic dataset, MCP tools) is pre-existing work. Everything in this repo - the agentic investigation layer, session memory, eval framework, and production analysis - is the work trial scope.

---

## What This Does

An analyst says: *"Investigate merchant mrc_00005."*

The agent runs a multi-turn investigation:

1. **Assess risk** - calls `assess_payment_risk` on recent transactions, surfaces risk score and graph signals
2. **Detect rings** - calls `detect_merchant_ring`, maps shared bank accounts and coordinated payer pools
3. **Synthesize** - accumulates evidence in structured session memory, produces a risk assessment with citations

The agent maintains a structured investigation state across turns - not chat history, but an evidence ledger: entities examined, hypotheses formed, tools called, findings accumulated.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/AbhinavKhareTech/fraud-investigation-agent.git
cd fraud-investigation-agent
pip install -e ".[dev]"

# 2. Set your LLM API key
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...

# 3. Run the investigation agent
python -m agent.cli

# Or start the API server
uvicorn api.server:app --reload
```

### Run the Eval Suite

```bash
python -m eval.run_eval
# Results written to eval/results/
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Architecture

```
Analyst (CLI / API)
    |
    v
+---------------------+
| Investigation       |    Structured session memory:
| Orchestrator        |    - entities examined
|                     |    - evidence collected
| (Custom state       |    - hypotheses + confidence
|  machine, ~250 LOC. |    - tools called + results
|  No LangGraph.)     |    - risk score progression
+---------------------+
    |           |
    v           v
+--------+  +--------+
| Tool 1 |  | Tool 2 |
| Risk   |  | Ring   |
| Assess |  | Detect |
+--------+  +--------+
    |           |
    v           v
+---------------------+
| BGI Trident         |
| PaymentRiskEngine   |
| (graph-native       |
|  fraud scoring)     |
+---------------------+
```

---

## Project Structure

```
fraud-investigation-agent/
├── README.md
├── ARCHITECTURE.md              # Decision doc (PS3 deliverable)
├── SCOPE.md                     # What I built, what I cut, why
├── pyproject.toml
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py          # Investigation state machine
│   ├── memory.py                # Structured investigation state
│   ├── tools.py                 # Wraps BGI Trident engine
│   ├── prompts.py               # System prompt + guardrails
│   ├── llm.py                   # LLM client abstraction
│   └── cli.py                   # Interactive CLI for demo
├── api/
│   ├── __init__.py
│   └── server.py                # FastAPI /investigate endpoint
├── eval/
│   ├── __init__.py
│   ├── scenarios.json           # 20 test cases (4 categories)
│   ├── run_eval.py              # Automated scoring pipeline
│   ├── results/                 # Before/after scores
│   └── analysis.md              # Failure modes (PS3 deliverable)
├── tests/
│   ├── test_orchestrator.py
│   ├── test_memory.py
│   └── test_tools.py
├── Makefile
└── docker-compose.yml
```

---

## Key Design Decisions

See [ARCHITECTURE.md](ARCHITECTURE.md) for full decision doc. Summary:

| Decision | Choice | Why |
|---|---|---|
| Framework | Custom orchestrator (~250 LOC) | PS3 says "minimal agent." LangGraph adds 3 deps for a 2-tool agent. |
| Session memory | Structured dict, not chat history | Investigation state != conversation log. Evidence ledger matters. |
| LLM | Model-agnostic (Anthropic/OpenAI) | Swap without rewriting agent logic. |
| Eval | Deterministic + LLM-as-judge hybrid | Task completion is binary. Reasoning quality needs judgment. |

---

## Eval Results

| Category | Scenarios | Pass Rate | Avg Completeness | Avg Accuracy |
|---|---|---|---|---|
| True Positive | 5 | 100% | 0.91 | 0.83 |
| True Negative | 5 | 0%* | 0.07 | 0.35 |
| Ambiguous | 5 | 0%* | 0.40 | 0.67 |
| Degraded | 5 | 100% | 1.00 | 0.75 |

*Rate-limited on free tier — true_negative and ambiguous pending full re-run. True positive and degraded categories fully validated.*

---

## Related

- [trident-payment-fraud](https://github.com/AbhinavKhareTech/trident-payment-fraud) - The BGI Trident fraud detection engine this agent is built on
- [trident-consumption-graph](https://github.com/AbhinavKhareTech/trident-consumption-graph) - BGI Trident applied to Swiggy's consumption domain

## Author

**Abhinav Khare** - Cofounder and CTO, AhinsaAI

## License

MIT
