# Failure Analysis

**PS3 Deliverable: Document at least 2 failure modes, fix 1 with before/after eval scores.**

---

## Failure Mode 1: XGBoost/Graph Signal Conflict (FIXED)

### The Bug

The agent relied on the `ensemble_score` from the BGI Trident engine without checking for conflicts between the XGBoost prong and the graph-native prong. When XGBoost assigned a low risk score (normal velocity and amount patterns) but the graph showed high-risk structural signals (shared bank accounts, ring membership), the ensemble averaged these down to a REVIEW instead of a BLOCK.

### Impact

False negatives on structurally sophisticated fraud rings that maintain normal transaction patterns to evade velocity-based detection.

### Root Cause

The `_assess_confidence()` method in the orchestrator used the ensemble score directly without examining the individual graph signals for structural risk indicators.

### Fix

Added conflict detection in `orchestrator.py::_assess_confidence()`:

```python
has_structural_risk = any(
    keyword in str(signals).upper()
    for keyword in ["SHARED_BANK", "RING_HIGH", "DEVICE_MULE"]
)
if has_structural_risk and score < 0.5:
    score = max(score, 0.55)  # Floor at REVIEW threshold
```

### Before/After Eval Scores

| Scenario | Before (accuracy) | After (accuracy) |
|---|---|---|
| tp-001 (mrc_00005, Ring B) | - | - |
| tp-004 (mrc_00006, Ring B peripheral) | - | - |
| am-003 (mrc_00009, ambiguous) | - | - |

*(Fill with actual scores after running eval suite)*

---

## Failure Mode 2: Context Window Saturation (DOCUMENTED, NOT FIXED)

### The Bug

After 8-10 turns of investigation, the state summary + conversation history approaches the LLM's effective context window. The LLM begins dropping earlier findings from its reasoning, producing conclusions that contradict evidence from earlier turns.

### Impact

Long, complex investigations (multi-merchant ring analysis spanning 10+ turns) produce worse verdicts than shorter investigations. More evidence should improve decisions, not degrade them.

### Root Cause

The `get_state_summary()` method dumps ALL findings, ALL tool calls, and ALL messages without compression. At turn 8 with 2 tools called per turn, this is ~16 tool results plus ~16 messages, totaling 3000-5000 tokens of context before the LLM even starts reasoning.

### What It Would Take to Fix

**Option A: Rule-based compression.** After each phase transition, compress completed findings into a 2-3 sentence summary. Drop raw tool results, keep entity risk scores and decisions. Estimated effort: 4-6 hours. Risk: loses nuance that matters in ambiguous cases.

**Option B: LLM-based summarization.** Separate (cheap model) LLM call to summarize investigation state at phase transitions. Estimated effort: 1 day. Risk: adds latency, cost, and a potential failure point.

### Why Not Fixed

Both options require evaluation to ensure compression does not introduce false negatives. The eval suite would need new 10+ turn scenarios. At least a full day beyond the 2-day scope.

---

## Observed But Not Investigated

- **Duplicate tool call prevention**: `has_called()` uses exact argument matching. LLM sometimes adds trailing whitespace to merchant IDs, bypassing the check. A normalized comparison would fix this.
- **Response length variance**: Agent produces 3-sentence responses for some scenarios and 15 sentences for similar ones. Prompt tuning issue, not architectural.
