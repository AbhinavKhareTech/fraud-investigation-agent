"""
Eval Pipeline - automated scoring across 20 scenarios.

Measures three dimensions:
1. Investigation Completeness: did the agent call the right tools?
2. Factual Accuracy: did the agent hallucinate data not in tool results?
3. Graceful Degradation: does the agent handle tool failures properly?

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --category true_positive
"""

import asyncio
import json
import re
import logging
import argparse
import time
from pathlib import Path
from typing import Any

from agent.orchestrator import InvestigationOrchestrator
from agent.tools import ToolRegistry, DEFAULT_MOCK_DATA
from agent.llm import LLMClient

logger = logging.getLogger(__name__)

SCENARIOS_PATH = Path(__file__).parent / "scenarios.json"


class EvalRunner:
    """Run eval scenarios and score agent responses."""

    def __init__(self, orchestrator: InvestigationOrchestrator):
        self.orchestrator = orchestrator
        self._original_tools = orchestrator.tools

    async def run_all(self, category: str | None = None) -> dict[str, Any]:
        with open(SCENARIOS_PATH) as f:
            data = json.load(f)

        scenarios = data["scenarios"]
        if category:
            scenarios = [s for s in scenarios if s["category"] == category]

        results = []
        for scenario in scenarios:
            self.orchestrator.tools = self._original_tools
            result = await self.run_scenario(scenario)
            results.append(result)
            scores_str = ", ".join(f"{k}={v:.1f}" for k, v in result["scores"].items())
            print(f"  {result['id']}: {result['verdict']} ({scores_str})")

        summary = self._compute_summary(results)
        return {"results": results, "summary": summary}

    async def run_scenario(self, scenario: dict) -> dict:
        scenario_id = scenario["id"]
        category = scenario["category"]
        expected = scenario["expected"]
        session_id = f"eval-{scenario_id}-{int(time.time())}"

        if "inject_failure" in scenario:
            self.orchestrator.tools = self._create_failing_tools(scenario["inject_failure"])

        try:
            result = await self.orchestrator.investigate(session_id, scenario["input"])
        except Exception as e:
            return {
                "id": scenario_id, "category": category, "verdict": "ERROR",
                "error": str(e), "scores": {"completeness": 0, "accuracy": 0},
            }

        scores = self._score_response(result, expected, category)
        verdict = "PASS" if all(v >= 0.5 for v in scores.values()) else "FAIL"

        return {
            "id": scenario_id, "category": category, "verdict": verdict,
            "scores": scores,
            "response_excerpt": result["response"][:300],
            "tools_called": result["tool_calls_made"],
            "evidence_gaps": result["evidence_gaps"],
        }

    def _score_response(self, result: dict, expected: dict, category: str) -> dict[str, float]:
        scores = {}
        min_tools = expected.get("min_tools_called", 1)
        tools_called = len(result["tool_calls_made"])
        scores["completeness"] = min(tools_called / max(min_tools, 1), 1.0)

        response = result["response"].lower()
        scores["accuracy"] = self._check_accuracy(response, expected, result)

        if category == "degraded":
            scores["degradation"] = self._check_degradation(result, expected)

        return scores

    def _check_accuracy(self, response: str, expected: dict, result: dict) -> float:
        score = 1.0
        if "expected_decision" in expected:
            if expected.get("should_detect", False):
                risk_words = ["block", "review", "risk", "suspicious", "flag", "fraud", "alert"]
                if not any(w in response for w in risk_words):
                    score -= 0.4
            else:
                safe_words = ["allow", "clear", "clean", "low risk", "no significant", "no fraud"]
                if not any(w in response for w in safe_words):
                    score -= 0.4

        # Hallucination check: fabricated merchant IDs
        mentioned = set(re.findall(r"mrc_\d{5}", response))
        known = set()
        for tc in result["tool_calls_made"]:
            args = tc.get("args", {})
            if "merchant_id" in args:
                known.add(args["merchant_id"])
        for key, val in DEFAULT_MOCK_DATA.items():
            members = val.get("ring_members", [])
            for m in members:
                if isinstance(m, dict) and "id" in m:
                    known.add(m["id"])
            detected = val.get("rings_detected", [])
            for r in detected:
                if isinstance(r, dict) and "members" in r:
                    known.update(r["members"])

        fabricated = mentioned - known
        if fabricated:
            score -= 0.3 * len(fabricated)

        return max(score, 0.0)

    def _check_degradation(self, result: dict, expected: dict) -> float:
        score = 1.0
        if expected.get("should_acknowledge_gap", False):
            has_gaps = len(result["evidence_gaps"]) > 0
            response = result["response"].lower()
            mentions_gap = any(
                w in response for w in [
                    "gap", "unavailable", "failed", "could not", "unable",
                    "missing", "error", "timeout", "timed out",
                ]
            )
            if not has_gaps and not mentions_gap:
                score -= 0.5

        if expected.get("should_still_report", True):
            if len(result["response"]) < 50:
                score -= 0.3

        return max(score, 0.0)

    def _create_failing_tools(self, failure_config: dict) -> ToolRegistry:
        registry = ToolRegistry.from_mock()
        failing_tool = failure_config["tool"]
        error_msg = failure_config["message"]
        error_class = {
            "TimeoutError": TimeoutError,
            "ConnectionError": ConnectionError,
            "ValueError": ValueError,
            "Exception": Exception,
        }.get(failure_config["error"], Exception)

        def _failing_tool(**kwargs):
            raise error_class(error_msg)

        registry._tools[failing_tool] = _failing_tool
        return registry

    def _compute_summary(self, results: list[dict]) -> dict:
        by_category: dict[str, dict] = {}
        for r in results:
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = {"pass": 0, "fail": 0, "error": 0, "scores": []}
            verdict_key = r["verdict"].lower()
            by_category[cat][verdict_key] = by_category[cat].get(verdict_key, 0) + 1
            by_category[cat]["scores"].append(r.get("scores", {}))

        summary: dict[str, Any] = {"total": len(results), "by_category": {}}
        total_pass = 0
        for cat, data in by_category.items():
            pass_count = data["pass"]
            total = pass_count + data["fail"] + data["error"]
            total_pass += pass_count
            avg_scores: dict[str, float] = {}
            for key in ["completeness", "accuracy", "degradation"]:
                vals = [s.get(key) for s in data["scores"] if key in s and s[key] is not None]
                if vals:
                    avg_scores[key] = sum(vals) / len(vals)
            summary["by_category"][cat] = {
                "pass_rate": pass_count / total if total else 0,
                "total": total, "avg_scores": avg_scores,
            }
        summary["overall_pass_rate"] = total_pass / len(results) if results else 0
        return summary


async def main():
    parser = argparse.ArgumentParser(description="Run fraud agent eval suite")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    import os
    tools = ToolRegistry.from_mock()
    provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
    llm = LLMClient(provider=provider)
    orchestrator = InvestigationOrchestrator(llm_client=llm, tool_registry=tools)

    runner = EvalRunner(orchestrator)
    label = f" (category: {args.category})" if args.category else ""
    print(f"\nRunning eval suite{label}")
    print("-" * 60)

    results = await runner.run_all(category=args.category)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Overall pass rate: {results['summary']['overall_pass_rate']:.1%}")
    for cat, data in results["summary"]["by_category"].items():
        print(f"  {cat}: {data['pass_rate']:.1%} ({data['total']} scenarios)")
        for metric, val in data["avg_scores"].items():
            print(f"    avg {metric}: {val:.2f}")

    output_path = args.output or f"eval/results/run_{int(time.time())}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
