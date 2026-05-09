"""Eval Pipeline - 20 scenarios, 3 scoring dimensions."""

import asyncio
import json
import os
import re
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict

from agent.orchestrator import InvestigationOrchestrator
from agent.tools import ToolRegistry, DEFAULT_MOCK_DATA
from agent.llm import LLMClient

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
SCENARIOS_PATH = Path(__file__).parent / "scenarios.json"


def detect_provider():
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def check_accuracy(response, expected, result):
    score = 1.0
    if "expected_decision" in expected:
        if expected.get("should_detect", False):
            if not any(w in response for w in ["block","review","risk","suspicious","flag","fraud","alert"]):
                score -= 0.4
        else:
            if not any(w in response for w in ["allow","clear","clean","low risk","no significant","no fraud"]):
                score -= 0.4
    mentioned = set(re.findall(r"mrc_\d{5}", response))
    known = set()
    for tc in result.get("tool_calls_made", []):
        mid = tc.get("args", {}).get("merchant_id")
        if mid:
            known.add(mid)
    for val in DEFAULT_MOCK_DATA.values():
        for m in val.get("ring_members", []):
            if isinstance(m, dict) and "id" in m:
                known.add(m["id"])
        for ring in val.get("rings_detected", []):
            if isinstance(ring, dict) and "members" in ring:
                known.update(ring["members"])
    fabricated = mentioned - known
    if fabricated:
        score -= 0.3 * len(fabricated)
    return max(score, 0.0)


def check_degradation(result, expected):
    score = 1.0
    if expected.get("should_acknowledge_gap", False):
        has_gaps = len(result.get("evidence_gaps", [])) > 0
        resp = result.get("response", "").lower()
        mentions = any(w in resp for w in ["gap","unavailable","failed","could not","unable","missing","error","timeout","timed out"])
        if not has_gaps and not mentions:
            score -= 0.5
    if expected.get("should_still_report", True):
        if len(result.get("response", "")) < 50:
            score -= 0.3
    return max(score, 0.0)


def create_failing_tools(cfg):
    registry = ToolRegistry.from_mock()
    ecls = {"TimeoutError": TimeoutError, "ConnectionError": ConnectionError,
            "ValueError": ValueError, "Exception": Exception}.get(cfg["error"], Exception)
    def _fail(**kw):
        raise ecls(cfg["message"])
    registry._tools[cfg["tool"]] = _fail
    return registry


async def run_scenario(orchestrator, scenario, original_tools):
    sid = scenario["id"]
    cat = scenario["category"]
    expected = scenario["expected"]
    session_id = f"eval-{sid}-{int(time.time())}"

    if "inject_failure" in scenario:
        orchestrator.tools = create_failing_tools(scenario["inject_failure"])
    else:
        orchestrator.tools = original_tools

    try:
        result = await orchestrator.investigate(session_id, scenario["input"])
    except Exception as e:
        return {"id": sid, "category": cat, "verdict": "ERROR", "error": str(e),
                "scores": {"completeness": 0, "accuracy": 0}, "response_excerpt": "",
                "tools_called": [], "evidence_gaps": []}

    scores = {}
    min_tools = expected.get("min_tools_called", 1)
    scores["completeness"] = min(len(result["tool_calls_made"]) / max(min_tools, 1), 1.0)
    scores["accuracy"] = check_accuracy(result["response"].lower(), expected, result)
    if cat == "degraded":
        scores["degradation"] = check_degradation(result, expected)

    verdict = "PASS" if all(v >= 0.5 for v in scores.values()) else "FAIL"
    return {"id": sid, "category": cat, "verdict": verdict, "scores": scores,
            "response_excerpt": result["response"][:300],
            "tools_called": result["tool_calls_made"],
            "evidence_gaps": result["evidence_gaps"]}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    provider = detect_provider()
    if not provider:
        print("ERROR: Set GEMINI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY")
        return

    llm = LLMClient(provider=provider)
    tools = ToolRegistry.from_mock()
    orchestrator = InvestigationOrchestrator(llm_client=llm, tool_registry=tools)
    original_tools = tools

    print(f"\nProvider: {provider} / {llm.model}")
    print("-" * 60)

    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)["scenarios"]
    if args.category:
        scenarios = [s for s in scenarios if s["category"] == args.category]

    results = []
    for i, scenario in enumerate(scenarios):
        print(f"[{i+1:02d}/{len(scenarios)}] {scenario['id']} ({scenario['category']}): {scenario['input'][:50]}...")
        result = await run_scenario(orchestrator, scenario, original_tools)
        results.append(result)
        mark = "PASS" if result["verdict"] == "PASS" else "FAIL"
        scores_str = " ".join(f"{k}={v:.2f}" for k,v in result["scores"].items())
        print(f"         {'✅' if mark=='PASS' else '❌'} {scores_str}")
        if i < len(scenarios) - 1:
            await asyncio.sleep(4)

    cats = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0, "scores": []})
    for r in results:
        cats[r["category"]][r["verdict"].lower()] += 1
        cats[r["category"]]["scores"].append(r["scores"])

    summary = {"total": len(results), "by_category": {}}
    total_pass = 0
    for cat, d in cats.items():
        pc = d["pass"]
        tot = pc + d["fail"] + d["error"]
        total_pass += pc
        avgs = {}
        for k in ["completeness", "accuracy", "degradation"]:
            vals = [s[k] for s in d["scores"] if k in s]
            if vals:
                avgs[k] = round(sum(vals)/len(vals), 2)
        summary["by_category"][cat] = {"pass_rate": round(pc/tot,2) if tot else 0,
                                       "passed": pc, "total": tot, "avg_scores": avgs}
    summary["overall_pass_rate"] = round(total_pass/len(results),2) if results else 0

    print("\n" + "=" * 60)
    print(f"Overall: {summary['overall_pass_rate']:.0%} ({total_pass}/{len(results)})")
    for cat in ["true_positive","true_negative","ambiguous","degraded"]:
        if cat in summary["by_category"]:
            d = summary["by_category"][cat]
            avgs = ", ".join(f"{k}={v:.2f}" for k,v in d["avg_scores"].items())
            print(f"  {cat}: {d['pass_rate']:.0%} ({d['passed']}/{d['total']}) -- {avgs}")

    output_path = args.output or f"eval/results/run_{int(time.time())}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
