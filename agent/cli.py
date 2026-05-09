"""
Interactive CLI for the fraud investigation agent.
Run: python -m agent.cli
"""

import asyncio
import os
import time
import logging

from agent.orchestrator import InvestigationOrchestrator
from agent.tools import ToolRegistry
from agent.llm import LLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def detect_provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    elif os.environ.get("OPENAI_API_KEY"):
        return "openai"
    elif os.environ.get("GROQ_API_KEY"):
        return "groq"
    return None


def try_load_engine():
    try:
        from bgi_trident.mcp.bgi_risk_engine import PaymentRiskEngine
        engine = PaymentRiskEngine(data_dir="src/data")
        engine.load()
        print("  [OK] BGI Trident engine loaded (live mode)")
        return ToolRegistry.from_engine(engine)
    except (ImportError, Exception):
        print("  [MOCK] Using mock tool responses")
        return ToolRegistry.from_mock()


async def main():
    print("\n" + "=" * 60)
    print("  BGI Trident - Fraud Investigation Agent")
    print("  PS3: Minimal Agent, Maximum Reliability")
    print("=" * 60)
    print("\n  Commands: <any text> | /state | /reset | /quit\n")

    tools = try_load_engine()
    provider = detect_provider()
    if not provider:
        print("\n  ERROR: Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY")
        return

    llm = LLMClient(provider=provider)
    orchestrator = InvestigationOrchestrator(llm_client=llm, tool_registry=tools)
    session_id = "cli-001"
    print(f"  Session: {session_id}")
    print(f"  LLM: {provider} / {llm.model}")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n[analyst] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/reset":
            session_id = f"cli-{int(time.time())}"
            print(f"  Session reset: {session_id}")
            continue
        if user_input == "/state":
            memory = orchestrator.get_or_create_session(session_id)
            print(memory.get_state_summary())
            continue

        print("\n  [investigating...]")
        result = await orchestrator.investigate(session_id, user_input)
        print(f"\n[agent] {result['response']}")
        print(f"\n  --- phase: {result['phase']} | findings: {result['findings_count']} | gaps: {len(result['evidence_gaps'])} | turn: {result['turn']} ---")
        for tc in result["tool_calls_made"]:
            print(f"  tool: {tc['tool']} [{'OK' if tc['success'] else 'FAILED'}]")


if __name__ == "__main__":
    asyncio.run(main())
