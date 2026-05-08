"""
Interactive CLI for the fraud investigation agent.
This is the primary demo interface for the work trial presentation.
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
logger = logging.getLogger(__name__)


def print_header():
    print("\n" + "=" * 60)
    print("  BGI Trident - Fraud Investigation Agent")
    print("  PS3: Minimal Agent, Maximum Reliability")
    print("=" * 60)
    print()
    print("  Commands:")
    print("    <any text>     - Investigate (e.g., 'Investigate mrc_00005')")
    print("    /state         - Show investigation state")
    print("    /reset         - Reset session")
    print("    /quit          - Exit")
    print()


def try_load_engine():
    """Attempt to load live BGI Trident engine, fall back to mock."""
    try:
        from bgi_trident.mcp.bgi_risk_engine import PaymentRiskEngine
        engine = PaymentRiskEngine(data_dir="src/data")
        engine.load()
        print("  [OK] BGI Trident engine loaded (live mode)")
        return ToolRegistry.from_engine(engine)
    except (ImportError, Exception) as e:
        logger.info(f"BGI Trident not available ({e}), using mock tools")
        print("  [MOCK] Using mock tool responses")
        print("         Install trident-payment-fraud for live mode")
        return ToolRegistry.from_mock()


async def main():
    print_header()

    tools = try_load_engine()

    if os.environ.get("ANTHROPIC_API_KEY"):
        provider = "anthropic"
    elif os.environ.get("OPENAI_API_KEY"):
        provider = "openai"
    else:
        print("\n  ERROR: Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        print("  Example: export ANTHROPIC_API_KEY=sk-ant-...")
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
        print(f"\n  --- phase: {result['phase']} | "
              f"findings: {result['findings_count']} | "
              f"gaps: {len(result['evidence_gaps'])} | "
              f"turn: {result['turn']} ---")

        if result["tool_calls_made"]:
            for tc in result["tool_calls_made"]:
                status = "OK" if tc["success"] else "FAILED"
                print(f"  tool: {tc['tool']} [{status}]")


if __name__ == "__main__":
    asyncio.run(main())
