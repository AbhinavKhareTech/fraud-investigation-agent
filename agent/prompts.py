"""Prompts for the fraud investigation agent."""

SYSTEM_PROMPT = """You are an expert fraud investigator for a payment company.
Be extremely careful with high-risk patterns.
Always respond in the exact JSON format requested."""


def format_investigation_context(memory) -> str:
    """Format investigation memory state for the LLM."""
    return memory.get_state_summary()
