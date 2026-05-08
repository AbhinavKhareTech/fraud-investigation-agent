"""
LLM Client abstraction - swap between Anthropic and OpenAI without
rewriting agent logic.

Design choice: thin wrapper, not an abstraction framework.
We need complete() and that is it.
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """Model-agnostic LLM client."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str | None = None,
        max_tokens: int = 2048,
    ):
        self.provider = provider
        self.max_tokens = max_tokens

        if provider == "anthropic":
            self.model = model or "claude-sonnet-4-20250514"
            self._init_anthropic()
        elif provider == "openai":
            self.model = model or "gpt-4o"
            self._init_openai()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _init_anthropic(self):
        import anthropic
        self.client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

    def _init_openai(self):
        import openai
        self.client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
    ) -> str:
        """Single completion call. Returns text content."""
        if self.provider == "anthropic":
            return await self._complete_anthropic(system, messages, temperature)
        return await self._complete_openai(system, messages, temperature)

    async def _complete_anthropic(
        self, system: str, messages: list[dict], temperature: float
    ) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            temperature=temperature,
        )
        return response.content[0].text

    async def _complete_openai(
        self, system: str, messages: list[dict], temperature: float
    ) -> str:
        full_messages = [{"role": "system", "content": system}] + messages
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=full_messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
