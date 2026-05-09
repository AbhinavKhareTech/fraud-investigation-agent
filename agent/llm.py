"""LLM Client abstraction - Anthropic, OpenAI, Groq, Deepseek, Qwen, Gemini."""

from __future__ import annotations
import os
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, provider="anthropic", model=None, max_tokens=2048):
        self.provider = provider
        self.max_tokens = max_tokens

        if provider == "anthropic":
            self.model = model or "claude-sonnet-4-20250514"
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        elif provider == "openai":
            self.model = model or "gpt-4o"
            import openai
            self.client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif provider == "groq":
            self.model = model or "llama-3.3-70b-versatile"
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=os.environ.get("GROQ_API_KEY"),  # set via env var
                base_url="https://api.groq.com/openai/v1",
            )
        elif provider == "deepseek":
            self.model = model or "deepseek-chat"
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
            )
        elif provider == "gemini":
            self.model = model or "gemini-2.0-flash"
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=os.environ.get("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def complete(self, system, messages, temperature=0.3):
        if self.provider == "anthropic":
            resp = await self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                system=system, messages=messages, temperature=temperature,
            )
            return resp.content[0].text
        else:
            full = [{"role": "system", "content": system}] + messages
            resp = await self.client.chat.completions.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=full, temperature=temperature,
            )
            return resp.choices[0].message.content
