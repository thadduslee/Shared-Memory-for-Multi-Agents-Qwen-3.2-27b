from __future__ import annotations

from ..types import LLMConfig
from .openai_compatible import OpenAICompatibleChatClient


class OpenRouterClient(OpenAICompatibleChatClient):
    """OpenRouter OpenAI-compatible chat-completions client.

    Model names use OpenRouter's ``vendor/model`` slugs, e.g. ``openai/gpt-4.1``,
    ``anthropic/claude-sonnet-4``, ``google/gemini-2.5-pro``.

    Override --api_base / --judge_api_base to point at a different gateway.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(
            config,
            provider_name="openrouter",
            default_api_base="https://openrouter.ai/api/v1",
            default_api_key_env="OPENROUTER_API_KEY",
            default_model="openai/gpt-4.1",
            extra_headers={
                # Optional attribution headers recommended by OpenRouter.
                "HTTP-Referer": "https://github.com/rzhub/GateMem",
                "X-Title": "GateMem",
            },
        )
