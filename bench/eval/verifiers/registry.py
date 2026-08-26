"""Pluggable judge-model registry for GateMem.

Swap judges with one CLI flag:

    python bench/scripts/rejudge.py ... --judge gpt41
    python bench/scripts/rejudge.py ... --judge claude_sonnet
    python bench/scripts/rejudge.py ... --judge qwen32b_local

Add a judge by appending one entry to JUDGE_PROFILES, or override any field
inline without touching this file:

    --judge gpt41 --judge_model openai/gpt-4.1-mini --judge_base_url ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base_verifier import (
    LLM_CLAUDE_SONNET,
    LLM_DEEPSEEK,
    LLM_GEMINI_PRO,
    LLM_GPT4O,
    LLM_GPT41,
    LLM_GPT41_MINI,
    LLM_QWEN32B_LOCAL,
)
from .gatemem_verifier import GateMemVerifier

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENAI_BASE = "https://api.openai.com/v1"

# Optional but recommended by OpenRouter for attribution / rate-limit tiering.
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/rzhub/GateMem",
    "X-Title": "GateMem",
}


@dataclass
class JudgeProfile:
    """Everything needed to instantiate a judge."""

    model: str
    base_url: str = OPENROUTER_BASE
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 512
    json_mode: bool = True
    timeout_s: float = 120.0
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_body: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


JUDGE_PROFILES: Dict[str, JudgeProfile] = {
    # ---- OpenRouter (default path) --------------------------------------
    "gpt41": JudgeProfile(
        model=LLM_GPT41,
        extra_headers=dict(OPENROUTER_HEADERS),
        note="GPT-4.1 via OpenRouter. Paper-comparable default.",
    ),
    "gpt41_mini": JudgeProfile(
        model=LLM_GPT41_MINI,
        extra_headers=dict(OPENROUTER_HEADERS),
        note="Cheaper GPT-4.1 tier. Good for pilot runs.",
    ),
    "gpt4o": JudgeProfile(
        model=LLM_GPT4O,
        extra_headers=dict(OPENROUTER_HEADERS),
        note="Judge used in the GateMem paper.",
    ),
    "claude_sonnet": JudgeProfile(
        model=LLM_CLAUDE_SONNET,
        extra_headers=dict(OPENROUTER_HEADERS),
        note="Cross-family judge for agreement analysis.",
    ),
    "gemini_pro": JudgeProfile(
        model=LLM_GEMINI_PRO,
        extra_headers=dict(OPENROUTER_HEADERS),
        note="Cross-family judge for agreement analysis.",
    ),
    "deepseek": JudgeProfile(
        model=LLM_DEEPSEEK,
        extra_headers=dict(OPENROUTER_HEADERS),
        note="Low-cost judge.",
    ),
    # ---- Direct OpenAI (no OpenRouter in between) ------------------------
    "gpt41_openai": JudgeProfile(
        model="gpt-4.1",
        base_url=OPENAI_BASE,
        api_key_env="OPENAI_API_KEY",
        note="GPT-4.1 straight from OpenAI. Model slug has no vendor prefix.",
    ),
    # ---- Local vLLM on your own GPUs -------------------------------------
    "qwen32b_local": JudgeProfile(
        model=LLM_QWEN32B_LOCAL,
        base_url="http://localhost:8000/v1",
        api_key_env="LOCAL_API_KEY",
        json_mode=False,  # vLLM guided decoding is off by default
        note="Your local Qwen2.5-32B server. Reproduces the original judge.",
    ),
    # ---- Offline smoke test, no network, no cost -------------------------
    "mock": JudgeProfile(
        model="mock",
        base_url="mock",
        api_key_env="UNUSED",
        note="Deterministic offline judge. Verifies plumbing before spending money.",
    ),
}


def list_judges() -> str:
    lines = []
    for key, p in JUDGE_PROFILES.items():
        lines.append(f"  {key:<16} {p.model:<34} {p.note}")
    return "\n".join(lines)


def build_verifier(
    judge_key: str,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    timeout_s: Optional[float] = None,
    json_mode: Optional[bool] = None,
    gate_by_action: bool = False,
) -> GateMemVerifier:
    """Instantiate a GateMemVerifier from a profile key, with CLI overrides."""
    if judge_key not in JUDGE_PROFILES:
        raise KeyError(
            f"Unknown judge {judge_key!r}. Available:\n{list_judges()}"
        )
    p = JUDGE_PROFILES[judge_key]
    return GateMemVerifier(
        model=model or p.model,
        base_url=base_url or p.base_url,
        api_key_env=api_key_env or p.api_key_env,
        temperature=p.temperature if temperature is None else temperature,
        max_output_tokens=(
            p.max_output_tokens if max_output_tokens is None else max_output_tokens
        ),
        timeout_s=p.timeout_s if timeout_s is None else timeout_s,
        json_mode=p.json_mode if json_mode is None else json_mode,
        extra_headers=p.extra_headers,
        extra_body=p.extra_body,
        gate_by_action=gate_by_action,
    )
