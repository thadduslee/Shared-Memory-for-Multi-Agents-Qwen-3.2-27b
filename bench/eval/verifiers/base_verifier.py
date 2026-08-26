"""Base class and shared constants for GateMem LLM verifiers (judges).

Design mirrors MiroMind's ``browsecomp_en_verifier.py``:
  - one prompt constant per benchmark,
  - one verifier class that formats the prompt, calls a judge model, and
    parses the raw text into structured labels,
  - retry with exponential backoff around the network call.

Differences from BrowseComp:
  - BrowseComp emits a single binary label (A/B). GateMem emits FOUR tri-state
    labels (action_ok / utility_ok / privacy_leak / deletion_leak) plus notes,
    because a checkpoint is graded on whichever governance axis applies.
  - The judge endpoint is fully pluggable (model + base_url + api key env),
    so the same verifier runs against OpenRouter, OpenAI, a local vLLM server,
    or anything else that speaks OpenAI /chat/completions.

Dependencies: ``requests`` only (already in GateMem's requirements.txt).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from typing import Any, Callable, Dict, Optional

import requests

# --------------------------------------------------------------------------
# Judge label constants
# --------------------------------------------------------------------------
# GateMem labels are tri-state: True / False / None ("not applicable").
# These aliases exist so downstream code reads clearly.
JUDGE_TRUE: bool = True
JUDGE_FALSE: bool = False
JUDGE_NA: None = None

# Parse-status markers written into judge_scores.jsonl.
PARSE_JSON = "json"
PARSE_REGEX = "regex"
PARSE_FAILED = "judge_output_not_parseable_as_structured_json"

# GateMem query types. NOTE the paper/code naming mismatch:
#   paper "Access Control"    -> code "privacy"
#   paper "Active Forgetting" -> code "safety"
QUERY_TYPE_UTILITY = "utility"
QUERY_TYPE_PRIVACY = "privacy"
QUERY_TYPE_SAFETY = "safety"

# Retry policy.
RETRY_MULTIPLIER = 1.0
RETRY_MAX_ATTEMPTS = 4
RETRY_MAX_WAIT_S = 30.0

# --------------------------------------------------------------------------
# Judge model slugs.
# OpenRouter-style "vendor/model" slugs. Swap freely; see registry.py.
# --------------------------------------------------------------------------
LLM_GPT41 = "openai/gpt-4.1"
LLM_GPT41_MINI = "openai/gpt-4.1-mini"
LLM_GPT4O = "openai/gpt-4o"
LLM_CLAUDE_SONNET = "anthropic/claude-sonnet-4"
LLM_GEMINI_PRO = "google/gemini-2.5-pro"
LLM_DEEPSEEK = "deepseek/deepseek-chat"
LLM_QWEN32B_LOCAL = "Qwen/Qwen2.5-32B-Instruct"

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


class VerifierError(RuntimeError):
    """Raised when the judge endpoint fails after all retries."""


class AuthError(RuntimeError):
    """Raised on 401/403. Never retried: a bad key will not fix itself."""


def retry_with_backoff(
    fn: Callable[..., Any],
    *,
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    multiplier: float = RETRY_MULTIPLIER,
) -> Callable[..., Any]:
    """Minimal stand-in for tenacity's retry decorator (no extra dependency)."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        last_exc: Optional[BaseException] = None
        for attempt in range(max_attempts):
            try:
                return fn(*args, **kwargs)
            except VerifierError as exc:
                last_exc = exc
                if attempt == max_attempts - 1:
                    break
                wait = min(RETRY_MAX_WAIT_S, multiplier * (2**attempt))
                time.sleep(wait + random.uniform(0, 0.4 * wait))
        raise last_exc if last_exc else VerifierError("retry wrapper exhausted")

    return wrapper


class BaseVerifier:
    """OpenAI-compatible /chat/completions judge client.

    Any endpoint exposing ``POST {base_url}/chat/completions`` works:
    OpenRouter, OpenAI, Together, DeepSeek, a local vLLM server, etc.

    Set ``base_url="mock"`` for an offline smoke test that makes no network
    calls and returns deterministic labels.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        timeout_s: float = 120.0,
        max_retries: int = RETRY_MAX_ATTEMPTS,
        json_mode: bool = True,
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self.mock = self.base_url == "mock"
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.json_mode = bool(json_mode)
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self.api_key_env = api_key_env

        if self.mock:
            self.api_key = "mock"
        else:
            self.api_key = api_key or os.getenv(api_key_env) or ""
            if not self.api_key:
                raise VerifierError(
                    f"Missing API key: set the {api_key_env} environment variable, "
                    f"e.g.  export {api_key_env}='sk-or-v1-...'"
                )

        # One session per worker thread: requests.Session is not designed for
        # concurrent use, and the judge runs under a ThreadPoolExecutor.
        self._local = threading.local()

    @property
    def _session(self) -> requests.Session:
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            adapter = requests.adapters.HTTPAdapter(max_retries=0)
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            self._local.session = sess
        return sess

    # -- networking --------------------------------------------------------

    def _post_once(self, prompt: str, *, json_mode: bool) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_body)

        try:
            resp = self._session.post(
                url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s
            )
        except Exception as exc:  # network-level failure -> retryable
            raise VerifierError(f"network_error: {exc}") from exc

        if resp.status_code in (401, 403):
            raise AuthError(
                f"Judge endpoint rejected the API key (HTTP {resp.status_code}).\n"
                f"  endpoint : {url}\n"
                f"  key env  : {self.api_key_env}\n"
                f"  key len  : {len(self.api_key)} chars, starts {self.api_key[:9]!r}\n"
                f"  response : {resp.text[:200]}\n"
                f"Check the variable is exported in THIS shell and holds a real key\n"
                f"(an OpenRouter key is 73 chars: 'sk-or-v1-' + 64 hex)."
            )
        if resp.status_code in RETRYABLE_STATUS:
            raise VerifierError(f"http_{resp.status_code}: {resp.text[:400]}")
        if not (200 <= resp.status_code < 300):
            # Non-retryable (bad key, unknown model, malformed request).
            raise RuntimeError(
                f"Judge endpoint error {resp.status_code} for model "
                f"{self.model!r} at {url}: {resp.text[:600]}"
            )
        return resp.json()

    def chat(self, prompt: str) -> Dict[str, Any]:
        """Call the judge. Returns {'text', 'usage', 'model', 'latency_s'}."""
        if self.mock:
            return self._mock_chat(prompt)

        json_mode = self.json_mode
        t0 = time.perf_counter()

        def _call() -> Dict[str, Any]:
            return self._post_once(prompt, json_mode=json_mode)

        try:
            data = retry_with_backoff(_call, max_attempts=self.max_retries)()
        except RuntimeError as exc:
            # Some providers reject response_format for some models. Retry once
            # in plain-text mode before giving up, so a judge swap never dies
            # purely on JSON-mode support.
            if json_mode and "response_format" in str(exc).lower():
                json_mode = False
                data = retry_with_backoff(
                    lambda: self._post_once(prompt, json_mode=False),
                    max_attempts=self.max_retries,
                )()
            else:
                raise

        latency = time.perf_counter() - t0
        return {
            "text": _extract_text(data) or "",
            "usage": _extract_usage(data),
            "model": data.get("model") or self.model,
            "latency_s": latency,
            "json_mode": json_mode,
        }

    def _mock_chat(self, prompt: str) -> Dict[str, Any]:
        """Deterministic offline response for smoke tests (no network, no cost)."""
        qtype = ""
        for line in prompt.splitlines():
            if line.startswith("query_type:"):
                qtype = line.split(":", 1)[1].strip()
                break
        obj = {
            "action_ok": True,
            "utility_ok": True if qtype == QUERY_TYPE_UTILITY else None,
            "privacy_leak": False if qtype == QUERY_TYPE_PRIVACY else None,
            "deletion_leak": False if qtype == QUERY_TYPE_SAFETY else None,
            "notes": "mock judge response",
        }
        return {
            "text": json.dumps(obj),
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "model": "mock",
            "latency_s": 0.0,
            "json_mode": True,
        }

    def preflight(self) -> None:
        """One cheap call to validate credentials and the model slug.

        Raises AuthError on a bad key and RuntimeError on a bad model name,
        both before any checkpoints are dispatched.
        """
        if self.mock:
            return
        saved = self.max_output_tokens
        self.max_output_tokens = 5
        try:
            self._post_once("Reply with the single word OK.", json_mode=False)
        except VerifierError as exc:
            # Transient at preflight time; let the real run retry properly.
            print(f"  [preflight] endpoint slow or busy, continuing: {exc}")
        finally:
            self.max_output_tokens = saved

    # -- interface ---------------------------------------------------------

    def verify(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "json_mode": self.json_mode,
        }


def _extract_text(data: Dict[str, Any]) -> Optional[str]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):  # some providers return content parts
        parts = [c.get("text", "") for c in content if isinstance(c, dict)]
        return "".join(parts).strip() or None
    return None


def _extract_usage(data: Dict[str, Any]) -> Dict[str, int]:
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": int(usage.get("total_tokens") or (inp + out)),
    }