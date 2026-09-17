"""Async, bounded, retrying HTTP client for OpenAI-compatible chat endpoints.

All three serving routes -- OpenRouter (Architect/Developer/Critic), the local
6-GPU vLLM cluster (Medical Evaluator) and OpenAI (Judge) -- speak the same
`POST {base_url}/chat/completions` wire format, so they share one client and
differ only in `RouteConfig`.

Three things here are load-bearing:

1.  ONE `httpx.AsyncClient` for the whole process, with an explicit connection
    pool.  Creating a client per request is the classic way to exhaust ephemeral
    ports under fan-out.
2.  A PER-ROUTE SEMAPHORE.  The vLLM cluster has a fixed number of GPU workers;
    sending it 579 concurrent requests does not make it faster, it makes it
    time out.  The semaphore is the backpressure.
3.  RETRY WITH FULL JITTER on 429/5xx.  Synchronised retries from a fan-out are
    a self-inflicted DDoS; the jitter decorrelates them.

Nothing in this module blocks the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

import config

log = logging.getLogger("orchestrator.llm")

# Status codes worth retrying: rate limit, and the transient server-side family.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# 402 IS TWO DIFFERENT ERRORS WEARING ONE STATUS CODE, and telling them apart is
# worth a whole run.
#
#   * "you are out of money"                     -- terminal. Retrying spends
#                                                   the wall clock to be told
#                                                   the same thing.
#   * "this request would exceed your available  -- TRANSIENT. OpenRouter is
#     credits given your current IN-FLIGHT          rate-limiting against a
#     requests; retry after they settle"           balance, and the response
#                                                   carries `Retry-After`.
#
# run-c993a6e93050 died on the second kind, at iteration 23, after 3 hours and
# 12.5M tokens. The response said `"reason":"in_flight_budget_exhausted"` and
# `"Retry-After":"120"`; 402 was not in RETRYABLE_STATUS, so the client returned
# an error immediately, `route_after_architect` treated any Architect failure as
# fatal, and the run ended. Waiting two minutes would have saved it.
_TRANSIENT_402_MARKERS = ("in_flight", "in-flight", "settle", "concurrent")


def _is_transient_payment_required(status: int, body: str, retry_after: str | None) -> bool:
    """Is this 402 the rate-limit kind rather than the out-of-money kind?

    Conservative in the direction that costs least: an explicit `Retry-After`
    means the provider is telling us WHEN to come back, which an account with no
    credit left has no reason to say. Absent that header, the body has to name
    the in-flight condition.
    """
    if status != 402:
        return False
    if retry_after:
        return True
    lowered = (body or "").lower()
    return any(marker in lowered for marker in _TRANSIENT_402_MARKERS)


@dataclass
class ChatResult:
    """A completed chat call, plus the accounting the budget guard needs."""

    text: str
    model: str
    route: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # The assistant's prose ALONE -- no rendered tool calls appended.
    #
    # `text` is the parsing surface: it carries tool calls re-rendered as the
    # fenced blocks the fenced-block callers expect. That is exactly wrong for a
    # caller that is going to REPLAY this message back to the model, which is
    # what the Developer's conversation does: the replayed turn would then hold
    # both a real tool call and a prose transcript of it, and the model reads
    # the transcript as a call it already made. So the two uses get two fields.
    content: str = ""
    # Native tool calls, verbatim: [{"id": ..., "name": ..., "arguments": {...}}].
    # `id` is load-bearing, not decoration -- every tool call must be answered by
    # a `tool` message quoting its id, or the NEXT request is a 400.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Carried so an empty reply can say WHY it was empty. "length" with no text
    # means the completion budget was consumed before the model said anything
    # visible -- a different problem, and a different fix, from a refusal or a
    # dead upstream, and indistinguishable from either without this field.
    finish_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


class LLMClientProtocol(Protocol):
    """The seam the mock implements.

    Nodes depend on this, never on `AsyncLLMClient` directly, which is what
    makes `MOCK_MODE` a swap rather than a rewrite.
    """

    async def chat(
        self,
        *,
        route: str,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        role: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult: ...

    async def aclose(self) -> None: ...


class AsyncLLMClient:
    """Real transport.  Used when `MOCK_MODE` is False."""

    def __init__(self, routes: dict[str, config.RouteConfig] | None = None) -> None:
        self._routes: dict[str, config.RouteConfig] = routes or {
            "openrouter": config.openrouter_route(),
            "vllm": config.vllm_route(),
            "openai": config.openai_route(),
        }
        # One semaphore per route: the vLLM cluster and OpenRouter have very
        # different capacity and must not share a budget.
        self._semaphores: dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(route.max_concurrency)
            for name, route in self._routes.items()
        }
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._usage_total: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    # -------------------- lifecycle --------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        config.HTTP_TIMEOUT_S, connect=config.HTTP_CONNECT_TIMEOUT_S
                    ),
                    limits=httpx.Limits(
                        max_connections=config.HTTP_MAX_CONNECTIONS,
                        max_keepalive_connections=config.HTTP_MAX_KEEPALIVE,
                    ),
                )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def usage_total(self) -> dict[str, int]:
        return dict(self._usage_total)

    # -------------------- request --------------------

    def _headers(self, route: config.RouteConfig) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(route.api_key_env, "")
        if not api_key and route.name == "vllm":
            # A local vLLM server accepts any bearer token; it only requires
            # the header to be present when started with --api-key.
            api_key = "sk-local-dummy-key"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(route.extra_headers)
        return headers

    @staticmethod
    def _backoff_delay(
        attempt: int, retry_after: str | None, max_delay: float | None = None
    ) -> float:
        """Exponential backoff with full jitter, honouring `Retry-After`.

        `max_delay` overrides the usual `HTTP_BACKOFF_MAX_S` clamp. It exists
        for the in-flight 402: the provider's stated wait is the authoritative
        number there -- it is how long the requests already in flight need to
        settle -- and clamping `Retry-After: 120` down to 30s spends every retry
        arriving too early to succeed.
        """
        ceiling_cap = config.HTTP_BACKOFF_MAX_S if max_delay is None else max_delay
        if retry_after:
            try:
                return min(float(retry_after), ceiling_cap)
            except ValueError:
                pass
        ceiling = min(config.HTTP_BACKOFF_BASE_S * (2**attempt), config.HTTP_BACKOFF_MAX_S)
        return random.uniform(0.0, ceiling)

    async def chat(
        self,
        *,
        route: str,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        role: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult:
        """`role` names the originating node (architect, judge, ...).

        Carried for cost attribution in the logs -- and it is what lets the mock
        client dispatch deterministically instead of guessing the role from
        prose, which is unreliable: the Developer's system prompt legitimately
        mentions the Architect.
        """
        route_cfg = self._routes.get(route)
        if route_cfg is None:
            return ChatResult(text="", model=model, route=route, error=f"unknown route {route!r}")

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        # OMIT the key rather than send `null`. `max_tokens=None` means "run
        # until you stop", and every provider spells that as the ABSENCE of the
        # field: OpenAI and OpenRouter tolerate an explicit null, but vLLM's
        # OpenAI-compatible server validates the body against a typed schema and
        # rejects a null for an `int` field outright. One `if` here is what makes
        # "uncapped" mean the same thing on all three routes.
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        # The Developer's ten DevToolbox tools travel here. Only a caller that
        # actually has a toolbox passes this: advertising tools to a node that
        # cannot execute them would invite calls nothing answers, and the
        # privilege model (harness/profiles.py) is that a node's tools are
        # exactly what it is handed.
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if route_cfg.name == "openrouter":
            # Pin which upstream hosts may serve this model; see the provider-
            # pinning section in config.py for the outage that makes this
            # necessary. Only the openrouter route understands the field --
            # vLLM and OpenAI would reject an unknown body key.
            preferences = config.openrouter_provider_preferences(model)
            if preferences:
                payload["provider"] = preferences
            # Cap the reasoning budget on reasoning models. Uncapped, they spend
            # all of `max_tokens` thinking and return empty `content`; see the
            # reasoning-budget section in config.py.
            reasoning = config.openrouter_reasoning(model)
            if reasoning:
                payload["reasoning"] = reasoning
        if route_cfg.is_vllm:
            # The vLLM-route twin of the reasoning cap above: stop a hybrid
            # thinking model from reasoning past HTTP_TIMEOUT_S and returning an
            # empty answer. Only vLLM understands this key; OpenRouter and
            # OpenAI would reject it.
            template_kwargs = config.vllm_chat_template_kwargs(model)
            if template_kwargs:
                payload["chat_template_kwargs"] = template_kwargs

        url = route_cfg.base_url.rstrip("/") + "/chat/completions"
        client = await self._ensure_client()
        last_error = "no attempt made"

        # The semaphore is held across retries on purpose: a request that is
        # being retried is still consuming that route's capacity budget.
        async with self._semaphores[route]:
            for attempt in range(config.HTTP_MAX_RETRIES):
                try:
                    response = await client.post(
                        url,
                        json=payload,
                        headers=self._headers(route_cfg),
                        timeout=timeout_s or config.HTTP_TIMEOUT_S,
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    # LOG IT. This retry used to be completely silent, so a run
                    # that was quietly burning HTTP_MAX_RETRIES x timeout on one
                    # hanging request looked identical to a run doing nothing at
                    # all -- diagnosing it took inspecting the process's open
                    # sockets, which is not a thing a log should require.
                    last_error = f"{type(exc).__name__}: {exc}"
                    delay = self._backoff_delay(attempt, None)
                    log.warning(
                        "role=%s route=%s %s attempt=%d/%d retrying in %.2fs",
                        role or "?", route, type(exc).__name__,
                        attempt + 1, config.HTTP_MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                # Headers are only consulted on an ERROR. A 200 has nothing to
                # say about retrying, and reading them unconditionally makes the
                # success path depend on a response attribute it never needed.
                retry_after: str | None = None
                transient_402 = False
                if response.status_code >= 400:
                    retry_after = getattr(response, "headers", {}).get("Retry-After")
                    transient_402 = _is_transient_payment_required(
                        response.status_code, response.text, retry_after
                    )

                if response.status_code in RETRYABLE_STATUS or transient_402:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    delay = self._backoff_delay(
                        attempt, retry_after,
                        # A stated in-flight wait is honoured in full, up to the
                        # per-call timeout: arriving early just burns a retry.
                        max_delay=config.HTTP_RETRY_AFTER_MAX_S if transient_402 else None,
                    )
                    log.warning(
                        "role=%s route=%s status=%s attempt=%d/%d retrying in %.2fs%s",
                        role or "?", route, response.status_code,
                        attempt + 1, config.HTTP_MAX_RETRIES, delay,
                        " (provider says the limit is on IN-FLIGHT requests, not "
                        "on your balance)" if transient_402 else "",
                    )
                    await asyncio.sleep(delay)
                    continue

                if response.status_code >= 400:
                    # 4xx other than the retryable set is a request bug; retrying
                    # a malformed request just wastes the budget.
                    return ChatResult(
                        text="", model=model, route=route,
                        error=f"HTTP {response.status_code}: {response.text[:500]}",
                    )

                return self._parse(response.json(), model=model, route=route,
                                   max_tokens=max_tokens)

        return ChatResult(text="", model=model, route=route, error=last_error)

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        """Everything the assistant actually said, not just `content`.

        `content` alone is the wrong field for a reasoning or tool-calling model
        and the mistake is SILENT -- it yields an empty string, the caller reports
        "empty response", and the turn is spent. Three shapes have to be handled:

        * `content` as a plain string (the ordinary case);
        * `content` as a LIST of typed parts, which is what an OpenAI-compatible
          gateway returns when the model emitted mixed output;
        * `tool_calls`, when the model answered in its native tool-call format
          instead of the fenced JSON the persona asked for. Those are rendered as
          the fenced action block the nodes already parse, so a Developer turn
          that chose a tool is honoured rather than discarded. This is the
          http-transport twin of `collect_assistant_text`'s recovery.

        `reasoning` / `reasoning_content` are deliberately NOT collected: a
        reasoning trace routinely holds a draft ```json block, and it would
        outrank the committed answer in the first-fence race.
        """
        parts = [AsyncLLMClient._message_content(message)]

        # Appended AFTER the prose so a real deliverable fence still wins the
        # first-fence race in `extract_json_block`.
        for call in AsyncLLMClient._message_tool_calls(message):
            action = {"tool": call["name"], "args": call["arguments"]}
            parts.append("```json\n" + json.dumps(action, ensure_ascii=False) + "\n```")
        return "\n\n".join(p for p in parts if p)

    @staticmethod
    def _message_content(message: dict[str, Any]) -> str:
        """The assistant's prose only, in whichever shape it arrived.

        Split out of `_message_text` because a caller that REPLAYS the turn back
        to the model must not replay a prose transcript of its own tool calls
        alongside the real ones -- see the note on `ChatResult.content`.
        """
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "\n\n".join(p for p in parts if p)
        return ""

    @staticmethod
    def _message_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
        """Native tool calls as `{"id", "name", "arguments"}`, arguments decoded.

        `arguments` arrives as a JSON *string* on every OpenAI-compatible
        provider. A model that streams a truncated object leaves it unparseable;
        that is an empty argument dict and a failed tool call, never an
        exception in the transport.
        """
        calls: list[dict[str, Any]] = []
        for index, call in enumerate(message.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else call
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            raw = function.get("arguments")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw or "{}")
                except json.JSONDecodeError:
                    log.warning("tool call %r arrived with unparseable arguments: %r",
                                name, raw[:200])
                    raw = {}
            calls.append({
                # Synthesised when the provider omits it, because the id is what
                # the answering `tool` message has to quote.
                "id": str(call.get("id") or f"call_{index}"),
                "name": name.strip(),
                "arguments": raw if isinstance(raw, dict) else {},
            })
        return calls

    def _parse(self, body: dict[str, Any], *, model: str, route: str,
               max_tokens: int | None = None) -> ChatResult:
        try:
            choice = body["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
            text = self._message_text(message)
            content = self._message_content(message)
            tool_calls = self._message_tool_calls(message)
            finish_reason = str(choice.get("finish_reason") or "")
        except (KeyError, IndexError, TypeError):
            return ChatResult(
                text="", model=model, route=route, raw=body,
                error=f"unexpected response shape: {str(body)[:300]}",
            )
        usage_raw = body.get("usage") or {}
        usage = {
            "input_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        }
        if not usage["total_tokens"]:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        for key, value in usage.items():
            self._usage_total[key] = self._usage_total.get(key, 0) + value
        if not text.strip() and not tool_calls and finish_reason == "length":
            # Worth a log line of its own: the request succeeded, the money was
            # spent, and the only evidence is a field the caller may not print.
            reasoning_tokens = int(
                (usage_raw.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            )
            # The advice MUST branch on what actually ran out, because the two
            # cases have opposite fixes and naming the wrong one sends the
            # reader to a knob that cannot help.
            #
            #   capped   -> the ceiling is the binding constraint; RAISE it.
            #              (The preflight probe in main.py is exactly this: a
            #              16-token ping that a reasoning model cannot answer
            #              inside its budget. Telling that reader to lower the
            #              reasoning effort is wrong -- and telling them "a
            #              token ceiling is not the lever" about a 16-token
            #              ceiling contradicts the same sentence's own numbers.)
            #   uncapped -> there is no ceiling left to raise; the CONTEXT
            #              WINDOW filled, and on a reasoning model the thinking
            #              channel is what filled it.
            #
            # The reasoning-effort hint is additionally gated on the openrouter
            # route and on reasoning tokens having actually been spent:
            # OPENROUTER_REASONING_EFFORT does nothing on vLLM or OpenAI, and
            # nothing at all when the model reported no reasoning tokens.
            if max_tokens is not None:
                budget = f"the {max_tokens}-token completion budget"
                fix = f"Raise this node's *_MAX_TOKENS (currently {max_tokens})"
            else:
                budget = "the model's context window (this call was UNCAPPED)"
                fix = "There is no token ceiling to raise -- shorten the prompt"
            if route == "openrouter" and reasoning_tokens:
                fix += ", or lower OPENROUTER_REASONING_EFFORT: reasoning tokens share this budget"
            log.error(
                "route=%s model=%s returned NO content: %s (%d tokens spent, %d of "
                "them reasoning) was exhausted first. %s.",
                route, model, budget, usage["output_tokens"], reasoning_tokens, fix,
            )
        return ChatResult(
            text=text, model=model, route=route, usage=usage, raw=body,
            finish_reason=finish_reason, content=content, tool_calls=tool_calls,
        )


# -------------------- process-wide accessor --------------------

_CLIENT: LLMClientProtocol | None = None


def get_llm_client() -> LLMClientProtocol:
    """Return the process-wide client, real or mock.

    This is the ONLY place that reads `MOCK_MODE` for LLM traffic.  Graph
    topology never branches on it (brief section 9).
    """
    global _CLIENT
    if _CLIENT is None:
        if config.MOCK_MODE:
            from mocks.llm import MockLLMClient

            _CLIENT = MockLLMClient()
            log.info("LLM client: MOCK (scenario=%s)", config.MOCK_SCENARIO)
        else:
            _CLIENT = AsyncLLMClient()
            log.info(
                "LLM client: REAL (openrouter=%s vllm=%s openai=%s)",
                config.OPENROUTER_BASE_URL, config.VLLM_BASE_URL, config.OPENAI_BASE_URL,
            )
    return _CLIENT


def reset_llm_client() -> None:
    """Drop the cached client.  Used by tests that flip MOCK_MODE."""
    global _CLIENT
    _CLIENT = None
