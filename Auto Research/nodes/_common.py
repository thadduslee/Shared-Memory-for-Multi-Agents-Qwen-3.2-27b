"""Shared node plumbing: structured logging, timing, artifacts, budget."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import config

log = logging.getLogger("orchestrator.node")


def setup_logging(level: str | None = None) -> None:
    """One-line-per-event structured logging.

    Deliberately not JSON by default: a research loop is watched by a human in
    a terminal far more often than it is parsed.  Every record still carries
    the four fields the brief asks for (iteration, phase, node, duration).
    """
    logging.basicConfig(
        level=getattr(logging, (level or config.LOG_LEVEL).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx logs every request at INFO, which drowns a 579-way fan-out.
    logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def node_span(
    node: str, iteration: int, phase: str, **extra: Any
) -> AsyncIterator[dict[str, Any]]:
    """Time a node and emit one structured record for it.

    Yields a mutable dict the node fills in (tokens, counts); on exit the dict
    becomes the `node_timings` entry appended to state via `operator.add`.
    """
    record: dict[str, Any] = {
        "node": node, "iteration": iteration, "phase": phase,
        "started_at": time.time(), **extra,
    }
    started = time.monotonic()
    log.info("-> %s (iter=%d phase=%s)", node, iteration, phase)
    try:
        yield record
    finally:
        record["duration_s"] = round(time.monotonic() - started, 3)
        log.info(
            "<- %s (iter=%d phase=%s) %.2fs tokens=%s",
            node, iteration, phase, record["duration_s"], record.get("tokens", 0),
        )


def write_artifact(path: Path, content: str | dict[str, Any] | list[Any]) -> Path:
    """Write one per-iteration artifact, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, (dict, list)):
        path.write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")
    else:
        path.write_text(str(content), encoding="utf-8")
    return path


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    return path


def budget_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Current wall-clock and token spend against the configured caps."""
    started = float(state.get("started_at") or time.monotonic())
    elapsed = time.monotonic() - started
    tokens = int((state.get("token_usage") or {}).get("total_tokens", 0))
    return {
        "elapsed_s": elapsed,
        "elapsed_frac": elapsed / config.MAX_WALLCLOCK_S if config.MAX_WALLCLOCK_S else 0.0,
        "total_tokens": tokens,
        "token_frac": tokens / config.MAX_TOTAL_TOKENS if config.MAX_TOTAL_TOKENS else 0.0,
        "over_wallclock": elapsed > config.MAX_WALLCLOCK_S,
        "over_tokens": tokens > config.MAX_TOTAL_TOKENS,
    }


def usage_delta(*usages: dict[str, Any] | None) -> dict[str, int]:
    """Sum several usage dicts into the shape `token_usage` reduces."""
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in usages:
        for key in total:
            total[key] += int((usage or {}).get(key, 0) or 0)
    return total
