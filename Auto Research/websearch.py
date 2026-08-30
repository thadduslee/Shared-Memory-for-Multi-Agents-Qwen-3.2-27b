"""Web-search adapter for the Architect's prior-art survey.

The brief requires the Architect to search the web before proposing.  No
first-party web-search plugin was confirmed in the DeepSeek Harness repo, so
this does NOT go through `dsh`; it is a separate, explicitly-swappable adapter.

# ASSUMPTION: the real path targets a generic JSON search API (Brave, Tavily,
#   SerpAPI, ...).  The request/response shape below is Tavily's, chosen because
#   it is the closest to a de-facto standard for agent search.  VERIFY against
#   whichever provider you fund and adjust `_parse` -- it is the only function
#   that knows the response shape.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

import config

log = logging.getLogger("orchestrator.websearch")

SEARCH_API_URL = os.environ.get("SEARCH_API_URL", "https://api.tavily.com/search")
SEARCH_API_KEY_ENV = "SEARCH_API_KEY"


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


class SearchProtocol(Protocol):
    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]: ...


class HttpSearchClient:
    """Real search over the shared async HTTP stack."""

    #: The Architect issues one search per proposal, so an unset key logged the
    #: same warning three times per iteration -- nine lines in a three-iteration
    #: run, all saying the same thing. Warn once, then say nothing.
    _warned_no_key = False

    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        import httpx

        api_key = os.environ.get(SEARCH_API_KEY_ENV, "")
        if not api_key:
            if not HttpSearchClient._warned_no_key:
                HttpSearchClient._warned_no_key = True
                log.warning(
                    "%s unset -- the Architect will propose without prior art for this "
                    "whole run. Set it in .env (Tavily: https://tavily.com) or override "
                    "SEARCH_API_URL for another provider.",
                    SEARCH_API_KEY_ENV,
                )
            return []
        try:
            async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT_S) as client:
                response = await client.post(
                    SEARCH_API_URL,
                    json={"api_key": api_key, "query": query, "max_results": max_results},
                )
                response.raise_for_status()
                return self._parse(response.json())
        except Exception as exc:  # noqa: BLE001 - search is advisory; never fail the node on it
            log.warning("web search failed (%s); continuing without prior art", exc)
            return []

    @staticmethod
    def _parse(body: dict[str, Any]) -> list[SearchHit]:
        return [
            SearchHit(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                snippet=str(item.get("content") or item.get("snippet") or "")[:500],
            )
            for item in (body.get("results") or [])
        ]


class MockSearchClient:
    """Fixed prior-art corpus.

    These are real, checkable references on multi-principal memory governance,
    RBAC-filtered retrieval and deletion -- the mock is offline, not fictional,
    so the Architect's mock design cites sources a reviewer can actually look up.
    """

    CORPUS = [
        SearchHit(
            "GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents",
            "https://arxiv.org/abs/2606.18829",
            "Defines Utility, Access Control and Active Forgetting, combined as "
            "MGS = U * (1 - A) * (1 - F) over 4 domains and 2218 checkpoints.",
        ),
        SearchHit(
            "Zanzibar: Google's Consistent, Global Authorization System",
            "https://research.google/pubs/pub48190/",
            "Relationship-based access control: authorization as a graph of tuples "
            "(object#relation@user) evaluated at read time. The model behind our "
            "relationships table and the covering-clinician inheritance rule.",
        ),
        SearchHit(
            "PostgreSQL Row-Level Security",
            "https://www.postgresql.org/docs/current/ddl-rowsecurity.html",
            "Policies attached to tables so the filter cannot be bypassed by a query path "
            "that forgot the WHERE clause -- the argument for a structural deny.",
        ),
        SearchHit(
            "Crypto-shredding for the right to erasure",
            "https://en.wikipedia.org/wiki/Crypto-shredding",
            "Encrypt per-record and destroy the key to render data unrecoverable without "
            "rewriting immutable storage; the standard GDPR Article 17 mechanism.",
        ),
        SearchHit(
            "Tombstones in log-structured stores",
            "https://en.wikipedia.org/wiki/Tombstone_(data_store)",
            "Deletion markers that preserve the fact of deletion. Needed to distinguish "
            "'deleted' from 'never existed' -- the distinction the confirm_yes_no attack probes.",
        ),
        SearchHit(
            "Machine unlearning and the limits of post-hoc deletion",
            "https://arxiv.org/abs/1912.03817",
            "Why filtering at read time is not deletion: residual influence survives in "
            "derived artifacts such as embeddings, summaries and caches.",
        ),
    ]

    async def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        terms = {t for t in query.lower().split() if len(t) > 3}
        scored = sorted(
            self.CORPUS,
            key=lambda hit: -len(terms & set((hit.title + " " + hit.snippet).lower().split())),
        )
        return scored[:max_results]


_CLIENT: SearchProtocol | None = None


def get_search_client() -> SearchProtocol:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = MockSearchClient() if config.MOCK_MODE else HttpSearchClient()
    return _CLIENT


def reset_search_client() -> None:
    global _CLIENT
    _CLIENT = None
    # The once-only warning is per-process state, so a test that resets the
    # client and expects to see the warning again gets to see it again.
    HttpSearchClient._warned_no_key = False
