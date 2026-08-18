#!/usr/bin/env python3
"""Disk cache for Pyserini BM25 search results."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from pyserini_client import DEFAULT_BASE_URL, DEFAULT_INDEX, project_root, search

DEFAULT_CACHE_DIR = project_root() / "runs/cache/bm25"


def _norm_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip().lower()


def _cache_key(query: str, *, hits: int, index: str) -> str:
    payload = f"{index}|{hits}|{_norm_query(query)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cached_search(
    query: str,
    *,
    hits: int = 100,
    base_url: str = DEFAULT_BASE_URL,
    index: str = DEFAULT_INDEX,
    cache_dir: Path | None = None,
    query_delay: float = 0.0,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Return BM25 candidates, loading from disk cache when available.

    If a shorter-hits result is requested and a deeper cache entry exists for
    the same query, we reuse and truncate it (saves API calls when tuning hits).
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(query, hits=hits, index=index)
    path = cache_dir / f"{key}.json"
    if path.exists() and not force_refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("candidates") or []

    # Try to reuse a deeper cached result for the same query.
    if not force_refresh:
        reused = _reuse_deeper_cache(cache_dir, query, hits=hits, index=index)
        if reused is not None:
            return reused

    if query_delay > 0:
        time.sleep(query_delay)
    candidates = search(query, hits=hits, base_url=base_url, index=index)
    path.write_text(
        json.dumps(
            {
                "query": query,
                "hits": hits,
                "index": index,
                "candidates": candidates,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return candidates


def _reuse_deeper_cache(
    cache_dir: Path,
    query: str,
    *,
    hits: int,
    index: str,
) -> list[dict[str, Any]] | None:
    """If any deeper hits cache exists for this query, truncate and return it."""
    needle = _norm_query(query)
    best: tuple[int, list[dict[str, Any]]] | None = None
    for path in cache_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("index") != index:
            continue
        if _norm_query(str(payload.get("query", ""))) != needle:
            continue
        cached_hits = int(payload.get("hits") or 0)
        if cached_hits < hits:
            continue
        candidates = payload.get("candidates") or []
        if best is None or cached_hits < best[0]:
            best = (cached_hits, candidates)
    if best is None:
        return None
    return best[1][:hits]
