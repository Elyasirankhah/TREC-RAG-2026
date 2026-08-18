#!/usr/bin/env python3
"""Reciprocal-rank fusion and candidate-pool cleanup helpers."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any


def rrf_fuse(
    rankings: list[list[dict]],
    *,
    rrf_k: int = 60,
    query_weights: list[float] | None = None,
    contrib_depth: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists with optional per-query weights and rank cutoff.

    Only ranks ``<= contrib_depth`` contribute (if set). Documents deeper than
    the cutoff for a weak facet should not pollute the fused ranking.
    """
    scores: dict[str, float] = defaultdict(float)
    for idx, candidates in enumerate(rankings):
        weight = 1.0 if not query_weights else query_weights[idx]
        for candidate in candidates:
            docid = candidate.get("docid")
            rank = candidate.get("rank")
            if not docid or not isinstance(rank, int):
                continue
            if contrib_depth is not None and rank > contrib_depth:
                continue
            scores[docid] += weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def collect_doc_texts(rankings: list[list[dict]]) -> dict[str, str]:
    """Build docid -> best available document text seen across all rankings."""
    texts: dict[str, str] = {}
    for candidates in rankings:
        for candidate in candidates:
            docid = candidate.get("docid")
            if not docid or docid in texts:
                continue
            text = _doc_text(candidate.get("doc"))
            if text:
                texts[docid] = text
    return texts


def _doc_text(doc: Any) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ("text", "contents", "body", "passage"):
            value = doc.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def consensus_counts(
    rankings: list[list[dict]],
    *,
    family_ids: list[str] | None = None,
    cutoff: int = 20,
) -> dict[str, int]:
    """Count how many distinct query *families* retrieve each doc in top-cutoff.

    Counting families (not near-duplicate query variants) avoids artificial
    inflation of the consensus bonus.
    """
    if family_ids is None:
        family_ids = [str(i) for i in range(len(rankings))]
    seen: dict[str, set[str]] = defaultdict(set)
    for family, candidates in zip(family_ids, rankings):
        for candidate in candidates:
            docid = candidate.get("docid")
            rank = candidate.get("rank")
            if not docid or not isinstance(rank, int):
                continue
            if rank <= cutoff:
                seen[docid].add(family)
    return {docid: len(fams) for docid, fams in seen.items()}


def fuse_with_consensus(
    rankings: list[list[dict]],
    *,
    rrf_k: int = 60,
    query_weights: list[float] | None = None,
    contrib_depth: int | None = 50,
    family_ids: list[str] | None = None,
    consensus_cutoff: int = 20,
    consensus_lambda: float = 0.0,
) -> list[tuple[str, float]]:
    """Weighted RRF plus optional log consensus bonus across query families."""
    base = dict(
        rrf_fuse(
            rankings,
            rrf_k=rrf_k,
            query_weights=query_weights,
            contrib_depth=contrib_depth,
        )
    )
    if consensus_lambda <= 0:
        return sorted(base.items(), key=lambda item: item[1], reverse=True)

    counts = consensus_counts(rankings, family_ids=family_ids, cutoff=consensus_cutoff)
    scores: dict[str, float] = {}
    for docid, score in base.items():
        scores[docid] = score + consensus_lambda * math.log1p(counts.get(docid, 0))
    # Also include docs that only appear via consensus edge cases (already in base).
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def coverage_aware_rrf(
    rankings: list[list[dict]],
    *,
    rrf_k: int = 60,
    query_weights: list[float] | None = None,
    contrib_depth: int | None = None,
    family_ids: list[str] | None = None,
    max_per_family: int = 40,
) -> list[tuple[str, float]]:
    """Weighted RRF that caps how many top docs each family can contribute.

    Inspired by the Phase-1 doc's coverage-aware fusion note (facet balance /
    redundancy control). Start with plain RRF; enable this if top ranks are
    dominated by one branch.
    """
    if family_ids is None:
        family_ids = [str(i) for i in range(len(rankings))]
    capped: list[list[dict]] = []
    for candidates in rankings:
        if contrib_depth is not None:
            candidates = [c for c in candidates if isinstance(c.get("rank"), int) and c["rank"] <= contrib_depth]
        capped.append(candidates[:max_per_family] if max_per_family > 0 else candidates)
    return rrf_fuse(capped, rrf_k=rrf_k, query_weights=query_weights, contrib_depth=None)


def dedupe_candidate_pools(
    rankings: list[list[dict]],
    doc_texts: dict[str, str],
    *,
    jaccard_threshold: float = 0.9,
    prefix_chars: int = 200,
) -> list[list[dict]]:
    """URL / near-duplicate cleanup ON THE CANDIDATE POOLS (doc: before RRF).

    Distinct docids whose text is near-identical are collapsed to a single
    canonical docid (the one with the best rank across channels). Redundant
    docids are removed from *every* channel ranking so fusion does not reward
    the same content twice. Same-doc appearances across channels (the genuine
    consensus signal) are preserved because they share one docid.
    """
    best_rank: dict[str, int] = {}
    for candidates in rankings:
        for candidate in candidates:
            docid = candidate.get("docid")
            rank = candidate.get("rank")
            if not docid or not isinstance(rank, int):
                continue
            if docid not in best_rank or rank < best_rank[docid]:
                best_rank[docid] = rank

    kept_tokens: list[set[str]] = []
    kept_prefixes: set[str] = set()
    redundant: set[str] = set()
    for docid in sorted(best_rank, key=lambda d: best_rank[d]):
        text = (doc_texts.get(docid) or "").strip()
        if not text:
            continue
        prefix = re.sub(r"\s+", " ", text[:prefix_chars]).lower()
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()[:1200]))
        duplicate = False
        if prefix and prefix in kept_prefixes:
            duplicate = True
        elif tokens:
            for prior in kept_tokens:
                inter = len(tokens & prior)
                union = len(tokens | prior) or 1
                if inter / union >= jaccard_threshold:
                    duplicate = True
                    break
        if duplicate:
            redundant.add(docid)
            continue
        if tokens:
            kept_tokens.append(tokens)
        if prefix:
            kept_prefixes.add(prefix)

    if not redundant:
        return rankings
    return [
        [c for c in candidates if c.get("docid") not in redundant]
        for candidates in rankings
    ]


def near_duplicate_filter(
    ranked: list[tuple[str, float]],
    doc_texts: dict[str, str],
    *,
    depth: int,
    jaccard_threshold: float = 0.85,
    prefix_chars: int = 280,
) -> list[tuple[str, float]]:
    """Drop near-duplicate passages while preserving fused order."""
    kept: list[tuple[str, float]] = []
    seen_tokens: list[set[str]] = []
    seen_prefixes: list[str] = []

    for docid, score in ranked:
        if len(kept) >= depth:
            break
        text = (doc_texts.get(docid) or "").strip()
        prefix = re.sub(r"\s+", " ", text[:prefix_chars]).lower() if text else ""
        tokens = set(re.findall(r"[a-z0-9]+", text.lower()[:1200])) if text else set()

        duplicate = False
        if prefix and any(prefix == p or (prefix and p and (prefix in p or p in prefix)) for p in seen_prefixes):
            duplicate = True
        elif tokens and seen_tokens:
            for prior in seen_tokens:
                inter = len(tokens & prior)
                union = len(tokens | prior) or 1
                if inter / union >= jaccard_threshold:
                    duplicate = True
                    break
        if duplicate:
            continue
        kept.append((docid, score))
        if tokens:
            seen_tokens.append(tokens)
        if prefix:
            seen_prefixes.append(prefix)
    return kept
