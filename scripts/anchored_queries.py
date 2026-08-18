#!/usr/bin/env python3
"""Query rewriting helpers for Phase-1 lexical branches."""

from __future__ import annotations

import re
from dataclasses import dataclass

from text_utils import content_terms, keyword_query, split_sentences, tokenize

CLAUSE_SPLIT_RE = re.compile(
    r"\b(?:i also|also want|along with|as well as|plus|in addition|"
    r"particularly concerning|specifically|could you explain|"
    r"can you help me understand|what should we|how should we|"
    r"i want to|i'm interested|im interested|i am interested|"
    r"i'm seeking|im seeking|i'm hoping|im hoping)\b",
    re.IGNORECASE,
)
QUESTION_START_RE = re.compile(
    r"^(?:how|what|why|when|where|which|who|can|could|should|would|do|does|did|is|are)\b",
    re.IGNORECASE,
)

# Soft filler stripped from facets (kept out of anchors too when possible).
FILLER = {
    "understand",
    "explain",
    "interested",
    "seeking",
    "hoping",
    "want",
    "help",
    "deeper",
    "particularly",
    "concerning",
    "including",
    "regarding",
    "please",
    "could",
    "would",
    "should",
}


@dataclass(frozen=True)
class QueryFamily:
    """One BM25 subquery with a role, weight, and family id for consensus."""

    text: str
    family: str          # e.g. original / core / vital / okay / alias
    weight: float
    label: str = ""


DEFAULT_WEIGHTS = {
    "original": 2.0,
    "core": 1.5,
    "vital": 1.25,
    "okay": 0.75,
    "alias": 0.50,
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,.;:?!")


def topic_anchor_terms(text: str, *, max_terms: int = 6) -> list[str]:
    """Pick main topic anchors from the opening of the narrative."""
    sentences = split_sentences(text)
    seed = sentences[0] if sentences else text
    terms_set = content_terms(seed)
    terms: list[str] = []
    seen: set[str] = set()
    for tok in tokenize(seed):
        if tok in FILLER or tok not in terms_set or tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
        if len(terms) >= max_terms:
            break
    if len(terms) < 2:
        bag = keyword_query(text, max_terms=max_terms).split()
        return bag[:max_terms]
    return terms


def concise_core_query(text: str, *, max_terms: int = 12) -> str:
    """Short keyword core that keeps the main subject."""
    return keyword_query(text, max_terms=max_terms)


def _facet_candidates(text: str) -> list[str]:
    facets: list[str] = []
    for sentence in split_sentences(text):
        cleaned = _clean(sentence)
        if len(cleaned) < 25:
            continue
        if cleaned.endswith("?") or QUESTION_START_RE.match(cleaned):
            facets.append(cleaned)
            continue
        # Split long sentences on commas / "and" lists for atomic facets.
        parts = re.split(r"\s*,\s*|\band\b", cleaned)
        if len(parts) >= 3:
            for part in parts:
                part = _clean(part)
                if len(part) >= 20:
                    facets.append(part)
        else:
            facets.append(cleaned)

    for part in CLAUSE_SPLIT_RE.split(text):
        cleaned = _clean(part)
        if len(cleaned) >= 30:
            facets.append(cleaned)
    return facets


def _anchor_facet(facet: str, anchors: list[str], *, max_words: int = 14) -> str:
    """Ensure the facet query retains topic anchors (anti-generic)."""
    facet_terms = [t for t in tokenize(facet) if t not in FILLER and len(t) > 2]
    # Drop pure stopword-y leftovers via content_terms on the facet string.
    facet_content = [t for t in facet_terms if t in content_terms(facet) or t in anchors]

    # Prepend any missing anchors (up to 3).
    missing = [a for a in anchors[:3] if a not in facet_content]
    combined = missing + facet_content
    # Deduplicate preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for term in combined:
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
        if len(out) >= max_words:
            break
    return " ".join(out)


def build_anchored_query_families(
    text: str,
    *,
    max_vital: int = 6,
    max_okay: int = 2,
    max_alias: int = 2,
    weights: dict[str, float] | None = None,
) -> list[QueryFamily]:
    """Build a compact, anchored query set for one narrative.

    Typical size: 1 original + 1 core + 4–6 vital + 1–3 okay/alias ≈ 8–10 queries.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    anchors = topic_anchor_terms(text)
    families: list[QueryFamily] = []
    seen: set[str] = set()

    def add(query: str, family: str, label: str = "") -> None:
        q = _clean(query)
        if len(q) < 8:
            return
        key = re.sub(r"\s+", " ", q).lower()
        if key in seen:
            return
        seen.add(key)
        families.append(
            QueryFamily(text=q, family=family, weight=weights[family], label=label or family)
        )

    # 1) Original full narrative
    add(text, "original", "original")

    # 2) Concise core
    core = concise_core_query(text)
    if core:
        add(core, "core", "core")

    # 3) Anchored vital facets from clauses / questions / comma lists
    raw_facets = _facet_candidates(text)
    vital_added = 0
    okay_added = 0
    for raw in raw_facets:
        anchored = _anchor_facet(raw, anchors)
        if not anchored:
            continue
        if vital_added < max_vital:
            add(anchored, "vital", f"vital-{vital_added + 1}")
            vital_added += 1
        elif okay_added < max_okay:
            add(anchored, "okay", f"okay-{okay_added + 1}")
            okay_added += 1

    # 4) Alias / entity-style bags: short and medium keyword variants
    alias_short = keyword_query(text, max_terms=6)
    alias_mid = keyword_query(text, max_terms=10)
    if alias_short:
        add(alias_short, "alias", "alias-short")
    if alias_mid and alias_mid != alias_short and alias_mid != core:
        add(alias_mid, "alias", "alias-mid")
        if max_alias <= 1:
            # trim extras handled by seen / max via add only twice above
            pass

    # Cap total soft: prefer original/core/vital over trailing okay/alias if huge.
    # Keep all we built; typically <= 1+1+6+2+2 = 12.
    return families


def families_as_queries(families: list[QueryFamily]) -> list[str]:
    return [f.text for f in families]


def families_weights(families: list[QueryFamily]) -> list[float]:
    return [f.weight for f in families]


def families_ids(families: list[QueryFamily]) -> list[str]:
    """Family ids for consensus counting (collapse vital-1/vital-2 → vital)."""
    return [f.family for f in families]
