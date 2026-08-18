#!/usr/bin/env python3
"""LLM planner that decomposes a narrative into BM25 query branches."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_client import llm_json, resolve_api_key, resolve_phase1_model
from pyserini_client import project_root

SPARSE_SYSTEM = (
    "You are an information retrieval planner. Your job is to rewrite a complex "
    "topic into a sparse-friendly search query."
)

SPARSE_USER = """Given the topic below, produce JSON with:
- "subquestions": 3 to 6 short subquestions
- "entity_terms": important exact-match entities, dates, places, abbreviations
- "keywords": high-value lexical terms and near-synonyms
- "sparse_big_query": one single query string made by concatenating the subquestions and the best keywords
Rules:
- Keep exact names and dates unchanged.
- Prefer precise nouns and technical terms.
- Do not add unsupported facts.
- Keep the sparse_big_query under 120 words.
Topic:
{topic}"""

DENSE_SYSTEM = "You generate dense-retrieval evidence sketches."

DENSE_USER = """Given the topic below, output JSON with:
- "evidence_sketches": 2 to 4 short passages, each 40 to 80 words
Rules:
- Each sketch should describe one likely answer facet.
- Write in plain factual prose as if it were a relevant passage.
- It is okay if details are approximate; do not invent named entities that are not already implied by the topic.
- Cover different facets, not paraphrases of the same facet.
Topic:
{topic}"""

WEIGHT_SYSTEM = "You assign retrieval weights for a hybrid search system."

WEIGHT_USER = """Given the topic below, output JSON with:
- "bm25_weight"
- "splade_weight"
- "dense_weight"
- "use_concat_sparse_query": true/false
- "use_dense_sketches": true/false
- "notes": one short sentence
Rules:
- If the topic contains exact entities, dates, quotes, legislation, or product/model names, increase sparse weights.
- If the topic is explanatory, comparative, or multi-faceted, increase SPLADE and dense weights.
- Weights must sum to 1.0.
Topic:
{topic}"""

DEFAULT_PLAN_CACHE = project_root() / "runs/cache/phase1_plans"


@dataclass
class Phase1Plan:
    topic: str
    subquestions: list[str] = field(default_factory=list)
    entity_terms: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    sparse_big_query: str = ""
    evidence_sketches: list[str] = field(default_factory=list)
    bm25_weight: float = 0.35
    splade_weight: float = 0.35
    dense_weight: float = 0.30
    use_concat_sparse_query: bool = True
    use_dense_sketches: bool = True
    notes: str = ""


@dataclass
class QueryBranch:
    family: str
    text: str
    weight: float


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    out: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                for key in ("text", "query", "sketch", "passage"):
                    raw = item.get(key)
                    if isinstance(raw, str) and raw.strip():
                        out.append(raw.strip())
                        break
    return out


def _clip_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _normalize_weights(bm25: float, splade: float, dense: float) -> tuple[float, float, float]:
    vals = [max(0.0, float(bm25)), max(0.0, float(splade)), max(0.0, float(dense))]
    total = sum(vals)
    if total <= 0:
        return 0.35, 0.35, 0.30
    return vals[0] / total, vals[1] / total, vals[2] / total


def _plan_cache_path(cache_dir: Path, topic: str, model: str | None) -> Path:
    norm = re.sub(r"\s+", " ", topic).strip().lower()
    key = f"{model or 'default'}|{norm}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _fallback_plan(topic: str) -> Phase1Plan:
    """Rule-based fallback if the LLM key is missing or the call fails."""
    sentences = [s.strip() for s in re.split(r"[.?!\n]+", topic) if len(s.strip()) >= 20]
    subqs = sentences[:5] or [topic]
    terms = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", topic)
    # Prefer longer / rarer-looking tokens roughly.
    uniq: list[str] = []
    for t in terms:
        low = t.lower()
        if low in {"the", "and", "for", "with", "that", "this", "from", "into", "about"}:
            continue
        if t not in uniq:
            uniq.append(t)
    keywords = uniq[:16]
    sparse = _clip_words(" ".join(subqs + keywords), 120)
    sketches = [
        _clip_words(s, 80)
        for s in subqs[:3]
    ]
    return Phase1Plan(
        topic=topic,
        subquestions=subqs,
        entity_terms=uniq[:8],
        keywords=keywords,
        sparse_big_query=sparse,
        evidence_sketches=sketches,
        notes="rule-based fallback (no LLM)",
    )


def plan_topic(
    topic: str,
    *,
    model: str | None = None,
    use_weights: bool = True,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> Phase1Plan:
    """Run the doc's sparse + dense (+ optional weight) prompts for one topic."""
    cache_dir = cache_dir or DEFAULT_PLAN_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _plan_cache_path(cache_dir, topic, model)
    if path.exists() and not force_refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Phase1Plan(**payload)

    if not resolve_api_key():
        plan = _fallback_plan(topic)
        path.write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2), encoding="utf-8")
        return plan

    sparse = llm_json(
        SPARSE_USER.format(topic=topic),
        model=model,
        system=SPARSE_SYSTEM,
        temperature=0.2,
        max_output_tokens=1200,
    )
    dense = llm_json(
        DENSE_USER.format(topic=topic),
        model=model,
        system=DENSE_SYSTEM,
        temperature=0.3,
        max_output_tokens=1200,
    )

    bm25_w, splade_w, dense_w = 0.35, 0.35, 0.30
    use_concat = True
    use_sketches = True
    notes = ""
    if use_weights:
        try:
            weights = llm_json(
                WEIGHT_USER.format(topic=topic),
                model=model,
                system=WEIGHT_SYSTEM,
                temperature=0.1,
                max_output_tokens=400,
            )
            bm25_w, splade_w, dense_w = _normalize_weights(
                float(weights.get("bm25_weight", bm25_w)),
                float(weights.get("splade_weight", splade_w)),
                float(weights.get("dense_weight", dense_w)),
            )
            use_concat = bool(weights.get("use_concat_sparse_query", True))
            use_sketches = bool(weights.get("use_dense_sketches", True))
            notes = str(weights.get("notes") or "")
        except Exception as exc:  # noqa: BLE001 — keep retrieval alive
            notes = f"weight prompt failed: {exc}"

    sparse_big = str(sparse.get("sparse_big_query") or "").strip()
    if not sparse_big:
        parts = _as_str_list(sparse.get("subquestions")) + _as_str_list(sparse.get("keywords"))
        sparse_big = _clip_words(" ".join(parts) or topic, 120)
    else:
        sparse_big = _clip_words(sparse_big, 120)

    plan = Phase1Plan(
        topic=topic,
        subquestions=_as_str_list(sparse.get("subquestions"))[:6],
        entity_terms=_as_str_list(sparse.get("entity_terms")),
        keywords=_as_str_list(sparse.get("keywords")),
        sparse_big_query=sparse_big,
        evidence_sketches=_as_str_list(dense.get("evidence_sketches"))[:4],
        bm25_weight=bm25_w,
        splade_weight=splade_w,
        dense_weight=dense_w,
        use_concat_sparse_query=use_concat,
        use_dense_sketches=use_sketches,
        notes=notes,
    )
    path.write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def keyword_bag(plan: Phase1Plan, *, max_terms: int = 24) -> str:
    seen: set[str] = set()
    terms: list[str] = []
    for term in plan.entity_terms + plan.keywords:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return " ".join(terms)


def build_query_branches(plan: Phase1Plan, *, include_narrative: bool = True) -> list[QueryBranch]:
    """Map the plan onto BM25 search branches with fusion weights.

    Mapping (API has BM25 only):
      bm25_weight   -> original narrative + keyword/entity bag
      splade_weight -> concatenated sparse_big_query (NCSU-style)
      dense_weight  -> evidence sketches (HyDE stand-in via BM25)
    """
    branches: list[QueryBranch] = []
    bm25_parts: list[str] = []

    if include_narrative and plan.topic.strip():
        bm25_parts.append(plan.topic.strip())
    bag = keyword_bag(plan)
    if bag:
        bm25_parts.append(bag)

    if bm25_parts:
        share = plan.bm25_weight / len(bm25_parts)
        if include_narrative and plan.topic.strip():
            branches.append(QueryBranch("bm25_narrative", plan.topic.strip(), share))
        if bag:
            branches.append(QueryBranch("bm25_keywords", bag, share))

    if plan.use_concat_sparse_query and plan.sparse_big_query.strip():
        branches.append(
            QueryBranch("splade_concat", plan.sparse_big_query.strip(), plan.splade_weight)
        )
    elif plan.subquestions:
        # Ablation path: per-subquestion sparse fusion if concat disabled.
        share = plan.splade_weight / len(plan.subquestions)
        for i, q in enumerate(plan.subquestions):
            branches.append(QueryBranch(f"splade_sub_{i}", q, share))

    if plan.use_dense_sketches and plan.evidence_sketches:
        share = plan.dense_weight / len(plan.evidence_sketches)
        for i, sketch in enumerate(plan.evidence_sketches):
            branches.append(QueryBranch(f"dense_sketch_{i}", sketch, share))

    # Drop empty / near-duplicate query strings while keeping highest weight.
    deduped: list[QueryBranch] = []
    best: dict[str, QueryBranch] = {}
    for branch in branches:
        key = re.sub(r"\s+", " ", branch.text).strip().lower()
        if not key:
            continue
        prev = best.get(key)
        if prev is None or branch.weight > prev.weight:
            best[key] = branch
    deduped = list(best.values())
    if not deduped:
        deduped = [QueryBranch("bm25_narrative", plan.topic.strip(), 1.0)]
    return deduped
