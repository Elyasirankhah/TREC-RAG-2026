#!/usr/bin/env python3
"""Keyword expansion and HyDE helpers for Phase-1 query planning."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from llm_client import chat_completion_json, llm_text, resolve_api_key
from pyserini_client import project_root
from text_utils import (
    content_terms,
    dedupe_queries,
    keyword_query,
    split_sentences,
)

CLAUSE_SPLIT_RE = re.compile(
    r"\b(?:i also|also want|along with|as well as|plus|in addition|"
    r"particularly concerning|specifically|could you explain|"
    r"can you help me understand|what should we|how should we)\b",
    re.IGNORECASE,
)
QUESTION_START_RE = re.compile(
    r"^(?:how|what|why|when|where|which|who|can|could|should|would|do|does|did|is|are)\b",
    re.IGNORECASE,
)


def extract_question_sentences(text: str) -> list[str]:
    questions: list[str] = []
    for sentence in split_sentences(text):
        cleaned = sentence.strip()
        if not cleaned:
            continue
        if cleaned.endswith("?"):
            questions.append(cleaned)
            continue
        if QUESTION_START_RE.match(cleaned) and len(cleaned) >= 30:
            questions.append(cleaned)
    return questions


def split_narrative_clauses(text: str, *, min_chars: int = 35) -> list[str]:
    parts = CLAUSE_SPLIT_RE.split(text)
    clauses: list[str] = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip(" ,.;")
        if len(cleaned) >= min_chars:
            clauses.append(cleaned)
    return clauses


def phrase_queries(text: str, *, max_phrases: int = 2) -> list[str]:
    terms = sorted(content_terms(text))
    if len(terms) < 2:
        return []

    bigram_counts: dict[str, int] = {}
    for idx in range(len(terms) - 1):
        bigram = f"{terms[idx]} {terms[idx + 1]}"
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    ranked = sorted(bigram_counts.items(), key=lambda item: (-item[1], -len(item[0])))
    phrases: list[str] = []
    for phrase, _count in ranked:
        if phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return phrases


def rule_based_expansions(text: str) -> list[str]:
    expansions: list[str] = []
    kw = keyword_query(text, max_terms=14)
    if kw:
        expansions.append(kw)

    # Shorter keyword bag for focused BM25 queries.
    short_kw = keyword_query(text, max_terms=8)
    if short_kw and short_kw != kw:
        expansions.append(short_kw)

    expansions.extend(phrase_queries(text))
    return expansions


def _cache_path(cache_dir: Path, text: str) -> Path:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"llm_queries_{digest}.json"


def _load_llm_cache(cache_dir: Path, text: str) -> list[str] | None:
    path = _cache_path(cache_dir, text)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries")
    if isinstance(queries, list):
        return [str(q) for q in queries if str(q).strip()]
    return None


def _save_llm_cache(cache_dir: Path, text: str, queries: list[str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, text)
    path.write_text(
        json.dumps({"queries": queries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_openai_key(root: Path | None = None) -> str | None:
    return resolve_api_key(root)


def llm_keyword_expansion(
    text: str,
    *,
    cache_dir: Path | None = None,
    model: str | None = None,
) -> list[str]:
    root = project_root()
    cache_dir = cache_dir or (root / "runs/cache/query_plan")
    cached = _load_llm_cache(cache_dir, text)
    if cached is not None:
        return cached

    api_key = _resolve_openai_key(root)
    if not api_key:
        return []

    prompt = (
        "You help build web search queries for a retrieval system.\n"
        "Given a user narrative, return JSON with:\n"
        '- "keywords": 10-14 comma-free search terms (nouns/phrases, no stopwords)\n'
        '- "queries": 2-4 short BM25 search strings (<= 20 words each)\n'
        "Focus on concrete entities, topics, and sub-questions.\n\n"
        f"Narrative:\n{text}\n"
    )

    try:
        parsed = chat_completion_json(prompt, model=model)
    except RuntimeError:
        return []

    queries: list[str] = []
    keywords = parsed.get("keywords")
    if isinstance(keywords, list):
        terms = [str(k).strip() for k in keywords if str(k).strip()]
        if terms:
            queries.append(" ".join(terms[:14]))
    elif isinstance(keywords, str) and keywords.strip():
        queries.append(keywords.strip())

    extra = parsed.get("queries")
    if isinstance(extra, list):
        queries.extend(str(q).strip() for q in extra if str(q).strip())

    queries = dedupe_queries(queries)
    if queries:
        _save_llm_cache(cache_dir, text, queries)
    return queries


def decompose_narrative(text: str) -> list[str]:
    """Rule-based narrative decomposition (Keystone / CFDA / GRILL style)."""
    queries: list[str] = []
    queries.append(text.strip())

    sentences = split_sentences(text)
    for sentence in sentences[:2]:
        if len(sentence) >= 25:
            queries.append(sentence)

    queries.extend(extract_question_sentences(text))
    queries.extend(split_narrative_clauses(text))
    queries.extend(rule_based_expansions(text))
    return dedupe_queries(queries)


def _hyde_cache_path(cache_dir: Path, text: str) -> Path:
    digest = hashlib.sha256(("hyde::" + text).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"hyde_{digest}.json"


def hyde_query(
    text: str,
    *,
    cache_dir: Path | None = None,
    model: str | None = None,
    max_words: int = 120,
) -> str:
    """HyDE-as-text: ask the LLM for a short hypothetical answer, used as a BM25 query.

    This is a cheap, index-free stand-in for HyDE dense embeddings: the generated
    answer text carries topical vocabulary that improves lexical (BM25) recall.
    Returns "" when no LLM key is configured or on failure.
    """
    root = project_root()
    cache_dir = cache_dir or (root / "runs/cache/query_plan")
    path = _hyde_cache_path(cache_dir, text)
    if path.exists():
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("hyde", ""))
        except json.JSONDecodeError:
            pass

    if not resolve_api_key(root):
        return ""

    prompt = (
        "Write a concise, factual passage (<= "
        f"{max_words} words) that could plausibly answer the user's information "
        "need below. Use concrete terminology, entities, and specifics. "
        "Do not add citations or preamble.\n\n"
        f"Information need:\n{text}\n"
    )
    try:
        answer = llm_text(
            prompt,
            model=model,
            system="You write dense, on-topic factual passages.",
            max_output_tokens=300,
        )
    except (RuntimeError, ValueError):
        return ""

    answer = " ".join(answer.split())
    if answer:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"hyde": answer}, ensure_ascii=False), encoding="utf-8")
    return answer


def gap_queries(
    text: str,
    covered_text: str,
    *,
    model: str | None = None,
    max_gaps: int = 3,
) -> list[str]:
    """GRILL-style gap analysis: given what we've already retrieved, ask the LLM
    which facets are still missing and turn them into new BM25 queries."""
    if not resolve_api_key():
        return []

    snippet = covered_text[:4000]
    prompt = (
        "A user has this information need:\n"
        f"{text}\n\n"
        "Here is a summary of text already retrieved so far:\n"
        f"{snippet}\n\n"
        "Identify facets of the information need that are NOT yet well covered. "
        f'Return JSON: {{"queries": [up to {max_gaps} short web-search strings '
        "(<= 12 words each) targeting the missing facets]}. "
        "If coverage is already good, return an empty list."
    )
    try:
        parsed = chat_completion_json(prompt, model=model)
    except (RuntimeError, ValueError):
        return []

    out = parsed.get("queries")
    if not isinstance(out, list):
        return []
    return dedupe_queries([str(q).strip() for q in out if str(q).strip()])[:max_gaps]


def build_retrieval_queries(
    text: str,
    *,
    max_queries: int = 8,
    use_llm: bool = False,
    use_hyde: bool = False,
    cache_dir: Path | None = None,
) -> list[str]:
    queries = decompose_narrative(text)
    if use_llm:
        queries.extend(llm_keyword_expansion(text, cache_dir=cache_dir))
    if use_hyde:
        hyde = hyde_query(text, cache_dir=cache_dir)
        if hyde:
            queries.append(hyde)
    queries = dedupe_queries(queries)
    return queries[:max_queries]
