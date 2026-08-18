#!/usr/bin/env python3
"""Lightweight text normalization and content-term helpers."""

from __future__ import annotations

import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "im",
    "you",
    "we",
    "they",
    "he",
    "she",
    "my",
    "our",
    "your",
    "their",
    "how",
    "what",
    "why",
    "when",
    "where",
    "which",
    "who",
    "can",
    "could",
    "would",
    "should",
    "about",
    "into",
    "also",
    "like",
    "so",
    "some",
    "any",
    "more",
    "understand",
    "help",
    "trying",
    "interested",
    "want",
    "seeking",
    "hoping",
    "explain",
    "gain",
    "deeper",
    "specifically",
    "along",
    "such",
    "both",
    "other",
    "forms",
    "learn",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2}


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "st",
        "sr",
        "jr",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "u.s",
        "u.k",
        "fig",
        "no",
        "eq",
        "cf",
    }
)


def split_sentences_with_offsets(text: str) -> list[dict]:
    """Sentence splitter that preserves original character offsets.

    Returns ``[{"char_start": int, "char_end": int, "text": str}, ...]``
    where ``text`` is the original substring (no whitespace normalization
    beyond stripping the returned text). Offsets are on the original input.

    Abbreviations like ``Dr.`` or ``e.g.`` no longer force a sentence break.
    This is important for evidence spans in scientific/policy corpora.
    """
    if not text:
        return []
    n = len(text)
    out: list[dict] = []
    start = 0
    i = 0
    while i < n:
        ch = text[i]
        if ch in ".!?":
            j = i + 1
            while j < n and text[j] in ".!?":
                j += 1
            after = j
            while after < n and text[after].isspace():
                after += 1
            look_back = text[max(0, i - 10) : i].lower()
            token_start = len(look_back)
            for k in range(len(look_back) - 1, -1, -1):
                if not (look_back[k].isalnum() or look_back[k] == "."):
                    token_start = k + 1
                    break
                if k == 0:
                    token_start = 0
            prev_token = look_back[token_start:].rstrip(".")
            is_abbrev = prev_token in _ABBREVIATIONS
            is_end = after >= n or (
                after < n
                and (text[after].isupper() or text[after].isdigit() or text[after] in "\"'(")
            )
            if not is_abbrev and is_end:
                sent = text[start:j]
                stripped = sent.strip()
                if stripped:
                    offset = sent.find(stripped[0])
                    cs = start + offset
                    ce = cs + len(stripped)
                    out.append({"char_start": cs, "char_end": ce, "text": stripped})
                start = after
                i = after
                continue
        i += 1
    if start < n:
        sent = text[start:n]
        stripped = sent.strip()
        if stripped:
            offset = sent.find(stripped[0])
            cs = start + offset
            ce = cs + len(stripped)
            out.append({"char_start": cs, "char_end": ce, "text": stripped})
    return out


def keyword_query(text: str, *, max_terms: int = 12) -> str:
    terms = [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]
    if not terms:
        return ""
    counts = Counter(terms)
    top = [t for t, _ in counts.most_common(max_terms)]
    return " ".join(top)


def dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = re.sub(r"\s+", " ", query).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(query.strip())
    return deduped
