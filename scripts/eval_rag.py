#!/usr/bin/env python3
"""Evaluate RAG JSONL answers against development nuggets / vital facts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pyserini_client import project_root

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "can", "could", "would",
    "should", "also", "than", "more", "most", "very", "such", "into", "their",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def content_terms(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 2}


def load_answers(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        qid = str(obj["metadata"]["narrative_id"])
        text = " ".join(sent.get("text", "") for sent in obj.get("answer", []))
        answers[qid] = text
    return answers


def load_nuggets(path: Path) -> dict[str, list[dict]]:
    nuggets: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        nuggets[str(obj["qid"])] = obj["nuggets"]
    return nuggets


def nugget_covered(nugget_text: str, answer_terms: set[str], threshold: float) -> bool:
    terms = content_terms(nugget_text)
    if not terms:
        return False
    return (len(terms & answer_terms) / len(terms)) >= threshold


def main() -> None:
    root = project_root()
    default_nuggets = (
        root
        / "trec-rag-data-main/trec-rag-2026/development-data/rag25-dev-nuggets/rag25-dev-nuggets.jsonl"
    )
    default_answers = root / "runs/dev/rag_output_rag25_dev.jsonl"

    parser = argparse.ArgumentParser(description="Lexical nugget coverage proxy.")
    parser.add_argument("--answers", type=Path, default=default_answers)
    parser.add_argument("--nuggets", type=Path, default=default_nuggets)
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Fraction of a nugget's content terms that must appear.")
    args = parser.parse_args()

    answers = load_answers(args.answers)
    nuggets = load_nuggets(args.nuggets)

    vital_hit = vital_total = okay_hit = okay_total = 0
    per_topic: list[tuple[str, float]] = []

    for qid, topic_nuggets in nuggets.items():
        answer_terms = content_terms(answers.get(qid, ""))
        t_vital_hit = t_vital_total = 0
        for nug in topic_nuggets:
            covered = nugget_covered(nug["text"], answer_terms, args.threshold)
            if nug.get("importance") == "vital":
                vital_total += 1
                t_vital_total += 1
                if covered:
                    vital_hit += 1
                    t_vital_hit += 1
            else:
                okay_total += 1
                if covered:
                    okay_hit += 1
        if t_vital_total:
            per_topic.append((qid, t_vital_hit / t_vital_total))

    print(f"Topics scored: {len(per_topic)}")
    print(f"Vital nugget recall: {vital_hit}/{vital_total} = "
          f"{(vital_hit / vital_total if vital_total else 0):.3f}")
    print(f"Okay nugget recall:  {okay_hit}/{okay_total} = "
          f"{(okay_hit / okay_total if okay_total else 0):.3f}")
    if per_topic:
        worst = sorted(per_topic, key=lambda x: x[1])[:5]
        print("Weakest topics (vital recall):", ", ".join(f"{q}={r:.2f}" for q, r in worst))
    print("\nNote: lexical proxy only. Use RAGDoll for real nugget/battle scoring.")


if __name__ == "__main__":
    main()
