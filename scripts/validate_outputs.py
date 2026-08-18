#!/usr/bin/env python3
"""Validate TREC RAG 2026 R and RAG submission files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from pyserini_client import load_topics_tsv, project_root


def validate_retrieval(run_path: Path, topic_ids: set[str]) -> list[str]:
    errors: list[str] = []
    per_topic_last_rank: dict[str, int] = {}
    per_topic_last_score: dict[str, float] = {}
    seen_topics: set[str] = set()

    for lineno, line in enumerate(run_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 6:
            errors.append(f"R line {lineno}: expected 6 columns, got {len(parts)}")
            continue
        topic_id, q0, _docid, rank_s, score_s, _run = parts
        seen_topics.add(topic_id)
        if q0 != "Q0":
            errors.append(f"R line {lineno}: column 2 must be 'Q0', got {q0!r}")
        try:
            rank = int(rank_s)
            score = float(score_s)
        except ValueError:
            errors.append(f"R line {lineno}: rank/score not numeric")
            continue
        if topic_id not in per_topic_last_rank:
            if rank != 1:
                errors.append(f"R topic {topic_id}: first rank must be 1, got {rank}")
        else:
            if rank != per_topic_last_rank[topic_id] + 1:
                errors.append(f"R topic {topic_id}: ranks not consecutive at line {lineno}")
            if score > per_topic_last_score[topic_id]:
                errors.append(f"R topic {topic_id}: score increased at line {lineno}")
        per_topic_last_rank[topic_id] = rank
        per_topic_last_score[topic_id] = score

    missing = topic_ids - seen_topics
    if missing:
        errors.append(f"R: {len(missing)} topics missing from run: {sorted(missing)[:5]}...")
    return errors


def validate_rag(rag_path: Path, topics: dict[str, str]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()

    for lineno, line in enumerate(rag_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"RAG line {lineno}: invalid JSON ({exc})")
            continue

        for key in ("metadata", "references", "answer"):
            if key not in obj:
                errors.append(f"RAG line {lineno}: missing top-level '{key}'")
        meta = obj.get("metadata", {})
        for key in ("team_id", "narrative_id", "narrative", "run_id", "run_desc"):
            if key not in meta:
                errors.append(f"RAG line {lineno}: metadata missing '{key}'")

        qid = str(meta.get("narrative_id", ""))
        seen.add(qid)
        if qid in topics and meta.get("narrative") != topics[qid]:
            errors.append(f"RAG {qid}: narrative does not match topic text exactly")

        references = obj.get("references", [])
        answer = obj.get("answer", [])
        cited: set[int] = set()
        total_words = 0
        for i, sent in enumerate(answer):
            text = sent.get("text", "")
            total_words += len(text.split())
            citations = sent.get("citations", [])
            if len(citations) > 3:
                errors.append(f"RAG {qid} sent {i}: >3 citations")
            for c in citations:
                if not isinstance(c, int) or c < 0 or c >= len(references):
                    errors.append(f"RAG {qid} sent {i}: citation {c} out of range")
                else:
                    cited.add(c)
        if total_words > 1024:
            errors.append(f"RAG {qid}: answer has {total_words} words (>1024)")
        # Uncited references are explicitly allowed in TREC RAG 2026
        # (rag-task.md: "uncited references do not hurt the score").
        # Do not treat them as validation errors.

    missing = set(topics) - seen
    if missing:
        errors.append(f"RAG: {len(missing)} topics missing: {sorted(missing)[:5]}...")
    return errors


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Validate R and/or RAG outputs.")
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, default=None)
    parser.add_argument("--rag", type=Path, default=None)
    args = parser.parse_args()

    topic_pairs = load_topics_tsv(args.topics)
    topics = {tid: text for tid, text in topic_pairs}

    all_errors: list[str] = []
    if args.retrieval:
        errs = validate_retrieval(args.retrieval, set(topics))
        print(f"Retrieval: {'OK' if not errs else str(len(errs)) + ' issues'}")
        all_errors += errs
    if args.rag:
        errs = validate_rag(args.rag, topics)
        print(f"RAG: {'OK' if not errs else str(len(errs)) + ' issues'}")
        all_errors += errs

    for err in all_errors[:40]:
        print("  -", err)
    if not all_errors:
        print("All checks passed.")


if __name__ == "__main__":
    main()
