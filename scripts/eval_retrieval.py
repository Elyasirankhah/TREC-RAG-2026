#!/usr/bin/env python3
"""Compute mean nDCG@k for a TREC runfile against qrels."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from pyserini_client import project_root
from trec_io import load_run


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        topic_id, _, docid, rel = parts
        qrels[topic_id][docid] = int(rel)
    return qrels


def dcg(rels: list[int]) -> float:
    score = 0.0
    for i, rel in enumerate(rels, start=1):
        if rel <= 0:
            continue
        score += (2**rel - 1) / math.log2(i + 1)
    return score


def ndcg_at_k(run_docs: list[str], qrel: dict[str, int], k: int) -> float:
    gains = [qrel.get(docid, 0) for docid in run_docs[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(gains) / denom


def main() -> None:
    root = project_root()
    default_qrels = (
        root
        / "trec-rag-data-main/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels"
    )
    default_run = root / "runs/dev/r_output_rag25_dev.tsv"

    parser = argparse.ArgumentParser(description="Compute mean nDCG@k for a runfile.")
    parser.add_argument("--run", type=Path, default=default_run)
    parser.add_argument("--qrels", type=Path, default=default_qrels)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    qrels = load_qrels(args.qrels)
    run = load_run(args.run)

    scores: list[float] = []
    for topic_id, topic_qrels in sorted(qrels.items()):
        if topic_id not in run:
            continue
        scores.append(ndcg_at_k(run[topic_id], topic_qrels, args.k))

    mean_ndcg = sum(scores) / len(scores) if scores else 0.0
    print(f"Topics evaluated: {len(scores)}")
    print(f"Mean nDCG@{args.k}: {mean_ndcg:.4f}")


if __name__ == "__main__":
    main()
