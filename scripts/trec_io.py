#!/usr/bin/env python3
"""TREC runfile read/write helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def write_run(
    output_path: Path,
    rows: list[tuple[str, str, int, float, str]],
    *,
    score_precision: int = 6,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = f"{{score:.{score_precision}f}}"
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for topic_id, docid, rank, score, run_id in rows:
            f.write(f"{topic_id} Q0 {docid} {rank} {fmt.format(score=score)} {run_id}\n")


def load_run(path: Path) -> dict[str, list[str]]:
    run: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        topic_id, _, docid, _, _, _ = parts
        run[topic_id].append(docid)
    return run
