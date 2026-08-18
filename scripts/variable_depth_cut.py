#!/usr/bin/env python3
"""Apply narrative-specific depth using UMBRELA grade thresholds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

Row = tuple[str, str, str, int, float, str]


def load_grades(paths: list[Path]) -> list[dict[str, dict[str, int]]]:
    """Load UMBRELA grade caches: {"grades": {topic: {docid: grade}}}."""
    caches: list[dict[str, dict[str, int]]] = []
    for p in paths:
        payload = json.loads(p.read_text(encoding="utf-8"))
        grades = payload.get("grades", payload)
        if not isinstance(grades, dict):
            raise SystemExit(f"{p}: expected a dict of topic->docid->grade")
        caches.append(grades)
    return caches


def combined_grade(
    caches: list[dict[str, dict[str, int]]],
    topic: str,
    docid: str,
    how: str,
) -> int:
    vals = [c.get(topic, {}).get(docid) for c in caches]
    present = [v for v in vals if v is not None]
    if not present:
        return -1  # ungraded; caller decides
    if how == "max":
        return max(present)
    if how == "min":
        return min(present)
    return round(sum(present) / len(present))


def load_run(path: Path) -> list[tuple[str, str, str, int, float, str]]:
    rows: list[tuple[str, str, str, int, float, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 6:
            raise SystemExit(f"{path}:{lineno} expected 6 cols, got {len(parts)}")
        topic, q0, docid, rank_s, score_s, run_tag = parts
        if q0 != "Q0":
            raise SystemExit(f"{path}:{lineno} col2 must be 'Q0' got {q0!r}")
        rows.append((topic, q0, docid, int(rank_s), float(score_s), run_tag))
    return rows


def group_by_topic(
    rows: list[tuple[str, str, str, int, float, str]]
) -> dict[str, list[tuple[str, str, str, int, float, str]]]:
    by_topic: dict[str, list[tuple[str, str, str, int, float, str]]] = defaultdict(list)
    for r in rows:
        by_topic[r[0]].append(r)
    for tid, rs in by_topic.items():
        rs.sort(key=lambda x: x[3])  # by rank
        expected = list(range(1, len(rs) + 1))
        if [x[3] for x in rs] != expected:
            raise SystemExit(f"topic {tid}: input ranks not consecutive from 1")
    return by_topic


def cut_topic_by_score(
    rows: list[Row],
    *,
    min_depth: int,
    max_depth: int,
    score_frac: float,
) -> list[Row]:
    if not rows:
        return []
    top_score = rows[0][4]
    kept: list[Row] = []
    for r in rows:
        depth = len(kept)
        if depth >= max_depth:
            break
        if depth < min_depth:
            kept.append(r)
            continue
        if top_score > 0 and r[4] < score_frac * top_score:
            break
        kept.append(r)
    return kept


def cut_topic_by_grade(
    rows: list[Row],
    *,
    topic: str,
    caches: list[dict[str, dict[str, int]]],
    threshold: int,
    combine: str,
    min_depth: int,
    max_depth: int,
    keep_ungraded: bool,
) -> tuple[list[Row], int]:
    """Keep docs the judges predict are relevant; survivors keep their order.

    Returns (kept_rows, n_ungraded_encountered).
    """
    head = rows[:max_depth]
    kept: list[Row] = []
    ungraded = 0
    for r in head:
        grade = combined_grade(caches, topic, r[2], combine)
        if grade < 0:
            ungraded += 1
            if keep_ungraded:
                kept.append(r)
            continue
        if grade >= threshold:
            kept.append(r)
    if len(kept) < min_depth:
        # Judges rejected almost everything; fall back to the ranking's own head
        # rather than submitting a near-empty list for this narrative.
        kept = head[:min_depth]
    return kept, ungraded


def rewrite_ranks(
    rows: list[tuple[str, str, str, int, float, str]],
    *,
    new_run_id: str | None,
) -> list[tuple[str, str, str, int, float, str]]:
    out: list[tuple[str, str, str, int, float, str]] = []
    prev_score = float("inf")
    for i, (topic, q0, docid, _rank, score, run_tag) in enumerate(rows, start=1):
        if score > prev_score:
            score = prev_score  # enforce non-increasing without discarding rows
        prev_score = score
        rid = new_run_id if new_run_id else run_tag
        out.append((topic, q0, docid, i, score, rid))
    return out


def format_row(row: tuple[str, str, str, int, float, str]) -> str:
    topic, q0, docid, rank, score, run_tag = row
    return f"{topic} {q0} {docid} {rank} {score:.6f} {run_tag}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--selector",
        choices=["grade", "score"],
        default="grade",
        help="grade=keep judge-predicted-relevant docs (recommended); score=score prefix.",
    )
    ap.add_argument("--min-depth", type=int, default=10)
    ap.add_argument("--max-depth", type=int, default=100)
    ap.add_argument(
        "--grades",
        type=Path,
        action="append",
        default=None,
        help="UMBRELA grade cache JSON; repeat for multiple judges.",
    )
    ap.add_argument("--grade-threshold", type=int, default=2)
    ap.add_argument(
        "--grade-combine", choices=["max", "min", "mean"], default="max"
    )
    ap.add_argument(
        "--drop-ungraded",
        action="store_true",
        help="Drop docs no judge scored (default keeps them).",
    )
    ap.add_argument(
        "--score-frac",
        type=float,
        default=0.30,
        help="Only used with --selector score.",
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="If set, replace col 6 (run tag) with this value.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.input == args.output:
        raise SystemExit("--output must differ from --input; will not overwrite the input.")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite to replace.")
    if not (1 <= args.min_depth <= args.max_depth):
        raise SystemExit("require 1 <= min_depth <= max_depth")
    if not (0.0 <= args.score_frac <= 1.0):
        raise SystemExit("require 0.0 <= score_frac <= 1.0")
    if args.selector == "grade" and not args.grades:
        raise SystemExit("--selector grade requires at least one --grades cache")

    rows = load_run(args.input)
    by_topic = group_by_topic(rows)

    caches = load_grades(args.grades) if args.grades else []
    if args.selector == "grade":
        graded_topics = {t for c in caches for t in c}
        missing = sorted(set(by_topic) - graded_topics)
        if missing:
            raise SystemExit(
                f"{len(missing)} topic(s) have no grades "
                f"(e.g. {missing[:5]}); judge them before cutting."
            )

    kept_by_topic: dict[str, list[Row]] = {}
    total_ungraded = 0
    for tid, trows in by_topic.items():
        if args.selector == "grade":
            kept, ungraded = cut_topic_by_grade(
                trows,
                topic=tid,
                caches=caches,
                threshold=args.grade_threshold,
                combine=args.grade_combine,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                keep_ungraded=not args.drop_ungraded,
            )
            total_ungraded += ungraded
        else:
            kept = cut_topic_by_score(
                trows,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                score_frac=args.score_frac,
            )
        kept_by_topic[tid] = rewrite_ranks(kept, new_run_id=args.run_id)

    depths = [len(v) for v in kept_by_topic.values()]
    depths.sort()
    mean_d = sum(depths) / max(1, len(depths))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for tid in sorted(kept_by_topic, key=lambda x: (len(x), x)):
            for row in kept_by_topic[tid]:
                f.write(format_row(row) + "\n")
    tmp.replace(args.output)

    policy = (
        f"grade>={args.grade_threshold} combine={args.grade_combine}"
        if args.selector == "grade"
        else f"score_frac={args.score_frac}"
    )
    print(
        f"Wrote {args.output} rows={sum(depths)} topics={len(depths)} "
        f"depth min/mean/max={depths[0] if depths else 0}/{mean_d:.1f}/"
        f"{depths[-1] if depths else 0} "
        f"min_depth={args.min_depth} max_depth={args.max_depth} "
        f"selector={args.selector} {policy}"
    )
    if args.selector == "grade" and total_ungraded:
        action = "dropped" if args.drop_ungraded else "kept"
        print(f"  {total_ungraded} ungraded doc(s) {action}")


if __name__ == "__main__":
    main()
