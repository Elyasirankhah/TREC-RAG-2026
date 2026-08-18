#!/usr/bin/env python3
"""Apply frozen ensemble weights (r/g/uq/um/cov) without retuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_client import resolve_phase1_model
from pyserini_client import load_topics_tsv, project_root
from robust_ensemble_tune import load_pointwise, load_ranked_run, load_umbrela_grades
from trec_io import write_run

DEFAULT_WEIGHTS = {"r": 0.3, "g": 0.1, "uq": 0.4, "um": 0.1, "cov": 0.1}


def parse_weights(items: list[str] | None) -> dict[str, float]:
    if not items:
        return dict(DEFAULT_WEIGHTS)
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"bad --weight {item!r}; expected key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main() -> None:
    root = project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topics", type=Path, required=True)
    ap.add_argument("--input-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--grades-cache", type=Path, default=root / "runs/cache/phase1_grades")
    ap.add_argument("--umbrela-qwen", type=Path, default=None)
    ap.add_argument("--umbrela-ministral", type=Path, default=None)
    ap.add_argument("--coverage-cache", type=Path, default=None)
    ap.add_argument("--umbrela", type=Path, default=None, help="Single-judge UMBRELA as signal 'u'.")
    ap.add_argument("--weight", action="append", default=None, help="key=value; repeatable.")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--write-top", type=int, default=100)
    ap.add_argument("--model", default=None, help="Must match grade-cache model key.")
    ap.add_argument("--run-id", default="fixed-blend")
    ap.add_argument(
        "--renormalize",
        action="store_true",
        help="Redistribute weights of missing signals over available ones.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite")

    model = resolve_phase1_model(root, override=args.model)
    topics = load_topics_tsv(args.topics)
    runs = load_ranked_run(args.input_run)
    weights = parse_weights(args.weight)

    uq = load_umbrela_grades(args.umbrela_qwen) if args.umbrela_qwen else {}
    um = load_umbrela_grades(args.umbrela_ministral) if args.umbrela_ministral else {}
    cov = load_umbrela_grades(args.coverage_cache) if args.coverage_cache else {}
    u = load_umbrela_grades(args.umbrela) if args.umbrela else {}

    available = {"r", "g"}
    if uq:
        available.add("uq")
    if um:
        available.add("um")
    if cov:
        available.add("cov")
    if u:
        available.add("u")

    missing = [k for k, w in weights.items() if w > 0 and k not in available]
    if missing and args.renormalize:
        keep = {k: w for k, w in weights.items() if k in available and w > 0}
        total = sum(keep.values())
        if total <= 0:
            raise SystemExit("no available signals to renormalize onto")
        weights = {k: w / total for k, w in keep.items()}
        print(f"renormalized weights (dropped {missing}): {weights}", flush=True)
    elif missing:
        print(
            f"WARNING: signals missing {missing}; their weight contributes 0. "
            f"Pass --renormalize to redistribute.",
            flush=True,
        )

    rows: list[tuple[str, str, int, float, str]] = []
    n_topics = 0
    for tid, _narrative in topics:
        candidates = runs.get(tid, [])
        if not candidates:
            continue
        n_topics += 1
        head = candidates[: args.depth]
        tail = candidates[args.depth :]
        docids = [d for d, _ in head]
        n = len(docids)
        sig: dict[str, dict[str, float]] = {
            "r": {d: (n - i) / max(1, n) for i, d in enumerate(docids)},
            "g": {
                d: load_pointwise(args.grades_cache, tid, model).get(d, 0.0) / 100.0
                for d in docids
            },
        }
        if "uq" in available:
            sig["uq"] = {d: float(uq.get(tid, {}).get(d, 0)) / 4.0 for d in docids}
        if "um" in available:
            sig["um"] = {d: float(um.get(tid, {}).get(d, 0)) / 4.0 for d in docids}
        if "cov" in available:
            sig["cov"] = {d: float(cov.get(tid, {}).get(d, 0)) / 4.0 for d in docids}
        if "u" in available:
            sig["u"] = {d: float(u.get(tid, {}).get(d, 0)) / 4.0 for d in docids}

        scored: list[tuple[str, float]] = []
        for d in docids:
            s = 0.0
            for key, weight in weights.items():
                if weight == 0.0 or key not in sig:
                    continue
                s += weight * sig[key][d]
            scored.append((d, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored.extend(tail)
        for rank, (docid, score) in enumerate(scored[: args.write_top], start=1):
            rows.append((tid, docid, rank, score, args.run_id))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_run(args.output, rows)
    print(
        f"Wrote {args.output} topics={n_topics} rows={len(rows)} "
        f"weights={weights} model={model}",
        flush=True,
    )


if __name__ == "__main__":
    main()
