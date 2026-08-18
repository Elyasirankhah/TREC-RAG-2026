#!/usr/bin/env python3
"""Drop citations judged not_support from a RAG JSONL run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval_rag_citations import (
    cache_key,
    load_pool_texts,
    load_run,
    resolve_citation,
)
from llm_client import resolve_phase1_model
from pyserini_client import project_root


def main() -> None:
    root = project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pool-texts", type=Path, action="append", default=None)
    ap.add_argument(
        "--cache", type=Path, default=root / "runs/cache/citation_support/dev.json"
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument(
        "--drop-partial",
        action="store_true",
        help="Also drop partial_support citations (raises precision, lowers recall).",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.run == args.output:
        raise SystemExit("--output must differ from --run")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite")
    if not args.cache.exists():
        raise SystemExit(
            f"No judgement cache at {args.cache}. Run eval_rag_citations.py first."
        )

    model = resolve_phase1_model(root, override=args.model)
    cache: dict[str, list[str]] = json.loads(args.cache.read_text(encoding="utf-8"))
    pool_paths = args.pool_texts or [
        root / "runs/cache/pool_texts/dev.json",
        root / "runs/cache/pool_texts/dev_deep.json",
    ]
    pool = load_pool_texts(pool_paths)
    records = load_run(args.run)

    drop_labels = {"not_support"}
    if args.drop_partial:
        drop_labels.add("partial_support")

    n_before = n_after = n_unjudged = 0
    n_emptied = 0
    out_lines: list[str] = []

    for rec in records:
        refs = rec.get("references", [])
        answer = rec.get("answer", [])

        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for i, obj in enumerate(answer):
            for c in obj.get("citations", []) or []:
                d = resolve_citation(c, refs)
                if d:
                    groups[d].append((i, obj.get("text", "")))

        labels_by_pair: dict[tuple[int, str], str] = {}
        for docid, items in groups.items():
            if not pool.get(docid):
                continue
            sentences = [s for _, s in items]
            key = cache_key(model, docid, sentences, args.max_chars)
            labels = cache.get(key)
            if labels is None:
                continue
            for (idx, _), lab in zip(items, labels):
                labels_by_pair[(idx, docid)] = lab

        for i, obj in enumerate(answer):
            cits = obj.get("citations", []) or []
            n_before += len(cits)
            kept = []
            for c in cits:
                d = resolve_citation(c, refs)
                lab = labels_by_pair.get((i, d)) if d else None
                if lab is None:
                    n_unjudged += 1
                    kept.append(c)  # never drop something we did not judge
                elif lab not in drop_labels:
                    kept.append(c)
            if cits and not kept:
                n_emptied += 1
            obj["citations"] = kept
            n_after += len(kept)

        out_lines.append(json.dumps(rec, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"  citations {n_before} -> {n_after}  (dropped {n_before - n_after})")
    print(f"  objects left uncited: {n_emptied}")
    if n_unjudged:
        print(f"  kept {n_unjudged} unjudged citation(s) untouched")


if __name__ == "__main__":
    main()
