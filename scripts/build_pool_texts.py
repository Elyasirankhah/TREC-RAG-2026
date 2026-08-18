#!/usr/bin/env python3
"""Build a flat docid->text cache for a TREC run (BM25 cache + document fetch)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cross_encoder_rerank import build_texts_for_docids
from pyserini_client import project_root
from robust_ensemble_tune import load_ranked_run


def fetch_missing(docids: set[str], existing: dict[str, str]) -> dict[str, str]:
    missing = [d for d in docids if d not in existing or not existing[d]]
    if not missing:
        return {}
    from pyserini_client import fetch_document

    out: dict[str, str] = {}
    for i, docid in enumerate(missing, 1):
        try:
            payload = fetch_document(docid)
        except Exception as exc:  # noqa: BLE001
            print(f"  fetch fail {docid}: {exc}", flush=True)
            continue
        doc = payload.get("doc", payload)
        if isinstance(doc, dict):
            text = doc.get("text") or doc.get("contents") or ""
        else:
            text = str(doc or "")
        text = " ".join(str(text).split())
        if text:
            out[docid] = text
        if i % 50 == 0 or i == len(missing):
            print(f"  fetched {i}/{len(missing)}", flush=True)
    return out


def main() -> None:
    root = project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=root / "runs/cache/bm25")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; pass --overwrite")

    runs = load_ranked_run(args.input_run)
    needed: set[str] = set()
    for docs in runs.values():
        for docid, _ in docs[: args.depth]:
            needed.add(docid)
    print(f"need {len(needed)} unique docids from {len(runs)} topics", flush=True)

    texts = build_texts_for_docids(needed, cache_dir=args.cache_dir, max_chars=args.max_chars)
    print(f"from BM25 cache: {len(texts)}", flush=True)
    if not args.no_fetch:
        fetched = fetch_missing(needed, texts)
        texts.update(fetched)
        print(f"after fetch: {len(texts)}", flush=True)

    missing = sorted(needed - set(texts))
    if missing:
        print(f"WARNING: {len(missing)} docids still have no text (e.g. {missing[:5]})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps({"_flat": texts}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(args.output)
    print(f"Wrote {args.output} docs={len(texts)}", flush=True)


if __name__ == "__main__":
    main()
