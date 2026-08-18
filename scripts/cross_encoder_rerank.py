#!/usr/bin/env python3
"""Cross-encoder reranking utilities and shared pool-text helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_retrieval import load_qrels, ndcg_at_k
from pyserini_client import load_topics_tsv, project_root
from rerank_phase1_doc import load_ranked_run
from trec_io import write_run


def _doc_text(doc: object) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ("text", "contents", "body", "passage"):
            value = doc.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def build_texts_for_docids(
    needed: set[str], *, cache_dir: Path, max_chars: int
) -> dict[str, str]:
    """One pass over the BM25 disk cache -> docid->text for the needed docids."""
    texts: dict[str, str] = {}
    for path in cache_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for cand in payload.get("candidates") or []:
            docid = cand.get("docid")
            if not docid or docid in texts or docid not in needed:
                continue
            text = " ".join(_doc_text(cand.get("doc")).split())[:max_chars]
            if text:
                texts[docid] = text
        if len(texts) >= len(needed):
            break
    return texts


def load_pool_text_cache(path: Path) -> dict[str, dict[str, str]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def chunk_text(text: str, *, chunk_chars: int, stride: int, max_chunks: int) -> list[str]:
    """Split into overlapping character windows for MaxP-style scoring."""
    words = text.split()
    if not words:
        return []
    joined = " ".join(words)
    if len(joined) <= chunk_chars:
        return [joined]
    chunks: list[str] = []
    start = 0
    while start < len(joined) and len(chunks) < max_chunks:
        chunks.append(joined[start : start + chunk_chars])
        start += stride
    return chunks


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def blend(
    candidates: list[tuple[str, float]],
    ce_scores: dict[str, float],
    *,
    alpha: float,
) -> list[tuple[str, float]]:
    """Blend cross-encoder score (per-topic min-max) with rank percentile."""
    n = len(candidates)
    docids = [docid for docid, _ in candidates]
    ce_norm_list = minmax([ce_scores.get(d, min(ce_scores.values()) if ce_scores else 0.0) for d in docids])
    ce_norm = dict(zip(docids, ce_norm_list))
    scored: list[tuple[str, float]] = []
    for index, (docid, _raw) in enumerate(candidates):
        retrieval = (n - index) / max(1, n)
        ce = ce_norm.get(docid, 0.0)
        scored.append((docid, alpha * ce + (1.0 - alpha) * retrieval))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Cross-encoder rerank Phase-1 pool.")
    parser.add_argument(
        "--input-run",
        type=Path,
        default=root / "runs/dev/r_output_rag25_dev_phase1_doc_tuned.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs/dev/phase1_cross_encoder",
    )
    parser.add_argument(
        "--topics",
        type=Path,
        default=root / "trec-rag-data-main/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=root
        / "trec-rag-data-main/trec-rag-2026/development-data/rag25-dev-umbrela-qrels"
        / "rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels",
    )
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--plan-model", default=None, help="LLM model used for plan cache key")
    parser.add_argument("--rerank-depth", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hits", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", type=Path, default=root / "runs/cache/bm25")
    parser.add_argument("--plan-cache-dir", type=Path, default=root / "runs/cache/phase1_plans")
    parser.add_argument("--score-cache", type=Path, default=root / "runs/cache/cross_encoder")
    parser.add_argument("--pool-text-cache", type=Path, default=root / "runs/cache/pool_texts/dev.json")
    parser.add_argument("--chunk", action="store_true", help="MaxP: score overlapping chunks, take max")
    parser.add_argument("--chunk-chars", type=int, default=1400)
    parser.add_argument("--chunk-stride", type=int, default=1000)
    parser.add_argument("--max-chunks", type=int, default=6)
    parser.add_argument("--max-topics", type=int, default=0)
    args = parser.parse_args()

    topics = load_topics_tsv(args.topics)
    if args.max_topics:
        topics = topics[: args.max_topics]
    input_runs = load_ranked_run(args.input_run)

    score_key = args.model.replace("/", "_")
    mode_tag = f"chunk{args.chunk_chars}s{args.chunk_stride}" if args.chunk else "head"
    score_cache_file = args.score_cache / f"{score_key}_d{args.rerank_depth}_{mode_tag}.json"
    cached_scores: dict[str, dict[str, float]] = {}
    if score_cache_file.exists():
        cached_scores = json.loads(score_cache_file.read_text(encoding="utf-8"))
        print(f"Loaded cached CE scores for {len(cached_scores)} topics")

    need_model = any(
        topic_id not in cached_scores for topic_id, _ in topics if input_runs.get(topic_id)
    )

    # Flat docid->text cache. Build in ONE pass over the BM25 cache for the
    # docids we actually rerank (input-run top-depth), not the whole pool.
    flat_texts: dict[str, str] = load_pool_text_cache(args.pool_text_cache).get("_flat", {}) if args.pool_text_cache.exists() else {}
    needed: set[str] = set()
    for topic_id, _ in topics:
        for docid, _s in input_runs.get(topic_id, [])[: args.rerank_depth]:
            needed.add(docid)
    if need_model and not needed.issubset(flat_texts.keys()):
        print(f"Building text map for {len(needed)} docids (one pass over BM25 cache)...", flush=True)
        flat_texts = build_texts_for_docids(needed, cache_dir=args.cache_dir, max_chars=8000)
        args.pool_text_cache.parent.mkdir(parents=True, exist_ok=True)
        args.pool_text_cache.write_text(json.dumps({"_flat": flat_texts}, ensure_ascii=False), encoding="utf-8")
        print(f"  found texts for {len(flat_texts)}/{len(needed)} docids", flush=True)

    encoder = None
    if need_model:
        from sentence_transformers import CrossEncoder

        device = args.device
        encoder = CrossEncoder(args.model, max_length=args.max_length, device=device)
        print(f"Loaded cross-encoder {args.model} on {encoder.model.device}")

    all_scores: dict[str, dict[str, float]] = dict(cached_scores)
    for index, (topic_id, topic) in enumerate(topics, start=1):
        candidates = input_runs.get(topic_id, [])[: args.rerank_depth]
        if not candidates:
            continue
        if topic_id in all_scores:
            print(f"[{index}/{len(topics)}] {topic_id}: cached CE scores", flush=True)
            continue

        texts = flat_texts

        # Build (query, passage) pairs; MaxP over chunks when --chunk is set.
        pairs: list[tuple[str, str]] = []
        owner: list[str] = []  # docid for each pair (for max-pooling)
        present_docids: list[str] = []
        for docid, _score in candidates:
            text = texts.get(docid)
            if not text:
                continue
            present_docids.append(docid)
            if args.chunk:
                pieces = chunk_text(
                    text,
                    chunk_chars=args.chunk_chars,
                    stride=args.chunk_stride,
                    max_chunks=args.max_chunks,
                )
            else:
                pieces = [" ".join(text.split())[: args.max_chars]]
            for piece in pieces:
                pairs.append((topic, piece))
                owner.append(docid)

        if not pairs:
            print(f"[{index}/{len(topics)}] {topic_id}: no pool texts (skip)", flush=True)
            all_scores[topic_id] = {}
            continue

        raw = encoder.predict(
            pairs,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        doc_score: dict[str, float] = {}
        for docid, s in zip(owner, raw):
            s = float(s)
            if docid not in doc_score or s > doc_score[docid]:
                doc_score[docid] = s
        all_scores[topic_id] = doc_score
        print(
            f"[{index}/{len(topics)}] {topic_id}: scored {len(present_docids)} docs "
            f"({len(pairs)} passages)",
            flush=True,
        )
        score_cache_file.parent.mkdir(parents=True, exist_ok=True)
        score_cache_file.write_text(json.dumps(all_scores, ensure_ascii=False), encoding="utf-8")

    qrels = load_qrels(args.qrels) if args.qrels.exists() else {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best: tuple[float, float, Path] | None = None
    for alpha in (0.30, 0.50, 0.70, 0.85, 1.00):
        rows: list[tuple[str, str, int, float, str]] = []
        scores: list[float] = []
        for topic_id, _topic in topics:
            candidates = input_runs.get(topic_id, [])
            head = candidates[: args.rerank_depth]
            tail = candidates[args.rerank_depth :]
            ranked = blend(head, all_scores.get(topic_id, {}), alpha=alpha)
            ranked.extend(tail)
            if topic_id in qrels:
                scores.append(ndcg_at_k([d for d, _ in ranked], qrels[topic_id], 30))
            for rank, (docid, score) in enumerate(ranked[:100], start=1):
                rows.append((topic_id, docid, rank, score, f"phase1-ce-a{alpha:.2f}"))
        out = args.output_dir / f"r_output_phase1_ce_a{alpha:.2f}.tsv"
        write_run(out, rows)
        mean = sum(scores) / len(scores) if scores else 0.0
        print(f"alpha={alpha:.2f} nDCG@30={mean:.4f} -> {out}")
        if best is None or mean > best[0]:
            best = (mean, alpha, out)

    if best:
        print(f"BEST nDCG@30={best[0]:.4f} alpha={best[1]:.2f} run={best[2]}")


if __name__ == "__main__":
    main()
