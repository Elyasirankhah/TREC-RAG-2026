#!/usr/bin/env python3
"""Listwise GPT reranking over a Phase-1 candidate head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval_retrieval import load_qrels, ndcg_at_k
from fusion import collect_doc_texts
from llm_client import llm_json, resolve_phase1_model
from phase1_planner import build_query_branches, plan_topic
from pyserini_client import load_topics_tsv, project_root
from rerank_phase1_doc import load_ranked_run
from search_cache import cached_search
from trec_io import write_run

PROMPT_VERSION = "listwise-v1"


def cache_path(cache_dir: Path, topic_id: str, model: str, depth: int) -> Path:
    digest = hashlib.sha256(
        f"{PROMPT_VERSION}|{model}|{topic_id}|{depth}".encode()
    ).hexdigest()
    return cache_dir / f"{digest}.json"


def listwise_order(
    topic: str,
    facets: list[str],
    candidates: list[tuple[str, str]],
    *,
    model: str,
    max_chars: int,
) -> list[str]:
    passages = []
    for idx, (_docid, text) in enumerate(candidates):
        snippet = " ".join(text.split())[:max_chars]
        passages.append(f"[{idx}] {snippet}")
    prompt = f"""Rank these passages from most to least useful for answering the
complete information need. Compare passages directly. Put specific,
fact-bearing passages that answer requested facets ahead of generic,
repetitive, tangential, or keyword-stuffed passages. A passage can rank highly
if it strongly answers one important facet; it need not answer every facet.

Information need:
{topic}

Requested facets:
{json.dumps(facets, ensure_ascii=False)}

Passages:
{chr(10).join(passages)}

Return every passage id exactly once, best first:
{{"ordered_ids":[0,1,2,...]}}"""
    parsed = llm_json(
        prompt,
        model=model,
        system="You are a strict TREC listwise passage reranker. Return valid JSON only.",
        temperature=0,
        max_output_tokens=max(500, len(candidates) * 12),
        timeout=240,
    )
    raw = parsed.get("ordered_ids") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        raise RuntimeError("Listwise reranker returned no ordered_ids")
    seen: set[int] = set()
    ordered: list[str] = []
    for value in raw:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            ordered.append(candidates[idx][0])
    for idx, (docid, _text) in enumerate(candidates):
        if idx not in seen:
            ordered.append(docid)
    return ordered


def blend_orders(
    base: list[tuple[str, float]],
    listwise: list[str],
    *,
    alpha: float,
) -> list[tuple[str, float]]:
    n = len(listwise)
    base_pos = {docid: idx for idx, (docid, _score) in enumerate(base[:n])}
    list_pos = {docid: idx for idx, docid in enumerate(listwise)}
    scored = []
    for docid, _score in base[:n]:
        base_pct = 100.0 * (n - base_pos[docid]) / max(1, n)
        list_pct = 100.0 * (n - list_pos.get(docid, n)) / max(1, n)
        scored.append((docid, alpha * list_pct + (1.0 - alpha) * base_pct))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored + base[n:]


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Listwise rerank Phase-1 run.")
    parser.add_argument(
        "--input-run",
        type=Path,
        default=root / "runs/dev/phase1_doc_rerank/r_output_phase1_rerank_a0.55.tsv",
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
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--max-chars", type=int, default=600)
    parser.add_argument("--hits", type=int, default=200)
    parser.add_argument("--model", default=None)
    parser.add_argument("--cache-dir", type=Path, default=root / "runs/cache/bm25")
    parser.add_argument("--plan-cache-dir", type=Path, default=root / "runs/cache/phase1_plans")
    parser.add_argument("--list-cache-dir", type=Path, default=root / "runs/cache/listwise")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs/dev/phase1_listwise",
    )
    args = parser.parse_args()

    model = args.model or resolve_phase1_model()
    topics = load_topics_tsv(args.topics)
    base_runs = load_ranked_run(args.input_run)
    orders: dict[str, list[str]] = {}

    for index, (topic_id, topic) in enumerate(topics, start=1):
        base = base_runs[topic_id]
        plan = plan_topic(topic, model=model, cache_dir=args.plan_cache_dir)
        branches = build_query_branches(plan)
        rankings = [
            cached_search(
                branch.text,
                hits=args.hits,
                cache_dir=args.cache_dir,
                query_delay=0,
            )
            for branch in branches
        ]
        texts = collect_doc_texts(rankings)
        head = [
            (docid, texts[docid])
            for docid, _score in base[: args.depth]
            if docid in texts
        ]
        path = cache_path(args.list_cache_dir, topic_id, model, args.depth)
        if path.exists():
            order = json.loads(path.read_text(encoding="utf-8"))["ordered_docids"]
            print(f"[{index}/{len(topics)}] {topic_id}: cached", flush=True)
        else:
            order = listwise_order(
                topic,
                plan.subquestions or [topic],
                head,
                model=model,
                max_chars=args.max_chars,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"ordered_docids": order}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[{index}/{len(topics)}] {topic_id}: ranked {len(order)}", flush=True)
        orders[topic_id] = order

    qrels = load_qrels(args.qrels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best: tuple[float, float, Path] | None = None
    for alpha in (0.25, 0.40, 0.55, 0.70, 0.85, 1.00):
        rows = []
        scores = []
        for topic_id, _topic in topics:
            ranked = blend_orders(base_runs[topic_id], orders[topic_id], alpha=alpha)
            scores.append(
                ndcg_at_k([docid for docid, _ in ranked], qrels[topic_id], 30)
            )
            for rank, (docid, score) in enumerate(ranked[:100], start=1):
                rows.append((topic_id, docid, rank, score, f"phase1-listwise-a{alpha:.2f}"))
        mean = sum(scores) / len(scores)
        out = args.output_dir / f"r_output_phase1_listwise_a{alpha:.2f}.tsv"
        write_run(out, rows)
        print(f"alpha={alpha:.2f} nDCG@30={mean:.4f} -> {out}")
        if best is None or mean > best[0]:
            best = (mean, alpha, out)
    assert best is not None
    print(f"BEST nDCG@30={best[0]:.4f} alpha={best[1]:.2f} run={best[2]}")


if __name__ == "__main__":
    main()
