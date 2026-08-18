#!/usr/bin/env python3
"""GPT pointwise grading over Phase-1 top-N candidates (restart-safe cache)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from eval_retrieval import load_qrels, ndcg_at_k
from fusion import collect_doc_texts
from llm_client import llm_json, resolve_phase1_model
from phase1_planner import build_query_branches, plan_topic
from pyserini_client import load_topics_tsv, project_root
from search_cache import cached_search
from trec_io import write_run

PROMPT_VERSION = "phase1-rerank-v2"


def load_ranked_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    runs: dict[str, list[tuple[str, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        topic_id, _, docid, rank, score, _ = parts
        runs.setdefault(topic_id, []).append((docid, float(score)))
    return runs


def cache_path(cache_dir: Path, topic_id: str, model: str) -> Path:
    digest = hashlib.sha256(f"{PROMPT_VERSION}|{model}|{topic_id}".encode()).hexdigest()
    return cache_dir / f"{digest}.json"


def load_grades(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("grades", {})
    return {str(k): float(v) for k, v in raw.items()}


def save_grades(path: Path, model: str, grades: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"version": PROMPT_VERSION, "model": model, "grades": grades},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def grade_batch(
    topic: str,
    facets: list[str],
    batch: list[tuple[str, str]],
    *,
    model: str,
    max_chars: int,
) -> dict[str, float]:
    passages = []
    for idx, (_docid, text) in enumerate(batch):
        snippet = " ".join(text.split())[:max_chars]
        passages.append(f"[{idx}] {snippet}")

    prompt = f"""You are a strict TREC passage relevance assessor.

Information need:
{topic}

Important facets:
{json.dumps(facets, ensure_ascii=False)}

Score each passage independently from 0 to 100:
- 0-10: irrelevant or wrong intent
- 11-30: same broad topic but not useful for this information need
- 31-50: useful background/context only
- 51-70: directly answers part of one requested facet
- 71-85: strong, specific evidence for a requested facet
- 86-100: exceptionally direct, detailed, and useful evidence

Judge only what the passage states. Penalize keyword stuffing, unsupported
claims, vague generic prose, tangents, and passages that merely repeat the
question. Do not inflate scores to make passages look relevant. Use the full
scale and score every id.

Passages:
{chr(10).join(passages)}

Return JSON only:
{{"grades":[{{"id":0,"score":0}}, ...]}}"""
    parsed = llm_json(
        prompt,
        model=model,
        system="Return only valid JSON. Assess relevance strictly and independently.",
        temperature=0,
        max_output_tokens=max(500, len(batch) * 60),
        timeout=180,
    )
    items = parsed.get("grades") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("Reranker returned no grades list")

    result: dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item["id"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < len(batch):
            result[batch[idx][0]] = max(0.0, min(100.0, score))
    return result


def blend_ranking(
    candidates: list[tuple[str, float]],
    grades: dict[str, float],
    *,
    llm_alpha: float,
) -> list[tuple[str, float]]:
    n = len(candidates)
    scored: list[tuple[str, float]] = []
    for index, (docid, _raw_score) in enumerate(candidates):
        # Rank percentile is robust to score-scale changes across query sets.
        retrieval = 100.0 * (n - index) / max(1, n)
        llm = grades.get(docid, 0.0)
        score = llm_alpha * llm + (1.0 - llm_alpha) * retrieval
        scored.append((docid, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Rerank Phase-1 doc candidates.")
    parser.add_argument(
        "--input-run",
        type=Path,
        default=root / "runs/dev/r_output_rag25_dev_phase1_doc_tuned.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs/dev/phase1_doc_rerank",
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
    parser.add_argument("--rerank-depth", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--hits", type=int, default=200)
    parser.add_argument("--model", default=None)
    parser.add_argument("--cache-dir", type=Path, default=root / "runs/cache/bm25")
    parser.add_argument("--plan-cache-dir", type=Path, default=root / "runs/cache/phase1_plans")
    parser.add_argument("--grade-cache-dir", type=Path, default=root / "runs/cache/phase1_grades")
    parser.add_argument("--max-topics", type=int, default=0)
    args = parser.parse_args()

    model = args.model or resolve_phase1_model()
    topic_rows = load_topics_tsv(args.topics)
    if args.max_topics:
        topic_rows = topic_rows[: args.max_topics]
    input_runs = load_ranked_run(args.input_run)
    all_grades: dict[str, dict[str, float]] = {}

    print(f"Model: {model}; topics={len(topic_rows)}; rerank_depth={args.rerank_depth}")
    for topic_index, (topic_id, topic) in enumerate(topic_rows, start=1):
        candidates = input_runs.get(topic_id, [])[: args.rerank_depth]
        if not candidates:
            continue
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
        path = cache_path(args.grade_cache_dir, topic_id, model)
        grades = load_grades(path)
        pending = [
            (docid, texts[docid])
            for docid, _ in candidates
            if docid not in grades and docid in texts
        ]
        facets = plan.subquestions or [topic]
        print(
            f"[{topic_index}/{len(topic_rows)}] {topic_id}: "
            f"cached={len(grades)} pending={len(pending)}",
            flush=True,
        )
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            batch_grades = grade_batch(
                topic,
                facets,
                batch,
                model=model,
                max_chars=args.max_chars,
            )
            grades.update(batch_grades)
            save_grades(path, model, grades)
            print(
                f"    graded {min(start + len(batch), len(pending))}/{len(pending)}",
                flush=True,
            )
        all_grades[topic_id] = grades

    qrels = load_qrels(args.qrels) if args.qrels.exists() else {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best: tuple[float, float, Path] | None = None
    for alpha in (0.40, 0.55, 0.70, 0.80, 0.90, 1.00):
        rows: list[tuple[str, str, int, float, str]] = []
        metric_scores: list[float] = []
        for topic_id, _topic in topic_rows:
            candidates = input_runs.get(topic_id, [])
            head = candidates[: args.rerank_depth]
            tail = candidates[args.rerank_depth :]
            ranked = blend_ranking(head, all_grades.get(topic_id, {}), llm_alpha=alpha)
            ranked.extend(tail)
            if topic_id in qrels:
                metric_scores.append(
                    ndcg_at_k([docid for docid, _ in ranked], qrels[topic_id], 30)
                )
            for rank, (docid, score) in enumerate(ranked[:100], start=1):
                rows.append((topic_id, docid, rank, score, f"phase1-rerank-a{alpha:.2f}"))
        out = args.output_dir / f"r_output_phase1_rerank_a{alpha:.2f}.tsv"
        write_run(out, rows)
        mean = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
        print(f"alpha={alpha:.2f} nDCG@30={mean:.4f} -> {out}")
        if best is None or mean > best[0]:
            best = (mean, alpha, out)

    if best:
        print(f"BEST nDCG@30={best[0]:.4f} alpha={best[1]:.2f} run={best[2]}")


if __name__ == "__main__":
    main()
