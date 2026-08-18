#!/usr/bin/env python3
"""API-backed UMBRELA pointwise judge (laptop fallback)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval_retrieval import load_qrels, ndcg_at_k
from llm_client import llm_text, resolve_phase1_model
from pyserini_client import project_root
from rerank_phase1_doc import load_ranked_run
from trec_io import write_run
from umbrela_aligned_rerank import (
    PROMPT_VERSION,
    atomic_write_json,
    blended_ranking,
    build_prompt,
    load_flat_texts,
    parse_grade,
)

STRICT_SUFFIX = (
    "\n\nReply with one line only in exactly this format: ##final score: X "
    "where X is an integer 0, 1, 2, 3, or 4. No explanation."
)


def load_narratives(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "id" not in payload:
        # Allow map form.
        return {str(k): v for k, v in payload.items() if isinstance(v, dict)}
    if not isinstance(payload, list):
        raise SystemExit(f"Unexpected narratives format in {path}")
    return {str(item["id"]): item for item in payload}


def narratives_fingerprint(path: Path, narratives: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(path.resolve().as_posix().encode())
    for tid in sorted(narratives):
        item = narratives[tid]
        subs = [
            str(s.get("text") or "").strip()
            for s in item.get("sub_narratives", [])
            if isinstance(s, dict)
        ]
        digest.update(tid.encode())
        digest.update("\0".join(subs).encode())
    return digest.hexdigest()[:16]


def judge_one(
    prompt: str,
    *,
    model: str,
    timeout: float,
    retries: int = 8,
) -> int:
    """Judge one passage; retry with backoff on Azure rate limits."""
    # Reasoning deployments consume part of this budget before emitting the
    # short final grade. A 64-token cap can therefore return empty text.
    output_tokens = 1200 if ("5.6" in model or "sol" in model.lower()) else 64
    retry_tokens = 800 if ("5.6" in model or "sol" in model.lower()) else 16
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            text = llm_text(
                prompt,
                model=model,
                system=None,
                temperature=0,
                max_output_tokens=output_tokens,
                timeout=timeout,
            )
            try:
                return parse_grade(text)
            except ValueError:
                text2 = llm_text(
                    prompt + STRICT_SUFFIX,
                    model=model,
                    system=None,
                    temperature=0,
                    max_output_tokens=retry_tokens,
                    timeout=timeout,
                )
                return parse_grade(text2)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            msg = str(exc).lower()
            rate_limited = (
                "429" in msg
                or "rate limit" in msg
                or "too_many_requests" in msg
            )
            if attempt >= retries or not rate_limited:
                break
            # Azure TPM/RPM: back off aggressively; honor typical 10–60s windows.
            sleep_s = min(90.0, 4.0 * (2**attempt))
            time.sleep(sleep_s)
    assert last_error is not None
    raise last_error


def evaluate_run(
    ranked: dict[str, list[tuple[str, float]]],
    qrels_path: Path,
    *,
    k: int = 30,
) -> float:
    qrels = load_qrels(qrels_path)
    scores = [
        ndcg_at_k([docid for docid, _ in ranked[tid]], qrels[tid], k)
        for tid in sorted(qrels)
        if tid in ranked
    ]
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Azure/OpenAI UMBRELA judge.")
    parser.add_argument(
        "--input-run",
        type=Path,
        default=root / "runs/dev/r_output_rag25_dev_phase1_doc_tuned.tsv",
    )
    parser.add_argument(
        "--pool-text-cache",
        type=Path,
        default=root / "runs/cache/pool_texts/dev.json",
    )
    parser.add_argument(
        "--narratives",
        type=Path,
        required=True,
        help="Narratives JSON (official or generated).",
    )
    parser.add_argument(
        "--score-cache",
        type=Path,
        required=True,
        help="Output grade cache (must be unique per narratives file).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write alpha-blended runs and print nDCG.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--max-topics", type=int, default=0)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--tag", default="umbrela-llm")
    args = parser.parse_args()

    model = args.model or resolve_phase1_model()
    narratives = load_narratives(args.narratives)
    fp = narratives_fingerprint(args.narratives, narratives)
    runs = load_ranked_run(args.input_run)
    texts = load_flat_texts(args.pool_text_cache)
    topic_ids = [tid for tid in runs if tid in narratives]
    if args.max_topics:
        topic_ids = topic_ids[: args.max_topics]

    cache_version = f"{PROMPT_VERSION}|llm|{model}|narr={fp}"
    cache: dict[str, Any] = {
        "version": cache_version,
        "model": model,
        "narratives": str(args.narratives),
        "narratives_fp": fp,
        "grades": {},
    }
    if args.score_cache.exists():
        loaded = json.loads(args.score_cache.read_text(encoding="utf-8"))
        if loaded.get("version") == cache_version:
            cache = loaded
            print(
                f"Resuming cache with {len(cache.get('grades', {}))} topics",
                flush=True,
            )
        else:
            print(
                "WARNING: existing cache version mismatch; starting fresh "
                f"(old={loaded.get('version')!r})",
                flush=True,
            )

    print(
        f"topics={len(topic_ids)} depth={args.depth} texts={len(texts)} "
        f"model={model} narr_fp={fp}",
        flush=True,
    )

    all_grades: dict[str, dict[str, int]] = cache.setdefault("grades", {})
    for topic_index, topic_id in enumerate(topic_ids, start=1):
        candidates = runs[topic_id][: args.depth]
        if args.max_docs:
            candidates = candidates[: args.max_docs]
        existing = {
            str(docid): int(grade)
            for docid, grade in all_grades.get(topic_id, {}).items()
        }
        pending = [(d, s) for d, s in candidates if d not in existing]
        print(
            f"[{topic_index}/{len(topic_ids)}] {topic_id}: "
            f"cached={len(existing)} pending={len(pending)}",
            flush=True,
        )
        if not pending:
            continue

        grades = dict(existing)
        failed: list[str] = []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for docid, _score in pending:
                text = texts.get(docid, "")
                if not text:
                    print(f"  WARNING: missing text {docid}", flush=True)
                    continue
                prompt = build_prompt(narratives[topic_id], text[: args.max_chars])
                futures[
                    executor.submit(
                        judge_one, prompt, model=model, timeout=args.timeout
                    )
                ] = docid
            done = 0
            for future in as_completed(futures):
                docid = futures[future]
                try:
                    grades[docid] = future.result()
                except Exception as exc:  # noqa: BLE001
                    failed.append(docid)
                    print(f"  WARNING: {docid} failed: {exc}", flush=True)
                done += 1
                if done % 20 == 0 or done == len(futures):
                    all_grades[topic_id] = grades
                    atomic_write_json(args.score_cache, cache)
                    print(
                        f"  {topic_id}: judged {done}/{len(futures)} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )
        all_grades[topic_id] = grades
        atomic_write_json(args.score_cache, cache)
        if failed:
            print(
                f"  WARNING: {len(failed)} failed; rerun to retry",
                flush=True,
            )

    atomic_write_json(args.score_cache, cache)
    print(f"Wrote grades -> {args.score_cache}", flush=True)

    if args.output_dir is None:
        return

    qrels_dir = (
        root
        / "trec-rag-data-main/trec-rag-2026/development-data"
        / "rag25-dev-umbrela-qrels"
    )
    qrels_files = sorted(qrels_dir.glob("*.qrels"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for alpha in (0.50, 0.70, 0.85, 0.95, 1.00):
        ranked: dict[str, list[tuple[str, float]]] = {}
        rows = []
        for topic_id in topic_ids:
            candidates = runs[topic_id][: args.depth]
            ranking = blended_ranking(
                candidates,
                all_grades.get(topic_id, {}),
                alpha=alpha,
            )
            ranked[topic_id] = ranking
            for rank, (docid, score) in enumerate(ranking, start=1):
                rows.append(
                    (topic_id, docid, rank, score, f"{args.tag}-a{alpha:.2f}")
                )
        out = args.output_dir / f"r_output_{args.tag}_a{alpha:.2f}.tsv"
        write_run(out, rows)
        assessor_scores = {
            path.stem: evaluate_run(ranked, path) for path in qrels_files
        }
        robust_mean = (
            sum(assessor_scores.values()) / len(assessor_scores)
            if assessor_scores
            else 0.0
        )
        results.append((robust_mean, alpha, assessor_scores, out))
        display = "  ".join(
            f"{name.replace('rag25-climbmix-umbrela-', '')}={score:.4f}"
            for name, score in assessor_scores.items()
        )
        print(f"alpha={alpha:.2f} robust_mean={robust_mean:.4f}  {display}")

    best = max(results, key=lambda item: item[0])
    print(
        f"BEST robust_mean={best[0]:.4f} alpha={best[1]:.2f} run={best[3]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
