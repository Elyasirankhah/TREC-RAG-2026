#!/usr/bin/env python3
"""UMBRELA-objective listwise rerank stack with fixed-alpha blend for test."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from eval_retrieval import load_qrels, ndcg_at_k
from llm_client import llm_json, resolve_model
from pyserini_client import load_topics_tsv, project_root
from rerank_phase1_doc import load_ranked_run
from trec_io import write_run
from umbrela_aligned_rerank import (
    atomic_write_json,
    load_flat_texts,
    load_or_download_narratives,
)

PROMPT_VERSION = "umbrela-listwise-topk-v1"


def cache_path(cache_dir: Path, topic_id: str, model: str, depth: int, narr_fp: str) -> Path:
    digest = hashlib.sha256(
        f"{PROMPT_VERSION}|{model}|{topic_id}|{depth}|{narr_fp}".encode()
    ).hexdigest()
    return cache_dir / f"{digest}.json"


def narratives_fp(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def listwise_umbrela(
    *,
    narrative: str,
    subs: list[str],
    candidates: list[tuple[str, str]],
    model: str,
) -> list[str]:
    from llm_client import llm_text

    passages = []
    for idx, (_docid, text) in enumerate(candidates):
        passages.append(f"[{idx}] {text}")
    prompt = f"""Rank these passages for a TREC RAG retrieval task.

Narrative:
{narrative.strip()}

Atomic sub-narratives (a passage is better if it answers MORE of these IN DETAIL;
mere mentions do not count; lots of off-topic content should rank lower):
{json.dumps(subs, ensure_ascii=False)}

Passages:
{chr(10).join(passages)}

Return ONLY valid JSON with every passage id exactly once, best first:
{{"ordered_ids":[0,1,2,...]}}"""

    raw_text = ""
    parsed: Any = None
    # gpt-5.6-sol spends many tokens on hidden reasoning; leave headroom for JSON.
    out_tokens = 4000 if "5.6" in model or "sol" in model.lower() else max(800, len(candidates) * 20)
    try:
        parsed = llm_json(
            prompt,
            model=model,
            system=(
                "You are a strict TREC UMBRELA listwise reranker. "
                "Optimize sub-narrative coverage detail. Return valid JSON only."
            ),
            temperature=0,
            max_output_tokens=out_tokens,
            timeout=300,
        )
    except Exception:
        raw_text = llm_text(
            prompt,
            model=model,
            system=(
                "Return ONLY a JSON object like "
                '{"ordered_ids":[0,1,2,...]} with all ids once. No prose.'
            ),
            temperature=0,
            max_output_tokens=out_tokens,
            timeout=300,
            want_json=False,
        )
        try:
            parsed = _loads_lenient_local(raw_text)
        except Exception:
            parsed = None

    ordered_ids: list[int] = []
    if isinstance(parsed, dict) and isinstance(parsed.get("ordered_ids"), list):
        for value in parsed["ordered_ids"]:
            try:
                ordered_ids.append(int(value))
            except (TypeError, ValueError):
                continue
    elif isinstance(parsed, list):
        for value in parsed:
            try:
                ordered_ids.append(int(value))
            except (TypeError, ValueError):
                continue
    else:
        # Fallback: bracket ids in order of appearance.
        text = raw_text or (json.dumps(parsed) if parsed is not None else "")
        for match in re.finditer(r"\[(\d+)\]|(?<!\d)(\d+)(?!\d)", text):
            val = match.group(1) or match.group(2)
            try:
                idx = int(val)
            except ValueError:
                continue
            if 0 <= idx < len(candidates) and idx not in ordered_ids:
                ordered_ids.append(idx)

    if not ordered_ids:
        raise RuntimeError(f"Could not parse listwise order from model: {raw_text[:400]!r}")

    seen: set[int] = set()
    ordered: list[str] = []
    for idx in ordered_ids:
        if 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            ordered.append(candidates[idx][0])
    for idx, (docid, _text) in enumerate(candidates):
        if idx not in seen:
            ordered.append(docid)
    return ordered


def _loads_lenient_local(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("Could not parse JSON from model output")


def blend(
    base: list[tuple[str, float]],
    listwise: list[str],
    *,
    alpha: float,
) -> list[tuple[str, float]]:
    n = len(listwise)
    base_head = base[:n]
    base_pos = {d: i for i, (d, _) in enumerate(base_head)}
    list_pos = {d: i for i, d in enumerate(listwise)}
    scored = []
    for docid, _ in base_head:
        b = (n - base_pos[docid]) / max(1, n)
        l = (n - list_pos[docid]) / max(1, n) if docid in list_pos else 0.0
        scored.append((docid, alpha * l + (1.0 - alpha) * b))
    scored.sort(key=lambda x: x[1], reverse=True)
    scored.extend(base[n:])
    return scored


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="UMBRELA listwise stack on WIN head.")
    parser.add_argument(
        "--input-run",
        type=Path,
        default=root
        / "runs/dev/robust_ensemble"
        / "r_output_WIN_dual_uq_um_cov_r0.3_g0.1_uq0.4_um0.1_cov0.1.tsv",
    )
    parser.add_argument(
        "--pool-text-cache",
        type=Path,
        default=root / "runs/cache/pool_texts/dev.json",
    )
    parser.add_argument(
        "--narratives",
        type=Path,
        default=root / "runs/cache/umbrela/trec25_narratives.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "runs/cache/umbrela_listwise",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs/dev/robust_listwise_umbrela",
    )
    parser.add_argument(
        "--qrels-dir",
        type=Path,
        default=root
        / "trec-rag-data-main/trec-rag-2026/development-data/rag25-dev-umbrela-qrels",
    )
    parser.add_argument(
        "--topics",
        type=Path,
        default=None,
        help="Topics TSV. Defaults to rag25-topics-dev.tsv for back-compat.",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=None,
        help="If set, write only this alpha blend and skip qrels sweep "
        "(required for blind test). Frozen WIN uses 0.55.",
    )
    parser.add_argument(
        "--run-id-prefix",
        default="umbrela-listwise",
        help="Prefix for run tag / output filenames.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--max-topics", type=int, default=0)
    args = parser.parse_args()

    model = args.model or resolve_model()
    narratives = load_or_download_narratives(
        args.narratives,
        "https://trec.nist.gov/data/rag/"
        "trec25_narratives_final_w_questions_w_sub_narratives_edit_20250822.json",
    )
    nfp = narratives_fp(args.narratives) if args.narratives.exists() else "na"
    runs = load_ranked_run(args.input_run)
    texts = load_flat_texts(args.pool_text_cache)
    topics_path = args.topics or (
        root / "trec-rag-data-main/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"
    )
    topics = load_topics_tsv(topics_path)
    topic_ids = [tid for tid, _ in topics if tid in runs and tid in narratives]
    if args.max_topics:
        topic_ids = topic_ids[: args.max_topics]
    if not topic_ids:
        raise SystemExit(
            f"No overlapping topics between run, narratives, and {topics_path}. "
            "On test, pass --narratives generated_sub_narratives_test.json "
            "and --topics trec_rag_2026_queries.tsv."
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"topics={len(topic_ids)} depth={args.depth} model={model} "
        f"chars={args.max_chars}",
        flush=True,
    )

    listwise_by_topic: dict[str, list[str]] = {}
    for idx, tid in enumerate(topic_ids, start=1):
        path = cache_path(args.cache_dir, tid, model, args.depth, nfp)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            ordered = [str(d) for d in payload.get("ordered_docids", [])]
            if len(ordered) == args.depth:
                listwise_by_topic[tid] = ordered
                print(f"[{idx}/{len(topic_ids)}] {tid}: cached", flush=True)
                continue

        item = narratives[tid]
        subs = [
            str(sub.get("text") or "").strip()
            for sub in item.get("sub_narratives", [])
            if isinstance(sub, dict) and str(sub.get("text") or "").strip()
        ]
        head = runs[tid][: args.depth]
        cands = []
        for docid, _score in head:
            body = " ".join(texts.get(docid, "").split())[: args.max_chars]
            cands.append((docid, body or "(no text)"))
        print(f"[{idx}/{len(topic_ids)}] {tid}: calling {model}...", flush=True)
        try:
            ordered = listwise_umbrela(
                narrative=str(item["narrative"]),
                subs=subs,
                candidates=cands,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}/{len(topic_ids)}] {tid}: FAILED ({exc}); keeping base order", flush=True)
            ordered = [docid for docid, _ in head]
        atomic_write_json(
            path,
            {
                "version": PROMPT_VERSION,
                "model": model,
                "depth": args.depth,
                "ordered_docids": ordered,
            },
        )
        listwise_by_topic[tid] = ordered
        print(f"[{idx}/{len(topic_ids)}] {tid}: done", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    alphas = (
        [float(args.fixed_alpha)]
        if args.fixed_alpha is not None
        else [0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0]
    )
    if args.fixed_alpha is not None and not (0.0 <= args.fixed_alpha <= 1.0):
        raise SystemExit("--fixed-alpha must be in [0, 1]")

    qrels_files = (
        [] if args.fixed_alpha is not None else sorted(args.qrels_dir.glob("*.qrels"))
    )
    qrels_all = {p.stem: load_qrels(p) for p in qrels_files}

    best = (-1.0, 0.0, None)
    for alpha in alphas:
        ranked: dict[str, list[str]] = {}
        rows = []
        tag = f"{args.run_id_prefix}-a{alpha:.2f}"
        for tid in topic_ids:
            blended = blend(runs[tid], listwise_by_topic[tid], alpha=alpha)
            ranked[tid] = [d for d, _ in blended]
            for rank, (docid, score) in enumerate(blended[:100], start=1):
                rows.append((tid, docid, rank, score, tag))
        out = args.output_dir / f"r_output_{args.run_id_prefix}_a{alpha:.2f}.tsv"
        write_run(out, rows)
        if args.fixed_alpha is not None:
            print(
                f"Wrote fixed-alpha={alpha:.2f} run={out} rows={len(rows)}",
                flush=True,
            )
            best = (0.0, alpha, out)
            continue
        assessor = {}
        for name, qrels in qrels_all.items():
            vals = [
                ndcg_at_k(ranked[tid], qrels[tid], 30)
                for tid in topic_ids
                if tid in qrels
            ]
            assessor[name] = sum(vals) / len(vals) if vals else 0.0
        robust = sum(assessor.values()) / len(assessor) if assessor else 0.0
        display = "  ".join(
            f"{n.replace('rag25-climbmix-umbrela-', '')}={s:.4f}"
            for n, s in assessor.items()
        )
        print(f"alpha={alpha:.2f} robust_mean={robust:.4f}  {display}", flush=True)
        if robust > best[0]:
            best = (robust, alpha, out)

    if args.fixed_alpha is not None:
        print(f"FIXED alpha={best[1]:.2f} run={best[2]}")
    else:
        print(f"BEST robust_mean={best[0]:.4f} alpha={best[1]:.2f} run={best[2]}")


if __name__ == "__main__":
    main()
