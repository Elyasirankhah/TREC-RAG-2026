#!/usr/bin/env python3
"""Ensemble signal loading and ranking helpers used by the frozen blend."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

from eval_retrieval import load_qrels, ndcg_at_k
from llm_client import resolve_phase1_model
from pyserini_client import load_topics_tsv, project_root
from rerank_phase1_doc import load_ranked_run
from trec_io import write_run

POINTWISE_VERSION = "phase1-rerank-v2"
LISTWISE_VERSION = "listwise-v1"


def load_pointwise(cache_dir: Path, topic_id: str, model: str) -> dict[str, float]:
    digest = hashlib.sha256(f"{POINTWISE_VERSION}|{model}|{topic_id}".encode()).hexdigest()
    path = cache_dir / f"{digest}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in (payload.get("grades") or {}).items()}


def load_listwise(cache_dir: Path, topic_id: str, model: str, depth: int) -> list[str]:
    digest = hashlib.sha256(f"{LISTWISE_VERSION}|{model}|{topic_id}|{depth}".encode()).hexdigest()
    path = cache_dir / f"{digest}.json"
    if not path.exists():
        return []
    return (json.loads(path.read_text(encoding="utf-8")) or {}).get("ordered_docids", [])


def minmax(mapping: dict[str, float], docids: list[str]) -> dict[str, float]:
    vals = [mapping[d] for d in docids if d in mapping]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {d: 0.5 for d in docids if d in mapping}
    return {d: (mapping[d] - lo) / (hi - lo) for d in docids if d in mapping}


def load_umbrela_grades(path: Path) -> dict[str, dict[str, float]]:
    """Load topic->docid->grade (0-4, possibly averaged float) from umbrela cache."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    grades = payload.get("grades", {})
    out: dict[str, dict[str, float]] = {}
    if isinstance(grades, dict):
        for tid, mapping in grades.items():
            if not isinstance(mapping, dict):
                continue
            out[str(tid)] = {str(d): float(g) for d, g in mapping.items()}
    return out


def weight_tag(w: dict[str, float]) -> str:
    parts = []
    for key in ("r", "g", "l", "c", "u", "uq", "um", "us", "cov"):
        if key in w:
            parts.append(f"{key}{w[key]:.1f}")
    return "_".join(parts)


def simplex_combos(keys: list[str], step: float) -> list[dict[str, float]]:
    """Non-negative weights on keys summing to 1.0."""
    if not keys:
        return [{}]
    steps = [round(i * step, 3) for i in range(int(round(1 / step)) + 1)]
    free = keys[:-1]
    last = keys[-1]
    combos: list[dict[str, float]] = []
    for vals in product(steps, repeat=len(free)):
        rem = round(1.0 - sum(vals), 3)
        if rem < -1e-9 or rem > 1 + 1e-9:
            continue
        w = {k: float(v) for k, v in zip(free, vals)}
        w[last] = float(rem)
        combos.append(w)
    return combos


def write_ranked_run(
    path: Path,
    topics: list[tuple[str, str]],
    topic_ids: list[str],
    rank_fn,
    weights: dict[str, float] | None,
    per_topic_weights: dict[str, dict[str, float]] | None,
    write_top: int,
    tag: str,
) -> None:
    rows: list[tuple[str, str, int, float, str]] = []
    for tid, _topic in topics:
        if tid not in topic_ids:
            continue
        w = per_topic_weights[tid] if per_topic_weights is not None else weights
        assert w is not None
        ranked = rank_fn(tid, w)
        for rank, (docid, score) in enumerate(ranked[:write_top], start=1):
            rows.append((tid, docid, rank, score, tag))
    write_run(path, rows)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Robust ensemble tuning (multi-qrels, LOTO).")
    parser.add_argument(
        "--input-run",
        type=Path,
        default=root / "runs/dev/r_output_rag25_dev_phase1_doc_tuned.tsv",
    )
    parser.add_argument(
        "--topics",
        type=Path,
        default=root / "trec-rag-data-main/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv",
    )
    parser.add_argument(
        "--qrels-dir",
        type=Path,
        default=root / "trec-rag-data-main/trec-rag-2026/development-data/rag25-dev-umbrela-qrels",
    )
    parser.add_argument("--model", default=None, help="LLM model for cache keys")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--listwise-depth", type=int, default=50)
    parser.add_argument("--ce-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--ce-mode", choices=("head", "chunk"), default="chunk")
    parser.add_argument(
        "--umbrela-cache",
        type=Path,
        default=None,
        help="Legacy single UMBRELA cache (signal u).",
    )
    parser.add_argument(
        "--umbrela-qwen",
        type=Path,
        default=None,
        help="Qwen UMBRELA cache (signal uq). Enables dual-judge mode with --umbrela-ministral.",
    )
    parser.add_argument(
        "--umbrela-ministral",
        type=Path,
        default=None,
        help="Ministral UMBRELA cache (signal um).",
    )
    parser.add_argument(
        "--umbrela-sol",
        type=Path,
        default=None,
        help="Sol/Codex-family UMBRELA cache (signal us).",
    )
    parser.add_argument(
        "--coverage-cache",
        type=Path,
        default=None,
        help="Coverage pseudo-UMBRELA grades cache (signal cov).",
    )
    parser.add_argument("--grades-cache", type=Path, default=root / "runs/cache/phase1_grades")
    parser.add_argument("--listwise-cache", type=Path, default=root / "runs/cache/listwise")
    parser.add_argument("--ce-cache", type=Path, default=root / "runs/cache/cross_encoder")
    parser.add_argument("--step", type=float, default=0.1, help="Simplex grid step (e.g. 0.1).")
    parser.add_argument(
        "--zero-lc",
        action="store_true",
        help="Force listwise and CE weights to 0 (smaller grid; matches prior best).",
    )
    parser.add_argument("--write-top", type=int, default=100, help="Write top-k to the output run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs/dev/robust_ensemble",
    )
    parser.add_argument(
        "--run-prefix",
        default="r_output_robust_ens",
        help="Output run filename prefix.",
    )
    args = parser.parse_args()

    # Defaults for dual mode if flags omitted but files exist.
    if args.umbrela_qwen is None and args.umbrela_ministral is None and args.umbrela_cache is None:
        legacy = root / "runs/cache/umbrela/qwen3.5-9b-v2.json"
        args.umbrela_cache = legacy

    dual = args.umbrela_qwen is not None or args.umbrela_ministral is not None
    if dual and (args.umbrela_qwen is None or args.umbrela_ministral is None):
        raise SystemExit("Dual mode requires both --umbrela-qwen and --umbrela-ministral")
    if dual and args.umbrela_cache is not None:
        print("NOTE: ignoring --umbrela-cache in dual-judge mode", flush=True)

    model = args.model or resolve_phase1_model()
    topics = load_topics_tsv(args.topics)
    input_runs = load_ranked_run(args.input_run)

    qrels_files = sorted(args.qrels_dir.glob("*.qrels"))
    if not qrels_files:
        raise SystemExit(f"No qrels found under {args.qrels_dir}")
    qrels_all = {path.stem: load_qrels(path) for path in qrels_files}

    if args.ce_mode == "chunk":
        ce_file = args.ce_cache / f"{args.ce_model.replace('/', '_')}_d{args.depth}_chunk1400s1000.json"
    else:
        ce_file = args.ce_cache / f"{args.ce_model.replace('/', '_')}_d{args.depth}.json"
    ce_all: dict[str, dict[str, float]] = {}
    if ce_file.exists():
        ce_all = {
            str(tid): {str(d): float(s) for d, s in mapping.items()}
            for tid, mapping in json.loads(ce_file.read_text(encoding="utf-8")).items()
        }

    u_grades = load_umbrela_grades(args.umbrela_cache) if args.umbrela_cache else {}
    uq_grades = load_umbrela_grades(args.umbrela_qwen) if args.umbrela_qwen else {}
    um_grades = load_umbrela_grades(args.umbrela_ministral) if args.umbrela_ministral else {}
    us_grades = load_umbrela_grades(args.umbrela_sol) if args.umbrela_sol else {}
    cov_grades = load_umbrela_grades(args.coverage_cache) if args.coverage_cache else {}

    signal_keys = ["r", "g"]
    if not args.zero_lc:
        signal_keys.extend(["l", "c"])
    if dual:
        signal_keys.extend(["uq", "um"])
    elif u_grades:
        signal_keys.append("u")
    if us_grades:
        signal_keys.append("us")
    if cov_grades:
        signal_keys.append("cov")

    topic_ids = [tid for tid, _t in topics if tid in input_runs]
    tails: dict[str, list[tuple[str, float]]] = {}
    per_topic: dict[str, dict[str, dict[str, float]]] = {}
    for tid, _topic in topics:
        candidates = input_runs.get(tid, [])
        if not candidates:
            continue
        head = candidates[: args.depth]
        tails[tid] = candidates[args.depth :]
        docids = [d for d, _ in head]
        n = len(docids)

        r = {d: (n - i) / max(1, n) for i, d in enumerate(docids)}
        grades = load_pointwise(args.grades_cache, tid, model)
        g = {d: grades.get(d, 0.0) / 100.0 for d in docids}
        order = load_listwise(args.listwise_cache, tid, model, args.listwise_depth)
        m = len(order)
        lpos = {d: i for i, d in enumerate(order)}
        l = {d: ((m - lpos[d]) / max(1, m) if d in lpos else 0.0) for d in docids}
        c = minmax(ce_all.get(tid, {}), docids)
        c = {d: c.get(d, 0.0) for d in docids}

        sig: dict[str, dict[str, float]] = {
            "r": r,
            "g": g,
            "l": l,
            "c": c,
            "docids": {d: 1.0 for d in docids},
        }
        if "u" in signal_keys:
            ug = u_grades.get(tid, {})
            sig["u"] = {d: float(ug.get(d, 0)) / 4.0 for d in docids}
        if "uq" in signal_keys:
            ug = uq_grades.get(tid, {})
            sig["uq"] = {d: float(ug.get(d, 0)) / 4.0 for d in docids}
        if "um" in signal_keys:
            ug = um_grades.get(tid, {})
            sig["um"] = {d: float(ug.get(d, 0)) / 4.0 for d in docids}
        if "us" in signal_keys:
            ug = us_grades.get(tid, {})
            sig["us"] = {d: float(ug.get(d, 0)) / 4.0 for d in docids}
        if "cov" in signal_keys:
            cg = cov_grades.get(tid, {})
            sig["cov"] = {d: float(cg.get(d, 0)) / 4.0 for d in docids}
        per_topic[tid] = sig

    def rank_topic(tid: str, w: dict[str, float]) -> list[tuple[str, float]]:
        sig = per_topic[tid]
        docids = list(sig["docids"].keys())
        scored = []
        for d in docids:
            s = 0.0
            for key, weight in w.items():
                if weight == 0.0:
                    continue
                s += weight * sig[key][d]
            scored.append((d, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored.extend(tails.get(tid, []))
        return scored

    def robust_for_topic(tid: str, ranked: list[str]) -> float:
        vals = []
        for _name, qrels in qrels_all.items():
            if tid in qrels:
                vals.append(ndcg_at_k(ranked, qrels[tid], 30))
        return sum(vals) / len(vals) if vals else 0.0

    combos = simplex_combos(signal_keys, float(args.step))
    if args.zero_lc:
        # Attach zero l/c so rank_topic / filenames stay consistent if desired.
        for w in combos:
            w.setdefault("l", 0.0)
            w.setdefault("c", 0.0)
    if not combos:
        raise SystemExit("No weight combos generated; check --step / signal keys")

    print(
        f"qrels_files={len(qrels_files)}  topics={len(topic_ids)}  "
        f"signals={signal_keys}  combos={len(combos)}  dual={dual}  "
        f"has_cov={bool(cov_grades)}",
        flush=True,
    )

    topic_scores: dict[str, list[float]] = {tid: [] for tid in topic_ids}
    overall: list[float] = []
    for w in combos:
        per = []
        for tid in topic_ids:
            ranking = rank_topic(tid, w)
            per.append(robust_for_topic(tid, [d for d, _s in ranking]))
        mean = sum(per) / len(per) if per else 0.0
        overall.append(mean)
        for tid, val in zip(topic_ids, per):
            topic_scores[tid].append(val)

    best_idx = max(range(len(combos)), key=lambda i: overall[i])
    best_w = combos[best_idx]
    best_mean = overall[best_idx]

    # LOTO: best weights on other topics; evaluate on held-out; also keep those weights.
    heldout_vals: list[float] = []
    loto_weights: dict[str, dict[str, float]] = {}
    for tid in topic_ids:
        other = [t for t in topic_ids if t != tid]
        best_train = None
        best_train_idx = 0
        for i in range(len(combos)):
            train_mean = sum(topic_scores[t][i] for t in other) / len(other)
            if best_train is None or train_mean > best_train:
                best_train = train_mean
                best_train_idx = i
        heldout_vals.append(topic_scores[tid][best_train_idx])
        loto_weights[tid] = combos[best_train_idx]
    loto_mean = sum(heldout_vals) / len(heldout_vals) if heldout_vals else 0.0

    # Per-assessor for best in-sample weights.
    assessor_means: dict[str, float] = {}
    for name, qrels in qrels_all.items():
        vals = []
        for tid in topic_ids:
            if tid not in qrels:
                continue
            ranking = [d for d, _ in rank_topic(tid, best_w)]
            vals.append(ndcg_at_k(ranking, qrels[tid], 30))
        assessor_means[name] = sum(vals) / len(vals) if vals else 0.0

    print(f"BEST in-sample robust_mean={best_mean:.4f}  weights={best_w}")
    print(f"LOTO robust_mean={loto_mean:.4f}")
    for name, score in assessor_means.items():
        short = name.replace("rag25-climbmix-umbrela-", "")
        print(f"  assessor {short}={score:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.run_prefix}_{weight_tag(best_w)}.tsv"
    write_ranked_run(
        out,
        topics,
        topic_ids,
        rank_topic,
        best_w,
        None,
        args.write_top,
        "robust-ensemble",
    )
    print(f"Wrote {out}")

    loto_out = args.output_dir / f"{args.run_prefix}_loto_composed_{weight_tag(best_w)}.tsv"
    write_ranked_run(
        loto_out,
        topics,
        topic_ids,
        rank_topic,
        None,
        loto_weights,
        args.write_top,
        "robust-ensemble-loto",
    )
    # Honest LOTO mean is held-out scores; also report in-sample score of composed run.
    composed_vals = [
        robust_for_topic(tid, [d for d, _ in rank_topic(tid, loto_weights[tid])])
        for tid in topic_ids
    ]
    composed_mean = sum(composed_vals) / len(composed_vals) if composed_vals else 0.0
    print(f"Wrote LOTO-composed run {loto_out}")
    print(f"LOTO-composed in-sample robust_mean={composed_mean:.4f}")


if __name__ == "__main__":
    main()
