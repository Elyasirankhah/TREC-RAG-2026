#!/usr/bin/env python3
"""Report robust nDCG across multiple UMBRELA assessor qrels."""

from __future__ import annotations

from pathlib import Path

from eval_retrieval import load_qrels, ndcg_at_k
from pyserini_client import load_topics_tsv, project_root
from rerank_phase1_doc import load_ranked_run


def main() -> None:
    root = project_root()
    topics = load_topics_tsv(
        root
        / "trec-rag-data-main/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"
    )
    qdir = root / "trec-rag-data-main/trec-rag-2026/development-data/rag25-dev-umbrela-qrels"
    qrels_all = {p.stem: load_qrels(p) for p in sorted(qdir.glob("*.qrels"))}

    runs = {
        "old_best_r0.3_g0.3_u0.4": root
        / "runs/dev/robust_ensemble/r_output_robust_ens_r0.3_g0.3_l0.0_c0.0_u0.4.tsv",
        "dual_uq_um": root
        / "runs/dev/robust_ensemble/r_output_dual_uq_um_r0.3_g0.1_l0.0_c0.0_uq0.4_um0.2.tsv",
        "dual_uq_um_cov": root
        / "runs/dev/robust_ensemble"
        / "r_output_dual_uq_um_cov_r0.3_g0.1_l0.0_c0.0_uq0.4_um0.1_cov0.1.tsv",
        "dual_cov_hard_safe": root
        / "runs/dev/robust_ensemble/r_output_dual_uq_um_cov_hard_inject_safe.tsv",
        "WIN_freeze": root
        / "runs/dev/robust_ensemble"
        / "r_output_WIN_dual_uq_um_cov_r0.3_g0.1_uq0.4_um0.1_cov0.1.tsv",
    }

    def robust(runpath: Path) -> tuple[float, dict[str, float]]:
        ranked = load_ranked_run(runpath)
        per_topic: list[float] = []
        per_ass: dict[str, list[float]] = {n: [] for n in qrels_all}
        for tid, _ in topics:
            if tid not in ranked:
                continue
            docs = [d for d, _ in ranked[tid]]
            vals: list[float] = []
            for name, qrels in qrels_all.items():
                if tid in qrels:
                    score = ndcg_at_k(docs, qrels[tid], 30)
                    vals.append(score)
                    per_ass[name].append(score)
            if vals:
                per_topic.append(sum(vals) / len(vals))
        mean = sum(per_topic) / len(per_topic) if per_topic else 0.0
        assessors = {
            name.replace("rag25-climbmix-umbrela-", ""): (
                sum(vals) / len(vals) if vals else 0.0
            )
            for name, vals in per_ass.items()
        }
        return mean, assessors

    print("SCOREBOARD (in-sample robust_mean @30)")
    header = f"{'run':40} {'robust':>8} {'codex':>8} {'ministral':>10} {'qwen':>8}"
    print(header)
    for name, path in runs.items():
        if not path.exists():
            print(f"{name:40} MISSING")
            continue
        mean, assessors = robust(path)
        print(
            f"{name:40} {mean:8.4f} "
            f"{assessors.get('codex-gpt5.5-medium-reasoning-v1', 0.0):8.4f} "
            f"{assessors.get('ministral-3-14b-instruct-2512-v2', 0.0):10.4f} "
            f"{assessors.get('qwen3.5-9b-v2', 0.0):8.4f}"
        )

    print()
    print("LOTO (from tuner grids; honest bar)")
    print("  old_best Qwen-only u:     LOTO 0.5851  in-sample 0.5968")
    print("  dual uq+um:               LOTO 0.5826  in-sample 0.5971")
    print("  dual uq+um+cov:           LOTO 0.5911  in-sample 0.5992  << WIN")
    print(
        "  hard_inject_safe:         in-sample 0.6007 "
        "(NOT honest LOTO; revert used qrels)"
    )
    print()
    print("Frozen winner:")
    print(
        "  runs/dev/robust_ensemble/"
        "r_output_WIN_dual_uq_um_cov_r0.3_g0.1_uq0.4_um0.1_cov0.1.tsv"
    )
    print(
        "  LOTO-composed: "
        "runs/dev/robust_ensemble/r_output_WIN_dual_uq_um_cov_loto_composed.tsv"
    )


if __name__ == "__main__":
    main()
