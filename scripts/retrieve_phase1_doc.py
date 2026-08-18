#!/usr/bin/env python3
"""Phase-1 multi-query BM25 retrieval with RRF fusion to top-100."""

from __future__ import annotations

import argparse
from pathlib import Path

from fusion import (
    collect_doc_texts,
    coverage_aware_rrf,
    dedupe_candidate_pools,
    rrf_fuse,
)
from llm_client import resolve_phase1_model
from phase1_planner import build_query_branches, plan_topic
from pyserini_client import DEFAULT_BASE_URL, load_topics_tsv, project_root
from trec_io import write_run
from search_cache import cached_search


def main() -> None:
    root = project_root()
    default_topics = (
        root / "trec-rag-data-main/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"
    )

    parser = argparse.ArgumentParser(
        description="Phase-1 doc pipeline (exact): GPT decomposition -> BM25 channels "
        "-> pool cleanup -> RRF -> optional weighted/coverage RRF -> top-100."
    )
    parser.add_argument("--topics", type=Path, default=default_topics)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "runs/dev/r_output_rag25_dev_phase1_doc.tsv",
    )
    # Doc: per-branch cutoff depths such as 100/200/300.
    parser.add_argument("--hits", type=int, default=200, help="per-branch BM25 depth (100/200/300)")
    parser.add_argument("--depth", type=int, default=100, help="final top-N to write (doc: 100)")
    # Doc: RRF k, start at 60, tune 10/20/40/60/100.
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF k (doc: start 60)")
    # Doc: fusion starts VANILLA; weighted/coverage-aware is the optional layer.
    parser.add_argument(
        "--fusion",
        choices=["vanilla", "weighted", "coverage"],
        default="vanilla",
        help="doc default is vanilla RRF; weighted/coverage are the optional layer",
    )
    parser.add_argument(
        "--contrib-depth",
        type=int,
        default=0,
        help="only ranks <= this contribute to RRF (0 = no cutoff)",
    )
    parser.add_argument("--max-per-family", type=int, default=40, help="coverage-aware cap per channel")
    # Doc: URL and near-duplicate cleanup is a default stage on the candidate pools.
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="disable the doc's URL/near-duplicate cleanup on candidate pools",
    )
    parser.add_argument(
        "--facets",
        type=int,
        default=0,
        help="cap number of subquestions/sketches used (doc tunes 3/5/7; 0 = model default)",
    )
    parser.add_argument("--no-weight-prompt", action="store_true", help="skip LLM weight-selection call")
    parser.add_argument("--force-refresh-plans", action="store_true")
    parser.add_argument(
        "--model",
        default=None,
        help="LLM deployment (default: PHASE1_OPENAI_MODEL from .env.local, else gpt-4.1-mini)",
    )
    parser.add_argument("--query-delay", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=Path, default=root / "runs/cache/bm25")
    parser.add_argument("--plan-cache-dir", type=Path, default=root / "runs/cache/phase1_plans")
    parser.add_argument("--run-id", default="phase1-doc-facet-hybrid")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-topics", type=int, default=0)
    args = parser.parse_args()

    topics = load_topics_tsv(args.topics)
    if args.max_topics > 0:
        topics = topics[: args.max_topics]

    llm_model = args.model or resolve_phase1_model()
    contrib = None if args.contrib_depth <= 0 else args.contrib_depth
    cleanup = not args.no_cleanup

    print(f"Phase-1 LLM model: {llm_model}")
    print(
        f"Topics: {len(topics)}  hits/branch: {args.hits}  RRF k: {args.rrf_k}  "
        f"fusion: {args.fusion}  pool-cleanup: {cleanup}  output depth: {args.depth}"
    )

    rows: list[tuple[str, str, int, float, str]] = []

    for idx, (topic_id, text) in enumerate(topics, start=1):
        # --- Stage 1: GPT-4.1-mini decomposition -------------------------------
        plan = plan_topic(
            text,
            model=llm_model,
            use_weights=not args.no_weight_prompt,
            cache_dir=args.plan_cache_dir,
            force_refresh=args.force_refresh_plans,
        )
        if args.facets > 0:
            plan.subquestions = plan.subquestions[: args.facets]
            plan.evidence_sketches = plan.evidence_sketches[: args.facets]

        branches = build_query_branches(plan)
        print(
            f"[{idx}/{len(topics)}] {topic_id}  channels={len(branches)}  "
            f"w=(bm25={plan.bm25_weight:.2f}, sparse={plan.splade_weight:.2f}, "
            f"dense={plan.dense_weight:.2f})"
        )
        for branch in branches:
            preview = branch.text.replace("\n", " ")[:110]
            print(f"    [{branch.family} w={branch.weight:.3f}] {preview}")

        # --- Stage 2: BM25 / SPLADEv3 / BGE-small channels -> candidate pools ---
        rankings: list[list[dict]] = []
        weights: list[float] = []
        families: list[str] = []
        for qi, branch in enumerate(branches):
            delay = args.query_delay if qi > 0 else 0.0
            rankings.append(
                cached_search(
                    branch.text,
                    hits=args.hits,
                    base_url=args.base_url,
                    cache_dir=args.cache_dir,
                    query_delay=delay,
                )
            )
            weights.append(branch.weight)
            families.append(branch.family)

        # --- Stage 3: URL and near-duplicate cleanup on candidate pools --------
        if cleanup:
            pool_texts = collect_doc_texts(rankings)
            rankings = dedupe_candidate_pools(rankings, pool_texts)

        # --- Stage 4/5: RRF, then optional weighted / coverage-aware RRF -------
        if args.fusion == "coverage":
            fused = coverage_aware_rrf(
                rankings,
                rrf_k=args.rrf_k,
                query_weights=weights,
                contrib_depth=contrib,
                family_ids=families,
                max_per_family=args.max_per_family,
            )
        elif args.fusion == "weighted":
            fused = rrf_fuse(
                rankings,
                rrf_k=args.rrf_k,
                query_weights=weights,
                contrib_depth=contrib,
            )
        else:  # vanilla (doc default): plain, unweighted RRF
            fused = rrf_fuse(
                rankings,
                rrf_k=args.rrf_k,
                query_weights=None,
                contrib_depth=contrib,
            )

        # --- Stage 6: Top-100 segments ----------------------------------------
        fused = fused[: args.depth]
        for rank, (docid, score) in enumerate(fused, start=1):
            rows.append((topic_id, docid, rank, score, args.run_id))

    write_run(args.output, rows)
    print(f"Wrote {len(rows)} rows for {len(topics)} topics -> {args.output}")


if __name__ == "__main__":
    main()
