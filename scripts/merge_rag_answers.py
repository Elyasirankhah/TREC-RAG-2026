#!/usr/bin/env python3
"""Merge multiple RAG JSONL runs under a word budget (no LLM)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyserini_client import project_root
from rag_pipeline import load_aspects
from text_utils import content_terms


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        tid = str(obj["metadata"]["narrative_id"])
        out[tid] = obj
    return out


def score_sentence(
    text: str,
    draft_terms: set[str],
    aspect_term_sets: list[set[str]],
) -> float:
    terms = content_terms(text)
    if not terms:
        return -1.0
    novel = len(terms - draft_terms)
    aspect_bonus = 0.0
    if aspect_term_sets:
        # Prefer sentences that hit still-weak aspects.
        for at in aspect_term_sets:
            if not at:
                continue
            before = len(at & draft_terms) / len(at)
            after = len(at & (draft_terms | terms)) / len(at)
            aspect_bonus += max(0.0, after - before)
    return novel + 3.0 * aspect_bonus


def merge_topic(
    records: list[dict],
    *,
    aspects: list[str] | None,
    word_budget: int,
    strategy: str = "novelty",
    anchor_budget: int = 750,
) -> tuple[list[str], list[dict]]:
    # Collect candidates with source refs remapped into a shared reference list.
    references: list[str] = []
    ref_index: dict[str, int] = {}
    candidates: list[dict] = []
    seen_norm: set[str] = set()

    for source_idx, rec in enumerate(records):
        refs = rec.get("references") or []
        for sent in rec.get("answer") or []:
            text = str(sent.get("text") or "").strip()
            if not text:
                continue
            norm = " ".join(text.lower().split())
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            local_cites = sent.get("citations") or []
            mapped: list[int] = []
            for c in local_cites:
                try:
                    ci = int(c)
                except (TypeError, ValueError):
                    continue
                if not (0 <= ci < len(refs)):
                    continue
                did = str(refs[ci])
                if did not in ref_index:
                    ref_index[did] = len(references)
                    references.append(did)
                mid = ref_index[did]
                if mid not in mapped:
                    mapped.append(mid)
            if not mapped:
                continue
            candidates.append(
                {
                    "text": text,
                    "citations": mapped[:3],
                    "source_idx": source_idx,
                }
            )

    aspect_term_sets = [content_terms(a) for a in (aspects or []) if content_terms(a)]
    draft_terms: set[str] = set()
    chosen: list[dict] = []
    words = 0
    remaining = list(candidates)

    def choose_best(eligible_sources: set[int] | None = None) -> bool:
        nonlocal words, draft_terms
        best_i = -1
        best_score = 0.0
        for i, sent in enumerate(remaining):
            if eligible_sources is not None and sent["source_idx"] not in eligible_sources:
                continue
            w = len(sent["text"].split())
            if words + w > word_budget:
                continue
            sc = score_sentence(sent["text"], draft_terms, aspect_term_sets)
            if sc > best_score:
                best_score = sc
                best_i = i
        if best_i < 0 or best_score <= 0:
            return False
        sent = remaining.pop(best_i)
        chosen.append(sent)
        words += len(sent["text"].split())
        draft_terms |= content_terms(sent["text"])
        return True

    if strategy == "anchor":
        # The dense v2 run (last input) is stronger alone than the old merge on
        # the frozen validation split. Preserve a coherent v2 backbone, then
        # spend the remaining budget only on novel claims from either run.
        anchor_source = len(records) - 1
        anchor_limit = min(anchor_budget, word_budget)
        while words < anchor_limit:
            before = words
            if not choose_best({anchor_source}):
                break
            if words > anchor_limit:
                sent = chosen.pop()
                words = before
                draft_terms = content_terms(" ".join(s["text"] for s in chosen))
                remaining.append(sent)
                break

    while remaining:
        if not choose_best():
            break

    # Remap refs to only those cited.
    used = sorted({c for s in chosen for c in s["citations"]})
    remap = {old: new for new, old in enumerate(used)}
    kept_refs = [references[old] for old in used]
    for s in chosen:
        s["citations"] = [remap[c] for c in s["citations"] if c in remap]
        s.pop("source_idx", None)
    chosen = [s for s in chosen if s["citations"]]
    return kept_refs, chosen


def main() -> None:
    root = project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            root / "runs/dev/rag_output_rag25_dev_claim_pack.jsonl",
            root / "runs/dev/rag_output_rag25_dev_claim_pack_v2.jsonl",
        ],
    )
    ap.add_argument(
        "--narratives",
        type=Path,
        default=root / "runs/cache/umbrela/trec25_narratives.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=root / "runs/dev/rag_output_rag25_dev_claim_pack_merged.jsonl",
    )
    ap.add_argument("--word-budget", type=int, default=1000)
    ap.add_argument(
        "--strategy",
        choices=["novelty", "anchor"],
        default="novelty",
        help="novelty=original global merge; anchor=reserve a v2 backbone first.",
    )
    ap.add_argument(
        "--anchor-budget",
        type=int,
        default=750,
        help="Words reserved for the last input when --strategy anchor.",
    )
    ap.add_argument("--team-id", default="my-team")
    ap.add_argument("--run-id", default="claim-pack-merged-v1v2")
    ap.add_argument(
        "--run-desc",
        default="Greedy merge of claim-pack v1+v2 under 1024-word budget (no LLM).",
    )
    args = ap.parse_args()

    runs = [load_jsonl(p) for p in args.inputs]
    aspects_by = load_aspects(args.narratives)
    topic_ids = sorted(set.intersection(*(set(r) for r in runs)))
    print(f"merging {len(args.inputs)} runs over {len(topic_ids)} topics", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for tid in topic_ids:
            recs = [r[tid] for r in runs]
            narrative = recs[0]["metadata"].get("narrative", "")
            refs, answer = merge_topic(
                recs,
                aspects=aspects_by.get(tid),
                word_budget=args.word_budget,
                strategy=args.strategy,
                anchor_budget=args.anchor_budget,
            )
            out = {
                "metadata": {
                    "team_id": args.team_id,
                    "narrative_id": tid,
                    "narrative": narrative,
                    "run_id": args.run_id,
                    "run_desc": args.run_desc,
                },
                "references": refs,
                "answer": answer,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(
                f"{tid}: sentences={len(answer)} refs={len(refs)} "
                f"words={sum(len(s['text'].split()) for s in answer)}",
                flush=True,
            )
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
