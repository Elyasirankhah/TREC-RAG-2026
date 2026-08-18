#!/usr/bin/env python3
"""Assign nugget / vital coverage labels to RAG answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_client import llm_json, resolve_phase1_model
from pyserini_client import project_root
from umbrela_aligned_rerank import atomic_write_json

EVAL_VERSION = "full-answer-v2"
LABEL_RANK = {"not": 0, "partial": 1, "support": 2}


def load_answers(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        tid = str(obj["metadata"]["narrative_id"])
        out[tid] = " ".join(s.get("text", "") for s in obj.get("answer", []))
    return out


def load_nuggets(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out[str(obj["qid"])] = obj["nuggets"]
    return out


def chunk_answer(answer: str, *, max_chars: int, overlap: int) -> list[str]:
    """Split a long answer into overlapping windows so no span is dropped."""
    answer = answer.strip()
    if not answer:
        return [""]
    if len(answer) <= max_chars:
        return [answer]
    chunks: list[str] = []
    start = 0
    n = len(answer)
    while start < n:
        end = min(n, start + max_chars)
        if end < n:
            # Prefer a sentence/word boundary near the end of the window.
            window = answer[start:end]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if cut >= max_chars // 2:
                end = start + cut + 1
        chunk = answer[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
        if start >= end:
            start = end
    return chunks or [answer[:max_chars]]


def best_label(labels: list[str]) -> str:
    best = "not"
    best_r = -1
    for lab in labels:
        r = LABEL_RANK.get(lab, 0)
        if r > best_r:
            best_r = r
            best = lab
    return best


def assign_batch_on_text(
    query: str,
    answer_text: str,
    nuggets: list[str],
    *,
    model: str,
) -> list[str]:
    numbered = "\n".join(f"[{i}] {n}" for i, n in enumerate(nuggets))
    prompt = (
        "You are NuggetAssigner. For each nugget, decide if the ANSWER captures it.\n"
        'Label each nugget as exactly one of: "support", "partial", "not".\n'
        "- support: the answer fully states the nugget fact\n"
        "- partial: related but incomplete / weaker\n"
        "- not: missing or contradicted\n"
        "Return JSON: {\"labels\": [\"support\"|\"partial\"|\"not\", ...]} "
        "with one label per nugget in order.\n\n"
        f"Query:\n{query}\n\n"
        f"Answer:\n{answer_text}\n\n"
        f"Nuggets:\n{numbered}"
    )
    parsed = llm_json(prompt, model=model, max_output_tokens=800, timeout=180.0)
    labels = parsed.get("labels") if isinstance(parsed, dict) else None
    if not isinstance(labels, list):
        return ["not"] * len(nuggets)
    out: list[str] = []
    for i in range(len(nuggets)):
        lab = str(labels[i]).strip().lower() if i < len(labels) else "not"
        if lab not in {"support", "partial", "not"}:
            lab = "not"
        out.append(lab)
    return out


def assign_batch(
    query: str,
    answer: str,
    nuggets: list[str],
    *,
    model: str,
    chunk_chars: int,
    chunk_overlap: int,
) -> list[str]:
    """Assign labels using the full answer (chunked; best label wins per nugget)."""
    chunks = chunk_answer(answer, max_chars=chunk_chars, overlap=chunk_overlap)
    per_chunk = [
        assign_batch_on_text(query, chunk, nuggets, model=model) for chunk in chunks
    ]
    merged: list[str] = []
    for i in range(len(nuggets)):
        merged.append(best_label([labs[i] for labs in per_chunk]))
    return merged


def main() -> None:
    root = project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument(
        "--nuggets",
        type=Path,
        default=root
        / "trec-rag-data-main/trec-rag-2026/development-data/rag25-dev-nuggets/rag25-dev-nuggets.jsonl",
    )
    ap.add_argument(
        "--topics",
        type=Path,
        default=root
        / "trec-rag-data-main/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Assignment cache JSON (default under runs/cache/rag_nugget_assign/).",
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk-chars", type=int, default=5500)
    ap.add_argument("--chunk-overlap", type=int, default=500)
    ap.add_argument("--max-topics", type=int, default=0)
    ap.add_argument(
        "--topic-ids",
        default="",
        help="Comma-separated topic ids to score (overrides --max-topics).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing cache even if version matches.",
    )
    args = ap.parse_args()

    model = args.model or resolve_phase1_model()
    answers = load_answers(args.answers)
    nuggets = load_nuggets(args.nuggets)
    topics = {
        line.split("\t", 1)[0]: line.split("\t", 1)[1].strip()
        for line in args.topics.read_text(encoding="utf-8").splitlines()
        if "\t" in line
    }
    cache_path = args.cache or (
        root
        / "runs/cache/rag_nugget_assign"
        / f"{args.answers.stem}__{model.replace('/', '_')}__{EVAL_VERSION}.json"
    )
    cache: dict = {
        "model": model,
        "eval_version": EVAL_VERSION,
        "chunk_chars": args.chunk_chars,
        "chunk_overlap": args.chunk_overlap,
        "topics": {},
    }
    if cache_path.exists() and not args.force:
        loaded = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            loaded.get("eval_version") == EVAL_VERSION
            and loaded.get("model") == model
            and int(loaded.get("chunk_chars", -1)) == args.chunk_chars
        ):
            cache = loaded
            print(
                f"Resuming cache {cache_path.name} "
                f"({len(cache.get('topics', {}))} topics)",
                flush=True,
            )
        else:
            print(
                f"Cache version/config mismatch; starting fresh "
                f"(had eval_version={loaded.get('eval_version')!r})",
                flush=True,
            )

    topic_ids = sorted(set(answers) & set(nuggets) & set(topics))
    if args.topic_ids.strip():
        want = {t.strip() for t in args.topic_ids.split(",") if t.strip()}
        topic_ids = [t for t in topic_ids if t in want]
    elif args.max_topics > 0:
        topic_ids = topic_ids[: args.max_topics]

    # Truncation diagnostic
    over = sum(1 for tid in topic_ids if len(answers.get(tid, "")) > args.chunk_chars)
    print(
        f"eval_version={EVAL_VERSION} model={model} topics={len(topic_ids)} "
        f"answers_longer_than_chunk={over}/{len(topic_ids)} "
        f"chunk_chars={args.chunk_chars}",
        flush=True,
    )

    vital_sup = vital_tot = 0
    vital_partial = 0
    all_sup = all_tot = 0
    per_topic: list[tuple[str, float]] = []

    for ti, tid in enumerate(topic_ids, start=1):
        topic_nugs = nuggets[tid]
        cached = cache.setdefault("topics", {}).get(tid)
        if (
            cached
            and cached.get("eval_version") == EVAL_VERSION
            and len(cached.get("labels", [])) == len(topic_nugs)
        ):
            labels = cached["labels"]
            print(f"[{ti}/{len(topic_ids)}] {tid}: cached", flush=True)
        else:
            labels = []
            texts = [str(n.get("text") or "") for n in topic_nugs]
            n_chunks = len(
                chunk_answer(
                    answers[tid],
                    max_chars=args.chunk_chars,
                    overlap=args.chunk_overlap,
                )
            )
            for i in range(0, len(texts), args.batch_size):
                batch = texts[i : i + args.batch_size]
                labels.extend(
                    assign_batch(
                        topics[tid],
                        answers[tid],
                        batch,
                        model=model,
                        chunk_chars=args.chunk_chars,
                        chunk_overlap=args.chunk_overlap,
                    )
                )
            cache.setdefault("topics", {})[tid] = {
                "labels": labels,
                "eval_version": EVAL_VERSION,
                "answer_chars": len(answers[tid]),
                "n_chunks": n_chunks,
            }
            cache["eval_version"] = EVAL_VERSION
            cache["model"] = model
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(cache_path, cache)
            print(
                f"[{ti}/{len(topic_ids)}] {tid}: assigned {len(labels)} "
                f"(chunks={n_chunks}, chars={len(answers[tid])})",
                flush=True,
            )

        t_vital_sup = t_vital_tot = 0
        for nug, lab in zip(topic_nugs, labels):
            all_tot += 1
            if lab == "support":
                all_sup += 1
            if nug.get("importance") == "vital":
                vital_tot += 1
                t_vital_tot += 1
                if lab == "support":
                    vital_sup += 1
                    t_vital_sup += 1
                elif lab == "partial":
                    vital_partial += 1
        if t_vital_tot:
            per_topic.append((tid, t_vital_sup / t_vital_tot))

    print(f"\nModel={model}  eval_version={EVAL_VERSION}  topics={len(topic_ids)}")
    print(
        f"Strict vital recall: {vital_sup}/{vital_tot} = "
        f"{(vital_sup / vital_tot if vital_tot else 0):.3f}"
    )
    print(
        f"Vital partial:       {vital_partial}/{vital_tot} = "
        f"{(vital_partial / vital_tot if vital_tot else 0):.3f}"
    )
    print(
        f"Vital missing(~):    {vital_tot - vital_sup - vital_partial}/{vital_tot}"
    )
    print(
        f"All-nugget support:  {all_sup}/{all_tot} = "
        f"{(all_sup / all_tot if all_tot else 0):.3f}"
    )
    if per_topic:
        worst = sorted(per_topic, key=lambda x: x[1])[:5]
        print(
            "Weakest topics:",
            ", ".join(f"{q}={r:.2f}" for q, r in worst),
        )
    print(f"Cache -> {cache_path}")


if __name__ == "__main__":
    main()
