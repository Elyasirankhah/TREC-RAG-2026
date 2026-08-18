#!/usr/bin/env python3
"""Judge citation support for RAG answers; writes a resumable judgement cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from llm_client import llm_json_with_retry, resolve_phase1_model
from pyserini_client import project_root

EVAL_VERSION = "citation-support-v1"
LABEL_WEIGHT = {"not_support": 0.0, "partial_support": 0.5, "support": 1.0}
VALID = set(LABEL_WEIGHT)

SYSTEM = (
    "You are a strict TREC RAG support assessor. For each numbered sentence, "
    "decide whether the PASSAGE supports it. Reply with a JSON object "
    '{"labels": [...]} whose list has exactly one label per sentence, in order. '
    'Each label is "support" (passage fully supports the sentence), '
    '"partial_support" (passage supports part of it), or '
    '"not_support" (passage does not support it). No other text.'
)


def load_run(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_pool_texts(paths: list[Path]) -> dict[str, str]:
    """Read docid->text caches, unwrapping the `_flat`/`texts` container keys.

    Later paths win, so callers can layer a topic-specific cache over a base.
    """
    out: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        texts = payload
        for wrapper in ("texts", "_flat"):
            if isinstance(texts, dict) and wrapper in texts:
                texts = texts[wrapper]
        if not isinstance(texts, dict):
            continue
        for docid, value in texts.items():
            if isinstance(value, str):
                out[docid] = value
            elif isinstance(value, dict):
                out[docid] = value.get("text") or value.get("contents") or ""
    return out


def resolve_citation(cit: object, references: list[str]) -> str | None:
    """Citations may be zero-based indices into references, or doc IDs."""
    if isinstance(cit, bool):
        return None
    if isinstance(cit, int):
        return references[cit] if 0 <= cit < len(references) else None
    if isinstance(cit, str):
        if cit in references:
            return cit
        if cit.isdigit():
            i = int(cit)
            return references[i] if 0 <= i < len(references) else None
    return None


def build_prompt(narrative: str, passage: str, sentences: list[str], max_chars: int) -> str:
    body = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    return (
        f"NARRATIVE:\n{narrative[:600]}\n\n"
        f"PASSAGE:\n{passage[:max_chars]}\n\n"
        f"SENTENCES ({len(sentences)}):\n{body}\n\n"
        f'Return {{"labels": [...]}} with exactly {len(sentences)} labels.'
    )


def cache_key(model: str, docid: str, sentences: list[str], max_chars: int) -> str:
    h = hashlib.sha256()
    h.update(f"{EVAL_VERSION}|{model}|{docid}|{max_chars}|".encode())
    for s in sentences:
        h.update(s.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _is_content_filter(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "content_filter" in msg or "content management policy" in msg


def judge(
    narrative: str,
    passage: str,
    sentences: list[str],
    *,
    model: str,
    max_chars: int,
) -> list[str]:
    """Judge one (document, sentences) batch.

    Returns "unknown" labels rather than raising when the provider refuses the
    passage, so a single filtered document cannot discard a whole evaluation.
    """
    try:
        payload = llm_json_with_retry(
            build_prompt(narrative, passage, sentences, max_chars),
            model=model,
            system=SYSTEM,
            temperature=0,
            max_output_tokens=64 + 12 * len(sentences),
            timeout=120.0,
            max_attempts=3,
        )
    except Exception as exc:  # noqa: BLE001 - provider errors vary by backend
        if _is_content_filter(exc):
            return ["unknown"] * len(sentences)
        raise
    labels = payload.get("labels") if isinstance(payload, dict) else payload
    if not isinstance(labels, list):
        return ["unknown"] * len(sentences)
    out = [
        str(x).strip().lower() if str(x).strip().lower() in VALID else "not_support"
        for x in labels
    ]
    if len(out) < len(sentences):
        out += ["not_support"] * (len(sentences) - len(out))
    return out[: len(sentences)]


def main() -> None:
    root = project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument(
        "--pool-texts",
        type=Path,
        action="append",
        default=None,
        help="docid->text cache; repeat to layer several (later wins).",
    )
    ap.add_argument(
        "--cache", type=Path, default=root / "runs/cache/citation_support/dev.json"
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--topics", default=None, help="Comma-separated subset.")
    ap.add_argument("--limit-topics", type=int, default=0)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    model = resolve_phase1_model(root, override=args.model)
    records = load_run(args.run)
    pool_paths = args.pool_texts or [
        root / "runs/cache/pool_texts/dev.json",
        root / "runs/cache/pool_texts/dev_deep.json",
    ]
    pool = load_pool_texts(pool_paths)
    if not pool:
        raise SystemExit(f"No document texts loaded from {pool_paths}")

    if args.topics:
        want = {t.strip() for t in args.topics.split(",")}
        records = [r for r in records if str(r["metadata"]["narrative_id"]) in want]
    if args.limit_topics:
        records = records[: args.limit_topics]

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, list[str]] = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))

    n_obj = 0
    n_uncited = 0
    n_cit = 0
    n_missing_text = 0
    n_filtered = 0
    prec_num = 0.0
    recall_num = 0.0
    per_topic: dict[str, dict] = {}
    dirty = False

    for rec in records:
        tid = str(rec["metadata"]["narrative_id"])
        narrative = rec["metadata"].get("narrative", "")
        refs = rec.get("references", [])
        answer = rec.get("answer", [])

        # group: docid -> list of (obj_index, sentence)
        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        obj_cits: list[list[str]] = []
        for i, obj in enumerate(answer):
            text = obj.get("text", "")
            docids = []
            for c in obj.get("citations", []) or []:
                d = resolve_citation(c, refs)
                if d:
                    docids.append(d)
                    groups[d].append((i, text))
            obj_cits.append(docids)

        labels_by_pair: dict[tuple[int, str], str] = {}
        for docid, items in groups.items():
            passage = pool.get(docid, "")
            if not passage:
                n_missing_text += len(items)
                for idx, _ in items:
                    labels_by_pair[(idx, docid)] = "unknown"
                continue
            sentences = [s for _, s in items]
            key = cache_key(model, docid, sentences, args.max_chars)
            if key in cache:
                labels = cache[key]
            else:
                labels = judge(
                    narrative,
                    passage,
                    sentences,
                    model=model,
                    max_chars=args.max_chars,
                )
                cache[key] = labels
                dirty = True
                if len(cache) % 25 == 0:
                    args.cache.write_text(json.dumps(cache), encoding="utf-8")
            if labels and all(x == "unknown" for x in labels):
                n_filtered += len(items)
            for (idx, _), lab in zip(items, labels):
                labels_by_pair[(idx, docid)] = lab

        t_obj = t_cit = 0
        t_prec = t_rec = 0.0
        for i, docids in enumerate(obj_cits):
            n_obj += 1
            t_obj += 1
            if not docids:
                n_uncited += 1
                continue
            weights = []
            for d in docids:
                lab = labels_by_pair.get((i, d), "unknown")
                if lab == "unknown":
                    continue
                w = LABEL_WEIGHT[lab]
                weights.append(w)
                prec_num += w
                t_prec += w
                n_cit += 1
                t_cit += 1
            if weights:
                best = max(weights)
                recall_num += best
                t_rec += best
        per_topic[tid] = {
            "objects": t_obj,
            "citations": t_cit,
            "precision": (t_prec / t_cit) if t_cit else 0.0,
            "recall": (t_rec / t_obj) if t_obj else 0.0,
        }

    if dirty:
        args.cache.write_text(json.dumps(cache), encoding="utf-8")

    precision = prec_num / n_cit if n_cit else 0.0
    recall = recall_num / n_obj if n_obj else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print(f"Run:    {args.run}")
    print(f"Model:  {model}   ({EVAL_VERSION})")
    print(f"Topics: {len(records)}   answer objects: {n_obj}   citations: {n_cit}")
    if n_uncited:
        print(f"Uncited objects: {n_uncited} (each scores 0 for recall)")
    if n_missing_text:
        print(f"WARNING: {n_missing_text} citation(s) had no document text; excluded.")
    if n_filtered:
        print(f"WARNING: {n_filtered} citation(s) blocked by content filter; excluded.")
    print()
    print(f"Weighted citation precision: {precision:.4f}")
    print(f"Weighted citation recall:    {recall:.4f}")
    print(f"F1:                          {f1:.4f}")
    print("\nWeakest topics by precision:")
    for tid, m in sorted(per_topic.items(), key=lambda kv: kv[1]["precision"])[:5]:
        print(
            f"  {tid:<10} prec={m['precision']:.3f} rec={m['recall']:.3f} "
            f"objs={m['objects']} cits={m['citations']}"
        )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "eval_version": EVAL_VERSION,
                    "run": str(args.run),
                    "model": model,
                    "objects": n_obj,
                    "citations": n_cit,
                    "uncited_objects": n_uncited,
                    "weighted_citation_precision": precision,
                    "weighted_citation_recall": recall,
                    "f1": f1,
                    "per_topic": per_topic,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
