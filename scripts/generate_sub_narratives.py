#!/usr/bin/env python3
"""Generate sub-narratives for UMBRELA-style judging when official subs are unavailable."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from llm_client import llm_json, resolve_phase1_model
from pyserini_client import load_topics_tsv, project_root

PROMPT_VERSION = "generated-subnarratives-v2-consensus"

SYSTEM = (
    "You decompose first-person information needs into atomic sub-narratives "
    "for passage grading. Return only valid JSON."
)

USER = """Decompose the narrative below into atomic sub-narratives.

A sub-narrative is one specific, independently answerable facet of the user's need.
Follow the official TREC RAG style:
- Produce 6 to 10 sub-narratives (prefer 7-9 when the narrative is multi-faceted).
- Each "text" is short (typically under 15 words): a noun phrase or a focused question.
- Cover distinct facets; do not paraphrase the same facet.
- Stay faithful to the narrative; do not invent entities, events, or claims not implied by it.
- Mark importance as "vital" for core facets and "okay" for secondary ones.
- Roughly half or more should be "vital".

Return JSON:
{{
  "sub_narratives": [
    {{"text": "...", "importance": "vital"}},
    {{"text": "...", "importance": "okay"}}
  ]
}}

{examples}
Narrative:
{narrative}
"""


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_official(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in payload}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def token_jaccard(a: str, b: str) -> float:
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def best_match_scores(generated: list[str], official: list[str]) -> tuple[float, float]:
    """Mean best Jaccard gen->off and off->gen (symmetric coverage)."""
    if not generated or not official:
        return 0.0, 0.0
    g2o = [
        max(token_jaccard(g, o) for o in official)
        for g in generated
    ]
    o2g = [
        max(token_jaccard(o, g) for g in generated)
        for o in official
    ]
    return sum(g2o) / len(g2o), sum(o2g) / len(o2g)


def pick_fewshots(
    official: dict[str, dict[str, Any]],
    *,
    exclude_ids: set[str],
    n: int,
) -> list[dict[str, Any]]:
    pool = [
        item
        for tid, item in official.items()
        if tid not in exclude_ids and item.get("sub_narratives")
    ]
    # Prefer mid-length examples for style.
    pool.sort(key=lambda item: abs(len(item.get("sub_narratives", [])) - 8))
    return pool[:n]


def format_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return ""
    blocks = ["Examples of good decompositions (style only; unrelated topics):"]
    for item in examples:
        subs = [
            {
                "text": str(sub.get("text") or "").strip(),
                "importance": str(sub.get("importance") or "okay"),
            }
            for sub in item.get("sub_narratives", [])
            if isinstance(sub, dict) and str(sub.get("text") or "").strip()
        ]
        blocks.append(
            "Narrative:\n"
            f"{str(item['narrative']).strip()}\n"
            "Sub-narratives JSON:\n"
            f"{json.dumps({'sub_narratives': subs}, ensure_ascii=False)}"
        )
    return "\n\n".join(blocks) + "\n\n"


def coerce_subs(raw: Any) -> list[dict[str, str]]:
    if isinstance(raw, dict):
        raw = raw.get("sub_narratives", raw.get("subnarratives", []))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list of sub_narratives, got {type(raw)}")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            importance = "vital"
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("sub_narrative") or "").strip()
            importance = str(item.get("importance") or "okay").strip().lower()
        else:
            continue
        if not text:
            continue
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        if importance not in {"vital", "okay"}:
            importance = "okay"
        out.append({"text": text, "importance": importance})
    if len(out) < 4:
        raise ValueError(f"Too few sub-narratives after cleanup: {len(out)}")
    return out[:12]


def generate_one(
    narrative: str,
    *,
    model: str,
    examples: list[dict[str, Any]],
    temperature: float,
) -> list[dict[str, str]]:
    prompt = USER.format(
        examples=format_examples(examples),
        narrative=narrative.strip(),
    )
    payload = llm_json(
        prompt,
        model=model,
        system=SYSTEM,
        temperature=temperature,
        max_output_tokens=1200,
        timeout=120.0,
    )
    return coerce_subs(payload)


def _cluster_subs(
    decomps: list[list[dict[str, str]]],
    *,
    jaccard_thresh: float = 0.45,
) -> list[dict[str, str]]:
    """Merge independent decompositions into a minimal spanning facet set.

    Near-duplicate texts (token Jaccard >= thresh) collapse into one cluster.
    Cluster importance is vital if any member is vital. Pure-okay facets are
    kept only when they do not duplicate a vital cluster (unique secondary
    facets); consumers can still filter with --vital-only downstream.
    """
    flat: list[dict[str, str]] = []
    for subs in decomps:
        flat.extend(subs)
    if not flat:
        return []

    clusters: list[dict[str, Any]] = []
    for sub in flat:
        text = sub["text"]
        imp = sub["importance"]
        matched = None
        for cluster in clusters:
            if token_jaccard(text, cluster["text"]) >= jaccard_thresh:
                matched = cluster
                break
        if matched is None:
            clusters.append(
                {
                    "text": text,
                    "importance": imp,
                    "votes": 1,
                    "vital_votes": 1 if imp == "vital" else 0,
                }
            )
            continue
        matched["votes"] += 1
        if imp == "vital":
            matched["vital_votes"] += 1
            matched["importance"] = "vital"
            # Prefer the shorter, tighter phrasing once vital is locked.
            if len(text) < len(matched["text"]):
                matched["text"] = text
        elif matched["importance"] != "vital" and len(text) < len(matched["text"]):
            matched["text"] = text

    # Prefer clusters seen in multiple decomps, then vital, then shorter text.
    clusters.sort(
        key=lambda c: (-c["votes"], -c["vital_votes"], len(c["text"]))
    )

    vital = [
        {"text": c["text"], "importance": "vital"}
        for c in clusters
        if c["importance"] == "vital"
    ]
    okay_only = [
        {"text": c["text"], "importance": "okay"}
        for c in clusters
        if c["importance"] != "vital"
    ]
    # Drop okay facets that nearly duplicate a kept vital.
    kept_okay: list[dict[str, str]] = []
    for okay in okay_only:
        if any(token_jaccard(okay["text"], v["text"]) >= jaccard_thresh for v in vital):
            continue
        kept_okay.append(okay)

    out = vital + kept_okay
    # Guardrail: never return an empty/too-thin set.
    if len(vital) < 4 and okay_only:
        for okay in okay_only:
            if okay in kept_okay:
                continue
            promoted = {"text": okay["text"], "importance": "vital"}
            if any(token_jaccard(promoted["text"], v["text"]) >= jaccard_thresh for v in out):
                continue
            out.append(promoted)
            if sum(1 for s in out if s["importance"] == "vital") >= 4:
                break
    return out[:12]


def consensus_subs(
    narrative: str,
    *,
    model: str,
    examples: list[dict[str, Any]],
    temperature: float,
    n_decomps: int,
) -> tuple[list[dict[str, str]], list[list[dict[str, str]]]]:
    decomps: list[list[dict[str, str]]] = []
    for i in range(max(1, n_decomps)):
        # Slight temperature bump on later passes to diversify facets.
        temp = temperature if i == 0 else min(0.7, temperature + 0.15 * i)
        decomps.append(
            generate_one(
                narrative,
                model=model,
                examples=examples,
                temperature=temp,
            )
        )
    return _cluster_subs(decomps), decomps


def compare_report(
    generated_items: list[dict[str, Any]],
    official: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for item in generated_items:
        tid = str(item["id"])
        if tid not in official:
            continue
        gen = [s["text"] for s in item["sub_narratives"]]
        off = [
            str(s.get("text") or "").strip()
            for s in official[tid].get("sub_narratives", [])
            if str(s.get("text") or "").strip()
        ]
        g2o, o2g = best_match_scores(gen, off)
        rows.append(
            {
                "topic_id": tid,
                "n_generated": len(gen),
                "n_official": len(off),
                "count_delta": len(gen) - len(off),
                "jaccard_gen_to_off": round(g2o, 4),
                "jaccard_off_to_gen": round(o2g, 4),
                "jaccard_harmonic": round(
                    (2 * g2o * o2g / (g2o + o2g)) if (g2o + o2g) > 0 else 0.0,
                    4,
                ),
            }
        )
    if not rows:
        return {"topics": 0}
    return {
        "topics": len(rows),
        "mean_n_generated": round(sum(r["n_generated"] for r in rows) / len(rows), 2),
        "mean_n_official": round(sum(r["n_official"] for r in rows) / len(rows), 2),
        "mean_jaccard_gen_to_off": round(
            sum(r["jaccard_gen_to_off"] for r in rows) / len(rows), 4
        ),
        "mean_jaccard_off_to_gen": round(
            sum(r["jaccard_off_to_gen"] for r in rows) / len(rows), 4
        ),
        "mean_jaccard_harmonic": round(
            sum(r["jaccard_harmonic"] for r in rows) / len(rows), 4
        ),
        "per_topic": rows,
    }


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Generate atomic sub-narratives.")
    parser.add_argument(
        "--topics",
        type=Path,
        default=root
        / "trec-rag-data-main/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "runs/cache/umbrela/generated_sub_narratives_dev.json",
    )
    parser.add_argument(
        "--official-narratives",
        type=Path,
        default=root / "runs/cache/umbrela/trec25_narratives.json",
        help="Optional official JSON for few-shots and fidelity comparison.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--fewshot", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-topics", type=int, default=0)
    parser.add_argument(
        "--n-decomps",
        type=int,
        default=2,
        help="Independent decompositions to merge into a consensus spanning set.",
    )
    parser.add_argument(
        "--vital-only",
        action="store_true",
        help="Write only vital (and promoted) facets after consensus merge.",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip fidelity comparison against official narratives.",
    )
    args = parser.parse_args()

    model = args.model or resolve_phase1_model()
    topics = load_topics_tsv(args.topics)
    if args.max_topics:
        topics = topics[: args.max_topics]

    official: dict[str, dict[str, Any]] = {}
    if args.official_narratives.exists():
        official = load_official(args.official_narratives)

    topic_ids = {tid for tid, _ in topics}
    examples = pick_fewshots(official, exclude_ids=topic_ids, n=args.fewshot)

    # Resume-safe: keep previously generated items with matching generator config.
    by_id: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for item in existing:
                    if isinstance(item, dict) and item.get("id") is not None:
                        by_id[str(item["id"])] = item
                print(f"Resuming with {len(by_id)} existing items", flush=True)
        except (json.JSONDecodeError, OSError):
            pass

    print(
        f"topics={len(topics)} model={model} fewshot={len(examples)} "
        f"n_decomps={args.n_decomps} vital_only={args.vital_only} "
        f"version={PROMPT_VERSION}",
        flush=True,
    )

    def _cache_ok(item: dict[str, Any]) -> bool:
        if not item.get("sub_narratives"):
            return False
        gen = item.get("generator") or {}
        return (
            gen.get("version") == PROMPT_VERSION
            and int(gen.get("n_decomps", 1)) == args.n_decomps
            and bool(gen.get("vital_only", False)) == bool(args.vital_only)
        )

    for index, (tid, narrative) in enumerate(topics, start=1):
        if tid in by_id and _cache_ok(by_id[tid]):
            print(f"[{index}/{len(topics)}] {tid}: cached", flush=True)
            continue
        print(
            f"[{index}/{len(topics)}] {tid}: generating "
            f"({args.n_decomps} decomps)...",
            flush=True,
        )
        if args.n_decomps <= 1:
            subs = generate_one(
                narrative,
                model=model,
                examples=examples,
                temperature=args.temperature,
            )
            raw_decomps = [subs]
        else:
            subs, raw_decomps = consensus_subs(
                narrative,
                model=model,
                examples=examples,
                temperature=args.temperature,
                n_decomps=args.n_decomps,
            )
        if args.vital_only:
            vitals = [s for s in subs if s["importance"] == "vital"]
            if len(vitals) >= 4:
                subs = vitals
            else:
                # Keep okay fillers only to satisfy a minimum vital-like set.
                fillers = [s for s in subs if s["importance"] != "vital"]
                subs = vitals + [
                    {"text": s["text"], "importance": "vital"} for s in fillers
                ]
                subs = subs[: max(4, len(vitals) + 2)]
        by_id[tid] = {
            "id": tid,
            "narrative": narrative,
            "sub_narratives": subs,
            "generator": {
                "version": PROMPT_VERSION,
                "model": model,
                "fewshot": len(examples),
                "n_decomps": args.n_decomps,
                "vital_only": bool(args.vital_only),
                "n_raw_decomps": [len(d) for d in raw_decomps],
            },
        }
        # Persist after each topic.
        ordered = [by_id[t] for t, _ in topics if t in by_id]
        # Keep any extras from previous runs.
        for oid, item in by_id.items():
            if oid not in {t for t, _ in topics}:
                ordered.append(item)
        atomic_write_json(args.output, ordered)
        n_vital = sum(1 for s in subs if s["importance"] == "vital")
        print(
            f"  -> {len(subs)} sub-narratives ({n_vital} vital)",
            flush=True,
        )

    items = [by_id[t] for t, _ in topics if t in by_id]
    atomic_write_json(args.output, items)
    print(f"Wrote {args.output} ({len(items)} topics)", flush=True)

    if not args.no_compare and official:
        report = compare_report(items, official)
        report_path = args.output.with_name(args.output.stem + "_fidelity.json")
        atomic_write_json(report_path, report)
        print(
            "Fidelity vs official: "
            f"topics={report.get('topics')} "
            f"mean_n_gen={report.get('mean_n_generated')} "
            f"mean_n_off={report.get('mean_n_official')} "
            f"jaccard_harmonic={report.get('mean_jaccard_harmonic')}",
            flush=True,
        )
        print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
