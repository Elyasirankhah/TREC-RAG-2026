#!/usr/bin/env python3
"""Claim-pack RAG generation: harvest atomic claims and emit cited sentences."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from cross_encoder_rerank import build_texts_for_docids
from llm_client import llm_json, resolve_api_key, resolve_model
from pyserini_client import load_topics_tsv, project_root
from rag_pipeline import finalize_answer, gather_evidence, load_aspects
from rerank_phase1_doc import load_ranked_run
from search_cache import DEFAULT_CACHE_DIR
from text_utils import content_terms
from umbrela_aligned_rerank import load_flat_texts


def _parse_claims_payload(
    raw: object,
    evidence: list[tuple[str, str]],
    *,
    id_offset: int = 0,
    max_claims: int,
    seen: set[str],
) -> list[dict]:
    if isinstance(raw, dict):
        raw = raw.get("claims")
    if not isinstance(raw, list):
        return []
    claims: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        text = re.sub(r"\s+", " ", text)
        if not text or len(text.split()) > 28:
            continue
        key = text.lower()
        if key in seen:
            continue
        # Near-dup: high content-term overlap with an existing claim.
        terms = content_terms(text)
        if terms and any(
            len(terms & content_terms(c["text"])) / max(1, len(terms)) >= 0.85
            for c in claims
        ):
            continue
        docids: list[str] = []
        for s in item.get("support") or []:
            try:
                idx = int(s) + id_offset
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(evidence):
                did = evidence[idx][0]
                if did not in docids:
                    docids.append(did)
        if not docids:
            continue
        seen.add(key)
        claims.append({"text": text, "docids": docids})
        if len(claims) >= max_claims:
            break
    return claims


def extract_claims(
    topic_text: str,
    evidence: list[tuple[str, str]],
    *,
    model: str,
    aspects: list[str] | None,
    max_claims: int = 70,
    focus_aspect: str | None = None,
    evidence_slice: list[tuple[int, tuple[str, str]]] | None = None,
) -> list[dict]:
    """Extract grounded claims. Optional focus_aspect / evidence_slice for dense mode."""
    if evidence_slice is None:
        indexed = list(enumerate(evidence))
        id_offset = 0
        lines = [f"[{i}] {sent[:300]}" for i, (_d, sent) in indexed]
        local_evidence = evidence
    else:
        # Local ids 0..k-1 map back to global evidence indices via id_offset handling
        lines = []
        local_evidence = evidence
        for local_i, (global_i, (_d, sent)) in enumerate(evidence_slice):
            lines.append(f"[{local_i}] {sent[:300]}")
        # Remap supports: we'll translate local -> global below
        id_offset = -1  # sentinel; handled specially

    aspects_block = ""
    if focus_aspect:
        aspects_block = (
            f"PRIORITY: extract claims that specifically address this aspect:\n"
            f"- {focus_aspect}\n"
            "Still only use the evidence; skip if unsupported.\n\n"
        )
    elif aspects:
        aspects_block = (
            "Cover these aspects whenever evidence allows (prefer breadth):\n"
            + "\n".join(f"- {a}" for a in aspects[:20])
            + "\n\n"
        )
    n_ask = max_claims if focus_aspect is None else min(12, max_claims)
    prompt = (
        f"Extract up to {n_ask} ATOMIC factual claims that help answer the "
        "information need. Each claim must be a single concrete fact (<= 22 words) "
        "that appears in the evidence (names, numbers, comparisons, mechanisms, "
        "causes, effects, policies, risks, actions, dates, quantities).\n"
        "Do NOT summarize into vague themes. Prefer many specific facts over a few "
        "broad ones. Quote entities/numbers faithfully. Skip titles and meta text.\n"
        "Every claim MUST cite at least one evidence id that contains the fact.\n"
        'Return JSON: {"claims": [{"text": ..., "support": [evidence_ids]}]}.\n\n'
        f"Information need:\n{topic_text}\n\n"
        f"{aspects_block}"
        f"Evidence:\n" + "\n".join(lines) + "\n"
    )
    try:
        parsed = llm_json(prompt, model=model, max_output_tokens=5000, timeout=240.0)
    except (RuntimeError, ValueError):
        return []

    if evidence_slice is None:
        return _parse_claims_payload(
            parsed, evidence, id_offset=0, max_claims=max_claims, seen=set()
        )

    # Translate local support ids -> global docids via evidence_slice.
    raw = parsed.get("claims") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        return []
    claims: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        text = re.sub(r"\s+", " ", text)
        if not text or len(text.split()) > 28:
            continue
        key = text.lower()
        if key in seen:
            continue
        docids: list[str] = []
        for s in item.get("support") or []:
            try:
                local_i = int(s)
            except (TypeError, ValueError):
                continue
            if 0 <= local_i < len(evidence_slice):
                global_i, (did, _sent) = evidence_slice[local_i]
                if did not in docids:
                    docids.append(did)
        if not docids:
            continue
        seen.add(key)
        claims.append({"text": text, "docids": docids})
        if len(claims) >= max_claims:
            break
    return claims


def _aspect_coverage(aspect: str, claims: list[dict], threshold: float = 0.35) -> float:
    at = content_terms(aspect)
    if not at:
        return 1.0
    blob = content_terms(" ".join(c["text"] for c in claims))
    return len(at & blob) / len(at)


def _merge_claims(*groups: list[dict], max_claims: int) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for c in group:
            key = c["text"].lower()
            if key in seen:
                continue
            terms = content_terms(c["text"])
            if terms and any(
                len(terms & content_terms(x["text"])) / max(1, len(terms)) >= 0.85
                for x in out
            ):
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= max_claims:
                return out
    return out


def extract_claims_dense(
    topic_text: str,
    evidence: list[tuple[str, str]],
    *,
    model: str,
    aspects: list[str] | None,
    max_claims: int = 90,
    batch_size: int = 18,
) -> list[dict]:
    """Global pass + sliding evidence batches + aspect-gap passes.

    Attacks the measured span->extract hole (~0.25 vital recall) by forcing
    the model to look at evidence in smaller windows and refill weak aspects.
    """
    base = extract_claims(
        topic_text,
        evidence,
        model=model,
        aspects=aspects,
        max_claims=min(55, max_claims),
    )
    batches: list[dict] = []
    if evidence:
        step = max(8, batch_size // 2)
        for start in range(0, len(evidence), step):
            slice_pairs = [
                (i, evidence[i]) for i in range(start, min(len(evidence), start + batch_size))
            ]
            if not slice_pairs:
                break
            batches.extend(
                extract_claims(
                    topic_text,
                    evidence,
                    model=model,
                    aspects=aspects,
                    max_claims=14,
                    evidence_slice=slice_pairs,
                )
            )
            if len(base) + len(batches) >= max_claims:
                break

    merged = _merge_claims(base, batches, max_claims=max_claims)

    # Aspect gap-fill: aspects poorly covered by current claims.
    gap_claims: list[dict] = []
    for aspect in aspects or []:
        if _aspect_coverage(aspect, merged) >= 0.35:
            continue
        # Prefer evidence sentences overlapping the aspect.
        at = content_terms(aspect)
        ranked = sorted(
            enumerate(evidence),
            key=lambda iv: len(content_terms(iv[1][1]) & at),
            reverse=True,
        )
        top = [(i, evidence[i]) for i, _ in ranked[:16] if len(content_terms(evidence[i][1]) & at) > 0]
        if not top:
            top = [(i, evidence[i]) for i in range(min(16, len(evidence)))]
        gap_claims.extend(
            extract_claims(
                topic_text,
                evidence,
                model=model,
                aspects=aspects,
                max_claims=10,
                focus_aspect=aspect,
                evidence_slice=top,
            )
        )
        merged = _merge_claims(merged, gap_claims, max_claims=max_claims)
        if len(merged) >= max_claims:
            break

    return _merge_claims(merged, gap_claims, max_claims=max_claims)


def pack_claims_verbatim(
    claims: list[dict],
    reference_index: dict[str, int],
) -> list[dict]:
    """Emit each claim as a cited sentence with no second paraphrase step."""
    sentences: list[dict] = []
    for c in claims:
        cites = [reference_index[d] for d in c["docids"] if d in reference_index][:3]
        if not cites:
            continue
        text = str(c["text"]).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        if not text.endswith((".", "!", "?")):
            text = text.rstrip(".") + "."
        sentences.append({"text": text, "citations": cites})
    return sentences


def pack_claims_to_sentences(
    topic_text: str,
    claims: list[dict],
    reference_index: dict[str, int],
    *,
    model: str,
    batch_size: int = 18,
) -> list[dict]:
    """One fluent cited sentence per claim (batched for reliability)."""
    sentences: list[dict] = []
    for start in range(0, len(claims), batch_size):
        batch = claims[start : start + batch_size]
        payload = []
        for i, c in enumerate(batch):
            cites = [reference_index[d] for d in c["docids"] if d in reference_index][:3]
            if not cites:
                continue
            payload.append({"id": i, "claim": c["text"], "allowed_citations": cites})
        if not payload:
            continue
        prompt = (
            "Convert each claim into ONE short fluent sentence (12-28 words) that "
            "PRESERVES the specific fact (names, numbers, comparisons). Do not "
            "merge claims. Do not drop details. No titles or bracket tags in text.\n"
            "Citations: use only allowed_citations for that claim (1-3 ints).\n\n"
            f"Information need:\n{topic_text}\n\n"
            f"Claims:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            'Return JSON: {"sentences": [{"id": <claim id>, "text": ..., '
            '"citations": [ints]}]}.'
        )
        try:
            parsed = llm_json(prompt, model=model, max_output_tokens=4000, timeout=240.0)
        except (RuntimeError, ValueError):
            # Fallback: emit claim text itself with first citation.
            for item in payload:
                sentences.append(
                    {
                        "text": item["claim"].rstrip(".") + ".",
                        "citations": item["allowed_citations"][:1],
                    }
                )
            continue
        raw = parsed.get("sentences") if isinstance(parsed, dict) else None
        by_id = {
            int(x["id"]): x
            for x in raw
            if isinstance(x, dict) and str(x.get("id", "")).isdigit()
        } if isinstance(raw, list) else {}
        for item in payload:
            cid = item["id"]
            allowed = set(item["allowed_citations"])
            row = by_id.get(cid)
            if row:
                text = str(row.get("text") or "").strip()
                text = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]\s*", " ", text).strip()
                text = re.sub(r"\s+", " ", text)
                cites: list[int] = []
                for c in row.get("citations") or []:
                    try:
                        ci = int(c)
                    except (TypeError, ValueError):
                        continue
                    if ci in allowed and ci not in cites:
                        cites.append(ci)
            else:
                text = item["claim"].rstrip(".") + "."
                cites = item["allowed_citations"][:1]
            if not text:
                continue
            if not cites:
                cites = item["allowed_citations"][:1]
            sentences.append({"text": text, "citations": cites[:3]})
    return sentences


def build_claim_pack_answer(
    topic_text: str,
    pool_fused: list[tuple[str, float]],
    doc_texts: dict[str, str],
    *,
    model: str,
    docs_considered: int,
    max_evidence: int,
    max_claims: int,
    word_budget: int,
    min_chars: int,
    max_chars: int,
    aspects: list[str] | None,
    dense_extract: bool = True,
    verbatim_pack: bool = False,
) -> tuple[list[str], list[dict], int]:
    query_terms = content_terms(topic_text)
    ordered = [d for d, _ in pool_fused]
    aspect_terms = [content_terms(a) for a in (aspects or []) if content_terms(a)]
    evidence = gather_evidence(
        doc_texts,
        ordered,
        query_terms,
        docs_considered=docs_considered,
        max_evidence=max_evidence,
        min_chars=min_chars,
        max_chars=max_chars,
        aspect_terms=aspect_terms or None,
    )
    if dense_extract:
        claims = extract_claims_dense(
            topic_text,
            evidence,
            model=model,
            aspects=aspects,
            max_claims=max_claims,
        )
    else:
        claims = extract_claims(
            topic_text,
            evidence,
            model=model,
            aspects=aspects,
            max_claims=max_claims,
        )
    if not claims:
        return [], [], 0

    references: list[str] = []
    reference_index: dict[str, int] = {}
    for c in claims:
        for did in c["docids"]:
            if did not in reference_index:
                reference_index[did] = len(references)
                references.append(did)

    if verbatim_pack:
        sentences = pack_claims_verbatim(claims, reference_index)
    else:
        sentences = pack_claims_to_sentences(
            topic_text, claims, reference_index, model=model
        )
    refs, answer = finalize_answer(references, sentences, word_budget)
    return refs, answer, len(claims)


def main() -> None:
    root = project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--topics",
        type=Path,
        default=root
        / "trec-rag-data-main/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv",
    )
    ap.add_argument(
        "--input-run",
        type=Path,
        default=root
        / "runs/dev/robust_ensemble/r_output_WIN_listwise_sol_a0.55.tsv",
    )
    ap.add_argument(
        "--pool-text-cache",
        type=Path,
        default=root / "runs/cache/pool_texts/dev.json",
    )
    ap.add_argument(
        "--narratives",
        type=Path,
        default=root / "runs/cache/umbrela/trec25_narratives.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=root / "runs/dev/rag_output_rag25_dev_claim_pack_v2.jsonl",
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--docs-considered", type=int, default=40)
    ap.add_argument("--max-evidence", type=int, default=80)
    ap.add_argument("--max-claims", type=int, default=90)
    ap.add_argument("--word-budget", type=int, default=1000)
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--max-chars", type=int, default=420)
    ap.add_argument("--team-id", default="my-team")
    ap.add_argument("--run-id", default="claim-pack-v2-dense")
    ap.add_argument(
        "--run-desc",
        default=(
            "Frozen WIN listwise evidence; dense claim harvest "
            "(global+batch+aspect-gap) + one cited sentence per claim."
        ),
    )
    ap.add_argument("--max-topics", type=int, default=0)
    ap.add_argument(
        "--topic-ids",
        default="",
        help="Comma-separated topic ids (overrides --max-topics).",
    )
    ap.add_argument(
        "--dense-extract",
        action="store_true",
        default=True,
        help="Global + evidence-batch + aspect-gap claim harvest (default on).",
    )
    ap.add_argument(
        "--no-dense-extract",
        action="store_true",
        help="Use single-pass claim extraction only.",
    )
    ap.add_argument(
        "--verbatim-pack",
        action="store_true",
        help="Skip fluent rephrase; emit extracted claims as cited sentences.",
    )
    args = ap.parse_args()

    if not resolve_api_key():
        raise SystemExit("No LLM API key configured.")
    model = resolve_model(override=args.model)
    dense = not args.no_dense_extract
    print(
        f"Generation model: {model} dense_extract={dense} "
        f"verbatim_pack={args.verbatim_pack}",
        flush=True,
    )

    topics = load_topics_tsv(args.topics)
    if args.topic_ids.strip():
        want = {t.strip() for t in args.topic_ids.split(",") if t.strip()}
        topics = [(tid, text) for tid, text in topics if tid in want]
    elif args.max_topics > 0:
        topics = topics[: args.max_topics]
    ranked = load_ranked_run(args.input_run)
    texts = load_flat_texts(args.pool_text_cache) if args.pool_text_cache.exists() else {}
    aspects_by = load_aspects(args.narratives)
    print(f"Pool texts={len(texts)} aspects_topics={len(aspects_by)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                tid_done = str((obj.get("metadata") or {}).get("narrative_id") or "")
            except json.JSONDecodeError:
                continue
            if tid_done:
                done.add(tid_done)
        if done:
            print(f"Resuming {args.output}: {len(done)} topics already written", flush=True)

    written = 0
    skipped = 0
    mode = "a" if done else "w"
    with args.output.open(mode, encoding="utf-8", newline="\n") as f:
        for idx, (tid, text) in enumerate(topics, start=1):
            if tid in done:
                skipped += 1
                print(f"[{idx}/{len(topics)}] {tid}: cached", flush=True)
                continue
            print(f"[{idx}/{len(topics)}] {tid}", flush=True)
            fused = ranked.get(tid, [])
            if not fused:
                print("  skip: no ranks", flush=True)
                continue
            needed = {d for d, _ in fused[: args.docs_considered]}
            missing = {d for d in needed if not texts.get(d)}
            if missing:
                texts.update(
                    build_texts_for_docids(
                        missing, cache_dir=DEFAULT_CACHE_DIR, max_chars=8000
                    )
                )
            doc_texts = {d: texts.get(d, "") for d, _ in fused}
            refs, answer, n_claims = build_claim_pack_answer(
                text,
                fused,
                doc_texts,
                model=model,
                docs_considered=args.docs_considered,
                max_evidence=args.max_evidence,
                max_claims=args.max_claims,
                word_budget=args.word_budget,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                aspects=aspects_by.get(tid),
                dense_extract=dense,
                verbatim_pack=args.verbatim_pack,
            )
            record = {
                "metadata": {
                    "team_id": args.team_id,
                    "narrative_id": tid,
                    "narrative": text,
                    "run_id": args.run_id,
                    "run_desc": args.run_desc,
                },
                "references": refs,
                "answer": answer,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            print(
                f"  claims={n_claims} sentences={len(answer)} refs={len(refs)} "
                f"words={sum(len(s['text'].split()) for s in answer)}",
                flush=True,
            )
    print(
        f"Wrote {written} new (skipped {skipped} cached) -> {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
