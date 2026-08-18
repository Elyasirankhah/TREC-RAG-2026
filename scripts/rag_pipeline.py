#!/usr/bin/env python3
"""Shared RAG evidence gathering, aspect loading, and answer finalize helpers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from clustering import cluster_texts
from cross_encoder_rerank import build_texts_for_docids
from llm_client import llm_json, resolve_api_key, resolve_model
from pyserini_client import DEFAULT_BASE_URL, load_topics_tsv, project_root
from rerank_phase1_doc import load_ranked_run
from retrieval_core import build_candidate_pool
from search_cache import DEFAULT_CACHE_DIR
from text_utils import content_terms, split_sentences, tokenize
from umbrela_aligned_rerank import load_flat_texts


def load_aspects(path: Path | None) -> dict[str, list[str]]:
    """Load sub-narrative aspect checklist per topic (not gold nuggets)."""
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else list(payload.values())
    out: dict[str, list[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        tid = str(item.get("id") or "")
        aspects = []
        for sub in item.get("sub_narratives") or []:
            if not isinstance(sub, dict):
                continue
            text = str(sub.get("text") or "").strip()
            if text:
                aspects.append(text)
        if tid and aspects:
            out[tid] = aspects
    return out


def gather_evidence(
    doc_texts: dict[str, str],
    ordered_docids: list[str],
    query_terms: set[str],
    *,
    docs_considered: int,
    max_evidence: int,
    min_chars: int,
    max_chars: int,
    aspect_terms: list[set[str]] | None = None,
) -> list[tuple[str, str]]:
    """Return diversified evidence sentences as (docid, sentence)."""
    per_doc: dict[str, list[tuple[float, str]]] = {}
    seen: set[str] = set()
    for docid in ordered_docids[:docs_considered]:
        text = doc_texts.get(docid, "")
        if not text:
            continue
        bucket: list[tuple[float, str]] = []
        for sentence in split_sentences(text):
            if not (min_chars <= len(sentence) <= max_chars):
                continue
            norm = " ".join(tokenize(sentence))
            if not norm or norm in seen:
                continue
            seen.add(norm)
            terms = content_terms(sentence)
            overlap = len(terms & query_terms)
            aspect_bonus = 0.0
            if aspect_terms:
                aspect_bonus = max(
                    (len(terms & at) / max(1, len(at)) for at in aspect_terms),
                    default=0.0,
                )
            if overlap == 0 and aspect_bonus < 0.15:
                continue
            score = overlap / (len(terms) ** 0.5 + 1e-6) + 1.5 * aspect_bonus
            bucket.append((score, sentence))
        bucket.sort(key=lambda x: -x[0])
        if bucket:
            per_doc[docid] = bucket

    # Round-robin across docs so one long doc cannot dominate.
    out: list[tuple[str, str]] = []
    doc_order = [d for d in ordered_docids[:docs_considered] if d in per_doc]
    idxs = {d: 0 for d in doc_order}
    while len(out) < max_evidence and doc_order:
        progressed = False
        for docid in list(doc_order):
            i = idxs[docid]
            bucket = per_doc[docid]
            if i >= len(bucket):
                doc_order.remove(docid)
                continue
            _score, sentence = bucket[i]
            idxs[docid] = i + 1
            out.append((docid, sentence))
            progressed = True
            if len(out) >= max_evidence:
                break
        if not progressed:
            break
    return out


def extract_nuggets(
    topic_text: str,
    evidence: list[tuple[str, str]],
    *,
    model: str | None,
    max_chars: int = 300,
    aspects: list[str] | None = None,
) -> list[dict]:
    """LLM nugget extraction. Returns [{text, facet, docids:[...]}] grounded in evidence."""
    if not evidence:
        return []

    lines = []
    for i, (_docid, sentence) in enumerate(evidence):
        lines.append(f"[{i}] {sentence[:max_chars]}")
    evidence_block = "\n".join(lines)
    aspects_block = ""
    if aspects:
        aspects_block = (
            "Aspect checklist (cover as many as evidence allows; use facet labels "
            "aligned to these aspects):\n"
            + "\n".join(f"- {a}" for a in aspects[:20])
            + "\n\n"
        )

    prompt = (
        "Extract many atomic information nuggets that help answer EVERY part of "
        "the user's information need (aim for 40-70 distinct facts if evidence allows).\n"
        "A nugget is a single, self-contained fact (<= 30 words). Prefer concrete "
        "names, numbers, mechanisms, causes, effects, tradeoffs, and actions.\n"
        "Skip section titles, boilerplate, and meta commentary.\n"
        "Only use facts supported by the numbered evidence. For each nugget give:\n"
        '  "text": the fact,\n'
        '  "facet": a 1-4 word topic label grouping similar nuggets '
        "(use diverse facet labels that map to distinct parts of the need),\n"
        '  "support": list of evidence ids that state this fact.\n\n'
        f"Information need:\n{topic_text}\n\n"
        f"{aspects_block}"
        f"Evidence:\n{evidence_block}\n\n"
        'Return JSON: {"nuggets": [{"text": ..., "facet": ..., "support": [ids]}]}.'
    )
    try:
        parsed = llm_json(prompt, model=model, max_output_tokens=4500, timeout=240.0)
    except (RuntimeError, ValueError):
        return []

    raw = parsed.get("nuggets") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        return []

    nuggets: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        facet = str(item.get("facet", "")).strip() or "general"
        support = item.get("support", [])
        docids: list[str] = []
        if isinstance(support, list):
            for s in support:
                try:
                    idx = int(s)
                except (ValueError, TypeError):
                    continue
                if 0 <= idx < len(evidence):
                    docid = evidence[idx][0]
                    if docid not in docids:
                        docids.append(docid)
        if not docids:
            continue
        nuggets.append({"text": text, "facet": facet, "docids": docids})
    return nuggets


def group_facets(nuggets: list[dict]) -> list[list[int]]:
    """Group nugget indices by facet label; fall back to TF-IDF clustering."""
    labels = {n["facet"].lower() for n in nuggets}
    if len(labels) > 1:
        groups: dict[str, list[int]] = {}
        for i, n in enumerate(nuggets):
            groups.setdefault(n["facet"].lower(), []).append(i)
        return list(groups.values())
    # Single/zero distinct labels -> cluster on nugget text instead.
    return cluster_texts([n["text"] for n in nuggets])


def rank_facets(
    facets: list[list[int]],
    nuggets: list[dict],
    query_terms: set[str],
) -> list[list[int]]:
    def facet_score(indices: list[int]) -> float:
        coverage = len(indices)
        relevance = 0.0
        for i in indices:
            terms = content_terms(nuggets[i]["text"])
            relevance = max(relevance, len(terms & query_terms))
        return coverage + relevance

    return sorted(facets, key=facet_score, reverse=True)


def generate_sentences(
    topic_text: str,
    facets: list[list[int]],
    nuggets: list[dict],
    reference_index: dict[str, int],
    *,
    model: str | None,
    max_facets: int,
    sentences_per_facet: int = 2,
) -> list[dict]:
    """Grounded sentences per facet, citations restricted to that facet's docids."""
    facet_payload = []
    for cid, indices in enumerate(facets[:max_facets]):
        allowed: list[int] = []
        texts: list[str] = []
        for i in indices:
            texts.append(nuggets[i]["text"])
            for docid in nuggets[i]["docids"]:
                ref = reference_index[docid]
                if ref not in allowed:
                    allowed.append(ref)
        facet_payload.append(
            {
                "cluster_id": cid,
                "nuggets": texts[:10],
                "allowed_citations": allowed[:6],
                "n_sentences": min(sentences_per_facet, max(1, (len(texts) + 1) // 2)),
            }
        )

    prompt = (
        "Write a thorough, well-cited answer that covers as many parts of the "
        "user's information need as the evidence allows.\n"
        f"For each cluster, write up to n_sentences fluent sentences "
        f"(each 18-45 words) that convey DISTINCT facts from its nuggets.\n"
        "Rules:\n"
        "- No titles, headings, bullet markers, or bracketed citation tags in text.\n"
        "- Cite using allowed_citations only (at most 3 integers per sentence).\n"
        "- Do not invent facts or citations; stay grounded in the nuggets.\n"
        "- Prefer concrete detail over vague summaries.\n"
        "- Order clusters as given; cover important facets before minor ones.\n\n"
        f"Information need:\n{topic_text}\n\n"
        f"Clusters:\n{json.dumps(facet_payload, ensure_ascii=False)}\n\n"
        'Return JSON: {"sentences": [{"cluster_id": <int>, "text": ..., '
        '"citations": [ref ints]}]}.'
    )
    try:
        parsed = llm_json(prompt, model=model, max_output_tokens=4500, timeout=240.0)
    except (RuntimeError, ValueError):
        return []

    raw = parsed.get("sentences") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        return []

    allowed_by_cluster = {p["cluster_id"]: set(p["allowed_citations"]) for p in facet_payload}
    sentences: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        # Strip accidental inline citation markers like [0] or [0, 1].
        text = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]\s*", " ", text).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        cid = item.get("cluster_id")
        allowed = allowed_by_cluster.get(cid, set())
        cites_raw = item.get("citations", [])
        citations: list[int] = []
        if isinstance(cites_raw, list):
            for c in cites_raw:
                try:
                    ci = int(c)
                except (ValueError, TypeError):
                    continue
                if ci in allowed and ci not in citations:
                    citations.append(ci)
        if not citations and allowed:
            citations = sorted(allowed)[:1]
        if not citations:
            continue
        sentences.append({"text": text, "citations": citations[:3]})
    return sentences


def finalize_answer(
    references: list[str],
    sentences: list[dict],
    word_budget: int,
) -> tuple[list[str], list[dict]]:
    final: list[dict] = []
    words = 0
    used_refs: set[int] = set()
    for sent in sentences:
        w = len(sent["text"].split())
        if words + w > word_budget:
            continue
        final.append(sent)
        words += w
        used_refs.update(sent["citations"])
    remap = {old: new for new, old in enumerate(sorted(used_refs))}
    kept_refs = [references[old] for old in sorted(used_refs)]
    for sent in final:
        sent["citations"] = [remap[c] for c in sent["citations"] if c in remap]
    final = [s for s in final if s["citations"]]
    return kept_refs, final


def gap_fill_sentences(
    topic_text: str,
    aspects: list[str],
    draft: list[dict],
    evidence: list[tuple[str, str]],
    references: list[str],
    reference_index: dict[str, int],
    *,
    model: str | None,
    max_new: int = 16,
) -> list[dict]:
    """Second pass: cover uncovered aspects with additional grounded sentences."""
    if not aspects or not evidence or max_new <= 0:
        return []
    draft_text = " ".join(s["text"] for s in draft)
    lines = []
    for i, (docid, sentence) in enumerate(evidence):
        if docid not in reference_index:
            reference_index[docid] = len(references)
            references.append(docid)
        ref = reference_index[docid]
        lines.append(f"[{i}] (ref={ref}) {sentence[:280]}")
    prompt = (
        "You are filling gaps in a cited draft answer.\n"
        "Given the information need, aspect checklist, draft, and evidence, "
        f"write up to {max_new} NEW fluent sentences (18-40 words each) that "
        "cover aspects the draft misses.\n"
        "Rules:\n"
        "- Only facts supported by the numbered evidence.\n"
        "- citations must be ref integers from the evidence lines (at most 3).\n"
        "- No titles, bullets, or bracket tags inside text.\n"
        "- Skip aspects already well covered by the draft.\n"
        "- Prefer concrete facts over vague restatements.\n\n"
        f"Information need:\n{topic_text}\n\n"
        "Aspects:\n"
        + "\n".join(f"- {a}" for a in aspects[:20])
        + f"\n\nDraft:\n{draft_text[:3500]}\n\n"
        f"Evidence:\n" + "\n".join(lines[:70]) + "\n\n"
        'Return JSON: {"sentences": [{"text": ..., "citations": [ref ints]}]}.'
    )
    try:
        parsed = llm_json(prompt, model=model, max_output_tokens=3500, timeout=240.0)
    except (RuntimeError, ValueError):
        return []
    raw = parsed.get("sentences") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        return []
    n_refs = len(references)
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        text = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]\s*", " ", text).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        cites: list[int] = []
        for c in item.get("citations") or []:
            try:
                ci = int(c)
            except (ValueError, TypeError):
                continue
            if 0 <= ci < n_refs and ci not in cites:
                cites.append(ci)
        if not cites:
            continue
        out.append({"text": text, "citations": cites[:3]})
        if len(out) >= max_new:
            break
    return out


def build_facet_answer(
    topic_text: str,
    pool_fused: list[tuple[str, float]],
    doc_texts: dict[str, str],
    *,
    model: str | None,
    docs_considered: int,
    max_evidence: int,
    max_facets: int,
    word_budget: int,
    min_chars: int,
    max_chars: int,
    sentences_per_facet: int = 2,
    aspects: list[str] | None = None,
    gap_fill: bool = True,
) -> tuple[list[str], list[dict]]:
    query_terms = content_terms(topic_text)
    ordered_docids = [docid for docid, _ in pool_fused]
    aspect_terms = [content_terms(a) for a in (aspects or []) if content_terms(a)]
    evidence = gather_evidence(
        doc_texts,
        ordered_docids,
        query_terms,
        docs_considered=docs_considered,
        max_evidence=max_evidence,
        min_chars=min_chars,
        max_chars=max_chars,
        aspect_terms=aspect_terms or None,
    )
    nuggets = extract_nuggets(
        topic_text, evidence, model=model, aspects=aspects
    )
    if not nuggets:
        return [], []

    facets = rank_facets(group_facets(nuggets), nuggets, query_terms)

    references: list[str] = []
    reference_index: dict[str, int] = {}
    for indices in facets[:max_facets]:
        for i in indices:
            for docid in nuggets[i]["docids"]:
                if docid not in reference_index:
                    reference_index[docid] = len(references)
                    references.append(docid)

    sentences = generate_sentences(
        topic_text,
        facets,
        nuggets,
        reference_index,
        model=model,
        max_facets=max_facets,
        sentences_per_facet=sentences_per_facet,
    )
    if not sentences:
        return [], []

    if gap_fill and aspects:
        extras = gap_fill_sentences(
            topic_text,
            aspects,
            sentences,
            evidence,
            references,
            reference_index,
            model=model,
            max_new=18,
        )
        sentences.extend(extras)

    return finalize_answer(references, sentences, word_budget)


def extractive_fallback(
    topic_text: str,
    pool_fused: list[tuple[str, float]],
    doc_texts: dict[str, str],
    *,
    docs_considered: int,
    max_sentences: int,
    word_budget: int,
    min_chars: int,
    max_chars: int,
) -> tuple[list[str], list[dict]]:
    query_terms = content_terms(topic_text)
    ordered_docids = [docid for docid, _ in pool_fused]
    evidence = gather_evidence(
        doc_texts,
        ordered_docids,
        query_terms,
        docs_considered=docs_considered,
        max_evidence=max_sentences,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    references: list[str] = []
    ref_index: dict[str, int] = {}
    answer: list[dict] = []
    words = 0
    for docid, sentence in evidence:
        w = len(sentence.split())
        if words + w > word_budget:
            continue
        if docid not in ref_index:
            ref_index[docid] = len(references)
            references.append(docid)
        answer.append({"text": sentence, "citations": [ref_index[docid]]})
        words += w
    return references, answer


def main() -> None:
    root = project_root()
    default_topics = root / "trec-rag-data-main/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv"

    parser = argparse.ArgumentParser(description="Subtask RAG Phase 2: facet-clustered cited generation.")
    parser.add_argument("--topics", type=Path, default=default_topics)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "runs/dev/rag_output_rag25_dev_facet.jsonl",
    )
    parser.add_argument(
        "--input-run",
        type=Path,
        default=None,
        help="Frozen Subtask-R TSV. When set, skip retrieval and use this ranking as evidence.",
    )
    parser.add_argument(
        "--pool-text-cache",
        type=Path,
        default=root / "runs/cache/pool_texts/dev.json",
        help="Docid->text cache used with --input-run.",
    )
    parser.add_argument("--hits", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--docs-considered", type=int, default=40)
    parser.add_argument("--max-evidence", type=int, default=80)
    parser.add_argument("--max-facets", type=int, default=20)
    parser.add_argument(
        "--sentences-per-facet",
        type=int,
        default=2,
        help="Max grounded sentences to request per facet cluster.",
    )
    parser.add_argument("--max-sentences", type=int, default=15, help="fallback sentence cap")
    parser.add_argument("--word-budget", type=int, default=1000)
    parser.add_argument(
        "--narratives",
        type=Path,
        default=root / "runs/cache/umbrela/trec25_narratives.json",
        help="Sub-narrative aspects for checklist + gap-fill (not gold nuggets).",
    )
    parser.add_argument(
        "--no-gap-fill",
        action="store_true",
        help="Disable second-pass aspect gap filling.",
    )
    parser.add_argument("--min-chars", type=int, default=60)
    parser.add_argument("--max-chars", type=int, default=400)
    parser.add_argument("--use-hyde", action="store_true")
    parser.add_argument("--query-delay", type=float, default=1.0)
    parser.add_argument("--team-id", default="my-team")
    parser.add_argument("--run-id", default="facet-clustered-rag")
    parser.add_argument(
        "--run-desc",
        default=(
            "Frozen Subtask-R head (or Phase-1 multi-query BM25+RRF); LLM "
            "information-nugget extraction grounded in retrieved sentences; "
            "facet clustering; one grounded, cited super-sentence per facet "
            "under 1024 words."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Generation model (default: resolve_model / gpt-5.6-sol).",
    )
    parser.add_argument("--cache-dir", type=Path, default=root / "runs/cache/query_plan")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-topics", type=int, default=0, help="0 = all topics")
    parser.add_argument("--force-extractive", action="store_true", help="skip LLM, extractive only")
    args = parser.parse_args()

    model = resolve_model(override=args.model) if not args.force_extractive else None
    use_llm = not args.force_extractive and bool(resolve_api_key())
    if args.force_extractive:
        print("Running extractive fallback only (no LLM).")
    elif not use_llm:
        print("WARNING: no LLM key found; using extractive fallback.")
    else:
        print(f"Generation model: {model}")

    topics = load_topics_tsv(args.topics)
    if args.max_topics > 0:
        topics = topics[: args.max_topics]

    aspects_by_topic = load_aspects(args.narratives)
    if aspects_by_topic:
        print(f"Loaded aspect checklists for {len(aspects_by_topic)} topics")

    ranked_run: dict[str, list[tuple[str, float]]] | None = None
    pool_texts: dict[str, str] = {}
    if args.input_run is not None:
        ranked_run = load_ranked_run(args.input_run)
        if args.pool_text_cache.exists():
            pool_texts = load_flat_texts(args.pool_text_cache)
        print(
            f"Using frozen R run {args.input_run} "
            f"(pool texts={len(pool_texts)} from {args.pool_text_cache})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for idx, (topic_id, text) in enumerate(topics, start=1):
            print(f"[{idx}/{len(topics)}] {topic_id}", flush=True)
            if ranked_run is not None:
                fused = ranked_run.get(topic_id, [])
                if not fused:
                    print(f"  WARNING: no ranks for {topic_id}; skipping", flush=True)
                    continue
                needed = {d for d, _ in fused[: max(args.docs_considered, 30)]}
                missing = {d for d in needed if d not in pool_texts or not pool_texts[d]}
                if missing:
                    filled = build_texts_for_docids(
                        missing,
                        cache_dir=DEFAULT_CACHE_DIR,
                        max_chars=8000,
                    )
                    pool_texts.update(filled)
                    still = missing - set(filled)
                    if still:
                        print(
                            f"  WARNING: missing texts for {len(still)} docs "
                            f"(e.g. {next(iter(still))})",
                            flush=True,
                        )
                doc_texts = {d: pool_texts.get(d, "") for d, _ in fused}
                pool_fused = fused
            else:
                pool = build_candidate_pool(
                    topic_id,
                    text,
                    hits=args.hits,
                    max_queries=args.max_queries,
                    use_hyde=args.use_hyde,
                    base_url=args.base_url,
                    cache_dir=args.cache_dir,
                    query_delay=args.query_delay,
                )
                pool_fused = pool.fused
                doc_texts = pool.doc_texts

            references: list[str] = []
            answer: list[dict] = []
            if use_llm:
                references, answer = build_facet_answer(
                    text,
                    pool_fused,
                    doc_texts,
                    model=model,
                    docs_considered=args.docs_considered,
                    max_evidence=args.max_evidence,
                    max_facets=args.max_facets,
                    word_budget=args.word_budget,
                    min_chars=args.min_chars,
                    max_chars=args.max_chars,
                    sentences_per_facet=args.sentences_per_facet,
                    aspects=aspects_by_topic.get(topic_id),
                    gap_fill=not args.no_gap_fill,
                )

            if not answer:
                references, answer = extractive_fallback(
                    text,
                    pool_fused,
                    doc_texts,
                    docs_considered=args.docs_considered,
                    max_sentences=args.max_sentences,
                    word_budget=args.word_budget,
                    min_chars=args.min_chars,
                    max_chars=args.max_chars,
                )

            record = {
                "metadata": {
                    "team_id": args.team_id,
                    "narrative_id": topic_id,
                    "narrative": text,
                    "run_id": args.run_id,
                    "run_desc": args.run_desc,
                },
                "references": references,
                "answer": answer,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(
                f"  sentences={len(answer)} refs={len(references)} "
                f"words={sum(len(s['text'].split()) for s in answer)}",
                flush=True,
            )

    print(f"Wrote {written} RAG objects -> {args.output}")


if __name__ == "__main__":
    main()
