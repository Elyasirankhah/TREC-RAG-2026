#!/usr/bin/env python3
"""UMBRELA prompt, narrative loading, and grade-cache helpers."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from eval_retrieval import load_qrels, ndcg_at_k
from pyserini_client import project_root
from rerank_phase1_doc import load_ranked_run
from trec_io import write_run

NARRATIVES_URL = (
    "https://trec.nist.gov/data/rag/"
    "trec25_narratives_final_w_questions_w_sub_narratives_edit_20250822.json"
)
PROMPT_VERSION = "rag25-umbrela-subnarrative-v2"
FINAL_SCORE_RE = re.compile(r"##\s*final\s+score\s*:\s*([0-4])", re.IGNORECASE)
FALLBACK_SCORE_RE = re.compile(r"(?<!\d)([0-4])(?!\d)")
SCORE_PHRASE_RES = (
    re.compile(r"(?:final\s+)?score\s*(?:is|:|=)\s*([0-4])\b", re.IGNORECASE),
    re.compile(r"assign(?:ed)?\s+(?:a\s+)?(?:final\s+)?(?:score|grade)\s+of\s+([0-4])\b", re.IGNORECASE),
    re.compile(r"\bgrade\s*(?:is|:|=)\s*([0-4])\b", re.IGNORECASE),
)
STRICT_SUFFIX = (
    "\n\nReply with one line only in exactly this format: ##final score: X "
    "where X is an integer 0, 1, 2, 3, or 4. No explanation."
)

UMBRELA_PROMPT = """Given a user first person narrative and a passage, you must provide a score on an integer scale of 0 to 4
with the following meaning:
0 - represents that the passage has nothing to do with the narrative,
1 - represents that the passage seems related to the query but does not contain any sub-narrative of an answer to it,
2 - represents that the passage contains detailed answer for the 1 sub-narrative of the narrative with enough description,
3 - represents that the passage contains detailed answer for the 2 to 3 sub-narratives of the narrative with enough description and
4 - represents that the passage contains detailed answer for the 4+ sub-narratives of the narrative with enough description.
Narrative: {narrative}
Sub-narratives: {sub_narratives}
Passage: {passage}
Instructions: Determine whether each sub-narrative is fully and properly explained or only briefly mentioned. Assign a final integer score (0-4) based on the number of sub-narratives addressed.
Rule: A passage must cover a sub-narrative in a detailed, proper way to count; mere mentions or vague references do not qualify. If there is some extra information not relevant to the narrative at all, downgrade the level by 1. If lot of extra information then downgrade by 2. Do not provide reasoning after listing answered parts; only give the score in the format: ##final score: X"""


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_or_download_narratives(path: Path, url: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        print(f"Downloading official RAG25 narratives -> {path}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 TREC-RAG-research-client/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
        path.write_bytes(raw)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in payload}


def load_flat_texts(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("_flat"), dict):
        return {str(k): str(v) for k, v in payload["_flat"].items()}
    # Backward-compatible topic->docid->text pool cache.
    out: dict[str, str] = {}
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, dict):
                out.update({str(k): str(v) for k, v in value.items()})
    return out


def build_prompt(item: dict[str, Any], passage: str) -> str:
    sub_narratives = [
        str(sub.get("text") or "").strip()
        for sub in item.get("sub_narratives", [])
        if isinstance(sub, dict) and str(sub.get("text") or "").strip()
    ]
    return UMBRELA_PROMPT.format(
        narrative=str(item["narrative"]).strip(),
        sub_narratives=json.dumps(sub_narratives, ensure_ascii=False),
        passage=passage,
    )


def parse_grade(text: str) -> int:
    match = FINAL_SCORE_RE.search(text)
    if match:
        return int(match.group(1))
    for pattern in SCORE_PHRASE_RES:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    matches = FALLBACK_SCORE_RE.findall(text)
    if matches:
        return int(matches[-1])
    raise ValueError(f"Could not parse 0-4 grade from response: {text[:300]!r}")


def post_chat(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    api_key: str | None,
    timeout: float,
    retries: int,
) -> tuple[int, str]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            message = payload["choices"][0]["message"]
            text = str(message.get("content") or "")
            if not text:
                text = str(
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or ""
                )
            return parse_grade(text), text
        except ValueError as exc:
            # Model replied with reasoning; ask once more for strict one-line score.
            if attempt == 0 and "Could not parse" in str(exc):
                strict_body = dict(body)
                strict_body["messages"] = [
                    {"role": "user", "content": prompt + STRICT_SUFFIX}
                ]
                strict_body["max_tokens"] = 16
                strict_request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(strict_body).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(strict_request, timeout=timeout) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    message = payload["choices"][0]["message"]
                    text = str(message.get("content") or "")
                    return parse_grade(text), text
                except (
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                    TimeoutError,
                    KeyError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    pass
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(10.0, 1.5 * (2**attempt)))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {detail[:500]}")
            if attempt >= retries:
                break
            time.sleep(min(10.0, 1.5 * (2**attempt)))
        except (
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(10.0, 1.5 * (2**attempt)))
    assert last_error is not None
    raise last_error


def topic_scores(
    topic_id: str,
    candidates: list[tuple[str, float]],
    *,
    item: dict[str, Any],
    texts: dict[str, str],
    existing: dict[str, int],
    endpoint: str,
    model: str,
    api_key: str | None,
    workers: int,
    timeout: float,
    retries: int,
    max_chars: int,
) -> tuple[dict[str, int], list[str], list[str]]:
    grades = dict(existing)
    missing_text: list[str] = []
    failed: list[str] = []
    jobs: dict[Any, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for docid, _score in candidates:
            if docid in grades:
                continue
            text = texts.get(docid, "")
            if not text:
                missing_text.append(docid)
                continue
            prompt = build_prompt(item, text[:max_chars])
            future = executor.submit(
                post_chat,
                endpoint=endpoint,
                model=model,
                prompt=prompt,
                api_key=api_key,
                timeout=timeout,
                retries=retries,
            )
            jobs[future] = docid
        completed = 0
        for future in as_completed(jobs):
            docid = jobs[future]
            try:
                grade, _raw = future.result()
                grades[docid] = grade
            except Exception as exc:  # noqa: BLE001 — preserve partial progress
                failed.append(docid)
                print(f"  WARNING: {docid} failed: {exc}", flush=True)
            completed += 1
            if completed % 20 == 0 or completed == len(jobs):
                print(
                    f"  {topic_id}: judged {completed}/{len(jobs)} new passages",
                    flush=True,
                )
    return grades, missing_text, failed


def blended_ranking(
    candidates: list[tuple[str, float]],
    grades: dict[str, int],
    *,
    alpha: float,
) -> list[tuple[str, float]]:
    n = len(candidates)
    scored: list[tuple[str, float]] = []
    for index, (docid, _raw) in enumerate(candidates):
        retrieval = (n - index) / max(1, n)
        predicted_relevance = grades.get(docid, 0) / 4.0
        score = alpha * predicted_relevance + (1.0 - alpha) * retrieval
        scored.append((docid, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)


def evaluate_run(
    ranked: dict[str, list[tuple[str, float]]],
    qrels_path: Path,
    *,
    k: int = 30,
) -> float:
    qrels = load_qrels(qrels_path)
    scores = [
        ndcg_at_k([docid for docid, _ in ranked[tid]], qrels[tid], k)
        for tid in sorted(qrels)
        if tid in ranked
    ]
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Exact UMBRELA-aligned reranker.")
    parser.add_argument(
        "--input-run",
        type=Path,
        default=root / "runs/dev/r_output_rag25_dev_phase1_doc_tuned.tsv",
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
    parser.add_argument("--narratives-url", default=NARRATIVES_URL)
    parser.add_argument(
        "--score-cache",
        type=Path,
        default=root / "runs/cache/umbrela/qwen3.5-9b-v2.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs/dev/umbrela_aligned",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--max-topics", type=int, default=0)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    narratives = load_or_download_narratives(args.narratives, args.narratives_url)
    runs = load_ranked_run(args.input_run)
    texts = load_flat_texts(args.pool_text_cache)
    topic_ids = [tid for tid in runs if tid in narratives]
    if args.max_topics:
        topic_ids = topic_ids[: args.max_topics]
    print(
        f"topics={len(topic_ids)} depth={args.depth} texts={len(texts)} "
        f"model={args.model}",
        flush=True,
    )

    if args.dry_run:
        tid = topic_ids[0]
        docid = runs[tid][0][0]
        print(build_prompt(narratives[tid], texts[docid])[:4000])
        return

    cache: dict[str, Any] = {
        "version": PROMPT_VERSION,
        "model": args.model,
        "grades": {},
    }
    if args.score_cache.exists():
        loaded = json.loads(args.score_cache.read_text(encoding="utf-8"))
        if loaded.get("version") == PROMPT_VERSION:
            cache = loaded
            print(
                f"Resuming cache with {len(cache.get('grades', {}))} topics",
                flush=True,
            )

    all_grades: dict[str, dict[str, int]] = cache.setdefault("grades", {})
    for topic_index, topic_id in enumerate(topic_ids, start=1):
        candidates = runs[topic_id][: args.depth]
        if args.max_docs:
            candidates = candidates[: args.max_docs]
        existing = {
            str(docid): int(grade)
            for docid, grade in all_grades.get(topic_id, {}).items()
        }
        pending = sum(docid not in existing for docid, _ in candidates)
        print(
            f"[{topic_index}/{len(topic_ids)}] {topic_id}: "
            f"cached={len(existing)} pending={pending}",
            flush=True,
        )
        grades, missing, failed = topic_scores(
            topic_id,
            candidates,
            item=narratives[topic_id],
            texts=texts,
            existing=existing,
            endpoint=args.endpoint,
            model=args.model,
            api_key=args.api_key,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
            max_chars=args.max_chars,
        )
        all_grades[topic_id] = grades
        atomic_write_json(args.score_cache, cache)
        if missing:
            print(f"  WARNING: missing text for {len(missing)} docs", flush=True)
        if failed:
            print(
                f"  WARNING: {len(failed)} judgments failed for topic {topic_id}; "
                "partial progress saved — rerun to retry failed docs",
                flush=True,
            )

    qrels_dir = (
        root
        / "trec-rag-data-main/trec-rag-2026/development-data"
        / "rag25-dev-umbrela-qrels"
    )
    qrels_files = sorted(qrels_dir.glob("*.qrels"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[float, float, dict[str, float], Path]] = []
    for alpha in (0.50, 0.70, 0.85, 0.95, 1.00):
        ranked: dict[str, list[tuple[str, float]]] = {}
        rows: list[tuple[str, str, int, float, str]] = []
        for topic_id in topic_ids:
            candidates = runs[topic_id][: args.depth]
            ranking = blended_ranking(
                candidates,
                all_grades.get(topic_id, {}),
                alpha=alpha,
            )
            ranked[topic_id] = ranking
            for rank, (docid, score) in enumerate(ranking, start=1):
                rows.append(
                    (
                        topic_id,
                        docid,
                        rank,
                        score,
                        f"umbrela-aligned-a{alpha:.2f}",
                    )
                )
        out = args.output_dir / f"r_output_umbrela_aligned_a{alpha:.2f}.tsv"
        write_run(out, rows)
        assessor_scores = {
            path.stem: evaluate_run(ranked, path)
            for path in qrels_files
        }
        robust_mean = (
            sum(assessor_scores.values()) / len(assessor_scores)
            if assessor_scores
            else 0.0
        )
        results.append((robust_mean, alpha, assessor_scores, out))
        display = "  ".join(
            f"{name.replace('rag25-climbmix-umbrela-', '')}={score:.4f}"
            for name, score in assessor_scores.items()
        )
        print(f"alpha={alpha:.2f} robust_mean={robust_mean:.4f}  {display}")

    best = max(results, key=lambda item: item[0])
    print(
        f"BEST robust_mean={best[0]:.4f} alpha={best[1]:.2f} run={best[3]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
