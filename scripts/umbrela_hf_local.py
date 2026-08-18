#!/usr/bin/env python3
"""Local Hugging Face UMBRELA judge (Qwen / Ministral) for HPC GPUs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

# Must be set before torch/cuDNN init on some B200/H200 driver stacks.
os.environ.setdefault("CUDNN_FRONTEND_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_retrieval import load_qrels, ndcg_at_k
from pyserini_client import project_root
from rerank_phase1_doc import load_ranked_run
from trec_io import write_run
from umbrela_aligned_rerank import (
    PROMPT_VERSION,
    STRICT_SUFFIX,
    atomic_write_json,
    blended_ranking,
    build_prompt,
    load_flat_texts,
    load_or_download_narratives,
    parse_grade,
)


def apply_template(tokenizer, prompt: str) -> str:
    """Chat-template a prompt; tolerate tokenizers without enable_thinking."""
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _generate(model, tokenizer, text: str, max_new_tokens: int) -> str:
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def generate_grade(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
) -> tuple[int, str]:
    raw = _generate(model, tokenizer, apply_template(tokenizer, prompt), max_new_tokens)
    try:
        return parse_grade(raw), raw
    except ValueError:
        # One strict retry.
        raw = _generate(
            model,
            tokenizer,
            apply_template(tokenizer, prompt + STRICT_SUFFIX),
            16,
        )
        return parse_grade(raw), raw


def _is_ministral3(model_id: str) -> bool:
    mid = model_id.lower()
    return "ministral-3" in mid or "ministral3" in mid


def load_model_bundle(model_id: str) -> tuple[Any, Any, Callable[..., str]]:
    """Load model + tokenizer; Ministral-3 needs Mistral3ForConditionalGeneration."""
    if _is_ministral3(model_id):
        try:
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            from transformers import Mistral3ForConditionalGeneration
        except ImportError as exc:
            raise SystemExit(
                "Ministral-3 requires: pip install -U mistral-common "
                "'transformers>=4.49.0'"
            ) from exc

        print(f"Loading Ministral-3 via Mistral3ForConditionalGeneration...", flush=True)
        # sdpa/flash can segfault on Blackwell; eager is slower but stable.
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        tokenizer = MistralTokenizer.from_hf_hub(model_id)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
            tie_word_embeddings=False,
        )
        model.eval()
        device = next(model.parameters()).device

        def generate_text(prompt: str, max_new_tokens: int) -> str:
            req = ChatCompletionRequest(
                messages=[{"role": "user", "content": prompt}]
            )
            tokenized = tokenizer.encode_chat_completion(req)
            input_ids = torch.tensor([tokenized.tokens], device=device)
            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=getattr(tokenizer, "pad_id", None) or 0,
                )
            new_ids = out[0][input_ids.shape[-1] :].tolist()
            if hasattr(tokenizer, "decode"):
                return str(tokenizer.decode(new_ids)).strip()
            return str(tokenizer.decode_ids(new_ids)).strip()

        return model, tokenizer, generate_text

    print("Loading via AutoModelForCausalLM...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    def generate_text(prompt: str, max_new_tokens: int) -> str:
        return _generate(model, tokenizer, apply_template(tokenizer, prompt), max_new_tokens)

    return model, tokenizer, generate_text


def generate_grade_with(
    generate_text: Callable[..., str],
    prompt: str,
    *,
    max_new_tokens: int = 32,
) -> tuple[int, str]:
    raw = generate_text(prompt, max_new_tokens)
    try:
        return parse_grade(raw), raw
    except ValueError:
        raw = generate_text(prompt + STRICT_SUFFIX, 16)
        return parse_grade(raw), raw


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
    parser = argparse.ArgumentParser(description="UMBRELA HF local (no vLLM).")
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
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--max-topics", type=int, default=0)
    parser.add_argument("--max-docs", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU. Run this inside an salloc B200 session.")

    narratives = load_or_download_narratives(
        args.narratives,
        "https://trec.nist.gov/data/rag/"
        "trec25_narratives_final_w_questions_w_sub_narratives_edit_20250822.json",
    )
    runs = load_ranked_run(args.input_run)
    texts = load_flat_texts(args.pool_text_cache)
    topic_ids = [tid for tid in runs if tid in narratives]
    if args.max_topics:
        topic_ids = topic_ids[: args.max_topics]

    print(
        f"topics={len(topic_ids)} depth={args.depth} texts={len(texts)} "
        f"cuda={torch.cuda.get_device_name(0)} model={args.model}",
        flush=True,
    )

    print("Loading HF model (no vLLM)...", flush=True)
    _model, _tokenizer, generate_text = load_model_bundle(args.model)

    cache: dict = {"version": PROMPT_VERSION, "model": args.model, "grades": {}}
    if args.score_cache.exists():
        loaded = json.loads(args.score_cache.read_text(encoding="utf-8"))
        if loaded.get("version") == PROMPT_VERSION:
            cache = loaded
            print(f"Resuming cache with {len(cache.get('grades', {}))} topics", flush=True)

    all_grades: dict[str, dict[str, int]] = cache.setdefault("grades", {})
    for topic_index, topic_id in enumerate(topic_ids, start=1):
        candidates = runs[topic_id][: args.depth]
        if args.max_docs:
            candidates = candidates[: args.max_docs]
        existing = {
            str(docid): int(grade)
            for docid, grade in all_grades.get(topic_id, {}).items()
        }
        pending = [(d, s) for d, s in candidates if d not in existing]
        print(
            f"[{topic_index}/{len(topic_ids)}] {topic_id}: "
            f"cached={len(existing)} pending={len(pending)}",
            flush=True,
        )
        grades = dict(existing)
        for i, (docid, _score) in enumerate(pending, start=1):
            text = texts.get(docid, "")
            if not text:
                print(f"  WARNING: missing text {docid}", flush=True)
                continue
            prompt = build_prompt(narratives[topic_id], text[: args.max_chars])
            try:
                grade, _raw = generate_grade_with(generate_text, prompt)
                grades[docid] = grade
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: {docid} failed: {exc}", flush=True)
            if i % 10 == 0 or i == len(pending):
                print(f"  {topic_id}: judged {i}/{len(pending)}", flush=True)
                all_grades[topic_id] = grades
                atomic_write_json(args.score_cache, cache)
        all_grades[topic_id] = grades
        atomic_write_json(args.score_cache, cache)

    qrels_dir = (
        root
        / "trec-rag-data-main/trec-rag-2026/development-data"
        / "rag25-dev-umbrela-qrels"
    )
    qrels_files = sorted(qrels_dir.glob("*.qrels"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for alpha in (0.50, 0.70, 0.85, 0.95, 1.00):
        ranked: dict[str, list[tuple[str, float]]] = {}
        rows = []
        for topic_id in topic_ids:
            ranking = blended_ranking(
                runs[topic_id][: args.depth],
                all_grades.get(topic_id, {}),
                alpha=alpha,
            )
            ranked[topic_id] = ranking
            for rank, (docid, score) in enumerate(ranking, start=1):
                rows.append(
                    (topic_id, docid, rank, score, f"umbrela-hf-a{alpha:.2f}")
                )
        out = args.output_dir / f"r_output_umbrela_aligned_a{alpha:.2f}.tsv"
        write_run(out, rows)
        assessor_scores = {
            path.stem: evaluate_run(ranked, path) for path in qrels_files
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
        print(f"alpha={alpha:.2f} robust_mean={robust_mean:.4f}  {display}", flush=True)

    best = max(results, key=lambda item: item[0])
    print(
        f"BEST robust_mean={best[0]:.4f} alpha={best[1]:.2f} run={best[3]}",
        flush=True,
    )


if __name__ == "__main__":
    # Allow importing sibling modules when run as a script.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
