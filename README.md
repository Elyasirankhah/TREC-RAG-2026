# TREC RAG 2026 — Data Mining Lab @ Yale

Retrieval and generation pipeline for the [TREC RAG 2026](https://trec-rag.github.io/) track  
(Subtask R and Subtask RAG).

| | |
|---|---|
| **Lab** | [Data Mining Lab @ Yale](https://github.com/Data-Mining-Lab-Yale) |
| **Team ID** | `DM-Lab-RAG-2026` |
| **Track** | [TREC RAG 2026](https://trec-rag.github.io/) |

This repository contains the Python code needed to run the lab’s ClimbMix retrieval stack and claim-pack RAG generator. Official track data and API credentials are not included.

---


## What this repo provides

- Multi-query BM25 retrieval over ClimbMix (planning, fusion, reranking)
- Dual UMBRELA judging helpers (API or local GPU)
- Listwise reranking and variable-depth run construction for Subtask R
- Claim-pack answer generation, merging, and citation repair for Subtask RAG
- Validators and development evaluation utilities

---

## Repository layout

```
TREC-RAG-2026/
  README.md
  requirements.txt
  .env.example
  scripts/          # Pipeline code
  hpc/              # Optional GPU UMBRELA launchers
  runs/             # Optional caches and development reference runs
```

Place the official `trec-rag-data-main` release at the repository root (or pass explicit paths) before scoring or validating.

---

## Setup

```powershell
cd TREC-RAG-2026
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env.local
```

Fill in `.env.local` with your own credentials. Never commit that file.

| Variable | Purpose |
|----------|---------|
| `PYSERINI_API_TOKEN` | ClimbMix BM25 / document fetch |
| `OPEN_AI_KEY` | LLM API key |
| `AZURE_OPENAI_ENDPOINT` | Azure endpoint (if used) |
| `OPENAI_MODEL` | Strong model for listwise / RAG |
| `PHASE1_OPENAI_MODEL` | Lighter model for planning / pointwise grades |

```powershell
$env:PYTHONPATH = "scripts"
```

---

## Pipeline overview

### Subtask R (retrieval)

1. Generate multiple BM25 queries from each narrative  
2. Fuse branch results with RRF into a candidate pool  
3. Score documents with retrieval rank, GPT pointwise grades, and dual UMBRELA judges  
4. Blend signals with fixed weights  
5. Listwise-rerank the head with a strong LLM  
6. Apply a variable-depth cut and validate the TREC runfile  

### Subtask RAG (generation)

1. Use a ranked evidence list from Subtask R  
2. Generate claim-pack answers (single-pass and dense variants)  
3. Merge under the word budget  
4. Repair unsupported citations and validate JSONL  

---

## Quick start

```powershell
$env:PYTHONPATH = "scripts"
$topics = "trec-rag-data-main/trec-rag-2026/test-data/trec_rag_2026_queries.tsv"

python "scripts/retrieve_phase1_doc.py" `
  --topics $topics `
  --output "runs/test/r_output_phase1.tsv" `
  --depth 100

python "scripts/build_pool_texts.py" `
  --input-run "runs/test/r_output_phase1.tsv" `
  --output "runs/test/pool_texts_phase1_doc.json" `
  --depth 100 --overwrite
```

Then continue with grading, UMBRELA judging, blending, listwise rerank, variable-depth cut (Subtask R), and claim-pack → merge → citation repair (Subtask RAG).

Optional GPU UMBRELA helpers:

```bash
export CONDA_ENV_BIN=/path/to/env/bin
export HF_HOME=/path/to/hf_cache
bash hpc/run_umbrela_hf.sh --help
bash hpc/run_umbrela_ministral.sh
```

---

## Development evaluation

With official development topics and UMBRELA qrels available:

```powershell
$env:PYTHONPATH = "scripts"
python "scripts/eval_retrieval.py" `
  --run "runs/dev/robust_ensemble/r_output_WIN_listwise_sol_a0.55.tsv" `
  --qrels-dir "trec-rag-data-main/trec-rag-2026/development-data/rag25-dev-umbrela-qrels" `
  --k 30
```

---

## Notes

- Secrets, virtual environments, BM25/grade caches, and regenerated large artifacts are ignored via `.gitignore`.
- Caches under `runs/cache/` are resume-safe and can be deleted for a clean rebuild.
- Experimental side paths that were not part of the frozen recipes are intentionally omitted.

---

## License / use

Intended for research and reproducibility related to TREC RAG 2026. Obtain ClimbMix / API access and track data under their respective terms.
