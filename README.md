# TREC RAG 2026 — DM-Lab-RAG-2026

Public code for Yale team **DM-Lab-RAG-2026** (TREC RAG 2026 Subtask R + Subtask RAG).

This repo ships the **GO-path scripts**, frozen recipes, and development reference runs.  
**Official test submissions and large test caches are not included** — rebuild them with the runbooks if needed.

**Secrets are not included.** Copy `.env.example` → `.env.local` and fill in tokens locally. Never commit `.env.local`.

---

## Submitted systems (Evalbase)

| Subtask | Run ID | Recipe summary |
|---------|--------|----------------|
| **R** | `DMLabWin055` | Phase-1 BM25+RRF → GPT grades → dual UMBRELA → fixed blend → listwise α=0.55 → variable depth (`min-depth 1`) |
| **RAG** | `DMLabRAGMerge` | Claim-pack v1+v2 over WIN top-40 → novelty merge → citation repair |

Test topics: `trec-rag-data-main/trec-rag-2026/test-data/trec_rag_2026_queries.tsv`  
SHA256: `72dc2fd358d3eeda973397ccd7a8775545b19a6deaefc67709167eee6a9f8a2c`

Exact commands: `final submission/R/RUNBOOK.md` and `final submission/RAG/RUNBOOK.md`.

---

## Layout

```
yale-trec-rag-2026/
  README.md
  requirements.txt
  .env.example
  scripts/                       # GO-path Python tools
  hpc/                           # GPU UMBRELA launchers
  final submission/R/RUNBOOK.md
  final submission/RAG/RUNBOOK.md
  runs/dev/robust_ensemble/      # Frozen rag25-dev reference runs
  runs/cache/umbrela/            # Official-style narratives helper (dev)
```

Official track data (`trec-rag-data-main`) is **not** in this repo. Download it separately and place or symlink it at the repo root before scoring/validating.

---

## Setup

```powershell
cd yale-trec-rag-2026
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env.local
# Edit .env.local with local tokens (never commit)
$env:PYTHONPATH = "scripts"
```

Variables (see `.env.example`):

- `PYSERINI_API_TOKEN` — ClimbMix BM25 / document fetch
- `OPEN_AI_KEY` + `AZURE_OPENAI_ENDPOINT` — Azure Responses API
- `OPENAI_MODEL` — strong model (listwise / RAG; default `gpt-5.6-sol`)
- `PHASE1_OPENAI_MODEL` — cheap model (planning / grades; default `gpt-4.1-mini`)

---

## Subtask R — frozen recipe (`DMLabWin055`)

Do **not** retune weights or α on test.

1. Phase-1 multi-query BM25 + RRF → top-100  
2. `gpt-4.1-mini` pointwise grades (`g`)  
3. Generated sub-narratives  
4. Local UMBRELA judges on GPU: Qwen (`uq`) + Ministral (`um`)  
5. Fixed blend with `--renormalize` when coverage is absent  
6. `gpt-5.6-sol` UMBRELA listwise top-30, `--fixed-alpha 0.55`  
7. Variable depth: grade≥2 with `max(uq,um)`, **`--min-depth 1`**  
8. Validate  

```powershell
$env:PYTHONPATH = "scripts"
$topics = "trec-rag-data-main/trec-rag-2026/test-data/trec_rag_2026_queries.tsv"

python "scripts/retrieve_phase1_doc.py" `
  --topics $topics `
  --output "runs/test/r_output_trec_rag_2026_phase1_doc.tsv" `
  --depth 100

python "scripts/rerank_phase1_doc.py" `
  --input-run "runs/test/r_output_trec_rag_2026_phase1_doc.tsv" `
  --topics $topics `
  --output "runs/test/r_output_trec_rag_2026_phase1_grades.tsv"

python "scripts/build_pool_texts.py" `
  --input-run "runs/test/r_output_trec_rag_2026_phase1_doc.tsv" `
  --output "runs/test/pool_texts_phase1_doc.json" `
  --depth 100 --overwrite

python "scripts/generate_sub_narratives.py" `
  --topics $topics `
  --output "runs/cache/umbrela/generated_sub_narratives_test.json"

# GPU UMBRELA (see hpc/*.sh) → qwen + ministral JSON caches

python "scripts/fixed_weight_blend.py" `
  --topics $topics `
  --input-run "runs/test/r_output_trec_rag_2026_phase1_doc.tsv" `
  --umbrela-qwen "runs/cache/umbrela/qwen3.5-9b-test-v1.json" `
  --umbrela-ministral "runs/cache/umbrela/ministral-14b-test-v1.json" `
  --grades-cache "runs/cache/phase1_grades" `
  --output "runs/test/r_output_trec_rag_2026_WIN_blend.tsv" `
  --run-id "WIN-blend" --renormalize --overwrite

python "scripts/umbrela_listwise_stack.py" `
  --input-run "runs/test/r_output_trec_rag_2026_WIN_blend.tsv" `
  --pool-text-cache "runs/test/pool_texts_phase1_doc.json" `
  --narratives "runs/cache/umbrela/generated_sub_narratives_test.json" `
  --topics $topics `
  --cache-dir "runs/cache/umbrela_listwise_test" `
  --output-dir "runs/test/umbrela_listwise" `
  --fixed-alpha 0.55 --depth 30 `
  --model "gpt-5.6-sol" `
  --run-id-prefix "DMLabWin055"

python "scripts/variable_depth_cut.py" `
  --input "runs/test/umbrela_listwise/r_output_DMLabWin055_a0.55.tsv" `
  --output "final submission/R/r_output_trec_rag_2026.tsv" `
  --selector grade `
  --grades "runs/cache/umbrela/qwen3.5-9b-test-v1.json" `
  --grades "runs/cache/umbrela/ministral-14b-test-v1.json" `
  --grade-threshold 2 --grade-combine max `
  --min-depth 1 --max-depth 100 `
  --run-id "DMLabWin055" --overwrite
```

HPC helpers: `hpc/run_umbrela_hf.sh`, `hpc/run_umbrela_ministral.sh` (set `CONDA_ENV_BIN` / `HF_HOME`).

---

## Subtask RAG — frozen recipe (`DMLabRAGMerge`)

Evidence: WIN listwise α=0.55 top-40 + pool texts + generated sub-narratives.

1. Claim-pack **v1** (`--no-dense-extract`)  
2. Claim-pack **v2** (dense extract)  
3. Merge under 1000-word budget (`--strategy novelty`)  
4. Citation support judge + drop `not_support`  
5. Validate  

See `final submission/RAG/RUNBOOK.md`.

---

## Development reference (rag25-dev)

Frozen R on 22-topic development (included):

- Blend: `runs/dev/robust_ensemble/r_output_WIN_dual_uq_um_cov_r0.3_g0.1_uq0.4_um0.1_cov0.1.tsv`
- Listwise: `runs/dev/robust_ensemble/r_output_WIN_listwise_sol_a0.55.tsv`  
  LOTO robust nDCG@30 ≈ **0.60** across three UMBRELA assessor qrels

```powershell
python "scripts/eval_retrieval.py" `
  --run "runs/dev/robust_ensemble/r_output_WIN_listwise_sol_a0.55.tsv" `
  --qrels-dir "trec-rag-data-main/trec-rag-2026/development-data/rag25-dev-umbrela-qrels" `
  --k 30
```

---

## Explicitly out of scope

- Quote-first RAG pipelines  
- Averaging `uq`+`um` into a single `u` before blend  
- Retuning WIN weights or listwise α on test  
- Padding variable depth to a conventional cutoff (e.g. min 10)  
- Full-corpus SPLADE / dense index builds (not used in the submitted R run)
