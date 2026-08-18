#!/usr/bin/env bash
# Local Hugging Face UMBRELA judge (Qwen by default). Requires CUDA + transformers.
# From pack root:
#   export HF_HOME=/path/to/hf_cache
#   export PATH=/path/to/conda_env/bin:$PATH
#   bash hpc/run_umbrela_hf.sh --help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [[ -n "${CONDA_ENV_BIN:-}" ]]; then
  export PATH="${CONDA_ENV_BIN}:${PATH}"
fi
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME" runs/cache/umbrela logs

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA — allocate a GPU node before running this script.")
print("GPU:", torch.cuda.get_device_name(0))
PY

python "scripts/umbrela_hf_local.py" "$@"
echo "Done."
