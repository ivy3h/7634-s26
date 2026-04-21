#!/usr/bin/env bash
# Create the phase2 conda env. Run this once on a login node.
# Uses --prefix so the env lives in /coc/pskynet6 where we have quota.
set -eo pipefail   # intentionally NOT -u: .bashrc references unset vars

ENV_PREFIX="${ENV_PREFIX:-/coc/pskynet6/jhe478/conda_envs/phase2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# Source conda without failing on unset-var warnings from user rc files.
CONDA_BASE="$(/nethome/jhe478/miniconda3/bin/conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda create -y --prefix "${ENV_PREFIX}" "python=${PYTHON_VERSION}"
conda activate "${ENV_PREFIX}"

# vLLM 0.11+ bundles a compatible torch. Install with CUDA 12.1 wheels.
pip install --no-cache-dir "vllm>=0.11.0,<0.13"
pip install --no-cache-dir "openai>=1.52" "pydantic>=2" "numpy<2.3"

python - <<'PY'
import importlib, sys
for mod in ("vllm", "openai", "torch"):
    m = importlib.import_module(mod)
    print(f"{mod}: {getattr(m, '__version__', 'unknown')}")
PY

echo "Env ready at: ${ENV_PREFIX}"
