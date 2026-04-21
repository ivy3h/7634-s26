#!/bin/bash
#SBATCH --job-name=phase2_game
#SBATCH --partition=overcap
#SBATCH --qos=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=logs/game_%j.log
#SBATCH --error=logs/game_%j.log

# Run the game client in batch mode using a scripted transcript.
# For interactive play, run main.py directly on a login node after
# exporting LLM_ENDPOINT yourself.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

ENV_NAME="${PHASE2_ENV:-/coc/pskynet6/jhe478/conda_envs/phase2}"
MODE="${PHASE2_MODE:-replay}"           # build | replay
SCRIPT_FILE="${PHASE2_SCRIPT:-transcripts/success_run.txt}"

if [[ -f logs/vllm_endpoint.txt ]]; then
  export LLM_ENDPOINT="$(cat logs/vllm_endpoint.txt)"
  echo "LLM_ENDPOINT=${LLM_ENDPOINT}"
else
  echo "WARNING: logs/vllm_endpoint.txt missing; LLM calls will fail" >&2
fi
export LLM_MODEL="${PHASE2_MODEL:-Qwen/Qwen3.5-9B}"

CONDA_BASE="$(/nethome/jhe478/miniconda3/bin/conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

case "${MODE}" in
  build)
    python main.py build --data-dir data
    ;;
  replay)
    python main.py replay "${SCRIPT_FILE}" --data-dir data --log-dir logs
    ;;
  *)
    echo "unknown MODE: ${MODE}" >&2
    exit 2
    ;;
esac
