#!/bin/bash
#SBATCH --job-name=phase2_vllm
#SBATCH --partition=overcap
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/vllm_server_%j.log
#SBATCH --error=logs/vllm_server_%j.log
#SBATCH --exclude=omgwth,robby,protocol,chappie,crushinator,qt-1,spd-13,tachikoma,johnny5,trublu

# Launch the vLLM OpenAI-compatible server for Phase II.
#
# Override any of these via `--export=ALL,VAR=val ...` on sbatch.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# --- configurables -----------------------------------------------------
MODEL="${PHASE2_MODEL:-Qwen/Qwen3-8B}"
ENV_NAME="${PHASE2_ENV:-/coc/pskynet6/jhe478/conda_envs/phase2}"
DTYPE="${PHASE2_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${PHASE2_MAX_MODEL_LEN:-8192}"
GPU_MEM_FRACTION="${PHASE2_GPU_MEM:-0.88}"

# Pick a free port in a range unlikely to collide with other jobs.
PORT="${PHASE2_PORT:-$(python -c 'import socket,random; s=socket.socket(); s.bind(("",0)); p=s.getsockname()[1]; s.close(); print(p)')}"
HOSTNAME_SHORT="$(hostname -s)"
ENDPOINT="http://${HOSTNAME_SHORT}:${PORT}/v1"

mkdir -p logs
echo "${ENDPOINT}" > logs/vllm_endpoint.txt
echo "Endpoint recorded: ${ENDPOINT}"
echo "Model: ${MODEL}"
echo "Dtype: ${DTYPE} | max_model_len: ${MAX_MODEL_LEN} | gpu_mem: ${GPU_MEM_FRACTION}"
echo "Host: $(hostname -f)    GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true)"

# --- env activation ----------------------------------------------------
CONDA_BASE="$(/nethome/jhe478/miniconda3/bin/conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# --- serve -------------------------------------------------------------
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_FRACTION}" \
    --enforce-eager \
    --api-key EMPTY
