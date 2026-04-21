#!/usr/bin/env bash
# Poll the vLLM endpoint until /v1/models returns 200, or TIMEOUT seconds elapse.
set -euo pipefail

TIMEOUT="${TIMEOUT:-600}"
ENDPOINT_FILE="${ENDPOINT_FILE:-logs/vllm_endpoint.txt}"

deadline=$(( $(date +%s) + TIMEOUT ))
while [[ ! -s "${ENDPOINT_FILE}" ]]; do
  if (( $(date +%s) > deadline )); then
    echo "timeout waiting for ${ENDPOINT_FILE}" >&2
    exit 1
  fi
  sleep 2
done

ENDPOINT="$(cat "${ENDPOINT_FILE}")"
echo "Endpoint: ${ENDPOINT}"

while ! curl -sf "${ENDPOINT}/models" -H "Authorization: Bearer EMPTY" >/dev/null; do
  if (( $(date +%s) > deadline )); then
    echo "timeout: server at ${ENDPOINT} never returned 200 /models" >&2
    exit 1
  fi
  sleep 3
done
echo "vLLM server is ready."
