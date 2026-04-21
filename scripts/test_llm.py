"""Smoke-test the local vLLM server.

Run after launch_vllm_server.sbatch reports ready and logs/vllm_endpoint.txt
is populated. Reads LLM_ENDPOINT and LLM_MODEL from the environment; falls
back to the endpoint file if LLM_ENDPOINT is unset.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

endpoint_file = ROOT / "logs" / "vllm_endpoint.txt"
if "LLM_ENDPOINT" not in os.environ and endpoint_file.exists():
    os.environ["LLM_ENDPOINT"] = endpoint_file.read_text().strip()

from llm_client import chat_simple, health_check  # noqa: E402

print("Health:", json.dumps(health_check(), indent=2))
print()
print("One-turn chat:")
print(chat_simple("In one sentence, what is a fountain pen?", max_tokens=80, temperature=0.3))
