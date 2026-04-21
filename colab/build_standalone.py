"""Build a self-contained Colab notebook that runs the whole Phase II system
without any git clone. Each module is embedded as a `%%writefile` cell, so
running the notebook top-to-bottom in Colab populates the working dir and
then drives the full pipeline.

Usage:
    python colab/build_standalone.py
    # -> writes colab/phase2_standalone.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "phase2_standalone.ipynb"


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def writefile_cell(target_path: str, src_path: Path) -> dict:
    """Produce a cell that writes `src_path`'s content to `target_path` in Colab."""
    content = src_path.read_text(encoding="utf-8")
    # Ensure the content ends with a newline.
    if not content.endswith("\n"):
        content += "\n"
    body = f"%%writefile {target_path}\n{content}"
    return code_cell(body)


# ---------------------------------------------------------------------------
# Build the cell list
# ---------------------------------------------------------------------------
cells: list[dict] = []

cells.append(md_cell(
    "# Phase II -- Interactive Detective Mystery (standalone Colab)\n"
    "\n"
    "**No git clone required.** Every source file, SLURM script, and scripted transcript used by the system is embedded below as a `%%writefile` cell. Running the notebook top-to-bottom will:\n"
    "\n"
    "1. Install vLLM + openai\n"
    "2. Write all Python modules to `/content/`\n"
    "3. Start a local vLLM OpenAI-compatible server\n"
    "4. Generate one fixed mystery story + plan + world\n"
    "5. Assemble a novel-style markdown story\n"
    "6. Replay both the successful and exception playthroughs\n"
    "7. Show the drama-manager decision log\n"
    "\n"
    "**Template:** 2 (Intervention & Accommodation). See `DESIGN.md` in the source repo for the answer to the template-specific question.\n"
    "\n"
    "**GPU required.** Runtime -> Change runtime type -> T4 / L4 / A100.\n"
))

cells.append(md_cell("## 1. Install dependencies\n\n~3-5 min on Colab. vLLM bundles its own torch+CUDA wheels."))
cells.append(code_cell(
    "!pip install --quiet 'vllm>=0.11,<0.13' 'openai>=1.52'\n"
    "import torch\n"
    "print('CUDA available:', torch.cuda.is_available())\n"
    "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')\n"
    "print('GPU memory:', f\"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\" if torch.cuda.is_available() else 'n/a')\n"
))

cells.append(md_cell("## 2. Pick a model that fits the GPU\n\nT4 16GB: `Qwen2.5-3B-Instruct`. L4 24GB: `Qwen2.5-7B-Instruct`. A100 40GB: `Qwen3-8B`."))
cells.append(code_cell(
    "import os, torch, pathlib\n"
    "os.chdir('/content')\n"
    "for d in ('data', 'logs', 'transcripts', 'scripts', 'tests'):\n"
    "    pathlib.Path(d).mkdir(exist_ok=True)\n"
    "\n"
    "gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0\n"
    "if gpu_gb >= 35:\n"
    "    MODEL = 'Qwen/Qwen3-8B'\n"
    "elif gpu_gb >= 22:\n"
    "    MODEL = 'Qwen/Qwen2.5-7B-Instruct'\n"
    "else:\n"
    "    MODEL = 'Qwen/Qwen2.5-3B-Instruct'\n"
    "\n"
    "PORT = 8000\n"
    "os.environ['LLM_MODEL'] = MODEL\n"
    "os.environ['LLM_ENDPOINT'] = f'http://localhost:{PORT}/v1'\n"
    "os.environ['LLM_API_KEY'] = 'EMPTY'\n"
    "print('Model    :', MODEL)\n"
    "print('Endpoint :', os.environ['LLM_ENDPOINT'])\n"
    "print('Workdir  :', os.getcwd())\n"
))

cells.append(md_cell("## 3. Write the Python modules\n\nEach cell below drops one source file into `/content/`. Run them in order."))

# List of modules + target paths.
modules = [
    ("llm_client.py", ROOT / "llm_client.py"),
    ("plan_types.py", ROOT / "plan_types.py"),
    ("phase1_story_generator.py", ROOT / "phase1_story_generator.py"),
    ("story_to_plan.py", ROOT / "story_to_plan.py"),
    ("world_builder.py", ROOT / "world_builder.py"),
    ("action_interpreter.py", ROOT / "action_interpreter.py"),
    ("drama_manager.py", ROOT / "drama_manager.py"),
    ("game_engine.py", ROOT / "game_engine.py"),
    ("main.py", ROOT / "main.py"),
]
for target, src in modules:
    cells.append(md_cell(f"### `{target}`"))
    cells.append(writefile_cell(target, src))

# Transcripts
cells.append(md_cell("### scripted transcripts"))
cells.append(writefile_cell("transcripts/success_run.txt", ROOT / "transcripts" / "success_run.txt"))
cells.append(writefile_cell("transcripts/exception_run.txt", ROOT / "transcripts" / "exception_run.txt"))

# Smoke test script (not strictly needed since we call llm_client directly, but include for parity).
cells.append(md_cell("### `scripts/test_llm.py` (smoke test)"))
cells.append(writefile_cell("scripts/test_llm.py", ROOT / "scripts" / "test_llm.py"))

# ---- pipeline cells ----
cells.append(md_cell(
    "## 4. Launch vLLM server in the background\n\n"
    "First run on a fresh model can take 3-8 min to download + load."
))
cells.append(code_cell(
    "import subprocess, pathlib, sys, os\n"
    "log_path = pathlib.Path('logs/vllm_server.log')\n"
    "log_path.write_text('')\n"
    "\n"
    "cmd = [\n"
    "    sys.executable, '-m', 'vllm.entrypoints.openai.api_server',\n"
    "    '--model', os.environ['LLM_MODEL'],\n"
    "    '--host', '0.0.0.0', '--port', str(PORT),\n"
    "    '--dtype', 'bfloat16',\n"
    "    '--max-model-len', '4096',\n"
    "    '--gpu-memory-utilization', '0.85',\n"
    "    '--enforce-eager',\n"
    "    '--api-key', 'EMPTY',\n"
    "]\n"
    "print('Launching:', ' '.join(cmd))\n"
    "server = subprocess.Popen(cmd, stdout=open(log_path, 'ab'), stderr=subprocess.STDOUT)\n"
    "print('pid =', server.pid)\n"
    "pathlib.Path('logs/vllm_endpoint.txt').write_text(os.environ['LLM_ENDPOINT'])\n"
))

cells.append(md_cell("## 5. Wait for the server to be ready"))
cells.append(code_cell(
    "import time, urllib.request\n"
    "\n"
    "def probe():\n"
    "    try:\n"
    "        req = urllib.request.Request(f\"{os.environ['LLM_ENDPOINT']}/models\",\n"
    "                                      headers={'Authorization': 'Bearer EMPTY'})\n"
    "        with urllib.request.urlopen(req, timeout=5) as r:\n"
    "            return r.status == 200\n"
    "    except Exception:\n"
    "        return False\n"
    "\n"
    "for i in range(240):  # up to ~20 min\n"
    "    if probe():\n"
    "        print('READY after', i*5, 's')\n"
    "        break\n"
    "    if i % 12 == 0:\n"
    "        print('... still waiting (', i*5, 's)', flush=True)\n"
    "    time.sleep(5)\n"
    "else:\n"
    "    raise RuntimeError('vLLM did not become ready in time -- check logs/vllm_server.log')\n"
))

cells.append(md_cell("## 6. Smoke test"))
cells.append(code_cell("!python scripts/test_llm.py"))

cells.append(md_cell(
    "## 7. Build the story + plan + world\n\n"
    "Phase I generation -> story_to_plan -> world_builder, all against the live Qwen server. "
    "~3-10 min on Qwen3-8B. Outputs saved to `data/`."
))
cells.append(code_cell("!python main.py build --data-dir data --min-points 15"))

cells.append(md_cell(
    "## 8. Assemble a novel-style story\n\n"
    "Produces `data/final_story.md` (Prologue + chapters + Resolution + Epilogue). "
    "Useful when you want the story only, without playing."
))
cells.append(code_cell("!python main.py assemble --data-dir data --out data/final_story.md"))

cells.append(md_cell("Render the assembled story inline:"))
cells.append(code_cell(
    "from IPython.display import Markdown, display\n"
    "display(Markdown(pathlib.Path('data/final_story.md').read_text(encoding='utf-8')))\n"
))

cells.append(md_cell("## 9. Replay a successful run"))
cells.append(code_cell("!python main.py replay transcripts/success_run.txt --data-dir data --log-dir logs"))

cells.append(md_cell("## 10. Replay an exception run (drama manager accommodates)"))
cells.append(code_cell("!python main.py replay transcripts/exception_run.txt --data-dir data --log-dir logs"))

cells.append(md_cell(
    "## 11. Inspect the drama manager log\n\n"
    "Every classification, threat check, removed event and replacement event is a JSON line in `logs/drama.jsonl`. "
    "This is the primary evidence of Template 2 behavior for the final video."
))
cells.append(code_cell(
    "import json\n"
    "for line in pathlib.Path('logs/drama.jsonl').read_text().splitlines()[-20:]:\n"
    "    entry = json.loads(line)\n"
    "    kind = entry.pop('kind')\n"
    "    print(kind, '-', json.dumps(entry)[:200])\n"
))

cells.append(md_cell(
    "## 12. (Optional) Interactive mode\n\n"
    "Input format:\n"
    "- <= 8 words per command (auto-truncated)\n"
    "- Free-form natural language; the LLM parses it into a structured action\n"
    "- Exits are shown at each turn; use those location names for movement\n"
    "- Type `quit` or `exit` to end\n"
    "\n"
    "Example commands: `examine the body`, `go to gallery`, `interview victoria harrington`, `accuse eleanor voss`, "
    "or exception actions like `smash the flute`, `wedge morgue door shut`."
))
cells.append(code_cell(
    "from game_engine import GameEngine, EngineConfig\n"
    "from story_to_plan import load_plan\n"
    "from world_builder import load_world\n"
    "\n"
    "plan = load_plan('data/plan.json')\n"
    "world = load_world('data/world.json')\n"
    "engine = GameEngine(plan, world, EngineConfig(log_dir=pathlib.Path('logs/interactive')))\n"
    "\n"
    "def ask(prompt):\n"
    "    return input(prompt)\n"
    "\n"
    "status = engine.run(get_input=ask, echo=print)\n"
    "print('\\n=== game ended:', status, '===')\n"
))

cells.append(md_cell(
    "## 13. Cleanup\n\n"
    "Kill the background vLLM server to free the GPU. Optional -- Colab will kill it when the runtime disconnects."
))
cells.append(code_cell(
    "try:\n"
    "    server.terminate()\n"
    "    server.wait(timeout=10)\n"
    "    print('server stopped')\n"
    "except Exception as e:\n"
    "    print('cleanup:', e)\n"
))

# ---- assemble notebook ----
nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT}  ({len(cells)} cells, {OUT.stat().st_size/1024:.1f} KB)")
