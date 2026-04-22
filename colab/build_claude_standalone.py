"""Build a Colab notebook that runs the full Phase II system against the
Anthropic Claude API -- no git clone, no GPU, no vLLM server.

Usage:
    python colab/build_claude_standalone.py
    # -> writes colab/phase2_claude_standalone.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "phase2_claude_standalone.ipynb"


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
    content = src_path.read_text(encoding="utf-8")
    if not content.endswith("\n"):
        content += "\n"
    body = f"%%writefile {target_path}\n{content}"
    return code_cell(body)


# ---------------------------------------------------------------------------
cells: list[dict] = []

cells.append(md_cell(
    "# Phase II -- Interactive Detective Mystery (Claude API / Colab)\n"
    "\n"
    "**No git clone, no GPU, no vLLM.** All files are embedded below as "
    "`%%writefile` cells; every LLM call goes through the Anthropic Claude API.\n"
    "\n"
    "### What you need before running\n"
    "\n"
    "1. An Anthropic API key -- get one at https://console.anthropic.com/\n"
    "2. In Colab, click the **key icon** in the left sidebar ('Secrets'), add a\n"
    "   new secret named **`ANTHROPIC_API_KEY`**, paste your key as the value,\n"
    "   toggle 'Notebook access' on.\n"
    "3. No GPU runtime required. CPU is fine because the model runs on\n"
    "   Anthropic's servers.\n"
    "\n"
    "### What this notebook does\n"
    "\n"
    "1. Installs the `anthropic` SDK.\n"
    "2. Writes every Python module + scripted transcript to `/content/`.\n"
    "3. Pulls your API key from Colab secrets.\n"
    "4. Generates one fixed mystery story + plan + world via Claude.\n"
    "5. Assembles a novel-style `final_story.md`.\n"
    "6. Replays both the successful and exception playthroughs.\n"
    "7. Shows the drama-manager decision log.\n"
))

# --- 1. install ---
cells.append(md_cell("## 1. Install the Anthropic SDK\n\n(~10 seconds.)"))
cells.append(code_cell("!pip install --quiet 'anthropic>=0.40'"))

# --- 2. set up secrets + dirs + model ---
cells.append(md_cell(
    "## 2. Load your API key + pick the Claude model\n\n"
    "Reads `ANTHROPIC_API_KEY` from Colab Secrets. If that fails, falls back "
    "to `getpass` so you can paste it once.\n\n"
    "Default model: `claude-sonnet-4-5` -- fastest reasonable Sonnet. Use "
    "`claude-opus-4-7` if you want the strongest narrative quality (slower, "
    "pricier)."
))
cells.append(code_cell(
    "import os, pathlib, getpass\n"
    "\n"
    "# Create the working directories the scripts expect.\n"
    "os.chdir('/content')\n"
    "for d in ('data', 'logs', 'transcripts', 'scripts', 'tests'):\n"
    "    pathlib.Path(d).mkdir(exist_ok=True)\n"
    "\n"
    "# Pick a Claude model.\n"
    "MODEL = 'claude-sonnet-4-5'   # or 'claude-opus-4-7' / 'claude-haiku-4-5-20251001'\n"
    "os.environ['LLM_MODEL'] = MODEL\n"
    "\n"
    "# Resolve the API key: prefer Colab Secrets, fall back to getpass.\n"
    "if 'ANTHROPIC_API_KEY' not in os.environ:\n"
    "    try:\n"
    "        from google.colab import userdata\n"
    "        os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')\n"
    "        print('API key loaded from Colab Secrets.')\n"
    "    except Exception as e:\n"
    "        print('Secrets unavailable, asking interactively:', e)\n"
    "        os.environ['ANTHROPIC_API_KEY'] = getpass.getpass('Paste ANTHROPIC_API_KEY: ')\n"
    "\n"
    "assert os.environ.get('ANTHROPIC_API_KEY'), 'ANTHROPIC_API_KEY is still empty'\n"
    "print('Model:', MODEL)\n"
    "print('Workdir:', os.getcwd())\n"
))

# --- 3. write modules ---
cells.append(md_cell(
    "## 3. Write the Python modules\n\n"
    "The next cells drop every source file into `/content/`. The only difference "
    "from the vLLM version is `llm_client.py` -- here it uses the Anthropic SDK "
    "with the exact same public surface (`chat`, `chat_simple`, `chat_json`), so "
    "none of the other modules need to change."
))

# llm_client — Claude flavor
cells.append(md_cell("### `llm_client.py` (Claude backend)"))
cells.append(writefile_cell("llm_client.py", HERE / "llm_client_claude.py"))

# all other modules, verbatim
modules = [
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

# transcripts
cells.append(md_cell("### scripted transcripts"))
cells.append(writefile_cell("transcripts/success_run.txt", ROOT / "transcripts" / "success_run.txt"))
cells.append(writefile_cell("transcripts/exception_run.txt", ROOT / "transcripts" / "exception_run.txt"))

# --- 4. smoke test ---
cells.append(md_cell("## 4. Smoke test\n\nOne round-trip to confirm the API key works."))
cells.append(code_cell(
    "import llm_client, importlib\n"
    "importlib.reload(llm_client)   # in case the %%writefile cell was re-run\n"
    "import json\n"
    "print(json.dumps(llm_client.health_check(), indent=2))\n"
))

# --- 5. build ---
cells.append(md_cell(
    "## 5. Build the story + plan + world\n\n"
    "Runs Phase I -> story_to_plan -> world_builder against Claude. Expect "
    "~3-8 min with Sonnet (faster than a local 8B model). The artifacts land "
    "in `data/`."
))
cells.append(code_cell("!python main.py build --data-dir data --min-points 15"))

# --- 6. assemble ---
cells.append(md_cell(
    "## 6. Assemble a novel-style story\n\n"
    "Produces `data/final_story.md`. ~2-4 min."
))
cells.append(code_cell("!python main.py assemble --data-dir data --out data/final_story.md"))
cells.append(md_cell("Render inline:"))
cells.append(code_cell(
    "from IPython.display import Markdown, display\n"
    "display(Markdown(pathlib.Path('data/final_story.md').read_text(encoding='utf-8')))\n"
))

# --- 7. replays ---
cells.append(md_cell("## 7. Replay a successful run"))
cells.append(code_cell("!python main.py replay transcripts/success_run.txt --data-dir data --log-dir logs"))

cells.append(md_cell("## 8. Replay an exception run (drama manager accommodates)"))
cells.append(code_cell("!python main.py replay transcripts/exception_run.txt --data-dir data --log-dir logs"))

# --- 9. drama log ---
cells.append(md_cell(
    "## 9. Inspect the drama manager log\n\n"
    "Every classification, commonsense threat check, removed event and "
    "replacement event is a JSON line in `logs/drama.jsonl`."
))
cells.append(code_cell(
    "import json\n"
    "for line in pathlib.Path('logs/drama.jsonl').read_text().splitlines()[-20:]:\n"
    "    entry = json.loads(line)\n"
    "    kind = entry.pop('kind')\n"
    "    print(kind, '-', json.dumps(entry)[:200])\n"
))

# --- 10. interactive ---
cells.append(md_cell(
    "## 10. (Optional) Interactive mode\n\n"
    "Input format:\n"
    "- <= 8 words per command\n"
    "- Free-form natural language; Claude parses it into a structured action\n"
    "- Exits are shown each turn; use those location names for movement\n"
    "- Type `quit` or `exit` to end\n"
    "\n"
    "Examples: `examine the body`, `go to gallery`, `interview eleanor voss`, "
    "`smash the flute`, `wedge morgue door shut`."
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

# ---------------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT}  ({len(cells)} cells, {OUT.stat().st_size/1024:.1f} KB)")
