"""Build a self-contained Colab notebook that hosts the Phase II detective game
as a live website backed by a local LLM (vLLM) and FastAPI.

Pre-generated Phase I files required in /content/data/ before running:
    plan.json       (from story_to_plan)
    case_file.json  (from phase1_story_generator)

The notebook builds world.json itself (world_builder, needs the LLM).

Usage:
    python colab/build_web_standalone.py
    # -> writes colab/phase2_web_standalone.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT  = HERE / "phase2_web_standalone.ipynb"


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
    return code_cell(f"%%writefile {target_path}\n{content}")


# ---------------------------------------------------------------------------
cells: list[dict] = []

# ── Title ──────────────────────────────────────────────────────────────────
cells.append(md_cell(
    "# Phase II — Interactive Detective Mystery (web server)\n"
    "\n"
    "Hosts the detective game as a **live website** backed by vLLM + FastAPI.\n"
    "Every player command goes through the real action-interpreter, drama-manager,\n"
    "and LLM narration stack.\n"
    "\n"
    "**Before running:** upload your Phase I output files to `/content/data/`:\n"
    "- `plot_points.json` — story beats from `phase1_story_generator`\n"
    "- `case_file.json`   — case metadata from `phase1_story_generator`\n"
    "\n"
    "The notebook runs the full Phase II pipeline: `story_to_plan` (→ `plan.json`) → `world_builder` (→ `world.json`).\n"
    "\n"
    "**GPU required.** Runtime → Change runtime type → T4 / L4 / A100.\n"
))

# ── 1. Install ─────────────────────────────────────────────────────────────
cells.append(md_cell("## 1. Install dependencies\n\n~3–5 min on a fresh runtime."))
cells.append(code_cell(
    "!pip install --quiet 'vllm>=0.11,<0.13' 'openai>=1.52' "
    "'fastapi>=0.110' 'uvicorn[standard]>=0.29'\n"
    "import torch\n"
    "print('CUDA available:', torch.cuda.is_available())\n"
    "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')\n"
    "print('GPU memory:', f\"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\""
    " if torch.cuda.is_available() else 'n/a')\n"
))

# ── 2. Pick model ──────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 2. Pick model and set up directories\n\n"
    "T4 16 GB → `Qwen2.5-3B-Instruct` · L4 24 GB → `Qwen2.5-7B-Instruct` · "
    "A100 40 GB → `Qwen3-8B`"
))
cells.append(code_cell(
    "import os, torch, pathlib\n"
    "os.chdir('/content')\n"
    "for d in ('data', 'logs', 'logs/web', 'web'):\n"
    "    pathlib.Path(d).mkdir(exist_ok=True)\n"
    "\n"
    "if torch.cuda.is_available():\n"
    "    props    = torch.cuda.get_device_properties(0)\n"
    "    gpu_gb   = props.total_memory / 1e9\n"
    "    cc_major = props.major\n"
    "else:\n"
    "    gpu_gb, cc_major = 0, 0\n"
    "\n"
    "if gpu_gb >= 35:\n"
    "    MODEL = 'Qwen/Qwen3-8B'\n"
    "elif gpu_gb >= 22:\n"
    "    MODEL = 'Qwen/Qwen2.5-7B-Instruct'\n"
    "else:\n"
    "    MODEL = 'Qwen/Qwen2.5-3B-Instruct'\n"
    "\n"
    "DTYPE    = 'bfloat16' if cc_major >= 8 else 'float16'\n"
    "PORT     = 8000\n"
    "WEB_PORT = 7860\n"
    "\n"
    "os.environ['LLM_MODEL']    = MODEL\n"
    "os.environ['LLM_ENDPOINT'] = f'http://localhost:{PORT}/v1'\n"
    "os.environ['LLM_API_KEY']  = 'EMPTY'\n"
    "os.environ['LLM_DTYPE']    = DTYPE\n"
    "\n"
    "print('Model    :', MODEL)\n"
    "print('Dtype    :', DTYPE, f'(CC {cc_major}.x)')\n"
    "print('Endpoint :', os.environ['LLM_ENDPOINT'])\n"
))

# ── 3. Write modules ───────────────────────────────────────────────────────
cells.append(md_cell("## 3. Write Python modules"))

modules = [
    ("llm_client.py",             ROOT / "llm_client.py"),
    ("plan_types.py",             ROOT / "plan_types.py"),
    ("phase1_story_generator.py", ROOT / "phase1_story_generator.py"),
    ("story_to_plan.py",          ROOT / "story_to_plan.py"),
    ("world_builder.py",          ROOT / "world_builder.py"),
    ("action_interpreter.py",     ROOT / "action_interpreter.py"),
    ("drama_manager.py",          ROOT / "drama_manager.py"),
    ("game_engine.py",            ROOT / "game_engine.py"),
    ("main.py",                   ROOT / "main.py"),
]
for target, src in modules:
    cells.append(writefile_cell(target, src))

cells.append(writefile_cell("web/build_game.py",  ROOT / "web" / "build_game.py"))
cells.append(writefile_cell("web/api_server.py",  ROOT / "web" / "api_server.py"))

# ── 4. Verify Phase I files ────────────────────────────────────────────────
cells.append(md_cell(
    "## 4. Verify input files\n\n"
    "`plan.json` and `world.json` are independent outputs checked separately:\n\n"
    "| Have | Step 7 behaviour |\n"
    "|------|------------------|\n"
    "| `plan.json` + `world.json` | loads both — no LLM needed |\n"
    "| `plan.json` only | loads plan, runs `world_builder` (needs vLLM) |\n"
    "| neither | needs `plot_points.json` + `case_file.json`; runs full pipeline |\n\n"
    "Upload pre-generated files to `/content/data/` to skip the corresponding step."
))
cells.append(code_cell(
    "import pathlib, json, sys\n"
    "\n"
    "have_world = pathlib.Path('data/world.json').exists()\n"
    "have_plan  = pathlib.Path('data/plan.json').exists()\n"
    "\n"
    "print(f'plan.json  : {\"found\" if have_plan else \"will be generated from Phase I files\"}')\n"
    "print(f'world.json : {\"found\" if have_world else \"will be generated (needs LLM)\"}')\n"
    "\n"
    "if not have_plan:\n"
    "    required = ['data/plot_points.json', 'data/case_file.json']\n"
    "    missing  = [f for f in required if not pathlib.Path(f).exists()]\n"
    "    if missing:\n"
    "        print('MISSING — needed to generate plan.json:')\n"
    "        for f in missing: print(' •', f)\n"
    "        sys.exit(1)\n"
    "    plot_points = json.loads(pathlib.Path('data/plot_points.json').read_text())\n"
    "    case_file   = json.loads(pathlib.Path('data/case_file.json').read_text())\n"
    "    print(f'plot_points.json : {len(plot_points)} beats')\n"
    "    print(f'case_file.json   : victim = {case_file.get(\"victim\", {}).get(\"name\", \"?\")}')\n"
    "\n"
    "print('Ready.')\n"
))

# ── 5. Launch vLLM ─────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 5. Launch vLLM server\n\n"
    "First run downloads model weights (3–8 min). Subsequent runs load from Colab cache."
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
    "    '--dtype', os.environ['LLM_DTYPE'],\n"
    "    '--max-model-len', '4096',\n"
    "    '--gpu-memory-utilization', '0.85',\n"
    "    '--enforce-eager',\n"
    "    '--api-key', 'EMPTY',\n"
    "]\n"
    "vllm_proc = subprocess.Popen(cmd, stdout=open(log_path, 'ab'), stderr=subprocess.STDOUT)\n"
    "print('vLLM pid =', vllm_proc.pid)\n"
))

# ── 6. Wait for vLLM ───────────────────────────────────────────────────────
cells.append(md_cell("## 6. Wait for vLLM to be ready"))
cells.append(code_cell(
    "import time, urllib.request, pathlib\n"
    "\n"
    "def probe():\n"
    "    try:\n"
    "        req = urllib.request.Request(\n"
    "            f\"{os.environ['LLM_ENDPOINT']}/models\",\n"
    "            headers={'Authorization': 'Bearer EMPTY'})\n"
    "        with urllib.request.urlopen(req, timeout=5) as r:\n"
    "            return r.status == 200\n"
    "    except Exception:\n"
    "        return False\n"
    "\n"
    "def _show_log(tail=60):\n"
    "    log = pathlib.Path('logs/vllm_server.log')\n"
    "    if log.exists():\n"
    "        lines = log.read_text(errors='replace').splitlines()\n"
    "        print('\\n── vllm_server.log (last', min(tail, len(lines)), 'lines) ──')\n"
    "        print('\\n'.join(lines[-tail:]))\n"
    "        print('── end of log ──')\n"
    "    else:\n"
    "        print('(log file not found)')\n"
    "\n"
    "for i in range(240):\n"
    "    if probe():\n"
    "        print('vLLM READY after', i * 5, 's')\n"
    "        break\n"
    "    # If the process died, no point waiting further — show log immediately\n"
    "    rc = vllm_proc.poll()\n"
    "    if rc is not None:\n"
    "        _show_log()\n"
    "        raise RuntimeError(f'vLLM process exited early with code {rc}')\n"
    "    if i % 12 == 0 and i > 0:\n"
    "        print(f'... still waiting ({i * 5}s)', flush=True)\n"
    "    time.sleep(5)\n"
    "else:\n"
    "    _show_log()\n"
    "    raise RuntimeError('vLLM did not become ready after 20 min — see log above')\n"
))

# ── 7. Build world ─────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 7. Phase II pipeline\n\n"
    "Each step is skipped if its output file already exists in `/content/data/`."
))
cells.append(code_cell(
    "import sys, pathlib\n"
    "sys.path.insert(0, '/content')\n"
    "\n"
    "from world_builder import build_world, load_world, save_world\n"
    "\n"
    "# ── Step 1: plan.json ────────────────────────────────────────────────────\n"
    "if pathlib.Path('data/plan.json').exists():\n"
    "    print('plan.json found — skipping story_to_plan.')\n"
    "    from story_to_plan import load_plan\n"
    "    plan = load_plan('data/plan.json')\n"
    "    print(f'  {len(plan.events)} events, {len(plan.causal_links)} causal links')\n"
    "else:\n"
    "    print('--- story_to_plan ---')\n"
    "    from phase1_story_generator import load_checkpoint\n"
    "    from story_to_plan import build_plan\n"
    "    case_file   = load_checkpoint('data/case_file.json')\n"
    "    plot_points = load_checkpoint('data/plot_points.json')\n"
    "    plan = build_plan(case_file, plot_points, out_path='data/plan.json')\n"
    "    print(f'  plan.json written: {len(plan.events)} events, {len(plan.causal_links)} causal links')\n"
    "\n"
    "# ── Step 2: world.json ───────────────────────────────────────────────────\n"
    "if pathlib.Path('data/world.json').exists():\n"
    "    print('world.json found — skipping world_builder (no LLM needed).')\n"
    "    world = load_world('data/world.json')\n"
    "else:\n"
    "    print('--- world_builder (LLM calls) ---')\n"
    "    world = build_world(plan)\n"
    "    save_world(world, 'data/world.json')\n"
    "    print('  world.json written')\n"
    "\n"
    "print(f'Ready: {len(world.locations)} locations')\n"
    "for lid, loc in world.locations.items():\n"
    "    print(f'  {loc.name}: exits={sorted(loc.adjacent)}')\n"
))

# ── 7b. Interactive test (optional) ───────────────────────────────────────
cells.append(md_cell(
    "## 7b. Interactive game test (optional)\n\n"
    "Run this cell to play a test session directly in the notebook before starting the web server.\n"
    "Every command shows a concise drama-manager log, the narration, and a location snapshot.\n"
    "Type `quit` to exit the loop and continue to step 8."
))
cells.append(code_cell(
    "import sys, pathlib\n"
    "sys.path.insert(0, '/content')\n"
    "\n"
    "from story_to_plan import load_plan\n"
    "from world_builder import load_world\n"
    "from game_engine import EngineConfig, GameEngine\n"
    "\n"
    "_t_plan   = load_plan('data/plan.json')\n"
    "_t_world  = load_world('data/world.json')\n"
    "_t_engine = GameEngine(\n"
    "    _t_plan, _t_world,\n"
    "    EngineConfig(narrate_with_llm=True, log_dir=pathlib.Path('logs/test')),\n"
    ")\n"
    "\n"
    "def _path_summary(eng):\n"
    "    if eng.drama.goal_satisfied(eng.state):\n"
    "        return 'GOAL SATISFIED — case ready to close'\n"
    "    rem = len(eng.drama.remaining)\n"
    "    return f'{rem} plan event(s) remaining — investigation ongoing'\n"
    "\n"
    "def _hint_commands(eng):\n"
    "    loc_id = eng.state['detective']['location']\n"
    "    hints = []\n"
    "    for eid in eng.drama.remaining:\n"
    "        ev = eng.drama.plan.events[eid]\n"
    "        if ev.location != loc_id:\n"
    "            continue\n"
    "        verb = ev.verb\n"
    "        if verb in ('examine', 'search', 'investigate', 'observe'):\n"
    "            verb = 'examine'\n"
    "        elif verb in ('interview', 'consult', 'confront', 'visit'):\n"
    "            verb = 'question'\n"
    "        arg_str = ''\n"
    "        for a in ev.args:\n"
    "            s = str(a)\n"
    "            if s.startswith('character.'):\n"
    "                name = eng.state.get(s, {}).get('name', s.split('.')[-1].replace('_', ' '))\n"
    "                arg_str = name.split()[-1]\n"
    "                break\n"
    "            elif s.startswith('evidence.'):\n"
    "                desc = eng.state.get(s, {}).get('description', '')\n"
    "                arg_str = ' '.join(desc.split()[:4]) if desc else s\n"
    "                break\n"
    "            elif not s.startswith('location.'):\n"
    "                arg_str = s[:35]\n"
    "                break\n"
    "        hint = f'{verb} {arg_str}'.strip() if arg_str else verb\n"
    "        if hint not in hints:\n"
    "            hints.append(hint)\n"
    "        if len(hints) >= 4:\n"
    "            break\n"
    "    return hints\n"
    "\n"
    "def _print_location(eng):\n"
    "    print(eng.render_location())\n"
    "    hints = _hint_commands(eng)\n"
    "    if hints:\n"
    "        print('  Try: ' + '  |  '.join(hints))\n"
    "\n"
    "print('=' * 56)\n"
    "print('  Interactive game test  (type quit to exit)')\n"
    "print('=' * 56)\n"
    "print()\n"
    "print(_t_engine.render_map())\n"
    "print()\n"
    "_print_location(_t_engine)\n"
    "\n"
    "while True:\n"
    "    try:\n"
    "        cmd = input('\\n> ').strip()\n"
    "    except (EOFError, KeyboardInterrupt):\n"
    "        print('\\n[session ended]')\n"
    "        break\n"
    "    if cmd.lower() in ('quit', 'exit', 'q'):\n"
    "        print('[session ended]')\n"
    "        break\n"
    "    if not cmd:\n"
    "        continue\n"
    "\n"
    "    result = _t_engine.step(cmd)\n"
    "    tag    = result['classification']\n"
    "\n"
    "    # ── DM log ───────────────────────────────────────────────────\n"
    "    dm_line = f'[DM] {tag}'\n"
    "    if result.get('triggered_event_id'):\n"
    "        dm_line += f\"  event={result['triggered_event_id']}\"\n"
    "    if result.get('moved_to'):\n"
    "        dm_line += f\"  moved→{result['moved_to']}\"\n"
    "    print(dm_line)\n"
    "\n"
    "    if tag == 'exceptional':\n"
    "        for entry in reversed(_t_engine.drama.log):\n"
    "            if entry.kind == 'accommodation':\n"
    "                p = entry.payload\n"
    "                print(f\"     removed  : {p.get('removed_events', [])}\")\n"
    "                print(f\"     added    : {p.get('replacement_event_ids', [])}\")\n"
    "                rat = (p.get('rationale') or '')[:200]\n"
    "                if rat:\n"
    "                    print(f'     rationale: {rat}')\n"
    "                break\n"
    "        for entry in reversed(_t_engine.drama.log):\n"
    "            if entry.kind == 'classification':\n"
    "                threats = [t.get('event_id') for t in entry.payload.get('soft_threats', [])]\n"
    "                if threats:\n"
    "                    print(f'     threats  : {threats}')\n"
    "                break\n"
    "\n"
    "    print(f'[DM] {_path_summary(_t_engine)}')\n"
    "\n"
    "    # ── Narration ────────────────────────────────────────────────\n"
    "    print()\n"
    "    for entry in result.get('log_entries', []):\n"
    "        if entry.get('title'):\n"
    "            print(f\"  {entry['title']}\")\n"
    "        print(f\"  {entry['text']}\")\n"
    "\n"
    "    # ── Location snapshot + hints ────────────────────────────────\n"
    "    print()\n"
    "    _print_location(_t_engine)\n"
    "\n"
    "    if result.get('game_over'):\n"
    "        print('\\n=== CASE SOLVED ===')\n"
    "        break\n"
))

# ── 8. Start web server ────────────────────────────────────────────────────
cells.append(md_cell("## 8. Start the FastAPI web server"))
cells.append(code_cell(
    "import subprocess, pathlib, socket\n"
    "sys.path.insert(0, '/content/web')\n"
    "\n"
    "def _wait_port_free(port, timeout=10):\n"
    "    deadline = time.time() + timeout\n"
    "    while time.time() < deadline:\n"
    "        try:\n"
    "            with socket.socket() as s:\n"
    "                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    "                s.bind(('0.0.0.0', port))\n"
    "            return True\n"
    "        except OSError:\n"
    "            time.sleep(0.3)\n"
    "    return False\n"
    "\n"
    "# Kill previous process (SIGKILL — cannot be ignored) then wait for port.\n"
    "if '_web_proc' in globals() and _web_proc is not None:\n"
    "    _web_proc.kill()\n"
    "    _web_proc.wait()\n"
    "    _web_proc = None\n"
    "_wait_port_free(WEB_PORT)\n"
    "\n"
    "log_path = pathlib.Path('logs/web/uvicorn.log')\n"
    "log_path.parent.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "_web_proc = subprocess.Popen(\n"
    "    [sys.executable, '-m', 'uvicorn', 'api_server:app',\n"
    "     '--host', '0.0.0.0', '--port', str(WEB_PORT), '--log-level', 'warning'],\n"
    "    cwd='/content/web',\n"
    "    stdout=open(log_path, 'ab'), stderr=subprocess.STDOUT,\n"
    ")\n"
    "time.sleep(2)\n"
    "if _web_proc.poll() is None:\n"
    "    print(f'Web server running on port {WEB_PORT}')\n"
    "else:\n"
    "    print(f'ERROR: web server exited immediately — check {log_path}')\n"
))

# ── 9. Print URL ───────────────────────────────────────────────────────────
cells.append(md_cell(
    "## 9. Open the game\n\n"
    "Click the URL below. It stays live as long as this Colab session is running."
))
cells.append(code_cell(
    "from google.colab.output import eval_js\n"
    "url = eval_js(f'google.colab.kernel.proxyPort({WEB_PORT})')\n"
    "print('='*60)\n"
    "print(url)\n"
    "print('='*60)\n"
))

# ── 10. Cleanup ────────────────────────────────────────────────────────────
cells.append(md_cell("## 10. Cleanup\n\nStops vLLM and frees the GPU."))
cells.append(code_cell(
    "try:\n"
    "    vllm_proc.terminate()\n"
    "    vllm_proc.wait(timeout=10)\n"
    "    print('vLLM stopped')\n"
    "except Exception as e:\n"
    "    print('cleanup:', e)\n"
))

# ── Assemble ───────────────────────────────────────────────────────────────
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
