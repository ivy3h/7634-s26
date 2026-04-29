"""FastAPI server for the LLM-powered detective game.

Serves a dynamically built game page (DATA injected from data/) with the
API_URL wired in, so the browser routes every command through the real
action-interpreter + drama-manager + LLM narration stack.

Usage (from repo root):
    uvicorn web.api_server:app --host 0.0.0.0 --port 8080

Or from /content in Colab after all modules are written:
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8080, reload=False)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Make sure the project root (or /content in Colab) is importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_engine import EngineConfig, GameEngine  # noqa: E402
from story_to_plan import load_plan               # noqa: E402
from world_builder import load_world              # noqa: E402

app = FastAPI(title="Detective Game API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: GameEngine | None = None
_html_cache: str | None = None
DATA_DIR = ROOT / "data"


def _normalize_plan_world() -> None:
    """Normalize character IDs across plan.json + world.json so backend and
    frontend always reference the same canonical character IDs.

    build_game.py performs the same remap in-memory (for the browser data
    blob) but never persists it.  Running this once after plan/world
    generation ensures the game engine reads the same IDs the browser shows.

    The normalization is idempotent: a `_char_normalized` flag in plan.json
    prevents it from running more than once per generated mystery.
    """
    import re as _re

    plan_path  = DATA_DIR / "plan.json"
    world_path = DATA_DIR / "world.json"
    case_path  = DATA_DIR / "case_file.json"

    if not all(p.exists() for p in [plan_path, world_path, case_path]):
        return

    try:
        plan  = json.loads(plan_path.read_text(encoding="utf-8"))
        world = json.loads(world_path.read_text(encoding="utf-8"))
        case  = json.loads(case_path.read_text(encoding="utf-8"))
    except Exception:
        return

    if plan.get("_char_normalized"):
        return

    # ── 1. Canonical character set from initial_state ──────────────────────
    chars: dict = {
        k: v for k, v in plan.get("initial_state", {}).items()
        if k.startswith("character.")
    }

    # ── 2. Ensure criminal is in the set ──────────────────────────────────
    crim_name = (case.get("criminal") or {}).get("name", "")
    if crim_name:
        crim_id = "character." + _re.sub(r"\W+", "_", crim_name.lower()).strip("_")
        if crim_id not in chars:
            entry: dict = {"name": crim_name, "alive": True, "available": True,
                           "role": "associate"}
            chars[crim_id] = entry
            plan.setdefault("initial_state", {})[crim_id] = entry

    # ── 3. First-name lookup ───────────────────────────────────────────────
    by_first: dict[str, list[str]] = {}
    for cid, fields in chars.items():
        name = fields.get("name") or cid.split(".", 1)[-1].replace("_", " ").title()
        first = (_re.split(r"\W+", name.lower()) + [""])[0]
        if first:
            by_first.setdefault(first, []).append(cid)

    # ── 4. Collect all character IDs used in events & location lists ───────
    all_ids: set[str] = set()
    for ev in plan.get("events", {}).values():
        for a in ev.get("args", []):
            if isinstance(a, str) and a.startswith("character."):
                all_ids.add(a)
    for loc in world.get("locations", {}).values():
        for c in loc.get("characters", []):
            if isinstance(c, str) and c.startswith("character."):
                all_ids.add(c)

    # ── 5. Build remap; register unmappable IDs so engine gets a name ─────
    remap: dict[str, str] = {}
    for bad_id in all_ids:
        if bad_id in chars:
            continue
        body  = bad_id.split(".", 1)[-1]
        first = body.split("_", 1)[0]
        candidates = by_first.get(first, [])
        if len(candidates) == 1:
            remap[bad_id] = candidates[0]
        else:
            # No unique match — register the character directly
            name = body.replace("_", " ").title()
            entry = {"name": name, "alive": True, "available": True, "role": "associate"}
            chars[bad_id] = entry
            plan.setdefault("initial_state", {})[bad_id] = entry

    # ── 6. Apply remap to events and location character lists ─────────────
    if remap:
        for ev in plan.get("events", {}).values():
            ev["args"] = [remap.get(a, a) if isinstance(a, str) else a
                          for a in ev.get("args", [])]
        for loc in world.get("locations", {}).values():
            loc["characters"] = [remap.get(c, c) for c in loc.get("characters", [])]

    # ── 7. Persist ────────────────────────────────────────────────────────
    plan["_char_normalized"] = True
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    world_path.write_text(json.dumps(world, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_engine() -> GameEngine:
    _normalize_plan_world()   # idempotent; persists canonical character IDs
    plan = load_plan(str(DATA_DIR / "plan.json"))
    world = load_world(str(DATA_DIR / "world.json"))
    detective_name = "Inspector Rothwell"
    case_path = DATA_DIR / "case_file.json"
    if case_path.exists():
        try:
            case = json.loads(case_path.read_text(encoding="utf-8"))
            detective_name = case.get("detective", {}).get("name", detective_name)
        except Exception:
            pass
    config = EngineConfig(
        narrate_with_llm=True,
        log_dir=ROOT / "logs" / "web",
        detective_name=detective_name,
    )
    return GameEngine(plan, world, config)


def _get_engine() -> GameEngine:
    global _engine
    if _engine is None:
        _engine = _make_engine()
    return _engine


def _build_html() -> str:
    """Return the game HTML with DATA from data/ and API_URL injected."""
    global _html_cache
    if _html_cache is not None:
        return _html_cache

    # Prefer build_game's template so DATA always matches the generated mystery.
    web_dir = ROOT / "web"
    if str(web_dir) not in sys.path:
        sys.path.insert(0, str(web_dir))
    try:
        from build_game import HTML_TEMPLATE, build_game_data  # type: ignore
        data = build_game_data()
        html = HTML_TEMPLATE.replace("__GAME_DATA__", json.dumps(data, ensure_ascii=False))
    except Exception:
        # Fallback: root index.html (may have a different mystery baked in)
        html = (ROOT / "index.html").read_text(encoding="utf-8")

    # Inject empty-string API_URL → JS detects this as API mode (relative URL).
    html = html.replace("let API_URL = null;", 'let API_URL = "";')
    _html_cache = html
    return html


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return HTMLResponse(_build_html())


@app.post("/api/new_game")
async def new_game():
    global _engine, _html_cache
    _engine = _make_engine()
    _html_cache = None  # bust so DATA is re-read on next page load
    return {"status": "ok"}


class StepRequest(BaseModel):
    command: str
    force_event_id: str | None = None


@app.post("/api/step")
async def api_step(req: StepRequest):
    eng = _get_engine()
    try:
        if req.force_event_id:
            return eng.step_force_event(req.force_event_id)
        return eng.step(req.command)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
