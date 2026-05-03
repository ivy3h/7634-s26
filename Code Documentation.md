# Code Documentation — Phase II Interactive Detective Mystery

## Architecture Overview

The system has two stages: a **planning stage** that converts Phase I story outputs into a structured plan and world, and an **execution stage** where a player interacts with the story through a web interface backed by a live LLM.

```
┌─────────────────────── PLANNING STAGE ───────────────────────┐
│                                                               │
│  case_file.json ──┐                                          │
│                   ├──► story_to_plan.py ──► plan.json        │
│  plot_points.json ┘         │                                 │
│                             └──► world_builder.py ──► world.json │
│                                                               │
└───────────────────────────────────────────────────────────────┘

┌─────────────────────── EXECUTION STAGE ──────────────────────┐
│                                                               │
│  Browser (index.html / build_game.py)                        │
│       │  POST /api/step                                       │
│       ▼                                                       │
│  web/api_server.py                                           │
│       │                                                       │
│       ▼                                                       │
│  game_engine.py ──► action_interpreter.py ──► llm_client.py  │
│       │                                                       │
│       └──► drama_manager.py                                   │
│                 │                                             │
│          Layer 1: _hard_violations()    (no LLM)             │
│          Layer 2: _commonsense_threats() (LLM)               │
│                 │                                             │
│          if exceptional: accommodate()  (LLM)                │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Module Map

### `plan_types.py` — Shared Data Structures

Defines the core dataclasses used throughout the system. Every other module imports from here.

| Class | Purpose | Key fields |
|---|---|---|
| `Condition` | Predicate over world state: `state[subject][attr] <op> value` | `subject`, `attr`, `op`, `value` |
| `Effect` | World-state mutation | `subject`, `attr`, `op` (`set`/`add`/`remove`), `value` |
| `Event` | One detective action step | `verb`, `args`, `location`, `preconditions`, `effects`, `reveals` |
| `CausalLink` | Dependency between two events | `producer`, `condition`, `consumer` |
| `Plan` | Complete partial-order plan | `events`, `causal_links`, `initial_state`, `goal` |

All classes implement `.to_dict()` / `.from_dict()` for JSON serialisation.

---

### `llm_client.py` — Single LLM Access Point

All LLM calls in the project go through this module. Swapping the backend (vLLM → Claude API) only requires changing environment variables.

```
LLM_ENDPOINT  (default: http://localhost:8000/v1)
LLM_MODEL     (default: Qwen/Qwen3-8B)
```

**Key functions:**

| Function | Purpose |
|---|---|
| `chat(messages, ...)` | Core call with 8-retry exponential backoff |
| `chat_simple(prompt, system=...)` | Single-turn convenience wrapper |
| `chat_json(prompt, ...)` | Requests JSON, re-prompts on parse failure (up to 3 retries) |
| `health_check()` | Smoke-test before starting the server |
| `_strip_think(text)` | Removes Qwen3 `<think>…</think>` reasoning blocks from output |

---

### `story_to_plan.py` — Phase I → Structured Plan

Converts the unstructured Phase I story outputs into a `Plan` with typed events, preconditions, effects, and causal links.

```
case_file.json + plot_points.json
         │
         ▼
extract_event_from_plot_point()   ← one LLM call per plot point
         │
         ▼
derive_causal_links()             ← pattern-match effects → preconditions (no LLM)
         │                          three passes: strict, reveal→reference, analyzed→reference
         ▼
build_initial_state()             ← seed world state from case_file
build_goal()                      ← goal conditions for case closure
         │
         ▼
     plan.json
```

**Entry point:** `build_plan(case_file, plot_points, out_path)` → `Plan`

**Loading saved plan:** `load_plan(path)` → `Plan`

---

### `world_builder.py` — Location Graph Construction

Builds a navigable world graph from the plan's location references. Uses LLM calls to determine which locations are adjacent and to generate descriptions.

```
plan.json (distinct locations)
         │
         ▼
_analyze_adjacency()    ← LLM: which location pairs are walkable?
         │
         ▼
build_world()           ← insert intermediary locations for unreachable pairs
         │                 seed each location with characters + evidence from plan
         ▼
     world.json
```

**Key classes:** `Location` (id, name, adjacent set, characters set, evidence set), `World` (locations dict, starting_location)

**Entry points:** `build_world(plan)`, `save_world(world, path)`, `load_world(path)`

---

### `action_interpreter.py` — Free-Text → Structured Action

Converts a player's raw text command into a structured dict that the drama manager can reason about. Does not know about causal links — only surface effects.

**Input:** raw string (e.g. `"look at the ring carefully"`)

**Output JSON shape:**
```json
{
  "verb": "examine",
  "args": ["evidence.E001"],
  "target_location": "",
  "effects": [{"subject": "evidence.E001", "attr": "discovered", "op": "set", "value": true}],
  "novel_state_vars": [],
  "plain_summary": "Examine the ladies' ring near the body"
}
```

**`novel_state_vars`** is the key field for accommodation: any physical state created by the action (e.g. `door.jammed`) that wasn't previously modelled is listed here, which triggers the commonsense threat check in the drama manager.

**Entry point:** `interpret_action(raw_input, world_summary)` → dict

The `world_summary` passed to this function is built by `game_engine.py`'s `_world_summary_for_interpreter()` and contains only entities present in the current scene, enforcing scene boundaries.

---

### `drama_manager.py` — Classification and Plan Repair

The core of the system. Classifies every action and repairs the plan when exceptions occur.

#### Classification pipeline (`classify()`)

```
parsed_action
      │
      ├──► _find_constituent_match()
      │         verb family match + token overlap with remaining events
      │         → constituent if matched
      │
      ├──► _hard_violations()           Layer 1 (no LLM, O(links × effects))
      │         checks if proposed effects negate any active CausalLink condition
      │         → exceptional/causal_violation if violated
      │
      └──► _commonsense_threats()       Layer 2 (one LLM call)
                only runs if: has effects OR novel_state_vars OR is_destructive
                LLM asked: does this action make any remaining event impossible?
                → exceptional/soft_threat if threatened
```

**Verb family matching** (`_VERB_FAMILIES`, used in `_find_constituent_match`):
- Inspection family: `examine`, `check`, `inspect`, `search`, `investigate`, `look`, …
- Analysis family: `analyze`, `test`, `process`
- Social family: `interview`, `question`, `talk`, `ask`, `consult`, `confront`, …
- Movement family: `move`, `go`, `walk`, `travel`, `head`

**Destructive verbs** (`_DESTRUCTIVE_VERBS`): `burn`, `destroy`, `shoot`, `kill`, `tamper`, `steal`, `hide`, … — these always produce `hard_block` (refused, no effects applied).

#### Action subtypes and outcomes

| Subtype | Meaning | Outcome |
|---|---|---|
| `constituent` | Matches a planned event | Event executed, plan advances |
| `consistent` | No plan match, no threats | Effects applied, plan unchanged |
| `exceptional/hard_block` | Destructive verb, no plan match | Refused — world unchanged |
| `exceptional/causal_violation` | Breaks a causal link | `accommodate()` called |
| `exceptional/soft_threat` | LLM says future events threatened | `accommodate()` called |

#### Accommodation (`accommodate()`)

```
threatened event IDs (from classification)
      │
      ├── remove from self.remaining
      │
      └── LLM call (ACCOMMODATION_PROMPT)
              given: world snapshot, goal, removed events, surviving links
              produces: 0–2 replacement events that restore goal reachability
              → replacement Event objects added to plan.events + self.remaining
```

**Log kinds written to `logs/drama.jsonl`:**
- `classification` — every action's verdict
- `executed_constituent` — plan event fired
- `applied_free_effects` — consistent/exceptional effects
- `accommodation` — plan repair with before/after snapshots
- `accommodation_failed` — LLM error during repair

---

### `game_engine.py` — Game Loop and State Management

Owns the mutable world state. Orchestrates the full per-turn pipeline and builds the API response.

#### Per-turn pipeline (`step()`)

```
raw text
   │
   ├── _world_summary_for_interpreter()   build scene context
   │
   ├── action_interpreter.interpret_action()
   │
   ├── drama_manager.classify()
   │
   ├── [if constituent] drama_manager.execute_constituent()
   ├── [if consistent]  drama_manager.apply_free_effects()
   └── [if exceptional/soft_threat or causal_violation]
           drama_manager.apply_free_effects()
           drama_manager.accommodate()
   │
   ├── _narrate()    LLM narration or _stub_narration() fallback
   │
   ├── _scene_state()   current location's live characters + discovered evidence
   │
   └── return {log_entries, dm_entry, scene_characters, scene_evidence, scene_leads, …}
```

#### Key methods

| Method | Purpose |
|---|---|
| `_scene_state()` | Returns (chars, evidence) actually present and discovered at current location — used for "You notice" and sidebar |
| `_next_hint_cmd()` | Generates the next actionable hint string from remaining plan events |
| `_evidence_desc(eid)` | Looks up evidence description from `plan.initial_state` |
| `_pending_scene_leads()` | Returns pending plan event targets at current location for sidebar |
| `step_force_event(event_id)` | Execute a plan event directly (hint chip click), bypassing interpreter |
| `render_map()` | ASCII location graph for CLI mode |

**`EngineConfig`** fields: `narrate_with_llm` (bool), `log_dir` (Path), `max_turns` (int)

---

### `web/api_server.py` — FastAPI Web Server

Thin HTTP wrapper around `GameEngine`. Serves the game HTML and handles `/api/step` calls.

```
GET  /              → serve HTML (built by build_game.py's HTML_TEMPLATE + build_game_data())
POST /api/new_game  → reset engine and bust HTML cache
POST /api/step      → eng.step(command) or eng.step_force_event(id)
GET  /api/state     → return current engine state snapshot
```

**HTML assembly:** On first page load, `_build_html()` imports `build_game.HTML_TEMPLATE` and `build_game.build_game_data()`, injects `DATA` as JSON, and replaces `let API_URL = null` with `let API_URL = ""` to enable API mode.

The engine instance is a module-level singleton (`_engine`), reset only on `/api/new_game`.

---

### `web/build_game.py` — Standalone HTML Game + API Template

Dual purpose:
1. **Run directly** (`python web/build_game.py`) → writes `web/game.html`, a fully self-contained offline game with a simplified in-browser drama manager (no LLM).
2. **Imported by `api_server.py`** → `HTML_TEMPLATE` and `build_game_data()` are used to serve the LLM-backed live version.

**`build_game_data()`** packs `plan.json`, `world.json`, `case_file.json`, and `final_story.md` into a single JS-friendly dict injected as `const DATA = {...}`.

**Frontend JS structure (inside `HTML_TEMPLATE`):**

| Function | Purpose |
|---|---|
| `renderSidebar()` | Sidebar: characters with `(spoke with)` annotation, evidence with `(analyzed)`/`(pending)` annotation, "Notable here" leads |
| `renderDmLog()` | Drama Manager log panel — constituent (✓), accommodated (⚠), refused (✗) |
| `_runCommandViaAPI()` | POST to `/api/step`, update `state`, re-render |
| `_mdToHtml()` | Minimal markdown renderer for the "Read Story" overlay |
| `freshState()` | Returns blank initial state (used on new game / API mode) |
| `greet()` | Opening narration and first scene render |

---

## Data Files

| File | Producer | Consumer | Contents |
|---|---|---|---|
| `data/case_file.json` | Phase I | `story_to_plan.py`, `build_game.py` | Victim, suspects, conspirators, evidence, criminal |
| `data/plot_points.json` | Phase I | `story_to_plan.py` | ~20 plot beats with action + narrative |
| `data/plan.json` | `story_to_plan.py` | `game_engine.py`, `drama_manager.py` | Events, causal links, initial state, goal |
| `data/world.json` | `world_builder.py` | `game_engine.py` | Location graph with adjacency, characters, evidence |
| `data/final_story.md` | `main.py assemble` or pipeline Step 3 | `build_game.py` ("Read Story") | Novel-style assembled prose |
| `data/story_bible.json` | Phase I | `story_to_plan.py` | Pinned names and constants |

---

## Colab Notebook (`Realtime LLM Browser System/Phase II Standalone Web System.ipynb`)

The notebook embeds all module source code in `%%writefile` cells and runs the full pipeline.

| Cell | Section | Key action |
|---|---|---|
| 0 | Header + upload instructions | Lists required data files to upload |
| 2 | Install dependencies | `pip install vllm openai fastapi uvicorn` |
| 4 | Setup directories | Create `/content/data/`, `logs/`, `web/` |
| 6–16 | Write Python modules | `%%writefile` cells for each `.py` file |
| 18 | Verify input files | Check `plan.json` / `world.json` exist |
| 20 | Launch vLLM server | `subprocess.Popen` on port 8000 |
| 22 | Wait for vLLM | Health-check loop |
| 24 | Phase II pipeline | Steps 1–3: build `plan.json`, `world.json`, `final_story.md` |
| 26 | Interactive CLI test | Optional in-notebook game session |
| 28 | Start web server | `uvicorn api_server:app` on port 7860 |
| 30 | Display URL | `eval_js(google.colab.kernel.proxyPort(7860))` |

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_ENDPOINT` | `http://localhost:8000/v1` | vLLM server URL |
| `LLM_MODEL` | `Qwen/Qwen3-8B` | Model served by vLLM |
| `LLM_API_KEY` | `EMPTY` | Required by OpenAI client; ignored by vLLM |
| `ANTHROPIC_API_KEY` | — | Required only for the Claude API backend variant |

---

## Log Files

| File | Format | Contents |
|---|---|---|
| `logs/drama.jsonl` | One JSON object per line | Every DM decision: classification, accommodation, goal reachability |
| `logs/turns.jsonl` | One JSON object per line | Per-turn record: raw input, parsed action, effects, narration |
| `logs/web/uvicorn.log` | Plain text | FastAPI server stdout/stderr |
| `logs/vllm_server.log` | Plain text | vLLM model loading and request log |
