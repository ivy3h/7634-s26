# Phase II — Interactive Detective Mystery (Template 2: Intervention & Accommodation)

An interactive text game built on top of the Phase I mystery generator. The player is the detective. Every command is classified as **constituent**, **consistent**, or **exceptional** by a drama manager that repairs the plan when the player breaks it.

## Layout

```
phase2/
├── llm_client.py              single point of LLM access (OpenAI-compatible, local vLLM)
├── phase1_story_generator.py  ported Phase I — no Anthropic dependency
├── plan_types.py              shared dataclasses (Event, Condition, Effect, CausalLink, Plan)
├── story_to_plan.py           Phase I JSON → partially ordered plan + causal links
├── world_builder.py           location graph + initial world state
├── action_interpreter.py      LLM parser: free-form text → structured action dict
├── drama_manager.py           classify + commonsense threat check + accommodate
├── game_engine.py             world state, I/O loop, narration
├── main.py                    build / play / replay entry point
├── scripts/
│   ├── setup_env.sh           one-shot conda env creation
│   ├── launch_vllm_server.sbatch   start vLLM on overcap / A40
│   ├── wait_for_server.sh     block until /v1/models returns 200
│   ├── run_game.sbatch        batch game client (reads logs/vllm_endpoint.txt)
│   └── test_llm.py            smoke test
├── tests/test_plan_types.py   offline unit tests
├── data/                      case_file.json, plot_points.json, plan.json, world.json
├── logs/                      vllm_endpoint.txt, drama.jsonl, turns.jsonl, *.log
└── transcripts/               example playthroughs
```

## Running on Colab (quickstart)

Three Colab entry points, all under `colab/`:

| Notebook | Backend | Needs GPU? | Needs git clone? | When to use |
|---|---|---|---|---|
| [`phase2_colab.ipynb`](colab/phase2_colab.ipynb) | local vLLM + Qwen | yes | yes | default cluster-like path, tracks future commits |
| [`phase2_standalone.ipynb`](colab/phase2_standalone.ipynb) | local vLLM + Qwen | yes | **no** | every module embedded; use when the repo is unreachable |
| [`phase2_claude_standalone.ipynb`](colab/phase2_claude_standalone.ipynb) | **Anthropic Claude API** | **no** | **no** | fastest end-to-end demo; needs an `ANTHROPIC_API_KEY` in Colab Secrets |

Pick one, set **Runtime → Change runtime type → GPU**, and run top-to-bottom. Both boot a local vLLM OpenAI-compatible server, generate one fixed story, **assemble a novel-style `final_story.md`** (no interaction needed), and replay both a successful playthrough and an exception playthrough — all inside the single runtime.

### Expected runtime and cost (Claude API version)

| Metric | Value |
|---|---|
| Total runtime | ~12 minutes |
| Estimated cost | ~$0.50 per full run |

### Successful Case

A successful output example is available in [`data/final_story.md`](data/final_story.md). The plot-point breakdown that drives it is in [`data/plot_points.json`](data/plot_points.json).

## Running on the Skynet cluster

### One-time setup

```bash
bash scripts/setup_env.sh
# creates /coc/pskynet6/jhe478/conda_envs/phase2 with vllm + openai + torch
```

### Cluster workflow

#### 1 — Start the vLLM server on a GPU node

```bash
sbatch scripts/launch_vllm_server.sbatch
# endpoint is written to logs/vllm_endpoint.txt
# tail logs/vllm_server_<jobid>.log for readiness
bash scripts/wait_for_server.sh
```

Override model / port / memory via env vars:

```bash
sbatch --export=ALL,PHASE2_MODEL=Qwen/Qwen3-8B,PHASE2_GPU_MEM=0.85 scripts/launch_vllm_server.sbatch
```

#### 2 — Smoke-test

```bash
export LLM_ENDPOINT="$(cat logs/vllm_endpoint.txt)"
export LLM_MODEL="Qwen/Qwen3-8B"
python scripts/test_llm.py
```

#### 3 — Build a fixed story + plan + world

```bash
python main.py build --data-dir data
```

This saves:

| File | Content |
|------|---------|
| `data/case_file.json`    | Phase I case file |
| `data/complexities.json` | Phase I cover narrative |
| `data/plot_points.json`  | Phase I detective-side plot points |
| `data/story_bible.json`  | Pinned names + constants |
| `data/plan.json`         | Partially-ordered plan — events, causal links, goal, initial state |
| `data/world.json`        | Location graph with adjacency, characters, evidence |

Re-run the engine against the **same** fixed story without regenerating:

```bash
python main.py build --data-dir data --skip-story   # rebuilds plan + world only
```

#### 4 — Play (interactive)

```bash
python main.py play --data-dir data --log-dir logs
```

Or run a scripted transcript (used for video demos / CI):

```bash
python main.py replay transcripts/success_run.txt --data-dir data
python main.py replay transcripts/exception_run.txt --data-dir data
```

All turns are logged in `logs/turns.jsonl`; drama-manager decisions (classification, threat checks, accommodation) are in `logs/drama.jsonl`.

## Plan data format

`data/plan.json` matches `plan_types.Plan.to_dict()`:

```json
{
  "events": {
    "E00": {
      "id": "E00",
      "actor": "detective",
      "verb": "examine",
      "args": ["evidence.E01"],
      "location": "location.gallery_main_hall",
      "preconditions": [{"subject":"detective","attr":"location","op":"==","value":"location.gallery_main_hall"}],
      "effects": [
        {"subject":"evidence.E01","attr":"discovered","op":"set","value":true},
        {"subject":"detective","attr":"knowledge","op":"add","value":"saw_pen"}
      ],
      "reveals": ["evidence.E01"],
      "description": "...",
      "narrative": "...",
      "source_plot_idx": 0
    }
  },
  "order": [["E00","E01"], ...],
  "causal_links": [{"producer":"E00","consumer":"E02","condition":{...}}],
  "initial_state": {"detective": {"location":"...","knowledge":[],"inventory":[]}, ...},
  "goal": [{"subject":"detective","attr":"knowledge","op":"contains","value":"identified:character.victoria_harrington"}]
}
```

## Module interfaces

- `llm_client.chat(messages, model?, max_tokens, temperature, retries, **kwargs) -> str`
- `llm_client.chat_json(prompt, system?, max_tokens, temperature, ...) -> dict|list`
- `story_to_plan.build_plan(case_file, plot_points, out_path?) -> Plan`
- `world_builder.build_world(plan, era="1920s London") -> World`
- `action_interpreter.interpret_action(raw_input, world_summary) -> dict`
- `drama_manager.DramaManager(plan).classify(parsed_action, state) -> dict`
- `drama_manager.DramaManager(plan).accommodate(parsed, classification, state, world_locations, characters) -> dict`
- `game_engine.GameEngine(plan, world).run(get_input=input, echo=print) -> str`

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `LLM_ENDPOINT` | `http://localhost:8000/v1` | Where `openai` client points |
| `LLM_MODEL`    | `Qwen/Qwen3-8B`          | Model name passed to vLLM |
| `LLM_API_KEY`  | `EMPTY`                    | vLLM ignores but `openai` client requires a non-empty value |
| `PHASE2_ENV`   | `/coc/pskynet6/jhe478/conda_envs/phase2` | Conda env for SLURM jobs |
| `PHASE2_GPU_MEM` | `0.88`                   | vLLM gpu_memory_utilization |
| `PHASE2_MAX_MODEL_LEN` | `8192`             | Context length cap |

See `DESIGN.md` for how the drama manager recognizes exceptions when user actions introduce state variables not present in the original plan.