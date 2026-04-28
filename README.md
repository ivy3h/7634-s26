# Phase II — Interactive Detective Mystery

A narrative AI game built on Template 2 (Intervention & Accommodation). The player is a detective investigating a murder; every command runs through a drama manager that classifies it as constituent, consistent, or exceptional, then uses a local LLM to narrate the outcome.

---

## Running the game in Google Colab

Open **`colab/phase2_web_standalone.ipynb`** in Google Colab.

### Prerequisites

**GPU runtime required.**
Runtime → Change runtime type → GPU (T4, L4, or A100). The notebook picks the right model size automatically.

**Phase I output files** from your Phase I story generator. You need either:

| Shortcut | Files to upload |
|---|---|
| Full pipeline | `plot_points.json` + `case_file.json` |
| Skip plan generation | `plan.json` + `case_file.json` |
| Skip everything | `plan.json` + `world.json` + `case_file.json` |

### Step-by-step

**1. Upload Phase I files**
In the Colab left sidebar, click the folder icon → navigate to `/content/data/` (create it if needed) → upload your files there before running any cell.

**2. Run all cells in order**
The notebook has 8 numbered sections:

| Section | What it does |
|---|---|
| 1. Install | Installs vLLM, FastAPI, uvicorn (~3–5 min first run) |
| 2. Pick model | Detects GPU memory, sets model (`Qwen2.5-3B/7B` or `Qwen3-8B`) |
| 3. Write modules | Writes all Python source files to `/content/` |
| 4. Verify input | Checks which files are present, reports what will be generated |
| 5. Launch vLLM | Starts the local LLM server (first run downloads weights, ~3–8 min) |
| 6. Wait for vLLM | Polls until the server is ready; prints the log if it crashes |
| 7. Run pipeline | Generates `plan.json` and/or `world.json` from your Phase I files if not already present |
| 8. Launch web server | Starts the FastAPI game server and prints a public URL via ngrok |

**3. Open the game URL**
After section 8 completes, a public `https://` URL is printed. Open it in any browser. The game runs fully in the browser; every command is sent to the Colab backend for LLM narration.

### Tips

- If vLLM crashes in section 6, the cell prints the last 40 lines of `logs/vllm_server.log` to help diagnose the issue (usually an out-of-memory error — switch to a larger GPU).
- The session lasts up to 12 hours on a free Colab account. Progress is not persisted across sessions.
- To restart with a fresh mystery, click **Reset** in the game UI or re-run section 7 onward.
- The `📖 Read story` link in the game footer opens the Phase I novel version inline (no upload needed — the story is embedded at build time from `data/final_story.md`).
