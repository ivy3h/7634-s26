"""Build a free-text interactive detective game as a single HTML file.

This is NOT a story reader. It is a playable text adventure. The user types
commands like `examine body`, `go to forensic lab`, `question gregory`,
`accuse vivienne`. A minimal in-browser game engine (pure JavaScript)
interprets the input against the same plan + world we generated for the
Phase II backend.

The engine intentionally implements a *simplified* version of
drama_manager's three classifications so the game is playable without an
LLM:
  - constituent -> match against a plan event at the current location
  - consistent  -> acknowledged but no plan progress
  - exceptional -> hand-picked destructive verbs are refused with noir
                   flavour text, simulating the real DM's repair step

Usage:
    python web/build_game.py
    # -> writes web/game.html
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "game.html"


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
STOPWORDS = {"the", "a", "an", "of", "in", "at", "to", "on", "and", "or", "for",
             "with", "by", "from", "into", "near", "behind"}


def _tokens_from(name: str) -> list[str]:
    return [t.lower() for t in re.split(r"\W+", name) if t and t.lower() not in STOPWORDS and len(t) > 1]


def _id_aliases(entity_id: str, pretty_name: str = "") -> list[str]:
    seen: set[str] = set()
    for part in entity_id.split(".", 1)[-1].split("_"):
        if part and part not in STOPWORDS:
            seen.add(part.lower())
    for tok in _tokens_from(pretty_name):
        seen.add(tok)
    if pretty_name:
        seen.add(pretty_name.lower())
    return sorted(seen)


# ---------------------------------------------------------------------------
# Narrative shortener for the in-game event text
# ---------------------------------------------------------------------------
def _short_narrative(text: str, max_chars: int = 900) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    # truncate at a sentence boundary
    cut = text[:max_chars]
    last = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if last > 400:
        return cut[: last + 1].strip()
    return cut.rstrip() + "…"


# ---------------------------------------------------------------------------
# Data packing
# ---------------------------------------------------------------------------
def build_game_data() -> dict:
    plan = json.loads((ROOT / "data/plan.json").read_text(encoding="utf-8"))
    world = json.loads((ROOT / "data/world.json").read_text(encoding="utf-8"))
    case = json.loads((ROOT / "data/case_file.json").read_text(encoding="utf-8"))

    events: dict[str, dict] = {}
    for eid, ev in plan["events"].items():
        events[eid] = {
            "id": eid,
            "verb": ev.get("verb", "act"),
            "args": ev.get("args", []),
            "location": ev.get("location", ""),
            "reveals": ev.get("reveals", []),
            "description": ev.get("description", ""),
            "narrative": _short_narrative(ev.get("narrative", "")),
        }

    characters: dict[str, dict] = {}
    for subj, fields in plan["initial_state"].items():
        if subj.startswith("character."):
            name = fields.get("name", subj.split(".", 1)[-1].replace("_", " ").title())
            characters[subj] = {
                "id": subj,
                "name": name,
                "role": fields.get("role", ""),
                "alive": fields.get("alive", True),
                "aliases": _id_aliases(subj, name),
            }

    # Ensure the real criminal is present as a selectable suspect even if
    # plan.initial_state didn't surface them as a distinct character.
    crim_name = case.get("criminal", {}).get("name", "")
    crim_id = "character." + re.sub(r"\W+", "_", crim_name.lower()).strip("_") if crim_name else ""
    if crim_id and crim_id not in characters:
        characters[crim_id] = {
            "id": crim_id,
            "name": crim_name,
            "role": "associate",   # neutral, not 'criminal' -- don't spoil
            "alive": True,
            "aliases": _id_aliases(crim_id, crim_name),
        }

    # Build per-character blurbs from case_file. Suspects are red herrings,
    # so their motives + alibis are safe to share. Conspirators' "role" is a
    # plot spoiler (it describes their complicity), so we expose only the
    # claimed alibi. The real criminal gets a neutral 'access/opportunity'
    # line that doesn't reveal their guilt.
    blurbs: dict[str, str] = {}
    for s in case.get("suspects", []):
        cid = "character." + re.sub(r"\W+", "_", s["name"].lower()).strip("_")
        parts = []
        if s.get("motive"):
            parts.append("Apparent motive — " + s["motive"])
        if s.get("alibi"):
            parts.append("Claimed alibi — " + s["alibi"])
        if parts:
            blurbs[cid] = "  ".join(parts)
    for s in case.get("conspirators", []):
        cid = "character." + re.sub(r"\W+", "_", s["name"].lower()).strip("_")
        if s.get("alibi"):
            blurbs[cid] = "Claimed alibi — " + s["alibi"]
    if crim_id:
        crim = case.get("criminal", {})
        opportunity = crim.get("opportunity", "")
        blurbs[crim_id] = ("Access — " + opportunity) if opportunity else "A close associate of the victim."

    for cid, ch in characters.items():
        ch["blurb"] = blurbs.get(cid, "")

    evidence_entities: dict[str, dict] = {}
    for subj, fields in plan["initial_state"].items():
        if subj.startswith("evidence."):
            desc = fields.get("description", subj)
            evidence_entities[subj] = {
                "id": subj,
                "description": desc,
                "aliases": _id_aliases(subj, desc),
            }

    locations: dict[str, dict] = {}
    for lid, loc in world["locations"].items():
        locations[lid] = {
            "id": lid,
            "name": loc["name"],
            "description": loc["description"],
            "adjacent": list(loc["adjacent"]),
            "characters": list(loc["characters"]),
            "evidence": list(loc["evidence"]),
            "aliases": _id_aliases(lid, loc["name"]),
        }

    # The real-criminal id used for the accusation win condition
    crim_name = case.get("criminal", {}).get("name", "")
    crim_id = "character." + re.sub(r"\W+", "_", crim_name.lower()).strip("_")

    return {
        "starting_location": world["starting_location"],
        "real_criminal_id": crim_id,
        "real_criminal_name": crim_name,
        "victim_name": case.get("victim", {}).get("name", ""),
        "detective_name": case.get("detective", {}).get("name", ""),
        "events": events,
        "characters": characters,
        "evidence": evidence_entities,
        "locations": locations,
        "goal_events_needed": 10,
    }


# ---------------------------------------------------------------------------
# HTML + JS
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Play: The Hartley Affair</title>
<style>
:root {
  --bg: #14100d;
  --panel: #1d1812;
  --panel2: #251f17;
  --paper: #f4ecdc;
  --ink: #1c1a17;
  --accent: #b38a4a;
  --accent-soft: #c9a878;
  --good: #4a9b5d;
  --warn: #c7883c;
  --danger: #c26060;
  --muted: #8a7b66;
  --rule: #2c251e;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; background: var(--bg); color: var(--paper);
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 16px; line-height: 1.55;
  height: 100vh;
  overflow: hidden;       /* lock the page; inner regions scroll themselves */
}
body { display: flex; flex-direction: column; }
header.masthead {
  padding: 18px 28px 14px;
  border-bottom: 1px solid var(--rule);
  background: linear-gradient(180deg, #1a140f 0%, var(--bg) 100%);
  display: flex; justify-content: space-between; align-items: baseline;
  flex-wrap: wrap; gap: 10px;
}
header.masthead h1 {
  margin: 0;
  font-family: "Playfair Display", "Didot", Georgia, serif;
  font-size: 26px;
  color: var(--accent);
  letter-spacing: 0.02em;
}
header.masthead .subtitle {
  margin: 0;
  color: var(--muted);
  font-style: italic;
  font-size: 13px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
header.masthead a { color: var(--accent-soft); }

.layout {
  display: grid;
  grid-template-columns: 280px 1fr 280px;
  flex: 1 1 auto; min-height: 0;   /* let the grid fill remaining viewport */
  overflow: hidden;                 /* children handle their own scroll */
}
aside {
  background: var(--panel);
  padding: 18px 18px 24px;
  overflow-y: auto;
  min-height: 0;
  border-color: var(--rule);
}
aside.left  { border-right: 1px solid var(--rule); }
aside.right { border-left:  1px solid var(--rule); }
aside h3 {
  font-family: "Playfair Display", Georgia, serif;
  font-size: 13px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-soft);
  margin: 0 0 8px;
}
aside h3:not(:first-child) { margin-top: 18px; }
aside ul { list-style: none; margin: 0; padding: 0; font-size: 13.5px; }
aside li { padding: 3px 0; color: var(--paper); }
aside .muted { color: var(--muted); font-style: italic; font-size: 13px; }
aside .exit-btn {
  background: transparent;
  color: var(--accent-soft);
  border: 1px dashed var(--rule);
  padding: 4px 8px;
  margin: 2px 0;
  cursor: pointer;
  font-family: inherit;
  font-size: 13px;
  display: block;
  width: 100%;
  text-align: left;
  border-radius: 2px;
}
aside .exit-btn:hover { background: rgba(179,138,74,0.12); border-style: solid; }

.center {
  display: flex; flex-direction: column;
  min-height: 0;
  background: var(--bg);
  overflow: hidden;
}
aside.left .scene-name {
  font-family: "Playfair Display", "Didot", Georgia, serif;
  font-size: 17px;
  color: var(--accent);
  letter-spacing: 0.02em;
  margin: 2px 0 4px;
  line-height: 1.3;
}
aside.left .scene-desc {
  font-size: 12.5px;
  color: var(--accent-soft);
  font-style: italic;
  line-height: 1.45;
  margin-bottom: 2px;
}
.log {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  padding: 22px 34px 20px;
  scroll-behavior: smooth;
}
.log .entry {
  margin-bottom: 16px;
  max-width: 780px;
}
.log .entry.system {
  color: var(--muted);
  font-style: italic;
  font-size: 13px;
  border-left: 2px solid var(--rule);
  padding-left: 10px;
}
.log .entry.user {
  color: var(--accent-soft);
  font-family: "Courier Prime", "Courier New", monospace;
  font-size: 14px;
  font-style: italic;
}
.log .entry.user::before {
  content: "> ";
  color: var(--accent);
  font-style: normal;
}
.log .entry.narration {
  background: #1a150f;
  padding: 14px 18px;
  border-left: 3px solid var(--accent);
  border-radius: 0 3px 3px 0;
  white-space: pre-wrap;
}
.log .entry.narration h4 {
  margin: 0 0 8px;
  font-family: "Playfair Display", Georgia, serif;
  font-size: 15px;
  color: var(--accent);
  letter-spacing: 0.04em;
}
.log .entry.outcome {
  color: var(--warn);
  font-size: 14px;
}
.log .entry.exception {
  color: var(--danger);
  font-size: 14px;
  border-left: 3px solid var(--danger);
  padding-left: 10px;
  background: rgba(194,96,96,0.06);
}
.log .entry.victory {
  color: var(--good);
  background: rgba(74,155,93,0.1);
  border: 1px solid var(--good);
  padding: 14px 18px;
  border-radius: 3px;
  font-weight: 600;
}

.hidden { display: none !important; }

.hints-panel {
  border-top: 1px solid var(--rule);
  background: var(--panel2);
  padding: 10px 20px 0;
}
.hints-toggle {
  background: transparent;
  color: var(--muted);
  border: 1px dashed var(--rule);
  padding: 6px 12px;
  font-family: inherit;
  font-size: 12px;
  letter-spacing: 0.04em;
  cursor: pointer;
  border-radius: 2px;
}
.hints-toggle:hover {
  color: var(--accent-soft);
  border-color: var(--accent-soft);
}
.hints-toggle.open {
  color: var(--accent);
  border-color: var(--accent);
  border-style: solid;
}
.hints-list {
  display: flex; flex-wrap: wrap; gap: 6px;
  margin: 8px 0 2px;
}
.hints-list .chip {
  background: var(--panel);
  color: var(--paper);
  border: 1px solid var(--rule);
  padding: 6px 12px;
  font-family: "Courier Prime", "Courier New", monospace;
  font-size: 13px;
  cursor: pointer;
  border-radius: 2px;
  transition: border-color .12s, background .12s;
}
.hints-list .chip:hover {
  border-color: var(--accent);
  background: #241d16;
}
.hints-list .chip:active { background: #2c2219; }
.hints-list .chip-label {
  font-family: "Courier Prime", "Courier New", monospace;
  font-size: 13px;
  color: var(--accent-soft);
}
.hints-list .chip .arrow { color: var(--muted); margin-right: 6px; }
.hints-note {
  font-size: 11.5px;
  color: var(--muted);
  margin: 6px 0 2px;
  font-style: italic;
}

.input-bar {
  display: flex; gap: 8px;
  padding: 12px 20px 16px;
  border-top: 1px solid var(--rule);
  background: var(--panel2);
}
.input-bar input {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid var(--rule);
  background: var(--bg);
  color: var(--paper);
  font-family: "Courier Prime", "Courier New", monospace;
  font-size: 15px;
  border-radius: 3px;
}
.input-bar input:focus { outline: 1px solid var(--accent); }
.input-bar button {
  padding: 10px 18px;
  background: var(--accent);
  color: var(--ink);
  border: none;
  font-family: inherit;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  letter-spacing: 0.05em;
  border-radius: 3px;
}
.input-bar button:disabled { opacity: 0.4; cursor: not-allowed; }
.input-bar .hint {
  padding: 0 20px 12px;
  font-size: 11.5px;
  color: var(--muted);
  letter-spacing: 0.04em;
}

.suspect-locked {
  color: var(--muted); font-style: italic; font-size: 13px; padding: 3px 0;
}
.suspect-row { padding: 2px 0; }
.suspect-row .suspect-name {
  background: transparent;
  color: var(--paper);
  border: none;
  padding: 2px 0;
  cursor: pointer;
  font-family: inherit;
  font-size: 13.5px;
  text-align: left;
  display: block;
  width: 100%;
}
.suspect-row .suspect-name:hover { color: var(--accent-soft); }
.suspect-row .suspect-name .chev {
  display: inline-block; width: 14px; color: var(--accent); font-size: 11px;
}
.suspect-row .blurb {
  margin: 4px 0 8px 14px;
  padding: 8px 10px;
  font-size: 12.5px;
  font-style: italic;
  color: var(--paper);
  border-left: 2px solid var(--accent);
  background: rgba(179,138,74,0.06);
  line-height: 1.5;
}

.knowledge li { position: relative; padding-left: 14px; }
.knowledge li::before { content: "●"; position: absolute; left: 0; color: var(--accent); font-size: 12px; top: 4px; }
.progress-bar {
  height: 4px; background: var(--rule); border-radius: 2px; overflow: hidden;
  margin-top: 8px;
}
.progress-bar > div { height: 100%; background: var(--accent); transition: width .3s; }

.footer-links {
  padding: 14px 28px; border-top: 1px solid var(--rule);
  background: var(--panel);
  text-align: center; font-size: 12px; color: var(--muted);
}
.footer-links a { color: var(--accent-soft); text-decoration: none; margin: 0 8px; }

@media (max-width: 980px) {
  .layout { grid-template-columns: 1fr; }
  aside.left, aside.right { border: none; border-bottom: 1px solid var(--rule); max-height: 200px; }
  .log { padding: 16px 14px; }
}
</style>
</head>
<body>
<header class="masthead">
  <div>
    <h1>The Hartley Affair &mdash; Play</h1>
    <p class="subtitle">You are Inspector Rothwell. Find the killer.</p>
  </div>
  <div style="font-size:12px; color: var(--muted);">
    <a href="./web/story_interactive.html">← Read the novel version</a>
  </div>
</header>

<div class="layout">
  <aside class="left">
    <h3>Current scene</h3>
    <div id="location-name" class="scene-name"></div>
    <div id="location-desc" class="scene-desc"></div>
    <h3>Exits</h3>
    <div id="exits"></div>
    <h3>Currently in room</h3>
    <ul id="characters-here"></ul>
    <h3>Notable here</h3>
    <ul id="evidence-here"></ul>
  </aside>

  <div class="center">
    <div id="log" class="log"></div>
    <div class="hints-panel">
      <button id="hints-toggle" class="hints-toggle" type="button">💡 Stuck? Show hints</button>
      <div id="hints-content" class="hidden">
        <div id="hints-list" class="hints-list"></div>
        <div class="hints-note">Click a hint to pre-fill the input — you still press Enter (and can edit) to submit.</div>
      </div>
    </div>
    <form class="input-bar" id="input-form">
      <input id="cmd" autocomplete="off" spellcheck="false"
             placeholder="Type a command (≤ 8 words). Try: examine body, go to london streets, question sutherland, accuse vivienne"
             maxlength="100" autofocus>
      <button type="submit">Enter</button>
      <button type="button" id="reset-btn" title="Start over" style="background: var(--panel); color: var(--paper);">Reset</button>
    </form>
  </div>

  <aside class="right">
    <h3>Detective's notebook</h3>
    <ul id="knowledge" class="knowledge"></ul>
    <h3>Suspects</h3>
    <ul id="suspects"></ul>
    <h3>Progress</h3>
    <div class="muted" style="font-size:12px;">
      <span id="events-triggered">0</span> / <span id="events-needed">0</span> plot events explored.
      You can attempt an accusation any time, but your case will only hold if
      you've uncovered enough evidence.
    </div>
    <div class="progress-bar"><div id="pbar" style="width:0%"></div></div>
  </aside>
</div>

<footer class="footer-links">
  <a href="./web/story_interactive.html">Novel version</a>
  <a href="https://github.com/ivy3h/7634-s26">Source</a>
  &middot; Template 2: Intervention &amp; Accommodation &middot; CS 7634
</footer>

<script>
const DATA = __GAME_DATA__;
const STORAGE_KEY = "hartley_affair_game_v1";

// -------------------- state --------------------
let state;
function freshState() {
  return {
    location: DATA.starting_location,
    lastLocation: null,          // so hints don't suggest retreating
    knowledge: [],
    inventory: [],
    executedEvents: [],
    evidenceFlags: {}, // id -> {discovered, analyzed, destroyed}
    charactersInterviewed: [],
    encounteredCharacters: [],   // seen in a visited location or referenced by an executed event
    turns: 0,
    gameOver: false,
  };
}
function encounter(cid) {
  if (!cid) return;
  if (!state.encounteredCharacters.includes(cid)) {
    state.encounteredCharacters.push(cid);
  }
}
function encounterAll(ids) { (ids || []).forEach(encounter); }
function loadState() {
  try {
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (s && s.location) return s;
  } catch (e) {}
  return freshState();
}
function saveState() { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }

state = loadState();

// -------------------- rendering --------------------
const logEl = document.getElementById("log");
function addLog(text, cls = "outcome", title = null) {
  const div = document.createElement("div");
  div.className = "entry " + cls;
  if (title) {
    const h = document.createElement("h4");
    h.textContent = title;
    div.appendChild(h);
  }
  const body = document.createElement("div");
  body.textContent = text;
  div.appendChild(body);
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
}

// -------------------- hint engine (non-prescriptive) --------------------
// The player always types commands freely in the input box. The hint
// panel is opt-in: it lists 3-5 commands the Inspector *could* try next,
// given the current state. Clicking a hint only pre-fills the input --
// it never auto-submits. This preserves open-ended input per the spec
// ("interactions should not be from a menu of options").
function firstAlias(entity) {
  if (!entity || !entity.aliases || !entity.aliases.length) return "";
  const clean = entity.aliases.filter(a => !a.includes(".") && a.length > 2);
  clean.sort((a, b) => b.length - a.length);
  return clean[0] || entity.aliases[0];
}
function locAlias(loc) { return loc ? loc.name.toLowerCase() : ""; }
function truncLabel(s, n = 34) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }

function buildHintCommands() {
  const hints = [];
  const loc = currentLoc();

  // A. Remaining plan events at this location.
  if (loc) {
    const localEvents = Object.values(DATA.events).filter(
      ev => ev.location === state.location && !state.executedEvents.includes(ev.id)
    );
    for (const ev of localEvents) {
      const firstArg = (ev.args || []).find(a => !String(a).startsWith("location."));
      let tgt = firstArg ? String(firstArg) : "";
      let argIsCharacter = false;
      if (tgt.startsWith("character.")) {
        argIsCharacter = true;
        const c = DATA.characters[tgt];
        tgt = c ? firstAlias(c) : tgt.split(".", 2)[1].replace(/_/g, " ");
      } else if (tgt.startsWith("evidence.")) {
        const e = DATA.evidence[tgt];
        tgt = e ? firstAlias(e) : tgt.split(".", 2)[1];
      } else if (tgt.startsWith("object.")) {
        tgt = tgt.split(".", 2)[1].replace(/_/g, " ");
      } else {
        // Free-text arg: snake_case -> human phrase for both display + command.
        tgt = String(tgt).replace(/_/g, " ").trim();
        tgt = tgt.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).slice(0, 4).join(" ");
      }
      // Normalize a few plan verbs so the chip routes to a working handler.
      let verbNorm = ev.verb;
      if (verbNorm === "investigate") verbNorm = "search";
      if (verbNorm === "visit" && argIsCharacter) verbNorm = "question";
      hints.push(`${verbNorm} ${tgt}`.trim());
      if (hints.length >= 3) break;
    }
  }

  // B. Characters still here that haven't been interviewed AND still have
  //    a remaining social event tied to them at this location. Avoid
  //    suggesting "dead" hints that would just print the generic polite
  //    fall-through.
  if (loc) {
    const socialFamily = new Set(["interview","question","consult","confront","visit"]);
    for (const cid of loc.characters) {
      const c = DATA.characters[cid];
      if (!c || !c.alive) continue;
      if (state.charactersInterviewed.includes(cid)) continue;
      const alias = firstAlias(c);
      const already = hints.some(h => h.toLowerCase().includes(alias));
      if (already || hints.length >= 4) continue;
      const hasPendingSocialEvent = Object.values(DATA.events).some(ev =>
        ev.location === state.location
        && !state.executedEvents.includes(ev.id)
        && socialFamily.has(ev.verb)
        && (ev.args || []).some(a => String(a).toLowerCase().includes(alias))
      );
      if (hasPendingSocialEvent) {
        hints.push("question " + alias);
      }
    }
  }

  // C. Exit suggestions, ranked by how much investigation work is waiting
  //    for the detective there. Skip the location we just came from so
  //    hints don't send you in loops.
  function eventsWaitingAt(locId) {
    return Object.values(DATA.events).filter(
      ev => ev.location === locId && !state.executedEvents.includes(ev.id)
    ).length;
  }
  if (loc && loc.adjacent.length && hints.length < 4) {
    const ranked = loc.adjacent
      .filter(id => id !== state.lastLocation)
      .map(id => ({id, score: eventsWaitingAt(id), loc: DATA.locations[id]}))
      .filter(x => x.loc)
      .sort((a, b) => b.score - a.score);
    // Offer at most two exits, prefer ones with work to do.
    for (const x of ranked.slice(0, 2)) {
      if (hints.length >= 4) break;
      hints.push("go to " + locAlias(x.loc));
    }
    // If we filtered out everything (only neighbor was lastLocation), allow it back.
    if (hints.length === 0) {
      const fallback = loc.adjacent.map(id => DATA.locations[id]).filter(Boolean)[0];
      if (fallback) hints.push("go to " + locAlias(fallback));
    }
  }

  // D. Fallback baseline.
  if (hints.length === 0) {
    hints.push("look");
  }

  // E. Endgame nudge.
  if (state.executedEvents.length >= DATA.goal_events_needed && !state.gameOver) {
    hints.push("accuse " + firstAlias(DATA.characters[DATA.real_criminal_id] || {name: DATA.real_criminal_name, aliases: [DATA.real_criminal_name.toLowerCase()]}));
  }

  // Keep it short.
  return Array.from(new Set(hints)).slice(0, 5);
}

function renderHints() {
  const list = document.getElementById("hints-list");
  list.innerHTML = "";
  if (state.gameOver) {
    const m = document.createElement("div");
    m.className = "hints-note";
    m.textContent = "Case closed. Press Reset below to start a new investigation.";
    list.appendChild(m);
    return;
  }
  const hints = buildHintCommands();
  hints.forEach(cmd => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.innerHTML = `<span class="arrow">&gt;</span>${cmd}`;
    chip.title = "Copy to input (Enter to submit)";
    chip.addEventListener("click", () => prefillInput(cmd));
    list.appendChild(chip);
  });
}

function prefillInput(cmd) {
  const el = document.getElementById("cmd");
  el.value = cmd;
  el.focus();
  // Place cursor at end.
  el.setSelectionRange(cmd.length, cmd.length);
}

function renderSidebar() {
  const loc = DATA.locations[state.location];
  document.getElementById("location-name").textContent = loc ? loc.name : state.location;
  document.getElementById("location-desc").textContent = loc ? loc.description : "";

  const exitsEl = document.getElementById("exits");
  exitsEl.innerHTML = "";
  if (loc && loc.adjacent.length) {
    loc.adjacent.forEach(adj => {
      const btn = document.createElement("button");
      btn.className = "exit-btn";
      btn.textContent = "→ " + (DATA.locations[adj] ? DATA.locations[adj].name : adj);
      btn.addEventListener("click", () => runCommand("go to " + (DATA.locations[adj] ? DATA.locations[adj].name : adj)));
      exitsEl.appendChild(btn);
    });
  } else {
    exitsEl.innerHTML = '<li class="muted">(no obvious exits)</li>';
  }

  const charsEl = document.getElementById("characters-here");
  charsEl.innerHTML = "";
  const here = loc ? loc.characters : [];
  if (!here.length) {
    charsEl.innerHTML = '<li class="muted">(no-one of interest)</li>';
  } else {
    here.forEach(cid => {
      const c = DATA.characters[cid];
      if (!c || !c.alive) return;
      const li = document.createElement("li");
      li.textContent = c.name + " (" + (c.role || "-") + ")";
      charsEl.appendChild(li);
    });
  }

  const evEl = document.getElementById("evidence-here");
  evEl.innerHTML = "";
  const evHere = loc ? loc.evidence : [];
  if (!evHere.length) {
    evEl.innerHTML = '<li class="muted">(nothing catches the eye)</li>';
  } else {
    evHere.forEach(eid => {
      const e = DATA.evidence[eid];
      if (!e) return;
      const li = document.createElement("li");
      const flag = state.evidenceFlags[eid] || {};
      if (flag.discovered) {
        li.textContent = "☑ " + truncate(e.description, 40);
        if (flag.destroyed) li.style.textDecoration = "line-through";
      } else {
        // Don't spoil what it is — just hint there's something to investigate.
        li.textContent = "? something catches the eye";
        li.className = "muted";
        li.style.fontStyle = "italic";
      }
      evEl.appendChild(li);
    });
  }

  const kEl = document.getElementById("knowledge");
  kEl.innerHTML = "";
  if (!state.knowledge.length) {
    kEl.innerHTML = '<li class="muted">(nothing yet; investigate)</li>';
  } else {
    state.knowledge.slice().reverse().forEach(k => {
      const li = document.createElement("li");
      li.textContent = k;
      kEl.appendChild(li);
    });
  }

  renderSuspectList();

  document.getElementById("events-triggered").textContent = state.executedEvents.length;
  document.getElementById("events-needed").textContent = DATA.goal_events_needed;
  const pct = Math.min(100, Math.round(100 * state.executedEvents.length / DATA.goal_events_needed));
  document.getElementById("pbar").style.width = pct + "%";
}

function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }

function renderSuspectList() {
  const susEl = document.getElementById("suspects");
  susEl.innerHTML = "";
  const candidates = Object.values(DATA.characters)
    .filter(c => c.alive && (c.role === "suspect" || c.role === "conspirator" || c.role === "associate"));
  const anyEncountered = candidates.some(c => state.encounteredCharacters.includes(c.id));
  if (!anyEncountered) {
    const li = document.createElement("li");
    li.className = "muted";
    li.style.fontStyle = "italic";
    li.textContent = "(no suspects known yet — investigate locations to uncover them)";
    susEl.appendChild(li);
    return;
  }
  candidates.forEach(c => {
    const li = document.createElement("li");
    if (!state.encounteredCharacters.includes(c.id)) {
      li.className = "suspect-locked";
      li.textContent = "? unidentified associate";
      susEl.appendChild(li);
      return;
    }
    li.className = "suspect-row";
    const btn = document.createElement("button");
    btn.className = "suspect-name";
    btn.type = "button";
    btn.innerHTML = '<span class="chev">▸</span>' + c.name;
    btn.addEventListener("click", () => toggleSuspectBlurb(li, btn, c));
    li.appendChild(btn);
    susEl.appendChild(li);
  });
}

function toggleSuspectBlurb(container, btn, c) {
  const existing = container.querySelector(".blurb");
  if (existing) {
    existing.remove();
    btn.querySelector(".chev").textContent = "▸";
    return;
  }
  const d = document.createElement("div");
  d.className = "blurb";
  d.textContent = c.blurb || "No notes yet.";
  container.appendChild(d);
  btn.querySelector(".chev").textContent = "▾";
}

// -------------------- command parsing --------------------
function interpret(raw) {
  const clean = raw.trim().toLowerCase().replace(/[^\w\s]/g, " ").replace(/\s+/g, " ").trim();
  if (!clean) return {verb: "noop"};
  const tokens = clean.split(" ");

  if (tokens[0] === "quit" || tokens[0] === "exit") return {verb: "quit"};
  if (clean === "look" || clean === "look around" || clean === "survey") return {verb: "look"};
  if (clean === "notebook" || clean === "notes") return {verb: "notebook"};
  if (clean === "help" || clean === "?") return {verb: "help"};

  // Go / move
  let m = clean.match(/^(?:go|move|walk|head|travel)\s+(?:to\s+|the\s+|into\s+)*(.+)$/);
  if (m) return {verb: "move", target: m[1].trim()};
  // Accuse
  m = clean.match(/^(?:accuse|arrest|charge|name)\s+(.+)$/);
  if (m) return {verb: "accuse", target: m[1].trim()};
  // Destroy-family (drama-manager exception)
  m = clean.match(/^(?:smash|destroy|break|shatter|burn|hide|steal|pocket|throw away)\s+(.+)$/);
  if (m) return {verb: "destroy", target: m[1].trim()};
  // Interview / question / talk to / confront / consult (expert or witness)
  m = clean.match(/^(?:interview|question|talk\s+to|ask|confront|speak\s+to|consult)\s+(.+)$/);
  if (m) return {verb: "interview", target: m[1].trim()};
  // Examine / inspect
  m = clean.match(/^(?:examine|inspect|look\s+at|study|observe|check)\s+(.+)$/);
  if (m) return {verb: "examine", target: m[1].trim()};
  // Analyze / test
  m = clean.match(/^(?:analyze|analyse|test|run)\s+(.+)$/);
  if (m) return {verb: "analyze", target: m[1].trim()};
  // Search / investigate
  m = clean.match(/^(?:search|explore|look\s+around|scour|investigate)\s+(.+)$/);
  if (m) return {verb: "search", target: m[1].trim()};
  // Visit
  m = clean.match(/^visit\s+(.+)$/);
  if (m) return {verb: "visit", target: m[1].trim()};
  // Take / pick up
  m = clean.match(/^(?:take|grab|pick\s+up|pocket)\s+(.+)$/);
  if (m) return {verb: "take", target: m[1].trim()};

  return {verb: "custom", target: clean};
}

// -------------------- entity matching --------------------
function matchEntity(target, pool) {
  if (!target) return null;
  const tgt = target.toLowerCase();
  const tgtTokens = new Set(tgt.split(" "));
  let best = null;
  let bestScore = 0;
  for (const e of pool) {
    let score = 0;
    for (const a of (e.aliases || [])) {
      if (tgt.includes(a)) score += a.length;
      else if (tgtTokens.has(a)) score += 2;
    }
    if (score > bestScore) { bestScore = score; best = e; }
  }
  return bestScore >= 3 ? best : null;
}

function matchLocation(target) {
  const loc = DATA.locations[state.location];
  const candidates = loc ? loc.adjacent.map(id => DATA.locations[id]).filter(Boolean) : [];
  // also allow matching all locations if user typed an obvious name
  const hit = matchEntity(target, candidates) || matchEntity(target, Object.values(DATA.locations));
  return hit;
}

function matchCharacter(target) {
  return matchEntity(target, Object.values(DATA.characters));
}

function matchEvidence(target) {
  return matchEntity(target, Object.values(DATA.evidence));
}

// -------------------- action handlers --------------------
function currentLoc() { return DATA.locations[state.location]; }

function eventAtHereMatching(verb, target) {
  // Find a plan event whose location == current AND verb matches AND args include an entity matching target
  const loc = state.location;
  const pool = Object.values(DATA.events).filter(ev => ev.location === loc);
  if (!pool.length) return null;

  // Verb family alignment
  const verbFamily = {
    "examine": ["examine", "search", "observe", "check", "analyze", "investigate"],
    "search":  ["search", "examine", "observe", "investigate"],
    "analyze": ["analyze", "examine", "investigate"],
    "interview": ["interview", "question", "consult", "confront", "visit"],
    "visit": ["visit", "interview", "investigate"],
  };
  const allowedVerbs = new Set(verbFamily[verb] || [verb]);

  const tgtLower = (target || "").toLowerCase();

  let best = null, bestScore = 0;
  for (const ev of pool) {
    if (state.executedEvents.includes(ev.id)) continue;
    if (!allowedVerbs.has(ev.verb)) continue;

    // Match arg tokens against target
    let score = 0;
    for (const arg of ev.args) {
      const argStr = String(arg).toLowerCase();
      const argTokens = argStr.split(/[_\s.]+/).filter(t => t.length > 2);
      for (const t of argTokens) {
        if (tgtLower.includes(t)) score += t.length;
      }
    }
    if (score > bestScore) { bestScore = score; best = ev; }
  }
  return bestScore >= 3 ? best : null;
}

function _extractPersonLike(s) {
  // Heuristic name-spotter for free-string event args. Picks up
  // "Dr. Helena Frost" / "Professor Vane" / "Lord Ashworth" style names.
  const m = String(s).match(
    /\b(?:Dr|Mr|Mrs|Ms|Lady|Lord|Inspector|Sir|Madam|Professor|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/
  );
  if (m) return m[0];
  // Fallback: two+ capitalised words in a row.
  const n = String(s).match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b/);
  return n ? n[0] : null;
}

function note(entry) {
  // Deduplicate: same line never appears twice in the notebook.
  if (!state.knowledge.includes(entry)) state.knowledge.push(entry);
}

function executeEvent(ev) {
  state.executedEvents.push(ev.id);
  const socialVerbs = new Set(["interview","consult","confront","question","visit"]);
  (ev.args || []).forEach(a => {
    const asStr = String(a);
    if (asStr.startsWith("character.")) {
      encounter(asStr);
      if (socialVerbs.has(ev.verb)) {
        const c = DATA.characters[asStr];
        if (c) note("Spoke with — " + c.name);
      }
    } else if (socialVerbs.has(ev.verb)) {
      // Free-string arg referencing a person off the main character list
      // (e.g. an off-screen consultant like Dr. Helena Frost).
      const name = _extractPersonLike(asStr);
      if (name) note("Consulted — " + name);
    }
  });
  // Apply reveals: mark evidence discovered (dedup via note()).
  // Some events in plan.json list reveals with the full 'evidence.E007' id,
  // others use just 'E007'. Normalize so both forms resolve to the same
  // entry in DATA.evidence.
  (ev.reveals || []).forEach(raw => {
    const rawStr = String(raw);
    const eid = rawStr.startsWith("evidence.") ? rawStr : "evidence." + rawStr;
    if (!state.evidenceFlags[eid]) state.evidenceFlags[eid] = {};
    state.evidenceFlags[eid].discovered = true;
    const e = DATA.evidence[eid];
    if (e) note("Evidence: " + truncate(e.description, 60));
    else   note("Lead: " + eid.replace(/^evidence\./, ""));
  });
  if (ev.verb === "analyze") {
    (ev.args || []).forEach(a => {
      const m = matchEvidence(String(a));
      if (m) {
        if (!state.evidenceFlags[m.id]) state.evidenceFlags[m.id] = {};
        state.evidenceFlags[m.id].analyzed = true;
      }
    });
  }
  // Any social-verb event on a named character counts as "interviewed"
  // so the hint engine stops nagging us to speak with them again.
  if (socialVerbs.has(ev.verb)) {
    (ev.args || []).forEach(a => {
      if (String(a).startsWith("character.")) {
        if (!state.charactersInterviewed.includes(a)) state.charactersInterviewed.push(a);
      }
    });
  }
  // Strip any leading markdown '# ...' lines the LLM may have left on the
  // plot-point prose (spurious chapter-style headings inside event text).
  const cleanNarrative = (ev.narrative || "").replace(/^(?:#+[^\n]*\n\s*)+/, "").trim();
  addLog(cleanNarrative, "narration", "— " + ev.description + " —");
  addLog("Your case grows clearer. (" + state.executedEvents.length + " / " + DATA.goal_events_needed + " plot events explored.)", "system");
}

function handleMove(target) {
  const loc = matchLocation(target);
  if (!loc) {
    addLog("You cannot find a way to " + target + " from here. (Click an exit in the left panel, or try one of the names listed there.)", "outcome");
    return;
  }
  const cur = currentLoc();
  if (!cur || !cur.adjacent.includes(loc.id)) {
    addLog("You'd have to pass through somewhere else first. " + loc.name + " isn't directly reachable from " + (cur ? cur.name : "here") + ".", "outcome");
    return;
  }
  state.lastLocation = state.location;
  state.location = loc.id;
  encounterAll(loc.characters);
  addLog("You make your way to " + loc.name + ".", "system");
  addLog(loc.description, "outcome");
}

function handleLook() {
  const loc = currentLoc();
  if (!loc) return;
  addLog(loc.description, "outcome");
  const names = loc.characters.map(id => (DATA.characters[id] || {}).name).filter(Boolean);
  if (names.length) addLog("You see " + names.join(", ") + ".", "outcome");
}

function handleExamine(target) {
  if (!target) { handleLook(); return; }
  // First try a plan event at this location
  const ev = eventAtHereMatching("examine", target);
  if (ev) { executeEvent(ev); return; }
  // Maybe user is pointing at evidence here, just not the right kind of examine
  const loc = currentLoc();
  if (loc) {
    for (const eid of loc.evidence) {
      const e = DATA.evidence[eid];
      if (!e) continue;
      for (const a of (e.aliases || [])) {
        if (target.toLowerCase().includes(a)) {
          addLog("You peer at the " + truncate(e.description, 50) + ", but nothing new reveals itself. Perhaps a specialist could analyze it elsewhere.", "outcome");
          return;
        }
      }
    }
  }
  addLog("You look, but " + target + " doesn't seem to be here — or Inspector Rothwell can't make sense of it from this angle.", "outcome");
}

function handleInterview(target) {
  // Try to match a plan event at this location first. Consult-type events
  // may reference off-screen experts (e.g. 'Dr. Helena Frost' called in to
  // authenticate a painting) who aren't in the room's character list.
  const evPre = eventAtHereMatching("interview", target);
  if (evPre) { executeEvent(evPre); return; }
  const c = matchCharacter(target);
  if (!c) {
    addLog("You call out for " + target + ", but no one answers. Try questioning someone listed in the left panel, or move to a location where that person is waiting.", "outcome");
    return;
  }
  const loc = currentLoc();
  if (!loc || !loc.characters.includes(c.id)) {
    addLog(c.name + " isn't here. You'll need to find them first.", "outcome");
    return;
  }
  // Nothing plan-relevant left to extract from this character at this
  // location. Mark them "interviewed" so the hint engine stops looping.
  if (!state.charactersInterviewed.includes(c.id)) state.charactersInterviewed.push(c.id);
  addLog(c.name + " answers politely but says nothing that moves the case forward here.", "outcome");
}

function handleAnalyze(target) {
  const ev = eventAtHereMatching("analyze", target);
  if (ev) { executeEvent(ev); return; }
  if (!currentLoc().id.includes("lab") && !currentLoc().id.includes("forensic")) {
    addLog("You'd need proper equipment. Try a forensic laboratory.", "outcome");
    return;
  }
  addLog("The analysis yields nothing useful.", "outcome");
}

function handleSearch(target) {
  const ev = eventAtHereMatching("search", target);
  if (ev) { executeEvent(ev); return; }
  addLog("You search " + (target || "the area") + " thoroughly but find nothing new.", "outcome");
}

function handleAccuse(target) {
  const c = matchCharacter(target);
  if (!c) {
    addLog("You announce an accusation, but the name doesn't register with anyone present. Be specific: try 'accuse <surname>'.", "outcome");
    return;
  }
  if (state.executedEvents.length < DATA.goal_events_needed) {
    addLog("You haven't gathered enough evidence yet. An accusation without proof will be dismissed. Keep investigating.", "exception");
    return;
  }
  if (c.id === DATA.real_criminal_id) {
    state.gameOver = true;
    addLog("You name " + c.name + " as the poisoner of " + DATA.victim_name + ". The hollow ring, the staged toast, the conspirators — you lay it all out. The room falls silent. The case is solved.", "victory", "THE CASE IS CLOSED");
    addLog("You have won. (Open the [novel version](./index.html) for the full narrative.)", "system");
  } else {
    addLog("You accuse " + c.name + " — but the evidence doesn't hold up under scrutiny. A wrongful arrest is a mark on your record, and the real killer may go free. Consider what other leads you haven't chased.", "exception");
  }
}

function handleDestroy(target) {
  // Drama-manager "exceptional" pantomime
  addLog("You reach to tamper with " + target + ", but catch yourself. An inspector who destroys evidence is no inspector at all. (The drama manager has intervened — your story still has a path to the truth.)", "exception");
}

// -------------------- dispatch --------------------
function runCommand(raw) {
  if (state.gameOver) {
    addLog("The case is already closed. Press Reset to begin a new investigation.", "system");
    return;
  }
  addLog(raw, "user");
  state.turns += 1;
  const action = interpret(raw);
  switch (action.verb) {
    case "noop":    break;
    case "quit":    addLog("You step back from the case. Press Reset to start over.", "system"); break;
    case "look":    handleLook(); break;
    case "notebook":
    case "notes":   addLog("See the right panel for your notebook entries.", "system"); break;
    case "help":
      addLog("Try any of: look · examine <thing> · go to <location> · question <person> · analyze <object> · search <place> · accuse <person>. You can also click an exit on the left panel to move.", "system");
      break;
    case "move":    handleMove(action.target); break;
    case "examine": handleExamine(action.target); break;
    case "interview": handleInterview(action.target); break;
    case "analyze": handleAnalyze(action.target); break;
    case "search":  handleSearch(action.target); break;
    case "visit":   handleMove(action.target); break;
    case "take":    addLog("Better not disturb evidence. Examine it in place instead.", "outcome"); break;
    case "accuse":  handleAccuse(action.target); break;
    case "destroy": handleDestroy(action.target); break;
    case "custom":
    default:
      addLog("Inspector Rothwell tilts his head. \"" + raw + "\" isn't a move he can make on this case — try a simpler verb like look, examine, go, question, analyze, or accuse.", "system");
  }
  saveState();
  renderSidebar();
  renderHints();
}

// -------------------- wiring --------------------
document.getElementById("input-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const el = document.getElementById("cmd");
  const raw = el.value;
  el.value = "";
  if (!raw.trim()) return;
  // Enforce 8-word ceiling like the backend engine
  const words = raw.trim().split(/\s+/).slice(0, 8);
  runCommand(words.join(" "));
});

document.getElementById("reset-btn").addEventListener("click", () => {
  if (!confirm("Start a new investigation? Your current progress will be lost.")) return;
  localStorage.removeItem(STORAGE_KEY);
  state = freshState();
  logEl.innerHTML = "";
  greet();
  renderSidebar();
  renderHints();
  document.getElementById("cmd").value = "";
  document.getElementById("cmd").focus();
});

// Toggle the hints panel open/closed.
document.getElementById("hints-toggle").addEventListener("click", () => {
  const tog = document.getElementById("hints-toggle");
  const box = document.getElementById("hints-content");
  const open = box.classList.toggle("hidden");
  if (open) {
    tog.classList.remove("open");
    tog.textContent = "💡 Stuck? Show hints";
  } else {
    tog.classList.add("open");
    tog.textContent = "💡 Hide hints";
    renderHints();  // refresh on open
  }
});

function greet() {
  addLog("CASE: The Hartley Affair", "system");
  addLog(
    "You are " + DATA.detective_name + " of Scotland Yard. It is past midnight. " +
    "Tonight, the host of a private gallery reception — " + DATA.victim_name + " — " +
    "collapsed in his own home during the opening toast. The attending physician called it heart failure; " +
    "you are unconvinced. A constable has just opened the door for you, and the body lies where it fell.",
    "narration",
    "— the call that brought you here —"
  );
  const loc = currentLoc();
  addLog(
    "You step into " + (loc ? loc.name : "the scene") + ". " + (loc ? loc.description : ""),
    "outcome"
  );
  addLog(
    "Open-ended investigation. Type any command (≤ 8 words). A good first move is 'examine body'. " +
    "You can move with 'go to <place>', 'question <name>', 'analyze <object>'. When you know the poisoner, 'accuse <name>'.",
    "system"
  );
}

renderSidebar();
renderHints();
if (state.executedEvents.length === 0 && state.turns === 0) greet();
else addLog("(Session resumed. " + state.executedEvents.length + " plot events already explored.)", "system");
document.getElementById("cmd").focus();
</script>
</body>
</html>
"""


def build() -> None:
    data = build_game_data()
    html_out = HTML_TEMPLATE.replace("__GAME_DATA__", json.dumps(data, ensure_ascii=False))
    OUT.write_text(html_out, encoding="utf-8")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
    print(f"Events: {len(data['events'])}  Locations: {len(data['locations'])}  Characters: {len(data['characters'])}")
    print(f"Starting location: {data['starting_location']}  ({data['locations'][data['starting_location']]['name']})")
    print(f"Win condition: accuse {data['real_criminal_name']} (id={data['real_criminal_id']}) after {data['goal_events_needed']} events")


if __name__ == "__main__":
    build()
