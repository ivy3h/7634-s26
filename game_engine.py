"""The text game engine: world state, I/O loop, action execution.

Responsibilities:
  - Own the mutable world state (initialized from plan.initial_state +
    world.locations).
  - Render a brief text description of the player's current location.
  - Call action_interpreter for every line of input.
  - Hand off to drama_manager for classification + accommodation.
  - Apply the resulting effects and narrate them back to the player.
  - Log everything.

The engine tries to stay dumb: it does not invent effects, it only applies
what the interpreter and drama manager produce. That keeps the decision
pipeline inspectable.
"""
from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from action_interpreter import interpret_action
from drama_manager import DramaManager
from llm_client import chat_simple
from plan_types import Effect, Plan
from world_builder import World


MAX_WORDS_PER_COMMAND = 8  # loose; spec suggests ~5


@dataclass
class EngineConfig:
    max_turns: int = 80
    prompt_label: str = "> "
    narrate_with_llm: bool = True
    log_dir: Path = Path("logs")
    detective_name: str = "Inspector Rothwell"


@dataclass
class TurnLog:
    turn: int
    raw: str
    parsed: dict[str, Any]
    classification: dict[str, Any]
    effects_applied: list[dict[str, Any]] = field(default_factory=list)
    accommodation: dict[str, Any] | None = None
    narration: str = ""


class GameEngine:
    def __init__(
        self,
        plan: Plan,
        world: World,
        config: EngineConfig | None = None,
    ) -> None:
        self.plan = plan
        self.world = world
        self.config = config or EngineConfig()
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        self.state: dict[str, dict[str, Any]] = copy.deepcopy(plan.initial_state)
        self.state.setdefault("detective", {})
        self.state["detective"]["location"] = world.starting_location

        # Normalize evidence IDs in every location: some world.json entries omit
        # the "evidence." prefix, which causes lookup failures in the scene sketch.
        for loc in self.world.locations.values():
            loc.evidence = [
                e if e.startswith("evidence.") else "evidence." + e
                for e in loc.evidence
            ]

        # Ensure the criminal character exists in state so name lookups work in
        # scene sketches (the criminal is not added to plan.initial_state by default).
        for loc in self.world.locations.values():
            for cid in loc.characters:
                if cid not in self.state:
                    name = cid.split(".", 1)[-1].replace("_", " ").title()
                    self.state[cid] = {"name": name, "role": "associate", "alive": True, "available": True}

        self.drama = DramaManager(plan, world=world, log_path=self.config.log_dir / "drama.jsonl")
        self.turn_logs: list[TurnLog] = []
        self.turn_log_path = self.config.log_dir / "turns.jsonl"
        # Truncate the file at game start so replays are clean.
        self.turn_log_path.write_text("", encoding="utf-8")

    # ---------------- rendering ---------------------------------------
    def render_location(self) -> str:
        loc_id = self.state["detective"]["location"]
        loc = self.world.locations.get(loc_id)
        if loc is None:
            return f"[The detective stands in an unmapped place: {loc_id}.]"
        parts = [f"-- {loc.name} --", loc.description]
        if loc.adjacent:
            adj_names = ", ".join(self.world.locations[a].name for a in sorted(loc.adjacent) if a in self.world.locations)
            parts.append(f"Exits: {adj_names}")
        chars_here = [self._char_name(cid) for cid in loc.characters if self._is_available(cid)]
        if chars_here:
            parts.append("You see: " + ", ".join(chars_here))
        evidence_here = [
            self.state.get(eid, {}).get("description", eid)
            for eid in loc.evidence
            if not self.state.get(eid, {}).get("destroyed", False)
            and not self.state.get(eid, {}).get("analyzed", False)
        ]
        if evidence_here:
            parts.append("Nearby: " + ", ".join(evidence_here[:3]))
        return "\n".join(parts)

    def render_map(self) -> str:
        """One-time overview of all locations and their connections."""
        lines = ["=== Location Map ==="]
        for lid in sorted(self.world.locations):
            loc = self.world.locations[lid]
            adj = [self.world.locations[a].name for a in sorted(loc.adjacent) if a in self.world.locations]
            lines.append(f"  {loc.name}")
            if adj:
                lines.append("    → " + ", ".join(adj))
        lines.append("===================")
        return "\n".join(lines)

    def _pending_event_hint(self, raw: str, loc_id: str) -> str | None:
        """Return a hint when the player mentioned a character with pending plan events.
        Returns None if the hint would just repeat what the player already typed."""
        raw_lower = raw.lower()
        for eid in self.drama.remaining:
            ev = self.plan.events.get(eid)
            if not ev:
                continue
            for arg in ev.args:
                if not str(arg).startswith("character."):
                    continue
                char_name = self.state.get(str(arg), {}).get("name", "")
                if not char_name:
                    continue
                if not any(tok in raw_lower for tok in char_name.lower().split() if len(tok) > 2):
                    continue
                verb = "question" if ev.verb not in {"examine", "search", "analyze"} else ev.verb
                surname = char_name.split()[-1]
                cmd = f"{verb} {surname}".lower()
                # Suppress if this is exactly what the player just typed
                if raw_lower.strip().startswith(cmd) or cmd in raw_lower:
                    return None
                if ev.location == loc_id:
                    # Use a special prefix so the JS can render this as a button
                    return f"HINT_CMD:{cmd}"
                loc = self.world.locations.get(ev.location or "")
                loc_name = loc.name if loc else ev.location
                return f"Head to {loc_name} to follow up on {char_name}."
        return None

    def _char_name(self, cid: str) -> str:
        return self.state.get(cid, {}).get("name", cid)

    def _evidence_desc(self, eid: str) -> str:
        """Return a human-readable description for an evidence id.
        Falls back to plan.initial_state so undiscovered items still show
        a proper description rather than the raw id like 'evidence.E003'."""
        desc = self.state.get(eid, {}).get("description", "")
        if not desc:
            desc = self.plan.initial_state.get(eid, {}).get("description", "")
        if not desc:
            desc = eid.replace("evidence.", "").replace("_", " ")
        return desc

    def _is_available(self, cid: str) -> bool:
        return self.state.get(cid, {}).get("alive", True) and self.state.get(cid, {}).get("available", True)

    def _world_summary_for_interpreter(self) -> dict[str, Any]:
        loc_id = self.state["detective"]["location"]
        loc = self.world.locations.get(loc_id)
        adj = sorted(loc.adjacent) if loc else []
        here_chars = [f"{cid} ({self._char_name(cid)})" for cid in (loc.characters if loc else [])]
        here_evidence = [
            eid for eid in (loc.evidence if loc else [])
            if not self.state.get(eid, {}).get("destroyed", False)
        ]
        return {
            "player_location": loc_id,
            "adjacent": adj,
            "here_objects": here_evidence,
            "here_characters": here_chars,
            "evidence_ids": [k for k in self.state if k.startswith("evidence.")],
            "inventory": self.state["detective"].get("inventory", []),
            "knowledge_snippets": self.state["detective"].get("knowledge", []),
        }

    # ---------------- input loop --------------------------------------
    def run(self, get_input=input, echo=print) -> str:
        echo(self.render_location())
        echo("")
        for turn in range(1, self.config.max_turns + 1):
            try:
                raw = get_input(self.config.prompt_label)
            except EOFError:
                echo("\n[input closed]")
                break
            if not raw.strip():
                continue
            if raw.strip().lower() in {"quit", "exit"}:
                echo("(detective signs off)")
                break
            raw = _truncate_input(raw)

            parsed = interpret_action(raw, self._world_summary_for_interpreter())
            classification = self.drama.classify(parsed, self.state)
            effects_applied: list[dict[str, Any]] = []
            accommodation_result: dict[str, Any] | None = None

            if classification["classification"] == "constituent":
                eid = classification["matched_event_id"]
                self.drama.execute_constituent(eid, self.state)
                effects_applied = [ef.to_dict() for ef in self.plan.events[eid].effects]
            elif classification["classification"] == "consistent":
                self.drama.apply_free_effects(parsed, self.state)
                self._apply_movement_if_any(parsed)
                effects_applied = parsed.get("effects", [])
            else:  # exceptional
                # Apply the free effects first so the world reflects what the
                # player actually did, then repair the plan around it.
                self.drama.apply_free_effects(parsed, self.state)
                self._apply_movement_if_any(parsed)
                effects_applied = parsed.get("effects", [])
                accommodation_result = self.drama.accommodate(
                    parsed,
                    classification,
                    self.state,
                    world_locations=list(self.world.locations.keys()),
                    characters=[s for s in self.state if s.startswith("character.")],
                )

            narration = self._narrate(parsed, classification, effects_applied, accommodation_result)

            self._log_turn(turn, raw, parsed, classification, effects_applied, accommodation_result, narration)
            echo(narration)
            echo("")

            if self.drama.goal_satisfied(self.state):
                echo(">>> The case is solved. <<<")
                return "solved"
        return "ended"

    # ---------------- web API (single-turn) --------------------------
    def step(self, raw: str) -> dict[str, Any]:
        """Process one player command; return a structured dict for the web frontend.

        Returns:
            log_entries: list of {text, cls, title, as_html} to append to the log div
            triggered_event_id: plan event id if constituent, else None
            moved_to: new location id if detective moved, else None
            new_knowledge: list of strings to add to the notebook
            new_evidence_flags: {evidence_id: {discovered, analyzed}} patches
            characters_encountered: character ids newly seen
            characters_interviewed: character ids newly spoken to
            classification: "constituent" | "consistent" | "exceptional"
            dm_entry: {kind, summary, detail} from latest drama-manager decision
            game_over: True when goal conditions are satisfied
        """
        raw = raw.strip()
        if not raw:
            return _empty_step()

        raw = _truncate_input(raw)
        prev_loc = self.state["detective"]["location"]
        turn = len(self.turn_logs) + 1

        self.drama.log_turn_start(turn, raw, prev_loc)
        parsed = interpret_action(raw, self._world_summary_for_interpreter())
        classification = self.drama.classify(parsed, self.state)
        effects_applied: list[dict[str, Any]] = []
        accommodation_result: dict[str, Any] | None = None
        triggered_event_id: str | None = None
        tag = classification["classification"]

        if tag == "constituent":
            eid = classification["matched_event_id"]
            triggered_event_id = eid
            self.drama.execute_constituent(eid, self.state)
            effects_applied = [ef.to_dict() for ef in self.plan.events[eid].effects]
        elif tag == "consistent":
            self.drama.apply_free_effects(parsed, self.state)
            self._apply_movement_if_any(parsed)
            effects_applied = parsed.get("effects", [])
        else:  # exceptional
            self.drama.apply_free_effects(parsed, self.state)
            self._apply_movement_if_any(parsed)
            effects_applied = parsed.get("effects", [])
            accommodation_result = self.drama.accommodate(
                parsed, classification, self.state,
                world_locations=list(self.world.locations.keys()),
                characters=[s for s in self.state if s.startswith("character.")],
            )

        narration = self._narrate(parsed, classification, effects_applied, accommodation_result)
        self._log_turn(turn, raw, parsed, classification, effects_applied, accommodation_result, narration)

        new_loc = self.state["detective"]["location"]
        moved_to: str | None = new_loc if new_loc != prev_loc else None

        log_entries: list[dict[str, Any]] = []
        new_knowledge: list[str] = []
        new_evidence_flags: dict[str, dict[str, Any]] = {}
        characters_encountered: list[str] = []
        characters_interviewed: list[str] = []
        _social = {"interview", "consult", "confront", "question", "visit"}

        if tag == "constituent" and triggered_event_id:
            ev = self.plan.events[triggered_event_id]
            log_entries.append({"text": narration, "cls": "narration",
                                 "title": f"— {ev.description} —", "as_html": False})
            log_entries.append({
                "text": f"Your case grows clearer. ({len(self.drama.executed)} plot events explored.)",
                "cls": "system", "title": None, "as_html": False,
            })
            for arg in ev.args:
                arg_s = str(arg)
                if arg_s.startswith("character."):
                    name = self.state.get(arg_s, {}).get(
                        "name", arg_s.split(".", 1)[-1].replace("_", " ").title()
                    )
                    characters_encountered.append(arg_s)
                    if ev.verb in _social:
                        characters_interviewed.append(arg_s)
                        new_knowledge.append(f"Spoke with — {name}")
            for raw_reveal in ev.reveals:
                eid2 = str(raw_reveal)
                if not eid2.startswith("evidence."):
                    eid2 = "evidence." + eid2
                desc = self._evidence_desc(eid2)
                new_knowledge.append(f"Evidence: {desc[:60]}")
                new_evidence_flags[eid2] = {"discovered": True}
            if ev.verb == "analyze":
                for arg in ev.args:
                    arg_s = str(arg)
                    if arg_s.startswith("evidence."):
                        new_evidence_flags.setdefault(arg_s, {})["analyzed"] = True
            loc_id = self.state["detective"]["location"]
            remaining_here = sum(
                1 for eid2 in self.drama.remaining
                if self.plan.events.get(eid2) and self.plan.events[eid2].location == loc_id
            )
            if remaining_here == 0:
                loc = self.world.locations.get(loc_id)
                log_entries.append({
                    "text": f"Nothing more to investigate at {loc.name if loc else loc_id}. "
                            "Try an exit from the left panel.",
                    "cls": "system", "title": None, "as_html": False,
                })
            if not self.drama.remaining:
                log_entries.append({
                    "text": "Every trail you could follow has been followed. "
                            "Your notebook is full. Type 'accuse <suspect>'.",
                    "cls": "narration", "title": "— the case, fully explored —", "as_html": False,
                })

        elif tag == "consistent":
            if moved_to:
                loc = self.world.locations.get(moved_to)
                loc_name = loc.name if loc else moved_to
                log_entries.append({"text": f"You make your way to {loc_name}.",
                                     "cls": "system", "title": None, "as_html": False})
                if loc:
                    log_entries.append({"text": loc.description,
                                         "cls": "outcome", "title": None, "as_html": False})
                    chars = [
                        self.state.get(cid, {}).get("name", cid)
                        for cid in loc.characters
                        if self.state.get(cid, {}).get("alive", True)
                    ]
                    items = [
                        self._evidence_desc(eid)
                        for eid in loc.evidence
                        if not self.state.get(eid, {}).get("destroyed", False)
                    ]
                    sketch = []
                    if chars:
                        sketch.append("Here with you: " + ", ".join(chars) + ".")
                    if items:
                        sketch.append("You notice: " + "; ".join(items) + ".")
                    if sketch:
                        log_entries.append({"text": " ".join(sketch),
                                             "cls": "outcome", "title": None, "as_html": False})
                    characters_encountered = list(loc.characters)
            else:
                log_entries.append({"text": narration, "cls": "outcome",
                                     "title": None, "as_html": False})
                hint = self._pending_event_hint(raw, self.state["detective"]["location"])
                if hint:
                    if hint.startswith("HINT_CMD:"):
                        cmd = hint[len("HINT_CMD:"):]
                        hint_html = (
                            f'<span style="opacity:.75">You might try: </span>'
                            f'<button class="chip inline-hint" data-cmd="{cmd}" '
                            f'style="margin-left:4px"><span class="arrow">&gt;</span>{cmd}</button>'
                        )
                        log_entries.append({"text": hint_html, "cls": "system",
                                             "title": None, "as_html": True})
                    else:
                        log_entries.append({"text": hint, "cls": "system",
                                             "title": None, "as_html": False})

        else:  # exceptional
            log_entries.append({"text": narration, "cls": "exception",
                                 "title": None, "as_html": False})

        # Latest drama-manager log entry for the DM panel.
        # Search backward so the most meaningful entry wins regardless of order.
        dm_entry: dict[str, Any] | None = None
        for entry in reversed(self.drama.log):
            p = entry.payload
            if entry.kind == "accommodation":
                removed = p.get("removed_events", [])
                added   = p.get("replacement_event_ids", [])
                verdict = p.get("goal_reachability", {}).get("verdict", "unknown")
                dm_entry = {
                    "kind": "exceptional",
                    "summary": f"plan modified — removed {len(removed)}, added {len(added)} event(s)",
                    "detail": f"goal: {verdict} | removed: {removed} | added: {added}",
                    "plan_change": f"removed {len(removed)}, added {len(added)} event(s)",
                    "goal_verdict": verdict,
                }
                break
            if entry.kind == "executed_constituent":
                eid  = p.get("event_id", "")
                desc = p.get("event_description", "")
                rem  = p.get("remaining_after", "?")
                dm_entry = {
                    "kind": "constituent",
                    "summary": f"plan event {eid} fired — {desc[:60]}",
                    "detail": f"reveals: {p.get('reveals', [])} | remaining: {rem}",
                    "event_desc": desc,
                    "reveals": p.get("reveals", []),
                    "remaining_after": rem,
                }
                break
            if entry.kind == "classification":
                cls_val = p.get("classification", "consistent")
                matched = p.get("matched_event_id") or "none"
                rem     = p.get("remaining_count", "?")
                active  = len(p.get("active_links", []))
                dm_entry = {
                    "kind": cls_val,
                    "summary": f"{cls_val} — matched={matched} | {rem} events remaining",
                    "detail": f"active causal links: {active} | cs_hint: {p.get('cs_hint', '')}",
                }
                break
        if dm_entry is None and self.drama.log:
            last = self.drama.log[-1]
            dm_entry = {"kind": last.kind, "summary": last.kind, "detail": ""}

        return {
            "log_entries": log_entries,
            "triggered_event_id": triggered_event_id,
            "moved_to": moved_to,
            "new_knowledge": new_knowledge,
            "new_evidence_flags": new_evidence_flags,
            "characters_encountered": characters_encountered,
            "characters_interviewed": characters_interviewed,
            "classification": tag,
            "dm_entry": dm_entry,
            "game_over": self.drama.goal_satisfied(self.state),
        }

    # ---------------- hint-forced execution (Bug-1 fix) ---------------
    def step_force_event(self, event_id: str) -> dict[str, Any]:
        """Execute a plan event directly as constituent, bypassing the action interpreter.
        Called when the player clicks a hint chip — guarantees constituent classification."""
        if event_id not in self.plan.events:
            return {**_empty_step(), "error": f"unknown event {event_id}"}
        if event_id not in self.drama.remaining:
            return {**_empty_step(), "error": f"event {event_id} already executed or removed"}

        ev = self.plan.events[event_id]
        cur_loc = self.state["detective"]["location"]
        if ev.location and ev.location != cur_loc:
            return {**_empty_step(), "error": f"not at required location {ev.location}"}

        turn = len(self.turn_logs) + 1
        short_args = " ".join(
            str(a) for a in ev.args[:2] if not str(a).startswith("location.")
        )
        raw = f"[hint] {ev.verb} {short_args}".strip()
        parsed = {
            "_raw": raw,
            "verb": ev.verb,
            "args": list(ev.args),
            "location": ev.location or cur_loc,
            "effects": [ef.to_dict() for ef in ev.effects],
            "plain_summary": f"{ev.verb} {short_args}".strip(),
            "preconditions": [],
            "reveals": list(ev.reveals),
        }
        classification = {
            "classification": "constituent",
            "matched_event_id": event_id,
            "hard_violations": [],
            "soft_threats": [],
        }

        self.drama.log_turn_start(turn, raw, cur_loc)
        self.drama.execute_constituent(event_id, self.state)
        effects_applied = [ef.to_dict() for ef in ev.effects]

        narration = self._narrate(parsed, classification, effects_applied, None)
        self._log_turn(turn, raw, parsed, classification, effects_applied, None, narration)

        new_knowledge: list[str] = []
        new_evidence_flags: dict[str, dict[str, Any]] = {}
        characters_encountered: list[str] = []
        characters_interviewed: list[str] = []
        log_entries: list[dict[str, Any]] = []
        _social = {"interview", "consult", "confront", "question", "visit"}

        log_entries.append({"text": narration, "cls": "narration",
                             "title": f"— {ev.description} —", "as_html": False})
        log_entries.append({
            "text": f"Your case grows clearer. ({len(self.drama.executed)} plot events explored.)",
            "cls": "system", "title": None, "as_html": False,
        })
        for arg in ev.args:
            arg_s = str(arg)
            if arg_s.startswith("character."):
                name = self.state.get(arg_s, {}).get(
                    "name", arg_s.split(".", 1)[-1].replace("_", " ").title()
                )
                characters_encountered.append(arg_s)
                if ev.verb in _social:
                    characters_interviewed.append(arg_s)
                    new_knowledge.append(f"Spoke with — {name}")
        for raw_reveal in ev.reveals:
            eid2 = str(raw_reveal)
            if not eid2.startswith("evidence."):
                eid2 = "evidence." + eid2
            desc = self._evidence_desc(eid2)
            new_knowledge.append(f"Evidence: {desc[:60]}")
            new_evidence_flags[eid2] = {"discovered": True}
        if ev.verb == "analyze":
            for arg in ev.args:
                arg_s = str(arg)
                if arg_s.startswith("evidence."):
                    new_evidence_flags.setdefault(arg_s, {})["analyzed"] = True

        loc_id = self.state["detective"]["location"]
        remaining_here = sum(
            1 for eid2 in self.drama.remaining
            if self.plan.events.get(eid2) and self.plan.events[eid2].location == loc_id
        )
        if remaining_here == 0:
            loc = self.world.locations.get(loc_id)
            log_entries.append({
                "text": f"Nothing more to investigate at {loc.name if loc else loc_id}. "
                        "Try an exit from the left panel.",
                "cls": "system", "title": None, "as_html": False,
            })
        if not self.drama.remaining:
            log_entries.append({
                "text": "Every trail you could follow has been followed. "
                        "Your notebook is full. Type 'accuse <suspect>'.",
                "cls": "narration", "title": "— the case, fully explored —", "as_html": False,
            })

        dm_entry = {
            "kind": "constituent",
            "summary": f"plan event {event_id} fired — {ev.description[:60]}",
            "detail": f"reveals: {ev.reveals} | remaining: {len(self.drama.remaining)}",
            "event_desc": ev.description,
            "reveals": ev.reveals,
            "remaining_after": len(self.drama.remaining),
        }
        return {
            "log_entries": log_entries,
            "triggered_event_id": event_id,
            "moved_to": None,
            "new_knowledge": new_knowledge,
            "new_evidence_flags": new_evidence_flags,
            "characters_encountered": characters_encountered,
            "characters_interviewed": characters_interviewed,
            "classification": "constituent",
            "dm_entry": dm_entry,
            "game_over": self.drama.goal_satisfied(self.state),
        }

    # ---------------- helpers -----------------------------------------
    _MOVEMENT_VERBS = frozenset({"move", "go", "walk", "travel", "head", "visit", "approach", "leave"})

    def _apply_movement_if_any(self, parsed: dict[str, Any]) -> None:
        # Only move the detective when the player actually used a movement verb.
        # Without this guard, "check X" or "inspect X" could trigger location
        # changes if the action interpreter extracted a target_location.
        verb = (parsed.get("verb") or "").lower()
        if verb not in self._MOVEMENT_VERBS:
            return
        target = parsed.get("target_location") or ""
        if not target:
            return
        cur_loc = self.state["detective"]["location"]
        cur = self.world.locations.get(cur_loc)
        if cur and target in cur.adjacent and target in self.world.locations:
            self.state["detective"]["location"] = target
            Effect("detective", "location", "set", target).apply(self.state)

    def _narration_system(self) -> str:
        """Build a system message that anchors the LLM to the real characters/locations."""
        char_names = []
        for k, v in self.state.items():
            if k.startswith("character."):
                name = v.get("name", k.split(".", 1)[-1].replace("_", " ").title())
                char_names.append(name)
        loc_names = [loc.name for loc in self.world.locations.values()]
        det = self.config.detective_name
        return (
            f"You are the narrator for a 1920s London detective noir mystery. "
            f"The player IS {det}. Always address them as 'you' — NEVER say "
            f"'the detective' or invent any other detective name. "
            f"ONLY use these characters: {', '.join(char_names) or 'none listed'}. "
            f"ONLY use these locations: {', '.join(loc_names[:12]) or 'none listed'}. "
            f"Never invent new people or places. Never reveal the real killer's identity."
        )

    def _next_move_hint(self) -> str | None:
        """One-sentence suggestion about what to try next, for embedding in narration."""
        loc_id = self.state["detective"]["location"]
        for eid in self.drama.remaining:
            ev = self.plan.events.get(eid)
            if not ev:
                continue
            if ev.location == loc_id:
                for arg in ev.args:
                    arg_s = str(arg)
                    if arg_s.startswith("character."):
                        cname = self.state.get(arg_s, {}).get("name", "")
                        if cname:
                            surname = cname.split()[-1]
                            return f"Perhaps questioning {cname} could reveal more."
                    elif not arg_s.startswith("location."):
                        # Strip object./evidence. prefixes so the hint reads naturally
                        pretty = re.sub(r"^(?:object|evidence)\.", "", arg_s).replace("_", " ")
                        verb = ev.verb if ev.verb not in {"interview", "consult"} else "question"
                        return f"You might try to {verb} the {pretty}."
        for eid in self.drama.remaining:
            ev = self.plan.events.get(eid)
            if not ev or not ev.location or ev.location == loc_id:
                continue
            loc = self.world.locations.get(ev.location)
            if loc:
                return f"There are leads worth pursuing at {loc.name}."
        return None

    def _narrate(
        self,
        parsed: dict[str, Any],
        classification: dict[str, Any],
        effects_applied: list[dict[str, Any]],
        accommodation_result: dict[str, Any] | None,
    ) -> str:
        if not self.config.narrate_with_llm:
            return self._stub_narration(parsed, classification)
        tag = classification["classification"]
        raw_cmd = parsed.get("_raw", "")
        summary = parsed.get("plain_summary", raw_cmd)
        loc_id = self.state["detective"]["location"]
        loc = self.world.locations.get(loc_id)
        loc_name = loc.name if loc else loc_id
        det = self.config.detective_name
        system = self._narration_system()

        if tag == "exceptional":
            next_hint = self._next_move_hint()
            hint_context = (
                f" The most useful next step would be: {next_hint}"
                if next_hint else ""
            )
            prompt = (
                f"You are {det} at {loc_name}. You tried: \"{raw_cmd}\" but something stopped you.\n\n"
                "Write 2 sentences of immersive 1920s-noir prose in second person ('you'). "
                "First, describe in one sentence why the action can't happen right now — "
                f"a physical obstacle, your professional judgment, or a practical constraint at {loc_name}. "
                f"Second, in one sentence, hint at one concrete thing you could do instead to move forward.{hint_context} "
                "No invented characters. No new locations. No mechanical labels."
            )
            max_t, temp = 120, 0.55
        elif tag == "constituent":
            eid = classification.get("matched_event_id")
            event_hint = ""
            if eid and eid in self.plan.events:
                event_hint = self.plan.events[eid].narrative[:500]
            next_hint = self._next_move_hint()
            hint_line = (
                f"\nEnd with a subtle one-sentence lead-in to the next step: {next_hint}"
                if next_hint else ""
            )
            prompt = (
                f"You are {det} at {loc_name}. You just: {summary}.\n"
                f"Plot reference (what was found): {event_hint}\n"
                f"World-state effects applied: {json.dumps(effects_applied)[:400]}\n\n"
                "Write 3-4 sentences of 1920s-noir prose describing what you discovered. "
                "Use 'you' to address the player. Stay faithful to the plot reference above. "
                "Do NOT invent new characters or locations."
                + hint_line
            )
            max_t, temp = 250, 0.65
        else:  # consistent — no plan progress
            next_hint = self._next_move_hint()
            hint_line = f" Then: {next_hint}" if next_hint else ""
            prompt = (
                f"You are {det} at {loc_name}. You tried: {summary}.\n"
                "Write ONE sentence in 1920s-noir style acknowledging this led nowhere new. "
                "Use 'you'. No new plot facts. No invented characters."
                + hint_line
            )
            max_t, temp = 100, 0.55

        try:
            return chat_simple(prompt, system=system, max_tokens=max_t, temperature=temp).strip()
        except Exception:
            return self._stub_narration(parsed, classification)

    @staticmethod
    def _stub_narration(parsed: dict[str, Any], classification: dict[str, Any]) -> str:
        return f"[{classification['classification']}] {parsed.get('plain_summary') or parsed.get('_raw', '')}"

    def _log_turn(
        self,
        turn: int,
        raw: str,
        parsed: dict[str, Any],
        classification: dict[str, Any],
        effects_applied: list[dict[str, Any]],
        accommodation_result: dict[str, Any] | None,
        narration: str,
    ) -> None:
        log = TurnLog(
            turn=turn, raw=raw, parsed=parsed, classification=classification,
            effects_applied=effects_applied, accommodation=accommodation_result, narration=narration,
        )
        self.turn_logs.append(log)
        with self.turn_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(log), default=str) + "\n")


_WORD_RE = re.compile(r"\s+")


def _empty_step() -> dict:
    return {
        "log_entries": [], "triggered_event_id": None, "moved_to": None,
        "new_knowledge": [], "new_evidence_flags": {}, "characters_encountered": [],
        "characters_interviewed": [], "classification": "noop", "dm_entry": None,
        "game_over": False,
    }


def _truncate_input(raw: str) -> str:
    tokens = _WORD_RE.split(raw.strip())
    if len(tokens) > MAX_WORDS_PER_COMMAND:
        tokens = tokens[:MAX_WORDS_PER_COMMAND]
    return " ".join(tokens)
