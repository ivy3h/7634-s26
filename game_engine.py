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

        self.drama = DramaManager(plan, log_path=self.config.log_dir / "drama.jsonl")
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

    def _char_name(self, cid: str) -> str:
        return self.state.get(cid, {}).get("name", cid)

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

    # ---------------- helpers -----------------------------------------
    def _apply_movement_if_any(self, parsed: dict[str, Any]) -> None:
        target = parsed.get("target_location") or ""
        if not target:
            return
        cur_loc = self.state["detective"]["location"]
        cur = self.world.locations.get(cur_loc)
        if cur and target in cur.adjacent and target in self.world.locations:
            self.state["detective"]["location"] = target
            Effect("detective", "location", "set", target).apply(self.state)

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
        event_hint = ""
        if tag == "constituent":
            eid = classification.get("matched_event_id")
            if eid and eid in self.plan.events:
                event_hint = self.plan.events[eid].narrative[:600]
        prompt = (
            "Write ONE short paragraph (3-5 sentences) of 1920s-noir prose narrating the outcome "
            "of the detective's action. Do NOT reveal the real killer. Stay grounded in what "
            "changed in the world.\n\n"
            f"Action summary: {parsed.get('plain_summary', parsed.get('_raw', ''))}\n"
            f"Classification: {tag}\n"
            f"Effects applied (structured): {json.dumps(effects_applied)[:600]}\n"
            f"Location: {self.state['detective']['location']}\n"
            f"Prior plot reference (only if constituent): {event_hint}\n"
            + ("Replacement events introduced: " + json.dumps(accommodation_result.get("replacement_events", []))[:600]
               if accommodation_result else "")
        )
        try:
            return chat_simple(prompt, max_tokens=220, temperature=0.7).strip()
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


def _truncate_input(raw: str) -> str:
    tokens = _WORD_RE.split(raw.strip())
    if len(tokens) > MAX_WORDS_PER_COMMAND:
        tokens = tokens[:MAX_WORDS_PER_COMMAND]
    return " ".join(tokens)
