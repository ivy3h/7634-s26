"""LLM-based parser that maps free-form user input into a structured action.

The parser returns a dict of the same shape as a plan `Event` (minus the id):
verb, args, location, preconditions, effects, reveals. The drama manager
then checks preconditions and either executes the action, flags it as an
exception, or triggers accommodation.

The parser *does not* know about causal links. Only about surface effects
it can infer from commonsense. The drama manager is the one that decides
whether those effects constitute a story exception.
"""
from __future__ import annotations

import json
from typing import Any

from llm_client import chat_json
from plan_types import Condition, Effect


PARSE_SYSTEM = """You are an action interpreter for a text adventure. Convert the player's short command into a structured action over the game world. Output only valid JSON. Never write explanation."""


PARSE_PROMPT = """World summary:
  Player location: {player_location}
  Adjacent locations: {adjacent}
  Objects here: {here_objects}
  Characters here: {here_characters}
  Known evidence ids: {evidence_ids}
  Player inventory: {inventory}
  Player knowledge snippets (partial): {knowledge_snippets}

Subject id conventions:
  detective                       the player
  character.<snake_name>
  location.<snake>
  evidence.<id>
  object.<snake>

Player's raw input (<=5 words expected, may exceed): "{raw}"

Produce this JSON:
{{
  "verb": "move | examine | interview | search | take | drop | use | analyze | confront | accuse | observe | wait | custom",
  "args": ["<subject_id or short phrase>", ...],
  "target_location": "<location id or empty string>",
  "preconditions": [{{"subject":"...", "attr":"...", "op":"==", "value":...}}, ...],
  "effects": [{{"subject":"...", "attr":"...", "op":"set|add|remove", "value":...}}, ...],
  "reveals": ["evidence.<id>", ...],
  "novel_state_vars": [
    {{"subject":"...", "attr":"...", "why":"short reason why this new state matters"}}, ...
  ],
  "plain_summary": "one-sentence description of what the player tries to do"
}}

Critical rules:
- For movement, set verb="move" and fill target_location with an adjacent id.
- For any physical manipulation that creates a NEW persistent state on an object
  (jamming, breaking, locking, hiding, destroying), list that state under
  "novel_state_vars" in addition to listing it in effects. This tells the
  drama manager the action introduced a state slot not previously modeled.
- If the player tries to "accuse" or "arrest" someone, add an effect updating
  detective.knowledge by adding "accused:<character.id>".
- Keep preconditions minimal — usually just where the player and target must be."""


def interpret_action(
    raw_input: str,
    world_summary: dict[str, Any],
) -> dict[str, Any]:
    """Parse one user command into a structured action dict.

    `world_summary` should include: player_location, adjacent (list of ids),
    here_objects, here_characters, evidence_ids, inventory, knowledge_snippets.
    """
    prompt = PARSE_PROMPT.format(
        raw=raw_input.strip(),
        player_location=world_summary.get("player_location", ""),
        adjacent=world_summary.get("adjacent", []),
        here_objects=world_summary.get("here_objects", []),
        here_characters=world_summary.get("here_characters", []),
        evidence_ids=world_summary.get("evidence_ids", []),
        inventory=world_summary.get("inventory", []),
        knowledge_snippets=world_summary.get("knowledge_snippets", [])[-6:],
    )
    try:
        parsed = chat_json(prompt, system=PARSE_SYSTEM, max_tokens=700, temperature=0.2)
    except (ValueError, json.JSONDecodeError):
        parsed = {
            "verb": "custom",
            "args": [raw_input.strip()],
            "target_location": "",
            "preconditions": [],
            "effects": [],
            "reveals": [],
            "novel_state_vars": [],
            "plain_summary": raw_input.strip(),
        }
    parsed.setdefault("novel_state_vars", [])
    parsed["_raw"] = raw_input
    return parsed


def structured_preconditions(parsed: dict[str, Any]) -> list[Condition]:
    out: list[Condition] = []
    for pc in parsed.get("preconditions", []):
        try:
            out.append(Condition.from_dict(pc))
        except Exception:  # noqa: BLE001 — malformed LLM output, skip this precondition only
            continue
    return out


def structured_effects(parsed: dict[str, Any]) -> list[Effect]:
    out: list[Effect] = []
    for ef in parsed.get("effects", []):
        try:
            out.append(Effect.from_dict(ef))
        except Exception:  # noqa: BLE001 — malformed LLM output, skip this effect only
            continue
    return out
