"""Convert Phase I output into a partially ordered plan with causal links.

Phase I gives us:
  - case_file.json: evidence[], conspirators[], suspects[], solving_timeline[]
  - plot_points.json: 20-ish {action, narrative, plot_type, collision}

Phase I does NOT give us preconditions / effects / locations. We reconstruct
them here, one event at a time, with a structured LLM call and then sanity
filter the output to our typed dataclasses. A second pass builds causal
links by pattern-matching producer effects to consumer preconditions.

Output: plan.json (dict matching Plan.to_dict()).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_client import chat_json
from plan_types import CausalLink, Condition, Effect, Event, Plan


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
EVENT_EXTRACTION_SYSTEM = """You are a plan engineer for an interactive mystery game. Your job is to convert a detective's free-text action into a structured plan event with explicit preconditions and effects over a shared world state.

Use these subject-id conventions:
  character.<snake_case_name>   e.g. character.victoria_harrington
  location.<snake_case>         e.g. location.gallery_main_hall
  object.<snake_case>           e.g. object.fountain_pen
  evidence.<evidence_id>        e.g. evidence.E03
  detective                     (the player; has attrs: location, knowledge, inventory)

Always output valid JSON. Never add explanation outside the JSON."""


EVENT_EXTRACTION_PROMPT = """Convert this detective action into a structured event.

Context:
  victim: {victim}
  detective: {detective}
  suspects: {suspects}
  conspirators: {conspirators}
  available evidence ids: {evidence_ids}

Plot point index: {idx}
Plot type: {plot_type}
Action (free text): {action}
Narrative excerpt (for context only, do not copy verbatim): {narrative_short}

Output JSON with exactly these keys:
{{
  "verb": "interview | examine | visit | search | analyze | consult | confront | observe | reconstruct",
  "args": [string, ...],
  "location": "location.<snake_case>",
  "preconditions": [
    {{"subject": "...", "attr": "...", "op": "==", "value": ...}}, ...
  ],
  "effects": [
    {{"subject": "...", "attr": "...", "op": "set|add|remove", "value": ...}}, ...
  ],
  "reveals": ["evidence.<id>", ...]
}}

Rules:
- At least ONE precondition should reference detective.location == "<location>".
- At least ONE effect should update detective.knowledge (op=add) with what the detective now knows.
- If the action examines or finds a physical object, include an effect on that object.
- If the action interviews a character, include a precondition that the character is available
  (character.<name>.available == true) and an effect updating that character's
  alibi_status or willingness_to_talk.
- Use lowercase snake_case for all subject ids. Keep effects minimal and specific.
- Do NOT invent evidence ids that are not in the provided list."""


CAUSAL_LINK_PROMPT = """You are validating causal links between plan events.

Producer event (E_i) — effects: {producer_effects}
Consumer event (E_j) — preconditions: {consumer_preconditions}

Question: for each consumer precondition, does a producer effect directly establish it?
Output JSON: {{"links": [{{"condition": <precondition object>, "established_by_producer": true|false}}]}}

Return ONLY the JSON."""


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------
def _slug(s: str) -> str:
    import re
    out = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return out or "unknown"


def _short(text: str, max_chars: int = 320) -> str:
    text = text.strip().replace("\n", " ")
    return text[:max_chars]


def extract_event_from_plot_point(
    idx: int,
    plot: dict[str, Any],
    case_file: dict[str, Any],
) -> Event:
    """Call the LLM once to infer structured event fields for one plot point."""
    ctx = {
        "victim": case_file["victim"]["name"],
        "detective": case_file["detective"]["name"],
        "suspects": [s["name"] for s in case_file["suspects"]],
        "conspirators": [c["name"] for c in case_file["conspirators"]],
        "evidence_ids": [e["id"] for e in case_file["evidence"]],
        "idx": idx,
        "plot_type": plot.get("plot_type", "progress"),
        "action": plot.get("action", ""),
        "narrative_short": _short(plot.get("narrative", "")),
    }
    prompt = EVENT_EXTRACTION_PROMPT.format(**ctx)
    try:
        parsed = chat_json(
            prompt,
            system=EVENT_EXTRACTION_SYSTEM,
            max_tokens=900,
            temperature=0.3,
        )
    except (ValueError, json.JSONDecodeError) as err:
        print(f"  [warn] event {idx} extraction failed ({err!r}); using fallback")
        parsed = _fallback_event_dict(plot)

    event_id = f"E{idx:02d}"
    preconditions = [Condition.from_dict(c) for c in parsed.get("preconditions", [])]
    effects = [Effect.from_dict(e) for e in parsed.get("effects", [])]
    # Guarantee the minimum contract: detective must be at the action location,
    # and the detective learns at least one thing from the event.
    location = parsed.get("location") or "location.unknown"
    if not any(pc.subject == "detective" and pc.attr == "location" for pc in preconditions):
        preconditions.append(Condition("detective", "location", "==", location))
    if not any(ef.subject == "detective" and ef.attr == "knowledge" for ef in effects):
        effects.append(Effect("detective", "knowledge", "add", f"learned_from_{event_id}"))

    return Event(
        id=event_id,
        actor="detective",
        verb=parsed.get("verb", "act"),
        args=list(parsed.get("args", [])),
        location=location,
        preconditions=preconditions,
        effects=effects,
        reveals=list(parsed.get("reveals", [])),
        description=plot.get("action", ""),
        narrative=plot.get("narrative", ""),
        source_plot_idx=idx,
    )


def _fallback_event_dict(plot: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fallback when the LLM returns unparseable JSON."""
    return {
        "verb": "act",
        "args": [],
        "location": "location.unknown",
        "preconditions": [],
        "effects": [],
        "reveals": [],
    }


# ---------------------------------------------------------------------------
# Causal link derivation
# ---------------------------------------------------------------------------
def _conditions_match(effect: Effect, precondition: Condition) -> bool:
    """Does this effect establish this precondition?"""
    if effect.subject != precondition.subject or effect.attr != precondition.attr:
        return False
    if precondition.op == "==" and effect.op == "set":
        return effect.value == precondition.value
    if precondition.op == "contains" and effect.op == "add":
        return effect.value == precondition.value
    if precondition.op == "!=" and effect.op == "set":
        return effect.value != precondition.value
    return False


def derive_causal_links(events: list[Event]) -> list[CausalLink]:
    """Walk events in order; for each consumer precondition, find the nearest
    prior event whose effect establishes it. That pair becomes a causal link.
    """
    links: list[CausalLink] = []
    for j, consumer in enumerate(events):
        for pc in consumer.preconditions:
            producer_id: str | None = None
            for i in range(j - 1, -1, -1):
                producer = events[i]
                if any(_conditions_match(ef, pc) for ef in producer.effects):
                    producer_id = producer.id
                    break
            if producer_id is not None:
                links.append(CausalLink(producer=producer_id, condition=pc, consumer=consumer.id))
    return links


# ---------------------------------------------------------------------------
# Initial state construction
# ---------------------------------------------------------------------------
def build_initial_state(case_file: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Seed world state with characters, evidence, and the detective.

    We don't yet know locations — world_builder.py will pin each character
    and evidence item to a starting location once the location graph exists.
    """
    state: dict[str, dict[str, Any]] = {}

    state["detective"] = {
        "location": "location.unknown",
        "knowledge": [],
        "inventory": [],
        "alive": True,
    }

    for s in case_file["suspects"]:
        sid = f"character.{_slug(s['name'])}"
        state[sid] = {
            "name": s["name"],
            "role": "suspect",
            "available": True,
            "alibi_status": "unverified",
            "alive": True,
            "willingness_to_talk": "neutral",
        }
    for c in case_file["conspirators"]:
        cid = f"character.{_slug(c['name'])}"
        state[cid] = {
            "name": c["name"],
            "role": "conspirator",
            "available": True,
            "alibi_status": "unverified",
            "alive": True,
            "willingness_to_talk": "evasive",
        }
    state[f"character.{_slug(case_file['victim']['name'])}"] = {
        "name": case_file["victim"]["name"],
        "role": "victim",
        "available": False,
        "alive": False,
    }

    for ev in case_file["evidence"]:
        eid = f"evidence.{ev['id']}"
        state[eid] = {
            "type": ev.get("type", "physical"),
            "description": ev.get("description", ""),
            "discovered": False,
            "analyzed": False,
            "destroyed": False,
            "location": "location.unknown",
        }

    return state


def build_goal(case_file: dict[str, Any]) -> list[Condition]:
    """Detective wins when they have identified the real criminal and linked
    the key evidence to that person (by name)."""
    real = _slug(case_file["criminal"]["name"])
    goal = [
        Condition("detective", "knowledge", "contains", f"identified:character.{real}"),
        Condition("detective", "knowledge", "contains", f"linked_evidence:character.{real}"),
    ]
    return goal


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def build_plan(
    case_file: dict[str, Any],
    plot_points: list[dict[str, Any]],
    out_path: str | Path | None = None,
) -> Plan:
    print(f"Extracting events from {len(plot_points)} plot points...")
    events: list[Event] = []
    for idx, plot in enumerate(plot_points):
        ev = extract_event_from_plot_point(idx, plot, case_file)
        events.append(ev)
        print(f"  {ev.id} verb={ev.verb:10s} loc={ev.location:32s} "
              f"pre={len(ev.preconditions)} eff={len(ev.effects)} reveals={ev.reveals}")

    print("Deriving causal links...")
    links = derive_causal_links(events)
    print(f"  {len(links)} causal links derived")

    order = [(events[i].id, events[i + 1].id) for i in range(len(events) - 1)]

    plan = Plan(
        events={ev.id: ev for ev in events},
        order=order,
        causal_links=links,
        initial_state=build_initial_state(case_file),
        goal=build_goal(case_file),
    )

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        print(f"Saved: {out}")

    return plan


def load_plan(path: str | Path) -> Plan:
    return Plan.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/plan.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    case_file = json.loads((data_dir / "case_file.json").read_text())
    plot_points = json.loads((data_dir / "plot_points.json").read_text())
    build_plan(case_file, plot_points, out_path=args.out)
