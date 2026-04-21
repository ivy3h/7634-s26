"""Build a graph-based world from a plan.

For every distinct location referenced in plan events, we create a node.
Adjacency is decided by a commonsense LLM call: two locations are adjacent
if someone would reasonably walk between them without crossing a third
place. If two *consecutive* events occur in locations that wouldn't be
adjacent in the real world (bedroom → restaurant), we insert one or two
intermediate locations so the user can actually traverse.

We also populate each location with the characters and evidence the plan
expects to be there, and initialize the detective at the story's first
location.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_client import chat_json
from plan_types import Plan


@dataclass
class Location:
    id: str
    name: str
    description: str = ""
    adjacent: set[str] = field(default_factory=set)
    characters: set[str] = field(default_factory=set)
    evidence: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "adjacent": sorted(self.adjacent),
            "characters": sorted(self.characters),
            "evidence": sorted(self.evidence),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Location":
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            adjacent=set(d.get("adjacent", [])),
            characters=set(d.get("characters", [])),
            evidence=set(d.get("evidence", [])),
        )


@dataclass
class World:
    locations: dict[str, Location] = field(default_factory=dict)
    starting_location: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "locations": {lid: loc.to_dict() for lid, loc in self.locations.items()},
            "starting_location": self.starting_location,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "World":
        return cls(
            locations={lid: Location.from_dict(v) for lid, v in d["locations"].items()},
            starting_location=d.get("starting_location", ""),
        )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
LOCATION_DESCRIBE_PROMPT = """Describe this mystery-story location in 1-2 sentences of atmospheric detail.
Location id: {loc_id}
Context (story era/setting): {era}
Output JSON: {{"name": "...", "description": "..."}}"""


ADJACENCY_PROMPT = """You are a commonsense spatial reasoner for a 1920s London murder mystery.

Below is a list of locations that appear in the story, in the order the
detective must visit them. For each pair of *consecutive* locations, answer
whether they would naturally be adjacent (someone could walk directly from
one to the other without traversing a third distinct place) and if not,
propose 1-2 intermediate locations that bridge them.

Locations: {locations}

Output JSON:
{{
  "pairs": [
    {{"a": "<id>", "b": "<id>",
      "adjacent": true|false,
      "intermediates": ["location.<snake_case>", ...]}},
    ...
  ],
  "extra_intermediate_descriptions": {{
    "location.<id>": "short description",
    ...
  }}
}}

Rules:
- Include one entry per consecutive pair.
- If adjacent is true, intermediates must be [].
- Only invent intermediate locations when the pair clearly wouldn't be adjacent."""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def _describe_location(loc_id: str, era: str) -> dict[str, str]:
    try:
        return chat_json(
            LOCATION_DESCRIBE_PROMPT.format(loc_id=loc_id, era=era),
            max_tokens=200,
            temperature=0.6,
        )
    except Exception:
        pretty = loc_id.split(".", 1)[-1].replace("_", " ").title()
        return {"name": pretty, "description": f"A location in the case: {pretty}."}


def _analyze_adjacency(loc_ids: list[str]) -> dict[str, Any]:
    try:
        return chat_json(
            ADJACENCY_PROMPT.format(locations=loc_ids),
            max_tokens=1500,
            temperature=0.3,
        )
    except Exception as err:
        print(f"  [warn] adjacency prompt failed ({err!r}); falling back to linear chain")
        return {
            "pairs": [{"a": a, "b": b, "adjacent": True, "intermediates": []}
                      for a, b in zip(loc_ids[:-1], loc_ids[1:])],
            "extra_intermediate_descriptions": {},
        }


def build_world(plan: Plan, era: str = "1920s London") -> World:
    event_ids_in_order = sorted(plan.events.keys())
    loc_sequence: list[str] = []
    for eid in event_ids_in_order:
        loc = plan.events[eid].location or "location.unknown"
        loc_sequence.append(loc)

    # Dedupe while preserving order, but remember the sequential visits so
    # we can reason about consecutive-pair adjacency.
    unique_locs = list(dict.fromkeys(loc_sequence))
    print(f"Unique plan locations: {len(unique_locs)}")

    world = World()
    for lid in unique_locs:
        info = _describe_location(lid, era)
        world.locations[lid] = Location(id=lid, name=info.get("name", lid), description=info.get("description", ""))

    # Consecutive pairs along the event trajectory — these are the ones that
    # must be reachable by the player without teleporting.
    pairs: list[tuple[str, str]] = []
    for a, b in zip(loc_sequence[:-1], loc_sequence[1:]):
        if a != b and (a, b) not in pairs and (b, a) not in pairs:
            pairs.append((a, b))
    print(f"Consecutive location transitions to resolve: {len(pairs)}")

    ids_for_llm = list({lid for pair in pairs for lid in pair})
    adjacency = _analyze_adjacency(ids_for_llm) if ids_for_llm else {"pairs": [], "extra_intermediate_descriptions": {}}

    # Insert intermediates first so we can describe them before wiring edges.
    for mid_id, mid_desc in (adjacency.get("extra_intermediate_descriptions") or {}).items():
        if mid_id not in world.locations:
            pretty = mid_id.split(".", 1)[-1].replace("_", " ").title()
            world.locations[mid_id] = Location(id=mid_id, name=pretty, description=mid_desc)

    verdict_by_pair = {(p["a"], p["b"]): p for p in adjacency.get("pairs", []) if "a" in p and "b" in p}

    for a, b in pairs:
        verdict = verdict_by_pair.get((a, b)) or verdict_by_pair.get((b, a))
        if verdict and not verdict.get("adjacent", True):
            mids = [m for m in verdict.get("intermediates", []) if isinstance(m, str)]
            chain = [a, *mids, b]
            for mid in mids:
                if mid not in world.locations:
                    pretty = mid.split(".", 1)[-1].replace("_", " ").title()
                    world.locations[mid] = Location(id=mid, name=pretty, description=f"A passage between areas of the story.")
            for x, y in zip(chain[:-1], chain[1:]):
                world.locations[x].adjacent.add(y)
                world.locations[y].adjacent.add(x)
        else:
            world.locations[a].adjacent.add(b)
            world.locations[b].adjacent.add(a)

    # Seed contents: every event that reveals an evidence places that
    # evidence in the event's location; characters mentioned in args are
    # placed there too.
    for ev in plan.events.values():
        loc_id = ev.location
        if loc_id in world.locations:
            for rev in ev.reveals:
                world.locations[loc_id].evidence.add(rev)

    # Pin characters to the location of the first event that references them.
    for ev in plan.events.values():
        for arg in ev.args:
            if arg.startswith("character.") and ev.location in world.locations:
                # Only set if not already placed (first mention wins).
                already_placed = any(arg in loc.characters for loc in world.locations.values())
                if not already_placed:
                    world.locations[ev.location].characters.add(arg)

    world.starting_location = loc_sequence[0] if loc_sequence else next(iter(world.locations))
    if world.starting_location not in world.locations:
        # Unknown fallback — create a generic "case_briefing_room".
        world.locations["location.case_briefing_room"] = Location(
            id="location.case_briefing_room",
            name="Case Briefing Room",
            description="A quiet room where the detective sorts notes between outings.",
        )
        for lid in list(world.locations.keys()):
            if lid == "location.case_briefing_room":
                continue
            world.locations["location.case_briefing_room"].adjacent.add(lid)
            world.locations[lid].adjacent.add("location.case_briefing_room")
        world.starting_location = "location.case_briefing_room"

    # Also hook initial_state["detective"]["location"] to the start.
    plan.initial_state.setdefault("detective", {})["location"] = world.starting_location

    # Make sure every evidence entry has its "location" attr patched from the world.
    for loc in world.locations.values():
        for ev_id in loc.evidence:
            if ev_id in plan.initial_state:
                plan.initial_state[ev_id]["location"] = loc.id

    return world


def save_world(world: World, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(world.to_dict(), indent=2), encoding="utf-8")
    print(f"Saved: {p}")


def load_world(path: str | Path) -> World:
    return World.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


if __name__ == "__main__":
    import argparse

    from story_to_plan import load_plan

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="data/plan.json")
    parser.add_argument("--out", default="data/world.json")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    world = build_world(plan)
    save_world(world, args.out)
    # Persist the patched initial_state back into plan.json so game_engine
    # sees the resolved starting location.
    Path(args.plan).write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    print(f"Updated: {args.plan}")
