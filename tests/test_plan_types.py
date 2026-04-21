"""Minimal unit tests that don't need a live LLM."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drama_manager import DramaManager
from plan_types import CausalLink, Condition, Effect, Event, Plan


def test_condition_and_effect_roundtrip() -> None:
    state: dict[str, dict] = {}
    Effect("door.library", "locked", "set", True).apply(state)
    assert Condition("door.library", "locked", "==", True).evaluate(state)
    Effect("door.library", "locked", "set", False).apply(state)
    assert Condition("door.library", "locked", "==", False).evaluate(state)

    Effect("detective", "knowledge", "add", "saw_pen").apply(state)
    Effect("detective", "knowledge", "add", "saw_glass").apply(state)
    assert Condition("detective", "knowledge", "contains", "saw_pen").evaluate(state)
    assert not Condition("detective", "knowledge", "contains", "nope").evaluate(state)


def test_plan_json_roundtrip() -> None:
    ev = Event(
        id="E00", actor="detective", verb="examine",
        args=["evidence.E01"], location="location.gallery",
        preconditions=[Condition("detective", "location", "==", "location.gallery")],
        effects=[Effect("detective", "knowledge", "add", "saw_pen")],
        reveals=["evidence.E01"],
        description="examine the pen",
    )
    plan = Plan(events={ev.id: ev}, order=[], causal_links=[], initial_state={}, goal=[])
    rt = Plan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert rt.events["E00"].verb == "examine"
    assert rt.events["E00"].preconditions[0].subject == "detective"


def test_hard_violation_detection(tmp_path) -> None:
    """If the player destroys an item that a later event requires to exist,
    the drama manager should flag a hard violation."""
    e1 = Event(
        id="E00", actor="detective", verb="examine",
        args=["evidence.E01"], location="location.gallery",
        preconditions=[],
        effects=[Effect("evidence.E01", "discovered", "set", True)],
    )
    e2 = Event(
        id="E01", actor="detective", verb="analyze",
        args=["evidence.E01"], location="location.lab",
        preconditions=[Condition("evidence.E01", "destroyed", "==", False)],
        effects=[],
    )
    plan = Plan(
        events={"E00": e1, "E01": e2},
        order=[("E00", "E01")],
        causal_links=[
            CausalLink(producer="E00", consumer="E01",
                       condition=Condition("evidence.E01", "destroyed", "==", False)),
        ],
        initial_state={"evidence.E01": {"destroyed": False}},
        goal=[],
    )
    dm = DramaManager(plan, log_path=tmp_path / "d.jsonl")
    dm.executed.append("E00"); dm.remaining.remove("E00")
    state = {"evidence.E01": {"destroyed": False}}
    parsed = {
        "verb": "destroy",
        "args": ["evidence.E01"],
        "effects": [{"subject": "evidence.E01", "attr": "destroyed", "op": "set", "value": True}],
        "novel_state_vars": [],
        "_raw": "destroy the pen",
    }
    violations = dm._hard_violations([Effect.from_dict(e) for e in parsed["effects"]])
    assert len(violations) == 1
    assert violations[0].consumer == "E01"


if __name__ == "__main__":
    test_condition_and_effect_roundtrip()
    test_plan_json_roundtrip()
    import tempfile
    test_hard_violation_detection(Path(tempfile.mkdtemp()))
    print("All tests passed.")
