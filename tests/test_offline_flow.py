"""Offline end-to-end test with the LLM mocked out.

Stubs `llm_client.chat_simple` and `chat_json` with deterministic fake
responses so we can exercise story_to_plan → world_builder →
drama_manager → game_engine without a live server. This is NOT a
correctness test for the generated stories; it just verifies that the
pipeline wiring works.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import llm_client as _llm  # noqa: E402


# ---------------------------------------------------------------------------
# Fake LLM
# ---------------------------------------------------------------------------
def _fake_json(prompt: str, **kwargs):
    if "plan event with explicit preconditions" in prompt or "structured event" in prompt:
        return {
            "verb": "examine",
            "args": ["evidence.E01"],
            "location": "location.gallery_main_hall",
            "preconditions": [{"subject": "detective", "attr": "location", "op": "==", "value": "location.gallery_main_hall"}],
            "effects": [
                {"subject": "evidence.E01", "attr": "discovered", "op": "set", "value": True},
                {"subject": "detective", "attr": "knowledge", "op": "add", "value": "saw_pen"},
            ],
            "reveals": ["evidence.E01"],
        }
    if "mystery-story location" in prompt:
        return {"name": "Gallery Main Hall", "description": "Polished marble, champagne flutes glinting."}
    if "commonsense spatial reasoner" in prompt:
        return {"pairs": [], "extra_intermediate_descriptions": {}}
    if "action interpreter for a text adventure" in prompt or "Player's raw input" in prompt:
        raw = ""
        for line in prompt.splitlines():
            if "raw input" in line:
                raw = line.split(":", 1)[-1].strip().strip('"')
                break
        if "smash" in raw.lower() or "destroy" in raw.lower():
            return {
                "verb": "custom",
                "args": ["evidence.E01"],
                "target_location": "",
                "preconditions": [],
                "effects": [{"subject": "evidence.E01", "attr": "destroyed", "op": "set", "value": True}],
                "reveals": [],
                "novel_state_vars": [{"subject": "evidence.E01", "attr": "destroyed", "why": "physical destruction"}],
                "plain_summary": "player destroys the evidence",
            }
        return {
            "verb": "examine",
            "args": ["evidence.E01"],
            "target_location": "",
            "preconditions": [],
            "effects": [{"subject": "detective", "attr": "knowledge", "op": "add", "value": "examined_pen"}],
            "reveals": ["evidence.E01"],
            "novel_state_vars": [],
            "plain_summary": "player examines the pen",
        }
    if "commonsense reasoner" in prompt or "render any remaining event impossible" in prompt:
        # Only flag threats if destroyed evidence is in the action.
        if "destroyed" in prompt and "true" in prompt:
            return {"threatened_events": [{"event_id": "E01", "reason": "evidence destroyed", "repairable": True}],
                    "overall_classification_hint": "exceptional"}
        return {"threatened_events": [], "overall_classification_hint": "consistent"}
    if "replacement events" in prompt:
        return {
            "replacement_events": [{
                "verb": "interview",
                "args": ["character.witness"],
                "location": "location.gallery_main_hall",
                "preconditions": [],
                "effects": [{"subject": "detective", "attr": "knowledge", "op": "add", "value": "witness_said_pen"}],
                "reveals": [],
                "description": "interview a late-arriving witness who saw the pen",
                "narrative": "A trembling witness describes what they saw.",
            }],
            "rationale": "route around destroyed physical evidence via witness testimony",
        }
    return {}


def _fake_simple(prompt: str, **kwargs):
    return "The detective studies the scene, eyes narrow."


_llm.chat_json = _fake_json  # type: ignore[assignment]
_llm.chat_simple = _fake_simple  # type: ignore[assignment]

# Monkeypatch already-imported references in modules that did `from llm_client import ...`.
import story_to_plan as _stp  # noqa: E402
import world_builder as _wb  # noqa: E402
import action_interpreter as _ai  # noqa: E402
import drama_manager as _dm  # noqa: E402
import game_engine as _ge  # noqa: E402

_stp.chat_json = _fake_json
_wb.chat_json = _fake_json
_ai.chat_json = _fake_json
_dm.chat_json = _fake_json
_ge.chat_simple = _fake_simple


from game_engine import EngineConfig, GameEngine  # noqa: E402
from story_to_plan import build_plan  # noqa: E402
from world_builder import build_world  # noqa: E402


def _tiny_case_file():
    return {
        "criminal": {"name": "Victoria Harrington", "motive": "", "means": "", "opportunity": ""},
        "victim": {"name": "Lord Edmund Ashworth", "background": ""},
        "conspirators": [{"name": "Dr Fleming", "role": "", "alibi": ""}],
        "suspects": [{"name": "Victoria Harrington", "motive": "", "alibi": ""},
                     {"name": "Lord Pemberton", "motive": "", "alibi": ""}],
        "evidence": [
            {"id": "E01", "type": "physical", "description": "hollow fountain pen", "real_meaning": "", "steps_to_uncover": 2},
            {"id": "E02", "type": "physical", "description": "champagne flute", "real_meaning": "", "steps_to_uncover": 2},
        ],
        "crime_timeline": [{"time": "9pm", "event": "poison in glass"}],
        "solving_timeline": [
            {"step": 1, "action": "examine evidence", "target_evidence": ["E01"], "max_actions": 3},
        ],
        "detective": {"name": "Inspector Morgan", "personal_stake": "", "deadline": "", "dire_consequence": ""},
    }


def _tiny_plot_points():
    return [
        {"action": "examine the fountain pen", "narrative": "The pen is heavy.", "collision": {"collision": False, "type": None, "target": None}, "prob": 1.0, "plot_type": "clue_start"},
        {"action": "analyze the pen", "narrative": "Analysis of the pen.", "collision": {"collision": False, "type": None, "target": None}, "prob": 0.9, "plot_type": "clue_followup"},
    ]


def test_full_pipeline(tmp_path):
    plan = build_plan(_tiny_case_file(), _tiny_plot_points(), out_path=tmp_path / "plan.json")
    world = build_world(plan)
    assert plan.events, "plan has events"
    assert world.locations, "world has locations"
    assert world.starting_location in world.locations

    engine = GameEngine(plan, world, EngineConfig(log_dir=tmp_path / "logs", narrate_with_llm=False))

    inputs = iter(["examine pen", "smash the pen", "quit"])
    def fake_input(prompt: str) -> str:
        return next(inputs)

    outputs: list[str] = []
    def fake_echo(*args, **kwargs):
        outputs.append(" ".join(str(a) for a in args))

    status = engine.run(get_input=fake_input, echo=fake_echo)
    assert status in {"solved", "ended"}

    # Drama log should contain at least one classification and one accommodation.
    drama_lines = (tmp_path / "logs" / "drama.jsonl").read_text().splitlines()
    kinds = [json.loads(line)["kind"] for line in drama_lines]
    assert "classification" in kinds
    assert "accommodation" in kinds or any(k == "accommodation_failed" for k in kinds)
    print(f"Pipeline ran. kinds={kinds}")


if __name__ == "__main__":
    import tempfile
    test_full_pipeline(Path(tempfile.mkdtemp()))
    print("Offline end-to-end test PASSED.")
