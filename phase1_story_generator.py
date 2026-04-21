"""Phase I story generator — ported from Phase_I_Final_Story_Generator.ipynb.

All Claude API calls replaced with `llm_client.chat_simple` / `chat_json`.
Prompts preserved verbatim except where Claude-specific assumptions had to
be loosened for the Qwen chat template (see inline notes).

Usage:
    from phase1_story_generator import generate_full_story
    artifacts = generate_full_story("A poisoning murder at a 1920s London gallery")
    # artifacts["case_file"], ["complexities"], ["plot_points"], ["story_bible"], ["story_md"]
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from llm_client import chat_json, chat_simple


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(data: Any, filename: str | Path) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}")


def load_checkpoint(filename: str | Path) -> Any:
    return json.loads(Path(filename).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stage 1: case file
# ---------------------------------------------------------------------------
CASE_FILE_PROMPT = """You are a crime story architect. Generate a detailed murder mystery case file.
Output ONLY valid JSON, no markdown fences, no explanation.

{
  "criminal": {"name": str, "motive": str, "means": str, "opportunity": str},
  "victim": {"name": str, "background": str},
  "conspirators": [{"name": str, "role": str, "alibi": str}],
  "suspects": [{"name": str, "motive": str, "alibi": str}],
  "evidence": [
    {"id": str, "type": "physical|digital|testimonial",
     "description": str, "real_meaning": str, "steps_to_uncover": 2}
  ],
  "crime_timeline": [{"time": str, "event": str}],
  "solving_timeline": [
    {"step": int, "action": str, "target_evidence": [str], "max_actions": 3}
  ],
  "detective": {"name": str, "personal_stake": str, "deadline": str, "dire_consequence": str}
}

Generate at least 3 conspirators, 4 suspects, 8 evidence items, 6 solving steps."""


def generate_case_file(user_prompt: str) -> dict[str, Any]:
    full = f"{CASE_FILE_PROMPT}\n\nCrime context: {user_prompt}"
    return chat_json(full, max_tokens=3000, temperature=0.7)


# ---------------------------------------------------------------------------
# Stage 2: cover narrative / complexities
# ---------------------------------------------------------------------------
COMPLEXITIES_PROMPT = """Given real crime facts, generate a fabricated cover narrative.
Output ONLY valid JSON.

{
  "fake_suspect": {"name": str, "framing_reason": str},
  "planted_evidence": [{"description": str, "points_to": "fake_suspect"}],
  "false_testimonies": [{"witness": str, "claim": str}],
  "fake_timeline": [{"time": str, "event": str}],
  "evidence_fabrications": {"<evidence_id>": "fabricated explanation"},
  "conspirator_alibis": {"<name>": "alibi story"}
}

Rules:
- fake_suspect must NOT be the real criminal
- Every evidence id must appear in evidence_fabrications
- Alibis must be internally consistent"""


def generate_complexities(case_file: dict[str, Any]) -> dict[str, Any]:
    facts_summary = json.dumps(
        {
            "criminal": case_file["criminal"]["name"],
            "crime_timeline": case_file["crime_timeline"],
            "evidence_ids": [e["id"] for e in case_file["evidence"]],
            "conspirators": [c["name"] for c in case_file["conspirators"]],
        }
    )
    prompt = f"{COMPLEXITIES_PROMPT}\n\nReal facts: {facts_summary}"
    return chat_json(prompt, max_tokens=2000, temperature=0.7)


# ---------------------------------------------------------------------------
# Stage 3: meta-controller (plot points)
# ---------------------------------------------------------------------------
class StoryState:
    def __init__(self, case_file: dict[str, Any], complexities: dict[str, Any], max_points: int = 18) -> None:
        self.countdown = max_points + 3
        self.plot_points: list[dict[str, Any]] = []
        self.action_history: list[str] = []
        self.success_prob = 1.0
        self.evidence_progress = {e["id"]: 0 for e in case_file["evidence"]}
        self.alibi_status = {c["name"]: "unverified" for c in case_file["conspirators"]}
        self.closed_paths: set[str] = set()
        self.milestones_completed: set[int] = set()
        self.case_file = case_file
        self.complexities = complexities

    def tick(self) -> None:
        self.countdown -= 1
        self.success_prob = max(0.05, self.success_prob - 0.01)

    def is_done(self, min_points: int = 15) -> bool:
        all_milestones = len(self.milestones_completed) >= len(self.case_file["solving_timeline"])
        return all_milestones and len(self.plot_points) >= min_points


def _collision_detect(action: str, case_file: dict[str, Any]) -> dict[str, Any]:
    action_lower = action.lower()
    investigative_verbs = [
        "interview", "question", "investigate", "follow",
        "confront", "check", "ask", "visit", "examine", "search",
    ]
    for conspirator in case_file["conspirators"]:
        name_parts = conspirator["name"].lower().split()
        if any(part in action_lower for part in name_parts):
            if any(v in action_lower for v in investigative_verbs):
                return {"collision": True, "type": "conspirator", "target": conspirator["name"]}
    for evidence in case_file["evidence"]:
        keywords = set(evidence["description"].lower().split()) - {"the", "a", "an", "of", "in", "at", "to"}
        action_words = set(action_lower.split())
        if len(keywords & action_words) >= 2:
            return {"collision": True, "type": "evidence", "target": evidence["id"]}
    return {"collision": False, "type": None, "target": None}


def _check_extra_requirements(alibi_checks_done, multistep_clues_done, case_file):
    all_suspects = [s["name"] for s in case_file["suspects"]]
    alibis_covered = all(s in alibi_checks_done for s in all_suspects)
    multistep_done = sum(1 for v in multistep_clues_done.values() if v >= 2)
    return alibis_covered and multistep_done >= 3


def _decide_plot_type(state, alibi_checks_done, multistep_clues_done, case_file, iteration):
    suspects = [s["name"] for s in case_file["suspects"]]
    unchecked_suspects = [s for s in suspects if s not in alibi_checks_done]
    unfinished_clues = [eid for eid, steps in multistep_clues_done.items() if 0 < steps < 2]
    unstarted_clues = [eid for eid, steps in multistep_clues_done.items() if steps == 0]
    if unchecked_suspects and iteration % 3 == 1:
        return "alibi_check"
    if unfinished_clues:
        return "clue_followup"
    if unstarted_clues and iteration % 4 == 0:
        return "clue_start"
    if state.success_prob < 0.4:
        return "obstacle"
    return "progress"


def _generate_action(state, milestone, plot_type, case_file, complexities,
                     alibi_checks, multistep_clues, story_bible):
    suspects = case_file["suspects"]
    unchecked = [s for s in suspects if s["name"] not in alibi_checks]
    unfinished = [eid for eid, v in multistep_clues.items() if 0 < v < 2]
    unstarted = [eid for eid, v in multistep_clues.items() if v == 0]

    constraint = (
        f"CASE: Murder of {story_bible['victim_name']}.\n"
        f"Detective: {story_bible['detective_name']} and partner {story_bible['partner_name']}.\n"
        "All actions must relate to THIS murder case only. No hospitals, no unrelated crimes."
    )

    if plot_type == "alibi_check":
        target = unchecked[0] if unchecked else suspects[0]
        target_name = target["name"] if isinstance(target, dict) else target
        prompt = f"""{constraint}

Generate a detective action where the detective investigates the alibi of suspect: {target_name}
The action should involve contacting witnesses, checking records, or visiting locations related to the murder.
Recent actions (avoid repeating): {state.action_history[-3:]}
Output ONLY the action description (2-3 sentences)."""

    elif plot_type == "clue_followup":
        eid = unfinished[0]
        evidence = next((e for e in case_file["evidence"] if e["id"] == eid), None)
        prompt = f"""{constraint}

Generate a detective action that is a FOLLOW-UP step on this evidence from the murder scene:
Evidence: {evidence['description'] if evidence else eid}
This is step 2. The detective digs deeper (lab analysis, expert consultation, cross-referencing).
Output ONLY the action description (2-3 sentences)."""

    elif plot_type == "clue_start":
        eid = unstarted[0] if unstarted else list(multistep_clues.keys())[0]
        evidence = next((e for e in case_file["evidence"] if e["id"] == eid), None)
        prompt = f"""{constraint}

Generate a detective action discovering this murder evidence for the first time:
Evidence: {evidence['description'] if evidence else eid}
Step 1: detective notices something unusual but does NOT fully understand it yet.
Output ONLY the action description (2-3 sentences)."""
    else:
        prompt = f"""{constraint}

Generate a detective action for the murder investigation.
Current milestone: {milestone['action']}
Plot type: {plot_type}
Recent actions (avoid repeating): {state.action_history[-3:]}
Time pressure: {state.countdown} steps remaining
Output ONLY the action description (2-3 sentences)."""

    return chat_simple(prompt, max_tokens=150, temperature=0.8)


def _generate_narrative(action, collision, state, plot_type, case_file,
                        complexities, alibi_checks, multistep_clues, story_bible):
    fake_suspect = story_bible["fake_suspect"]
    constraint_header = f"""STRICT RULES - NEVER VIOLATE:
- This story is ONLY about the murder of: {story_bible['victim_name']}
- Detective: {story_bible['detective_name']} and partner {story_bible['partner_name']} (names never change)
- Real criminal (DO NOT reveal): {story_bible['real_criminal']}
- Fake suspect being framed: {fake_suspect} (names never change)
- Conspirators (exact names only): {story_bible['conspirator_names']}
- Key evidence: {story_bible['key_evidence']}
- Murder method: {story_bible['murder_method']}
- FORBIDDEN: Do NOT introduce hospitals, psychiatric wards, ambulances,
  unrelated victims, international crime networks, or any new murder cases.
- All red herrings must relate ONLY to the original murder of {story_bible['victim_name']}.
- Character names must NEVER change between chapters.

"""
    if plot_type == "alibi_check":
        instruction = (
            f"Write an alibi verification scene about the murder of {story_bible['victim_name']}. "
            "The detective checks a suspect's alibi for the night of the murder. "
            "A conspirator subtly provides false confirmation. "
            f"The detective ends up misled, suspicion points toward {fake_suspect}."
        )
    elif plot_type in ("clue_start", "clue_followup"):
        step = "initial discovery" if plot_type == "clue_start" else "deeper investigation"
        instruction = (
            f"Write a multi-step clue investigation ({step}) about the murder of {story_bible['victim_name']}. "
            "The detective examines physical evidence from the murder scene. "
            "Partial findings raise more questions. Do NOT reveal the full truth yet. "
            f"The clue should hint toward {fake_suspect} being guilty (misleadingly)."
        )
    elif collision["collision"]:
        instruction = (
            "Write a conspirator intervention scene. "
            f"A conspirator from the murder of {story_bible['victim_name']} smoothly misdirects the detective. "
            f"Detective almost gets close to real criminal {story_bible['real_criminal']}, then gets redirected toward {fake_suspect}."
        )
    elif plot_type == "obstacle":
        instruction = (
            f"Write an obstacle scene in the murder investigation of {story_bible['victim_name']}. "
            "A clue goes cold or a witness recants their statement about the murder. "
            "Detective frustrated, time running out to solve THIS case."
        )
    else:
        instruction = (
            f"Write a progress scene in the murder investigation of {story_bible['victim_name']}. "
            f"Small discovery points toward {fake_suspect} (the wrong person). "
            f"Dramatic irony: reader knows {story_bible['real_criminal']} is the real killer."
        )
    prompt = f"""{constraint_header}Write a suspenseful mystery plot point (2-3 paragraphs, 3rd person).
Detective's action: {action}
Writing instruction: {instruction}
Output ONLY the narrative paragraphs. Literary prose with dialogue."""
    return chat_simple(prompt, max_tokens=400, temperature=0.85)


def _update_tracking(plot_type, action, alibi_checks, multistep_clues, case_file):
    action_lower = action.lower()
    for suspect in case_file["suspects"]:
        name_parts = suspect["name"].lower().split()
        if any(p in action_lower for p in name_parts):
            if "alibi" in action_lower or plot_type == "alibi_check":
                if suspect["name"] not in alibi_checks:
                    alibi_checks[suspect["name"]] = "checked_false"
    for evidence in case_file["evidence"]:
        keywords = set(evidence["description"].lower().split()) - {"the", "a", "an", "of", "in"}
        if len(keywords & set(action_lower.split())) >= 2 or plot_type in ("clue_start", "clue_followup"):
            if plot_type == "clue_start" and multistep_clues.get(evidence["id"], 0) == 0:
                multistep_clues[evidence["id"]] = 1
                break
            if plot_type == "clue_followup" and multistep_clues.get(evidence["id"], 0) == 1:
                multistep_clues[evidence["id"]] = 2
                break


def _build_story_bible(case_file: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_summary": f"Murder of {case_file['victim']['name']}",
        "victim_name": case_file["victim"]["name"],
        "detective_name": case_file["detective"]["name"],
        "partner_name": "Detective Martinez",
        "real_criminal": case_file["criminal"]["name"],
        "fake_suspect": case_file.get("fake_suspect", {}).get("name", "unknown"),
        "conspirator_names": [c["name"] for c in case_file["conspirators"]],
        "suspect_names": [s["name"] for s in case_file["suspects"]],
        "key_evidence": [e["description"] for e in case_file["evidence"][:3]],
        "murder_method": case_file["criminal"]["means"],
        "murder_location": case_file["victim"]["background"],
    }


def run_meta_controller(
    case_file: dict[str, Any],
    complexities: dict[str, Any],
    story_bible: dict[str, Any],
    min_points: int = 20,
    max_iter: int = 60,
) -> list[dict[str, Any]]:
    state = StoryState(case_file, complexities)
    milestone_idx = 0
    solving_tl = case_file["solving_timeline"]
    alibi_checks_done: dict[str, str] = {}
    multistep_clues_done = {e["id"]: 0 for e in case_file["evidence"]}

    for iteration in range(max_iter):
        if state.is_done(min_points) and _check_extra_requirements(
            alibi_checks_done, multistep_clues_done, case_file
        ):
            break
        state.tick()
        current_milestone = solving_tl[min(milestone_idx, len(solving_tl) - 1)]
        plot_type = _decide_plot_type(
            state, alibi_checks_done, multistep_clues_done, case_file, iteration
        )
        action = _generate_action(
            state, current_milestone, plot_type, case_file, complexities,
            alibi_checks_done, multistep_clues_done, story_bible,
        )
        state.action_history.append(action)
        collision = _collision_detect(action, case_file)
        narrative = _generate_narrative(
            action, collision, state, plot_type, case_file, complexities,
            alibi_checks_done, multistep_clues_done, story_bible,
        )
        state.plot_points.append(
            {
                "action": action, "narrative": narrative, "collision": collision,
                "prob": state.success_prob, "plot_type": plot_type,
            }
        )
        if collision["collision"]:
            state.success_prob = max(0.05, state.success_prob - 0.08)
        elif plot_type == "progress":
            state.success_prob = min(1.0, state.success_prob + 0.02)
        _update_tracking(plot_type, action, alibi_checks_done, multistep_clues_done, case_file)
        actions_on_ms = sum(
            1 for a in state.action_history
            if any(w in a.lower() for w in current_milestone["action"].lower().split()[:2])
        )
        if actions_on_ms >= current_milestone.get("max_actions", 3):
            state.milestones_completed.add(milestone_idx)
            milestone_idx = min(milestone_idx + 1, len(solving_tl) - 1)
        print(
            f"  Plot #{len(state.plot_points):2d} [{plot_type:15s}] "
            f"| prob={state.success_prob:.0%} "
            f"| collision={'YES' if collision['collision'] else 'no'}"
        )
    return state.plot_points


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def generate_full_story(
    user_prompt: str = "A poisoning murder at a prestigious 1920s London art gallery opening",
    out_dir: str | Path = "data",
    min_points: int = 20,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating case file...")
    case_file = generate_case_file(user_prompt)
    save_checkpoint(case_file, out_dir / "case_file.json")

    print("Generating complexities...")
    complexities = generate_complexities(case_file)
    case_file["fake_suspect"] = complexities.get("fake_suspect", {})
    save_checkpoint(complexities, out_dir / "complexities.json")

    story_bible = _build_story_bible(case_file)
    save_checkpoint(story_bible, out_dir / "story_bible.json")

    print("Running meta-controller...")
    plot_points = run_meta_controller(case_file, complexities, story_bible, min_points=min_points)
    save_checkpoint(plot_points, out_dir / "plot_points.json")

    return {
        "case_file": case_file,
        "complexities": complexities,
        "story_bible": story_bible,
        "plot_points": plot_points,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="A poisoning murder at a prestigious 1920s London art gallery opening")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--min-points", type=int, default=20)
    args = parser.parse_args()
    generate_full_story(args.prompt, args.out_dir, args.min_points)
