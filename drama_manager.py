"""Intervention and Accommodation drama manager (Template 2).

Classifies every user action as constituent / consistent / exceptional and
repairs the plan when exceptions occur.

Key design points:

1. **Open-action problem**. The interpreter may produce effects on state
   variables that were never in the original plan (e.g. jam door with
   chair). Commonsense says those still threaten future events — a blocked
   door blocks anyone who must pass through it, even if no pre-declared
   causal link mentions "jammed". We solve this by running a *commonsense
   threat query* for every user action: we ask the LLM, given the effects
   and the remaining plan events, whether any future event would become
   impossible under normal physical/social reasoning. If yes, the action
   is exceptional even when no hard causal link is broken.

2. **Accommodation**. When something is exceptional, we remove the events
   that are no longer reachable and the links that depended on them, then
   ask the LLM to generate replacement events that re-establish the goal
   preconditions from the current world state. Replacement events are
   constrained to the existing world (characters + locations); they may
   modify *unrevealed* crime details but not the original crime outcome.

3. **Inspectable logs**. Every decision — classification, threat query,
   removed events, replacement events — is appended to an in-memory log
   and a JSONL file, so the final video can scroll the log for evidence of
   behind-the-scenes behavior.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_client import chat_json
from plan_types import CausalLink, Condition, Effect, Event, Plan


# ---------------------------------------------------------------------------
# Log entry
# ---------------------------------------------------------------------------
@dataclass
class LogEntry:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.payload}


# ---------------------------------------------------------------------------
# Commonsense threat / accommodation prompts
# ---------------------------------------------------------------------------
THREAT_SYSTEM = """You are a commonsense reasoner for an interactive detective story.
You assess whether a just-performed action makes future plan events impossible.
Output only valid JSON."""


THREAT_PROMPT = """User action (structured):
{parsed_action}

Remaining plan events (summarized; id, verb, args, location, key preconditions):
{remaining_events}

Active causal link conditions (must hold until the listed consumer event):
{active_links}

Question: does the user's action render any remaining event impossible or
meaningfully harder for a realistic actor (detective or NPC) to perform,
even if no hard causal link is formally broken?

Consider physical blocks ("jammed", "locked permanently", "broken"),
social blocks ("witness dead", "suspect fled the country"), and epistemic
blocks ("evidence destroyed").

Output JSON:
{{
  "threatened_events": [
    {{"event_id": "<id>",
      "reason": "short commonsense reason",
      "repairable": true|false}},
    ...
  ],
  "overall_classification_hint": "constituent | consistent | exceptional"
}}"""


ACCOMMODATION_SYSTEM = """You are a partial-order-plan repair agent. You output only valid JSON.
Your job: given a broken detective plan and the current world state, propose
replacement events that restore the detective's path to the goal. You may
modify unrevealed details of the crime but never its core outcome."""


ACCOMMODATION_PROMPT = """Current world snapshot (relevant slice): {world_snapshot}

Goal conditions: {goal}

Events just removed (unreachable): {removed_events}
Active causal links still standing: {surviving_links}
Characters present in the world: {characters}
Locations in the world: {locations}
Evidence still not destroyed: {live_evidence}

Produce replacement events that keep the story moving toward the goal.
Output JSON:
{{
  "replacement_events": [
    {{"id": "R01",
      "verb": "interview|examine|search|analyze|consult|observe|confront|reconstruct",
      "args": ["<subject id>", ...],
      "location": "location.<snake>",
      "preconditions": [{{"subject":"...", "attr":"...", "op":"==", "value":...}}, ...],
      "effects": [{{"subject":"...", "attr":"...", "op":"set|add|remove", "value":...}}, ...],
      "reveals": ["evidence.<id>", ...],
      "description": "one sentence the detective would say while doing this",
      "narrative": "one short paragraph of prose"
    }},
    ...
  ],
  "rationale": "2-3 sentences explaining how these events repair the plan"
}}

Constraints:
- Do NOT reveal the real criminal directly; preserve mystery.
- Prefer 1-3 replacement events (not a whole new plan).
- Each effect should mirror the contract of the removed events where possible,
  so the goal remains reachable."""


# ---------------------------------------------------------------------------
# Drama manager
# ---------------------------------------------------------------------------
class DramaManager:
    def __init__(self, plan: Plan, log_path: str | Path = "logs/drama.jsonl") -> None:
        self.plan = plan
        self.executed: list[str] = []
        self.remaining: list[str] = sorted(plan.events.keys())
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log: list[LogEntry] = []
        self._next_repair_idx = 1

    # ---- logging ---------------------------------------------------------
    def _log(self, kind: str, **payload: Any) -> None:
        entry = LogEntry(kind=kind, payload=payload)
        self.log.append(entry)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")

    # ---- active causal links ---------------------------------------------
    def active_links(self) -> list[CausalLink]:
        """Links whose producer has fired but whose consumer has not."""
        out: list[CausalLink] = []
        for cl in self.plan.causal_links:
            if cl.producer in self.executed and cl.consumer not in self.executed and cl.consumer in self.remaining:
                out.append(cl)
        return out

    # ---- classification --------------------------------------------------
    def classify(
        self,
        parsed_action: dict[str, Any],
        state: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Return {classification, matched_event_id|None, threats, details}."""
        constituent_match = self._find_constituent_match(parsed_action)
        proposed_effects = [Effect.from_dict(e) for e in parsed_action.get("effects", []) if isinstance(e, dict)]
        hard_violations = self._hard_violations(proposed_effects)

        # Commonsense threat check — only needed if not already an obvious
        # hard violation, because the LLM call is expensive.
        soft_threats: list[dict[str, Any]] = []
        if not hard_violations and (proposed_effects or parsed_action.get("novel_state_vars")):
            soft_threats = self._commonsense_threats(parsed_action)

        classification = self._decide_classification(constituent_match, hard_violations, soft_threats)

        result = {
            "classification": classification,
            "matched_event_id": constituent_match,
            "hard_violations": [cl.to_dict() for cl in hard_violations],
            "soft_threats": soft_threats,
        }
        self._log(
            "classification",
            action=parsed_action,
            classification=classification,
            matched_event_id=constituent_match,
            hard_violations=[cl.to_dict() for cl in hard_violations],
            soft_threats=soft_threats,
        )
        return result

    def _find_constituent_match(self, parsed_action: dict[str, Any]) -> str | None:
        """Return the first remaining event whose verb + args substantially
        overlap with the parsed action."""
        raw = (parsed_action.get("_raw") or "").lower()
        parsed_verb = parsed_action.get("verb", "").lower()
        parsed_args_lower = {str(a).lower() for a in parsed_action.get("args", [])}
        for eid in self.remaining:
            ev = self.plan.events[eid]
            # Skip if the plan event requires a specific location the player
            # isn't at — that's clearly not this action.
            if parsed_verb and ev.verb.lower() != parsed_verb and not raw:
                continue
            ev_args_lower = {str(a).lower() for a in ev.args}
            if parsed_args_lower & ev_args_lower:
                return eid
            # Fallback: description substring match on the raw input.
            if raw and any(tok in ev.description.lower() for tok in raw.split() if len(tok) > 3):
                return eid
        return None

    def _hard_violations(self, proposed_effects: list[Effect]) -> list[CausalLink]:
        """Which active causal links are negated by these effects?"""
        violated: list[CausalLink] = []
        for cl in self.active_links():
            for ef in proposed_effects:
                if self._effect_negates_condition(ef, cl.condition):
                    violated.append(cl)
                    break
        return violated

    @staticmethod
    def _effect_negates_condition(effect: Effect, condition: Condition) -> bool:
        if effect.subject != condition.subject or effect.attr != condition.attr:
            return False
        if condition.op == "==" and effect.op == "set":
            return effect.value != condition.value
        if condition.op == "!=" and effect.op == "set":
            return effect.value == condition.value
        if condition.op == "contains" and effect.op == "remove":
            return effect.value == condition.value
        if condition.op == "not_contains" and effect.op == "add":
            return effect.value == condition.value
        return False

    def _commonsense_threats(self, parsed_action: dict[str, Any]) -> list[dict[str, Any]]:
        remaining_summary = [
            {
                "id": eid,
                "verb": self.plan.events[eid].verb,
                "args": self.plan.events[eid].args,
                "location": self.plan.events[eid].location,
                "preconditions": [pc.to_dict() for pc in self.plan.events[eid].preconditions],
            }
            for eid in self.remaining[:10]
        ]
        active = [cl.to_dict() for cl in self.active_links()]
        prompt = THREAT_PROMPT.format(
            parsed_action=json.dumps(parsed_action)[:1200],
            remaining_events=json.dumps(remaining_summary)[:2500],
            active_links=json.dumps(active)[:1500],
        )
        try:
            parsed = chat_json(prompt, system=THREAT_SYSTEM, max_tokens=600, temperature=0.2)
        except Exception as err:  # noqa: BLE001 — falling back is safer than crashing
            self._log("threat_query_failed", error=repr(err))
            return []
        return parsed.get("threatened_events", []) if isinstance(parsed, dict) else []

    @staticmethod
    def _decide_classification(
        constituent_match: str | None,
        hard_violations: list[CausalLink],
        soft_threats: list[dict[str, Any]],
    ) -> str:
        any_unrepairable_soft = any(not t.get("repairable", True) for t in soft_threats)
        if hard_violations or soft_threats:
            return "exceptional"
        if constituent_match:
            return "constituent"
        _ = any_unrepairable_soft  # reserved for future tiering
        return "consistent"

    # ---- execution -------------------------------------------------------
    def execute_constituent(self, event_id: str, state: dict[str, dict[str, Any]]) -> None:
        event = self.plan.events[event_id]
        for ef in event.effects:
            ef.apply(state)
        self.executed.append(event_id)
        if event_id in self.remaining:
            self.remaining.remove(event_id)
        self._log("executed_constituent", event_id=event_id)

    def apply_free_effects(
        self,
        parsed_action: dict[str, Any],
        state: dict[str, dict[str, Any]],
    ) -> None:
        for ef_dict in parsed_action.get("effects", []):
            try:
                Effect.from_dict(ef_dict).apply(state)
            except Exception:  # noqa: BLE001 — malformed LLM output, skip this effect only
                continue
        self._log("applied_free_effects", effects=parsed_action.get("effects", []))

    # ---- accommodation ---------------------------------------------------
    def accommodate(
        self,
        parsed_action: dict[str, Any],
        classification: dict[str, Any],
        state: dict[str, dict[str, Any]],
        world_locations: list[str],
        characters: list[str],
    ) -> dict[str, Any]:
        """Remove unreachable events + generate replacements."""
        threatened_ids = {t["event_id"] for t in classification.get("soft_threats", []) if "event_id" in t}
        for cl_dict in classification.get("hard_violations", []):
            threatened_ids.add(cl_dict.get("consumer"))
        threatened_ids.discard(None)

        removed_events: list[str] = []
        for eid in list(self.remaining):
            if eid in threatened_ids:
                self.remaining.remove(eid)
                removed_events.append(eid)
        self.plan.causal_links = [
            cl for cl in self.plan.causal_links
            if cl.producer in self.remaining or cl.producer in self.executed
            if cl.consumer in self.remaining
        ]

        # Ask the LLM for replacement events. Keep world snapshot lean.
        live_evidence = [k for k, v in state.items() if k.startswith("evidence.") and not v.get("destroyed", False)]
        prompt = ACCOMMODATION_PROMPT.format(
            world_snapshot=json.dumps(_compact_state(state))[:2500],
            goal=json.dumps([c.to_dict() for c in self.plan.goal]),
            removed_events=json.dumps([self.plan.events[eid].to_dict() for eid in removed_events if eid in self.plan.events])[:2000],
            surviving_links=json.dumps([cl.to_dict() for cl in self.plan.causal_links])[:1500],
            characters=json.dumps(characters),
            locations=json.dumps(world_locations),
            live_evidence=json.dumps(live_evidence),
        )
        try:
            parsed = chat_json(prompt, system=ACCOMMODATION_SYSTEM, max_tokens=1500, temperature=0.4)
        except Exception as err:  # noqa: BLE001 — worst case: skip repair and let goal remain reachable by other paths
            self._log("accommodation_failed", error=repr(err), removed_events=removed_events)
            return {"removed_events": removed_events, "replacement_events": [], "rationale": "repair failed"}

        replacement_dicts = parsed.get("replacement_events", []) if isinstance(parsed, dict) else []
        replacements: list[Event] = []
        for rd in replacement_dicts:
            rid = f"R{self._next_repair_idx:02d}"
            self._next_repair_idx += 1
            try:
                ev = Event(
                    id=rid,
                    actor="detective",
                    verb=rd.get("verb", "act"),
                    args=list(rd.get("args", [])),
                    location=rd.get("location", "location.unknown"),
                    preconditions=[Condition.from_dict(c) for c in rd.get("preconditions", [])],
                    effects=[Effect.from_dict(e) for e in rd.get("effects", [])],
                    reveals=list(rd.get("reveals", [])),
                    description=rd.get("description", ""),
                    narrative=rd.get("narrative", ""),
                )
            except Exception:  # noqa: BLE001 — skip malformed replacement, continue with others
                continue
            self.plan.events[rid] = ev
            self.remaining.append(rid)
            replacements.append(ev)

        self._log(
            "accommodation",
            removed_events=removed_events,
            replacement_event_ids=[e.id for e in replacements],
            rationale=parsed.get("rationale", "") if isinstance(parsed, dict) else "",
        )
        return {
            "removed_events": removed_events,
            "replacement_events": [e.to_dict() for e in replacements],
            "rationale": parsed.get("rationale", "") if isinstance(parsed, dict) else "",
        }

    # ---- goal check ------------------------------------------------------
    def goal_satisfied(self, state: dict[str, dict[str, Any]]) -> bool:
        return all(c.evaluate(state) for c in self.plan.goal)


def _compact_state(state: dict[str, dict[str, Any]], max_chars: int = 2000) -> dict[str, Any]:
    """Skip bulky fields so the LLM prompt doesn't blow past context."""
    compact: dict[str, Any] = {}
    for sid, slots in state.items():
        if sid.startswith("evidence.") or sid == "detective" or sid.startswith("character."):
            compact[sid] = slots
        else:
            compact[sid] = {k: v for k, v in slots.items() if k not in {"description"}}
    text = json.dumps(compact)
    if len(text) > max_chars:
        compact = {sid: slots for sid, slots in compact.items() if sid == "detective" or sid.startswith("evidence.")}
    return compact
