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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_client import chat_json
from plan_types import CausalLink, Condition, Effect, Event, Plan


# ---------------------------------------------------------------------------
# Verb synonym families & token helpers for constituent matching
# ---------------------------------------------------------------------------

# Each frozenset is a family of interchangeable verbs. If the plan event verb
# and the parsed verb land in the same family, they are treated as equivalent.
_VERB_FAMILIES: list[frozenset[str]] = [
    # physical inspection — examine / check / inspect / search / investigate all count
    frozenset({"examine", "check", "inspect", "observe", "look", "study",
               "review", "search", "investigate", "explore", "scour"}),
    # scientific analysis
    frozenset({"analyze", "analyse", "test", "process"}),
    # social interaction — question / interview / consult / confront / accuse / visit all count
    frozenset({"interview", "question", "talk", "speak", "ask",
               "consult", "confront", "accuse", "arrest", "charge", "visit", "meet"}),
    # movement
    frozenset({"move", "go", "walk", "travel", "head"}),
]

_STOP = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "are", "was",
    "has", "have", "been", "his", "her", "its", "into", "upon", "over",
    "under", "around", "within", "when", "then", "also", "only", "just",
})


def _verb_in_same_family(ev_verb: str, parsed_verb: str) -> bool:
    """Return True if two verbs belong to the same synonym family."""
    if ev_verb == parsed_verb:
        return True
    for family in _VERB_FAMILIES:
        if ev_verb in family and parsed_verb in family:
            return True
    return False


def _tok(text: str) -> set[str]:
    """Lowercase content tokens: ≥4 chars, not a stop word."""
    return {t for t in re.split(r"\W+", text.lower())
            if len(t) >= 4 and t not in _STOP}


# Verbs/keywords that are always exceptional for a detective — no LLM call
# needed to decide; the keyword match is sufficient.
_DESTRUCTIVE_VERBS: frozenset[str] = frozenset({
    "burn", "destroy", "smash", "break", "shatter", "tamper",
    "contaminate", "conceal", "steal", "forge", "alter", "corrupt",
    "shoot", "kill", "attack", "assault", "threaten", "bribe",
    "dispose", "hide", "cover",
})
_DESTRUCTIVE_RE = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in sorted(_DESTRUCTIVE_VERBS)) + r")\b",
    re.IGNORECASE,
)


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
the MINIMUM number of replacement events that restore the detective's path to
the goal. You may modify unrevealed crime details but never the core outcome.
CRITICAL RULES — violating any of these makes your output unusable:
1. ONLY use characters from the provided characters list. Never invent new people.
2. ONLY use locations from the provided locations list. Never invent new places.
3. Prefer 1 replacement event. Use 2 only if truly necessary. Never use 3+.
4. If the goal is already reachable through surviving events, output 0 replacements.
5. Each replacement must be immediately achievable by the detective with current resources."""


ACCOMMODATION_PROMPT = """Current world snapshot: {world_snapshot}

Goal conditions that must still be reachable: {goal}

Events just removed (unreachable due to exceptional action): {removed_events}
Surviving causal links: {surviving_links}

ALLOWED characters (use ONLY these exact ids): {characters}
ALLOWED locations (use ONLY these exact ids): {locations}
Undestroyed evidence: {live_evidence}

If the goal conditions are still satisfiable by the surviving events alone,
output zero replacement events (empty list). Otherwise produce the minimum
replacements needed — usually just 1.

Output JSON:
{{
  "replacement_events": [
    {{"id": "R01",
      "verb": "interview|examine|search|analyze|consult|confront",
      "args": ["<one of the ALLOWED character or evidence ids above>"],
      "location": "<one of the ALLOWED location ids above>",
      "preconditions": [],
      "effects": [{{"subject":"detective", "attr":"knowledge", "op":"add", "value":"<slug>"}}],
      "reveals": [],
      "description": "one sentence the detective would do",
      "narrative": "one short paragraph of 1920s-noir prose, staying in the established story context"
    }}
  ],
  "rationale": "one sentence explaining the repair"
}}

Hard constraints:
- Do NOT invent characters, locations, or evidence IDs not in the ALLOWED lists above.
- Do NOT reveal the real criminal. Preserve mystery.
- Keep effects minimal — one or two knowledge additions at most."""


# ---------------------------------------------------------------------------
# Drama manager
# ---------------------------------------------------------------------------
class DramaManager:
    def __init__(self, plan: Plan, world: Any = None, log_path: str | Path = "logs/drama.jsonl") -> None:
        self.plan = plan
        self.world = world
        self.executed: list[str] = []
        self.remaining: list[str] = sorted(plan.events.keys())
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log: list[LogEntry] = []
        self._next_repair_idx = 1
        # Build predecessor index from plan.order for ordering enforcement.
        # predecessors[eid] = {set of event ids that must execute before eid}
        self._predecessors: dict[str, set[str]] = {eid: set() for eid in plan.events}
        for producer, consumer in plan.order:
            if consumer in self._predecessors:
                self._predecessors[consumer].add(producer)

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
        constituent_match = self._find_constituent_match(parsed_action, state)
        proposed_effects = []
        for _e in parsed_action.get("effects", []):
            if not isinstance(_e, dict):
                continue
            try:
                proposed_effects.append(Effect.from_dict(_e))
            except (KeyError, TypeError):
                pass  # malformed LLM effect — skip
        hard_violations = self._hard_violations(proposed_effects)

        # Detect destructive inputs by keyword — these are always exceptional
        # regardless of whether the LLM action interpreter modelled any effects.
        raw_input = (parsed_action.get("_raw") or "").lower()
        is_destructive = (
            parsed_action.get("verb", "").lower() in _DESTRUCTIVE_VERBS
            or bool(_DESTRUCTIVE_RE.search(raw_input))
        )

        # Commonsense threat check — skip when a hard violation is already
        # confirmed; always run for destructive inputs (effects may be empty).
        soft_threats: list[dict[str, Any]] = []
        cs_hint = "consistent"
        needs_cs = not hard_violations and (
            is_destructive or proposed_effects or parsed_action.get("novel_state_vars")
        )
        if needs_cs:
            soft_threats, cs_hint = self._commonsense_threats(parsed_action)

        # Destructive inputs with no constituent match are always exceptional,
        # overriding a "consistent" commonsense hint.
        if is_destructive and not constituent_match:
            cs_hint = "exceptional"

        classification = self._decide_classification(constituent_match, hard_violations, soft_threats, cs_hint)

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
            cs_hint=cs_hint,
            active_links=[cl.to_dict() for cl in self.active_links()],
            remaining_count=len(self.remaining),
            executed_count=len(self.executed),
        )
        return result

    def _find_constituent_match(
        self,
        parsed_action: dict[str, Any],
        state: dict[str, dict[str, Any]],
    ) -> str | None:
        """Return the first remaining event whose verb + args substantially
        overlap with the parsed action, respecting the detective's current
        location and character availability."""
        raw = (parsed_action.get("_raw") or "").lower()
        parsed_verb = parsed_action.get("verb", "").lower()
        cur_loc = state.get("detective", {}).get("location", "")

        for eid in self.remaining:
            ev = self.plan.events[eid]

            # Skip if the event is tied to a different location.
            if ev.location and cur_loc and ev.location != cur_loc:
                continue

            # Respect partial-order constraints: all declared predecessors must
            # already be executed (or absent from the plan) before this event
            # can match.
            required = self._predecessors.get(eid, set())
            if required and not required.issubset(set(self.executed) | (set(self.plan.events) - set(self.remaining))):
                continue

            # Skip if any required character is not at the detective's location
            # or is no longer available.
            if self.world and cur_loc:
                loc_obj = self.world.locations.get(cur_loc)
                chars_here = set(loc_obj.characters) if loc_obj else set()
                ev_chars = {str(a) for a in ev.args if str(a).startswith("character.")}
                if ev_chars:
                    available_here = {
                        cid for cid in chars_here
                        if state.get(cid, {}).get("alive", True)
                        and state.get(cid, {}).get("available", True)
                    }
                    if not ev_chars & available_here:
                        continue

            ev_verb = ev.verb.lower()
            verb_matches = (not parsed_verb) or _verb_in_same_family(ev_verb, parsed_verb)

            # Token-level content match: tolerates LLM paraphrasing and the
            # mismatch between short user words ("ring", "body") and full plan
            # arg phrases ("ladies' ring with flower design").
            ev_tokens = _tok(" ".join(str(a) for a in ev.args) + " " + ev.description)
            input_tokens = _tok(raw) | _tok(" ".join(str(a) for a in parsed_action.get("args", [])))
            content_match = bool(input_tokens & ev_tokens)

            if verb_matches and content_match:
                return eid
            # Softer fallback: strong content match even if verb is off
            # (guards against LLM verb drift, e.g. "observe" for "search").
            if content_match and len(input_tokens & ev_tokens) >= 2:
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

    def _commonsense_threats(
        self, parsed_action: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str]:
        """Return (threatened_events, overall_classification_hint)."""
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
            return [], "consistent"
        if not isinstance(parsed, dict):
            return [], "consistent"
        threats = parsed.get("threatened_events", [])
        hint = parsed.get("overall_classification_hint", "consistent")
        return threats, hint

    @staticmethod
    def _decide_classification(
        constituent_match: str | None,
        hard_violations: list[CausalLink],
        soft_threats: list[dict[str, Any]],
        cs_hint: str = "consistent",
    ) -> str:
        # Hard causal-link violations always win — the plan structure is broken.
        if hard_violations:
            return "exceptional"
        # A constituent match means the player is performing exactly the planned
        # action. Soft threats from the commonsense reasoner should yield to an
        # explicit plan match; they only matter for free (non-plan) actions.
        if constituent_match:
            return "constituent"
        # For free actions, escalate on commonsense threats.
        if soft_threats or cs_hint == "exceptional":
            return "exceptional"
        return "consistent"

    # ---- execution -------------------------------------------------------
    def execute_constituent(self, event_id: str, state: dict[str, dict[str, Any]]) -> None:
        event = self.plan.events[event_id]
        for ef in event.effects:
            ef.apply(state)
        self.executed.append(event_id)
        if event_id in self.remaining:
            self.remaining.remove(event_id)
        self._log(
            "executed_constituent",
            event_id=event_id,
            event_verb=event.verb,
            event_description=event.description,
            effects_applied=[ef.to_dict() for ef in event.effects],
            reveals=list(event.reveals),
            remaining_after=len(self.remaining),
            executed_total=len(self.executed),
        )

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

        # Snapshot plan state before removal for logging.
        plan_before = [
            {"id": eid, "verb": self.plan.events[eid].verb,
             "desc": self.plan.events[eid].description[:70]}
            for eid in self.remaining if eid in self.plan.events
        ]
        active_links_before = [cl.to_dict() for cl in self.active_links()]

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

        plan_after = [
            {"id": eid, "verb": self.plan.events[eid].verb,
             "desc": self.plan.events[eid].description[:70]}
            for eid in self.remaining if eid in self.plan.events
        ]
        self._log(
            "accommodation",
            removed_events=removed_events,
            removed_descriptions=[
                self.plan.events[eid].description for eid in removed_events if eid in self.plan.events
            ],
            replacement_event_ids=[e.id for e in replacements],
            replacement_descriptions=[e.description for e in replacements],
            rationale=parsed.get("rationale", "") if isinstance(parsed, dict) else "",
            plan_before=plan_before,
            plan_after=plan_after,
            active_links_before=active_links_before,
            goal_reachability=self.goal_reachability(),
        )
        return {
            "removed_events": removed_events,
            "replacement_events": [e.to_dict() for e in replacements],
            "rationale": parsed.get("rationale", "") if isinstance(parsed, dict) else "",
        }

    # ---- goal check ------------------------------------------------------
    def goal_satisfied(self, state: dict[str, dict[str, Any]]) -> bool:
        return all(c.evaluate(state) for c in self.plan.goal)

    def goal_reachability(self) -> dict[str, Any]:
        """Heuristic: which goal conditions can still be satisfied by remaining events?"""
        reachable, blocked = [], []
        for cond in self.plan.goal:
            can_satisfy = any(
                any(self._effect_satisfies_condition(ef, cond)
                    for ef in self.plan.events[eid].effects)
                for eid in self.remaining if eid in self.plan.events
            )
            (reachable if can_satisfy else blocked).append(
                f"{cond.subject}.{cond.attr} {cond.op} {cond.value}"
            )
        verdict = "reachable" if not blocked else ("no_path" if not reachable else "partially_blocked")
        return {"reachable": reachable, "blocked": blocked, "verdict": verdict}

    @staticmethod
    def _effect_satisfies_condition(effect: Effect, condition: Condition) -> bool:
        if effect.subject != condition.subject or effect.attr != condition.attr:
            return False
        if condition.op == "==" and effect.op == "set":
            return effect.value == condition.value
        if condition.op == "contains" and effect.op == "add":
            return effect.value == condition.value
        if condition.op == "not_contains" and effect.op == "remove":
            return effect.value == condition.value
        return False

    # ---- turn-start marker (called by GameEngine) ------------------------
    def log_turn_start(self, turn: int, raw: str, location: str) -> None:
        self._log(
            "turn_start",
            turn=turn,
            command=raw,
            detective_location=location,
            remaining_count=len(self.remaining),
            executed_count=len(self.executed),
            remaining_events=[
                {"id": eid, "verb": self.plan.events[eid].verb,
                 "desc": self.plan.events[eid].description[:70]}
                for eid in self.remaining if eid in self.plan.events
            ],
        )


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
