"""Shared dataclasses for the plan representation.

An `Event` is a single step the detective (or an NPC) takes. It has explicit
preconditions and effects expressed as predicates over world-state subjects.
A `CausalLink` (producer, condition, consumer) indicates that the producer
event establishes `condition`, which the consumer event requires — so the
condition must remain true across the span between them. The drama manager
watches those spans.

World state is a flat dict[str, dict[str, Any]]: state[subject_id][attr].
Subjects are identifiers like "character.victoria_harrington",
"location.gallery_main_hall", "object.fountain_pen", "detective.inventory".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Condition:
    """A predicate over world state: `state[subject][attr] <op> value`.

    `op` ∈ {"==", "!=", "contains", "not_contains", ">=", "<="}.
    """
    subject: str
    attr: str
    op: str
    value: Any

    def evaluate(self, state: dict[str, dict[str, Any]]) -> bool:
        slot = state.get(self.subject, {}).get(self.attr)
        if self.op == "==":
            return slot == self.value
        if self.op == "!=":
            return slot != self.value
        if self.op == "contains":
            return slot is not None and self.value in slot
        if self.op == "not_contains":
            return slot is None or self.value not in slot
        if self.op == ">=":
            return slot is not None and slot >= self.value
        if self.op == "<=":
            return slot is not None and slot <= self.value
        raise ValueError(f"unknown op: {self.op}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Condition":
        return cls(subject=d["subject"], attr=d["attr"], op=d["op"], value=d["value"])


@dataclass
class Effect:
    """A world-state mutation.

    `op` ∈ {"set", "add", "remove"}. "add"/"remove" treat the slot as a
    set / list of items; "set" overwrites.
    """
    subject: str
    attr: str
    op: str
    value: Any

    def apply(self, state: dict[str, dict[str, Any]]) -> None:
        bucket = state.setdefault(self.subject, {})
        if self.op == "set":
            bucket[self.attr] = self.value
            return
        cur = bucket.get(self.attr)
        if self.op == "add":
            if isinstance(cur, list):
                if self.value not in cur:
                    cur.append(self.value)
            elif isinstance(cur, set):
                cur.add(self.value)
            else:
                bucket[self.attr] = [self.value]
        elif self.op == "remove":
            if isinstance(cur, list) and self.value in cur:
                cur.remove(self.value)
            elif isinstance(cur, set) and self.value in cur:
                cur.discard(self.value)
        else:
            raise ValueError(f"unknown op: {self.op}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Effect":
        return cls(subject=d["subject"], attr=d["attr"], op=d["op"], value=d["value"])


@dataclass
class Event:
    id: str
    actor: str
    verb: str
    args: list[str] = field(default_factory=list)
    location: str = ""
    preconditions: list[Condition] = field(default_factory=list)
    effects: list[Effect] = field(default_factory=list)
    reveals: list[str] = field(default_factory=list)
    description: str = ""
    narrative: str = ""
    source_plot_idx: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "actor": self.actor,
            "verb": self.verb,
            "args": self.args,
            "location": self.location,
            "preconditions": [c.to_dict() for c in self.preconditions],
            "effects": [e.to_dict() for e in self.effects],
            "reveals": self.reveals,
            "description": self.description,
            "narrative": self.narrative,
            "source_plot_idx": self.source_plot_idx,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        return cls(
            id=d["id"],
            actor=d.get("actor", "detective"),
            verb=d.get("verb", "act"),
            args=list(d.get("args", [])),
            location=d.get("location", ""),
            preconditions=[Condition.from_dict(c) for c in d.get("preconditions", [])],
            effects=[Effect.from_dict(e) for e in d.get("effects", [])],
            reveals=list(d.get("reveals", [])),
            description=d.get("description", ""),
            narrative=d.get("narrative", ""),
            source_plot_idx=d.get("source_plot_idx"),
        )


@dataclass
class CausalLink:
    """(producer_event, condition, consumer_event)

    Condition must remain true from the moment `producer` executes until
    `consumer` executes. If something in between breaks it, we have an
    exception.
    """
    producer: str
    condition: Condition
    consumer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "consumer": self.consumer,
            "condition": self.condition.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CausalLink":
        return cls(
            producer=d["producer"],
            consumer=d["consumer"],
            condition=Condition.from_dict(d["condition"]),
        )


@dataclass
class Plan:
    events: dict[str, Event] = field(default_factory=dict)
    order: list[tuple[str, str]] = field(default_factory=list)
    causal_links: list[CausalLink] = field(default_factory=list)
    initial_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    goal: list[Condition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": {eid: ev.to_dict() for eid, ev in self.events.items()},
            "order": [list(edge) for edge in self.order],
            "causal_links": [cl.to_dict() for cl in self.causal_links],
            "initial_state": self.initial_state,
            "goal": [c.to_dict() for c in self.goal],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Plan":
        return cls(
            events={eid: Event.from_dict(e) for eid, e in d.get("events", {}).items()},
            order=[tuple(edge) for edge in d.get("order", [])],
            causal_links=[CausalLink.from_dict(cl) for cl in d.get("causal_links", [])],
            initial_state=d.get("initial_state", {}),
            goal=[Condition.from_dict(c) for c in d.get("goal", [])],
        )
