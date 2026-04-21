# DESIGN: Recognizing exceptions under open-action inputs

This document answers the Template-2 specific question:

> The original work on accommodation assumed a fixed set of user actions
> with known effects. When the user can enter any action, how can you
> recognize exceptions? The user may come up with actions that have
> effects that do not violate an active causal span, but commonsense
> would dictate that the event was rendered impossible.

## The jam-door example, rephrased for our mystery

Our plan contains an event *Interview Fleming in the morgue office*. Its
pre-declared preconditions are `detective.location == morgue_office` and
`character.fleming.available == true`. There is an active causal link
whose condition is `character.fleming.available == true`.

The player types: `wedge morgue door shut`.

Nothing in our plan mentions "door wedged". The interpreter, when it
parses `wedge morgue door shut`, will emit an effect on a brand-new state
variable — `object.morgue_door.wedged = true`. No active causal link
references that variable. Classical accommodation (Riedl/Saretto/Young)
would miss this: no link is formally broken, so the action looks
*consistent*, and the story would later crash when Fleming tries to enter
the morgue that has been sealed from the outside.

## Our two-layer exception detector

We detect exceptions in two passes:

### Layer 1: hard-link violation (O(|active_links| × |effects|))

`DramaManager._hard_violations` compares the parsed effects against the
set of currently active causal-link conditions. If any effect inverts a
condition that must still hold, the action is exceptional. This is the
classical check and runs without calling the LLM.

### Layer 2: commonsense threat query (one LLM call per non-trivial action)

If layer 1 did not flag anything *and* the parsed action either (a) has
at least one effect or (b) declares at least one `novel_state_var`, we
send a single structured prompt to the LLM:

```
User action (structured): ...
Remaining plan events (id, verb, args, location, preconditions): ...
Active causal link conditions: ...

Does the user's action render any remaining event impossible or
meaningfully harder for a realistic actor to perform, even if no hard
causal link is formally broken?

Consider physical blocks (jammed, locked permanently, broken), social
blocks (witness dead, suspect fled the country), and epistemic blocks
(evidence destroyed).
```

The LLM returns `threatened_events: [{event_id, reason, repairable}]`.
Any non-empty list reclassifies the action as exceptional.

Two design choices matter:

1. **We ask the LLM about effect *consequences*, not about user intent.**
   The prompt feeds it the structured effect list that the interpreter
   produced plus the remaining plan events — both small, bounded inputs.
   The LLM is doing pure physical/social commonsense over a stated world
   delta. This keeps the call cheap (< 600 tokens) and auditable: both
   sides of the reasoning are in the log.

2. **We promote `novel_state_vars` as first-class input to the prompt.**
   The interpreter is instructed to list any attribute it introduces that
   was not in the existing state. Those novel variables are exactly the
   signal that "this action might break something through a state we
   didn't model." Surfacing them explicitly gives the LLM the hook it
   needs to reason about them.

The two layers compose: hard link violations are cheap and unambiguous;
the commonsense query catches the open-action cases the fixed-link
analysis misses.

## Accommodation

Once we know which remaining events are threatened:

1. Remove them from `plan.remaining` and drop any causal links whose
   producer or consumer is now gone.
2. Call the LLM again with a repair prompt (`ACCOMMODATION_PROMPT`): here
   is the world snapshot, here are the goal conditions, here are the
   characters and locations that still exist, here are the removed
   events. Produce 1–3 replacement events whose preconditions are met by
   the current state and whose effects move the detective toward the
   goal.
3. Validate each replacement by constructing `Event` dataclasses — if
   any field is malformed, we skip it but keep the others. Surviving
   replacements are appended to `plan.remaining` with ids `R01`, `R02`, …
4. Everything is written to `logs/drama.jsonl` as a structured record of
   what was removed, what was added, and why.

We constrain the repair prompt to prefer changes that modify *unrevealed*
details of the crime (e.g. a previously unmentioned witness appears with
the same information the blocked evidence would have provided), rather
than changing the outcome of the crime itself — that matches the
template guidance: the original crime is fixed; only its undisclosed
details may drift.

## Why this is lightweight

The classifier makes at most one LLM call per user action, and only when
the action has non-trivial effects. Idle actions (e.g. `look around`,
`go north`) return with verb==move and no effects, hit layer 1 with
nothing to check, and short-circuit to "consistent" without ever
touching the model. This keeps the interactive loop responsive — on a
local Qwen-9B vLLM server, a turn with a single threat query lands in
~1-3 seconds on 1× A40.

## What the logs show

`logs/drama.jsonl` is one JSON object per line. During the video
demonstration, the key kinds to scroll are:

- `classification` — every user turn: parsed action, whether we fell
  through layer 1 or layer 2, the resulting label.
- `applied_free_effects` — effects from consistent / exceptional
  actions that weren't already applied by a plan event.
- `accommodation` — which event ids were dropped, which replacement
  ids were added, and the LLM's one-line rationale for the repair.

`logs/turns.jsonl` has the player-facing view: raw input, parsed action,
narration, effects applied.

Together they cover every behind-the-scenes decision and are the primary
evidence for the final video.
