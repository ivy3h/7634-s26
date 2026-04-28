"""Apply the same JS bug-fix patches to index.html that were made to build_game.py."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(ROOT, "index.html")

with open(INDEX, encoding="utf-8") as f:
    html = f.read()

REPLACEMENTS = [
    # 1. buildHintCommands – plan-event chips store eventId
    (
        "      const finalHint = `${verbNorm} ${tgt}`.trim().split(/\\s+/).slice(0, 6).join(\" \");\n"
        "      hints.push(finalHint);\n"
        "      if (hints.length >= 3) break;\n"
        "    }\n"
        "  }\n\n"
        "  // B. Characters still here that haven't been interviewed AND still have\n"
        "  //    a remaining social event tied to them at this location. Avoid\n"
        "  //    suggesting \"dead\" hints that would just print the generic polite\n"
        "  //    fall-through.\n"
        "  if (loc) {\n"
        "    const socialFamily = new Set([\"interview\",\"question\",\"consult\",\"confront\",\"visit\"]);\n"
        "    for (const cid of loc.characters) {\n"
        "      const c = DATA.characters[cid];\n"
        "      if (!c || !c.alive) continue;\n"
        "      if (state.charactersInterviewed.includes(cid)) continue;\n"
        "      const alias = firstAlias(c);\n"
        "      const already = hints.some(h => h.toLowerCase().includes(alias));\n"
        "      if (already || hints.length >= 4) continue;\n"
        "      const hasPendingSocialEvent = Object.values(DATA.events).some(ev =>\n"
        "        ev.location === state.location\n"
        "        && !state.executedEvents.includes(ev.id)\n"
        "        && socialFamily.has(ev.verb)\n"
        "        && (ev.args || []).some(a => String(a).toLowerCase().includes(alias))\n"
        "      );\n"
        "      if (hasPendingSocialEvent) {\n"
        "        hints.push(\"question \" + alias);\n"
        "      }\n"
        "    }\n"
        "  }",

        "      const finalHint = `${verbNorm} ${tgt}`.trim().split(/\\s+/).slice(0, 6).join(\" \");\n"
        "      hints.push({cmd: finalHint, eventId: ev.id});\n"
        "      if (hints.length >= 3) break;\n"
        "    }\n"
        "  }\n\n"
        "  // B. Characters still here that haven't been interviewed AND still have\n"
        "  //    a remaining social event tied to them at this location. Avoid\n"
        "  //    suggesting \"dead\" hints that would just print the generic polite\n"
        "  //    fall-through.\n"
        "  if (loc) {\n"
        "    const socialFamily = new Set([\"interview\",\"question\",\"consult\",\"confront\",\"visit\"]);\n"
        "    for (const cid of loc.characters) {\n"
        "      const c = DATA.characters[cid];\n"
        "      if (!c || !c.alive) continue;\n"
        "      if (state.charactersInterviewed.includes(cid)) continue;\n"
        "      const alias = firstAlias(c);\n"
        "      const already = hints.some(h => h.cmd.toLowerCase().includes(alias));\n"
        "      if (already || hints.length >= 4) continue;\n"
        "      const pendingSocialEv = Object.values(DATA.events).find(ev =>\n"
        "        ev.location === state.location\n"
        "        && !state.executedEvents.includes(ev.id)\n"
        "        && socialFamily.has(ev.verb)\n"
        "        && (ev.args || []).some(a => String(a).toLowerCase().includes(alias))\n"
        "      );\n"
        "      if (pendingSocialEv) {\n"
        "        hints.push({cmd: \"question \" + alias, eventId: pendingSocialEv.id});\n"
        "      }\n"
        "    }\n"
        "  }",
    ),

    # 2. buildHintCommands – movement chips (section C)
    (
        '      hints.push("go to " + locAlias(x.loc));\n'
        "    }\n"
        "  }\n\n"
        "  // D. Long-range nudge.",

        '      hints.push({cmd: "go to " + locAlias(x.loc), eventId: null});\n'
        "    }\n"
        "  }\n\n"
        "  // D. Long-range nudge.",
    ),

    # 3. buildHintCommands – section D long-range nudge
    (
        '        const cmd = "go to " + locAlias(targetLoc);\n'
        '        if (!hints.includes(cmd)) hints.push(cmd);\n',

        '        const moveCmd = "go to " + locAlias(targetLoc);\n'
        '        if (!hints.some(h => h.cmd === moveCmd)) hints.push({cmd: moveCmd, eventId: null});\n',
    ),

    # 4. buildHintCommands – section E fallback + dedup + return
    (
        '  if (hints.length === 0) {\n'
        '    hints.push("look");\n'
        "  }\n\n"
        "  // Deliberately no 'accuse X' hint — the detective must choose their own\n"
        "  // suspect. Accusation is always available via free text, no hint needed.\n\n"
        "  // Keep it short.\n"
        "  return Array.from(new Set(hints)).slice(0, 5);\n"
        "}",

        '  if (hints.length === 0) {\n'
        '    hints.push({cmd: "look", eventId: null});\n'
        "  }\n\n"
        "  // Deliberately no 'accuse X' hint — the detective must choose their own\n"
        "  // suspect. Accusation is always available via free text, no hint needed.\n\n"
        "  // Deduplicate by cmd text and cap at 5.\n"
        "  const _seen = new Set();\n"
        "  return hints.filter(h => { if (_seen.has(h.cmd)) return false; _seen.add(h.cmd); return true; }).slice(0, 5);\n"
        "}",
    ),

    # 5. renderHints – auto-submit with eventId
    (
        "  const hints = buildHintCommands();\n"
        "  hints.forEach(cmd => {\n"
        "    const chip = document.createElement(\"button\");\n"
        "    chip.type = \"button\";\n"
        "    chip.className = \"chip\";\n"
        "    chip.innerHTML = `<span class=\"arrow\">&gt;</span>${cmd}`;\n"
        "    chip.title = \"Copy to input (Enter to submit)\";\n"
        "    chip.addEventListener(\"click\", () => prefillInput(cmd));\n"
        "    list.appendChild(chip);\n"
        "  });",

        "  const hints = buildHintCommands();\n"
        "  hints.forEach(h => {\n"
        "    const chip = document.createElement(\"button\");\n"
        "    chip.type = \"button\";\n"
        "    chip.className = \"chip\";\n"
        "    chip.innerHTML = `<span class=\"arrow\">&gt;</span>${h.cmd}`;\n"
        "    if (h.eventId) {\n"
        "      chip.title = \"Click to perform this action (guaranteed plot progress)\";\n"
        "      chip.addEventListener(\"click\", () => runCommand(h.cmd, h.eventId));\n"
        "    } else {\n"
        "      chip.title = \"Click to fill input, then press Enter to move\";\n"
        "      chip.addEventListener(\"click\", () => prefillInput(h.cmd));\n"
        "    }\n"
        "    list.appendChild(chip);\n"
        "  });",
    ),

    # 6. _runCommandViaAPI – add forceEventId parameter and body field
    (
        "async function _runCommandViaAPI(raw) {\n"
        "  const cmdEl   = document.getElementById(\"cmd\");\n"
        "  const submitEl = document.querySelector(\"#input-form button[type=submit]\");\n"
        "  cmdEl.disabled = true;\n"
        "  if (submitEl) submitEl.disabled = true;\n"
        "  try {\n"
        "    const base = API_URL || \"\";\n"
        "    const resp = await fetch(base + \"/api/step\", {\n"
        "      method: \"POST\",\n"
        "      headers: {\"Content-Type\": \"application/json\"},\n"
        "      body: JSON.stringify({command: raw}),\n"
        "    });",

        "async function _runCommandViaAPI(raw, forceEventId) {\n"
        "  const cmdEl   = document.getElementById(\"cmd\");\n"
        "  const submitEl = document.querySelector(\"#input-form button[type=submit]\");\n"
        "  cmdEl.disabled = true;\n"
        "  if (submitEl) submitEl.disabled = true;\n"
        "  try {\n"
        "    const base = API_URL || \"\";\n"
        "    const body = {command: raw};\n"
        "    if (forceEventId) body.force_event_id = forceEventId;\n"
        "    const resp = await fetch(base + \"/api/step\", {\n"
        "      method: \"POST\",\n"
        "      headers: {\"Content-Type\": \"application/json\"},\n"
        "      body: JSON.stringify(body),\n"
        "    });",
    ),

    # 7. runCommand – add forceEventId parameter, API path, and JS force path
    (
        "// -------------------- dispatch --------------------\n"
        "async function runCommand(raw) {\n"
        "  if (state.gameOver) {\n"
        "    addLog(\"The case is already closed. Press Reset to begin a new investigation.\", \"system\");\n"
        "    return;\n"
        "  }\n"
        "  addLog(raw, \"user\");\n"
        "  state.turns += 1;\n\n"
        "  const action = interpret(raw);\n"
        "  const isAccuse = action.verb === \"accuse\";\n\n"
        "  if (API_URL !== null && !isAccuse) {\n"
        "    await _runCommandViaAPI(raw);\n"
        "    saveState();\n"
        "    renderSidebar();\n"
        "    renderHints();\n"
        "    const g = document.getElementById(\"guide-content\");\n"
        "    if (g && !g.classList.contains(\"hidden\")) renderGuide();\n"
        "    const d = document.getElementById(\"dmlog-content\");\n"
        "    if (d && !d.classList.contains(\"hidden\")) renderDmLog();\n"
        "    return;\n"
        "  }\n\n"
        "  switch (action.verb) {",

        "// -------------------- dispatch --------------------\n"
        "async function runCommand(raw, forceEventId) {\n"
        "  forceEventId = forceEventId || null;\n"
        "  if (state.gameOver) {\n"
        "    addLog(\"The case is already closed. Press Reset to begin a new investigation.\", \"system\");\n"
        "    return;\n"
        "  }\n"
        "  addLog(raw, \"user\");\n"
        "  state.turns += 1;\n\n"
        "  const action = interpret(raw);\n"
        "  const isAccuse = action.verb === \"accuse\";\n\n"
        "  if (API_URL !== null && !isAccuse) {\n"
        "    await _runCommandViaAPI(raw, forceEventId);\n"
        "    saveState();\n"
        "    renderSidebar();\n"
        "    renderHints();\n"
        "    const g = document.getElementById(\"guide-content\");\n"
        "    if (g && !g.classList.contains(\"hidden\")) renderGuide();\n"
        "    const d = document.getElementById(\"dmlog-content\");\n"
        "    if (d && !d.classList.contains(\"hidden\")) renderDmLog();\n"
        "    return;\n"
        "  }\n\n"
        "  // JS-only path: if a plan event ID was forced (hint chip), execute directly.\n"
        "  if (forceEventId && DATA.events[forceEventId]) {\n"
        "    const forcedEv = DATA.events[forceEventId];\n"
        "    if (!state.executedEvents.includes(forceEventId) && forcedEv.location === state.location) {\n"
        "      executeEvent(forcedEv);\n"
        "      saveState();\n"
        "      renderSidebar();\n"
        "      renderHints();\n"
        "      const g = document.getElementById(\"guide-content\");\n"
        "      if (g && !g.classList.contains(\"hidden\")) renderGuide();\n"
        "      const d = document.getElementById(\"dmlog-content\");\n"
        "      if (d && !d.classList.contains(\"hidden\")) renderDmLog();\n"
        "      return;\n"
        "    }\n"
        "  }\n\n"
        "  switch (action.verb) {",
    ),
]

for i, (old, new) in enumerate(REPLACEMENTS):
    if old not in html:
        print(f"MISSING pattern {i+1}: {repr(old[:60])}")
    else:
        html = html.replace(old, new, 1)
        print(f"OK patch {i+1}")

with open(INDEX, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\nindex.html updated ({len(html.encode('utf-8'))} bytes)")
