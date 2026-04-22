"""Bake data/final_story.md into a single self-contained HTML reader.

Usage:
    python web/build_interactive.py
    # -> writes web/story_interactive.html

Open the generated file in any browser -- no server, no dependencies.

Features baked into the output:
  * 1920s-noir styling (dark parchment + sepia accents).
  * Chapter navigation with progress tracker.
  * "Spoiler lock" on Prologue / Resolution / Epilogue until the reader
    finishes chapters 1-11 and submits a guess.
  * Who-done-it widget with five suspects; correct answer reveals all
    three locked sections with a flourish, wrong answer still reveals
    them but tags the guess as incorrect.
  * Evidence tracker that fills in automatically as the reader
    progresses.
  * localStorage so progress survives page reload.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STORY = ROOT / "data" / "final_story.md"
OUT = HERE / "story_interactive.html"


# ---------------------------------------------------------------------------
# Minimal markdown -> HTML
# ---------------------------------------------------------------------------
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITAL = re.compile(r"\*([^*]+)\*")


def md_para_to_html(block: str) -> str:
    block = block.strip()
    if not block:
        return ""
    # Escape HTML first, then reintroduce italics / bold.
    esc = html.escape(block)
    esc = _BOLD.sub(r"<strong>\1</strong>", esc)
    esc = _ITAL.sub(r"<em>\1</em>", esc)
    # Preserve single line breaks inside a paragraph as <br>.
    esc = esc.replace("\n", "<br>")
    return f"<p>{esc}</p>"


def parse_story(md_text: str) -> list[dict]:
    """Split markdown into sections at top-level '# ' headings."""
    sections: list[dict] = []
    current: dict | None = None
    buf: list[str] = []

    def flush() -> None:
        if current is None:
            return
        raw_body = "\n".join(buf).strip()
        # Convert paragraphs: split on blank lines.
        paragraphs = [md_para_to_html(p) for p in re.split(r"\n\s*\n", raw_body) if p.strip()]
        current["html"] = "\n".join(paragraphs)
        current["raw_len"] = len(raw_body)
        sections.append(current)

    for line in md_text.splitlines():
        if line.startswith("# "):
            flush()
            current = {"title": line[2:].strip()}
            buf = []
        else:
            if current is not None:
                buf.append(line)
    flush()
    return sections


# ---------------------------------------------------------------------------
# Section classification + short labels for the nav
# ---------------------------------------------------------------------------
def classify(title: str) -> tuple[str, str]:
    """Return (kind, nav_label). kind in {prologue, chapter, resolution, epilogue}."""
    low = title.lower()
    if low.startswith("prologue"):
        return "prologue", "Prologue"
    if low.startswith("epilogue"):
        return "epilogue", "Epilogue"
    if "resolution" in low:
        return "resolution", "Resolution"
    m = re.match(r"chapter\s+(\d+)", low)
    if m:
        return "chapter", f"Chapter {int(m.group(1))}"
    return "chapter", title


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Hartley Affair &mdash; Interactive Mystery</title>
<style>
:root {
  --bg: #14100d;
  --panel: #1f1a15;
  --paper: #f4ecdc;
  --ink: #1c1a17;
  --accent: #b38a4a;
  --accent-soft: #c9a878;
  --danger: #b34a4a;
  --good: #4a9b5d;
  --muted: #8a7b66;
  --rule: #2c251e;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--paper);
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 17px;
  line-height: 1.65;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
header.masthead {
  padding: 28px 40px 18px;
  border-bottom: 1px solid var(--rule);
  background: linear-gradient(180deg, #1a140f 0%, var(--bg) 100%);
}
header.masthead h1 {
  margin: 0;
  font-family: "Playfair Display", "Didot", "Bodoni 72", serif;
  font-size: 34px;
  letter-spacing: 0.02em;
  color: var(--accent);
  text-shadow: 0 1px 0 #0006;
}
header.masthead .subtitle {
  margin: 4px 0 0;
  color: var(--muted);
  font-style: italic;
  font-size: 14px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.layout {
  display: grid;
  grid-template-columns: 260px 1fr 260px;
  gap: 0;
  flex: 1;
  min-height: 0;
}
aside.nav, aside.tracker {
  padding: 26px 20px;
  background: var(--panel);
  overflow-y: auto;
}
aside.nav { border-right: 1px solid var(--rule); }
aside.tracker { border-left: 1px solid var(--rule); }
aside h3 {
  font-family: "Playfair Display", Georgia, serif;
  font-size: 14px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-soft);
  margin: 0 0 12px;
}
ol.toc {
  list-style: none;
  margin: 0 0 14px;
  padding: 0;
}
ol.toc li {
  padding: 8px 10px;
  margin: 4px -10px;
  cursor: pointer;
  border-radius: 4px;
  font-size: 15px;
  color: var(--paper);
  border-left: 3px solid transparent;
  transition: background .15s, border-color .15s;
}
ol.toc li:hover { background: rgba(179, 138, 74, 0.12); }
ol.toc li.active {
  background: rgba(179, 138, 74, 0.22);
  border-left-color: var(--accent);
}
ol.toc li.read::after {
  content: " ✓";
  color: var(--good);
}
ol.toc li.locked {
  opacity: 0.4;
  cursor: not-allowed;
}
ol.toc li.locked::after {
  content: " \1F512";
  color: var(--muted);
}
.progress {
  font-size: 13px;
  color: var(--muted);
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px dashed var(--rule);
}
.progress strong { color: var(--paper); }

main {
  overflow-y: auto;
  padding: 40px 56px 60px;
  background: var(--bg);
  scroll-behavior: smooth;
}
article.reader {
  max-width: 720px;
  margin: 0 auto;
  background: var(--paper);
  color: var(--ink);
  padding: 56px 64px 72px;
  border-radius: 3px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.45), 0 1px 2px rgba(0,0,0,0.3);
  position: relative;
}
article.reader::before {
  content: "";
  position: absolute;
  inset: 6px;
  border: 1px solid rgba(28,26,23,0.15);
  pointer-events: none;
  border-radius: 2px;
}
article.reader h2 {
  font-family: "Playfair Display", Georgia, serif;
  font-size: 30px;
  margin: 0 0 6px;
  color: var(--accent);
  letter-spacing: 0.01em;
  border-bottom: 1px solid rgba(28,26,23,0.18);
  padding-bottom: 12px;
}
article.reader .ch-meta {
  font-size: 12px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 24px;
}
article.reader p {
  margin: 0 0 14px;
  text-align: justify;
  hyphens: auto;
}
article.reader p:first-of-type::first-letter {
  font-family: "Playfair Display", Georgia, serif;
  font-size: 42px;
  font-weight: 700;
  color: var(--accent);
  float: left;
  line-height: 0.9;
  margin: 6px 6px -4px 0;
}
article.reader em { color: #3e2f1c; }

nav.controls {
  max-width: 720px;
  margin: 24px auto 0;
  display: flex;
  justify-content: space-between;
  gap: 10px;
}
nav.controls button {
  background: var(--panel);
  color: var(--paper);
  border: 1px solid var(--rule);
  padding: 10px 18px;
  font-family: inherit;
  font-size: 14px;
  letter-spacing: 0.06em;
  cursor: pointer;
  transition: background .15s, color .15s;
}
nav.controls button:disabled { opacity: 0.35; cursor: not-allowed; }
nav.controls button:not(:disabled):hover {
  background: var(--accent);
  color: var(--ink);
}

/* Evidence tracker (right column) */
ul.evidence {
  list-style: none;
  margin: 0;
  padding: 0;
}
ul.evidence li {
  padding: 8px 10px;
  margin: 0 -10px;
  font-size: 13px;
  color: var(--muted);
  border-left: 3px solid var(--rule);
  transition: color .2s, border-color .2s;
}
ul.evidence li.discovered {
  color: var(--paper);
  border-left-color: var(--accent);
}
ul.evidence li.discovered::before {
  content: "● ";
  color: var(--accent);
}
ul.evidence li:not(.discovered)::before {
  content: "○ ";
}

/* Guess widget */
section.guess {
  max-width: 720px;
  margin: 40px auto 0;
  background: #2a2219;
  border: 1px solid var(--accent);
  padding: 28px 36px;
  border-radius: 4px;
}
section.guess h2 {
  font-family: "Playfair Display", Georgia, serif;
  color: var(--accent);
  margin: 0 0 10px;
  font-size: 24px;
}
section.guess p { margin: 0 0 18px; color: var(--paper); }
.suspect-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px;
}
.suspect-grid button {
  padding: 14px 12px;
  background: var(--panel);
  color: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 3px;
  font-family: inherit;
  font-size: 15px;
  cursor: pointer;
  text-align: left;
  transition: border-color .15s, background .15s;
}
.suspect-grid button:hover { border-color: var(--accent); background: #241d16; }
.suspect-grid button.chosen.correct { border-color: var(--good); background: rgba(74,155,93,0.15); }
.suspect-grid button.chosen.wrong   { border-color: var(--danger); background: rgba(179,74,74,0.12); }
#guess-result {
  margin-top: 18px;
  padding: 14px 16px;
  border-left: 3px solid var(--accent);
  background: rgba(179, 138, 74, 0.08);
  font-size: 15px;
  color: var(--paper);
  display: none;
}
#guess-result.shown { display: block; }
#guess-result.correct { border-left-color: var(--good); }
#guess-result.wrong   { border-left-color: var(--danger); }

.hidden { display: none !important; }

footer.meta {
  text-align: center;
  padding: 20px 16px 32px;
  font-size: 12px;
  color: var(--muted);
  border-top: 1px solid var(--rule);
  background: var(--panel);
}
footer.meta a { color: var(--accent-soft); text-decoration: none; }

@media (max-width: 980px) {
  .layout { grid-template-columns: 1fr; }
  aside.nav, aside.tracker { border: none; border-bottom: 1px solid var(--rule); }
  main { padding: 24px 14px 40px; }
  article.reader { padding: 32px 28px 44px; }
}
</style>
</head>
<body>
<header class="masthead">
  <h1>The Hartley Affair</h1>
  <p class="subtitle">An Interactive Mystery &middot; CS 7634 Phase II &middot; Template 2</p>
</header>

<div class="layout">
  <aside class="nav">
    <h3>Chapters</h3>
    <ol class="toc" id="toc"></ol>
    <div class="progress">
      Chapters read: <strong id="read-count">0</strong> / <strong id="chapter-total">0</strong><br>
      <button id="reset-btn" style="margin-top:10px; font-size: 11px; background: transparent; color: var(--muted); border: 1px solid var(--rule); padding: 4px 8px; cursor: pointer;">Reset progress</button>
    </div>
  </aside>

  <main>
    <article class="reader" id="reader">
      <!-- rendered on load -->
    </article>

    <nav class="controls">
      <button id="prev-btn">&larr; Previous</button>
      <button id="next-btn">Next &rarr;</button>
    </nav>

    <section class="guess hidden" id="guess">
      <h2>Name the killer</h2>
      <p>You've followed the trail to its apparent end. Before you read the Inspector's final verdict, who do <em>you</em> think poisoned Edmund Hartley?</p>
      <div class="suspect-grid" id="suspect-grid"></div>
      <div id="guess-result"></div>
    </section>
  </main>

  <aside class="tracker">
    <h3>Evidence &amp; Leads</h3>
    <ul class="evidence" id="evidence"></ul>
    <h3 style="margin-top:22px;">Suspects</h3>
    <ul class="evidence" id="suspect-list"></ul>
  </aside>
</div>

<footer class="meta">
  Interactive reader generated from <code>data/final_story.md</code>.
  Template 2 project &middot; <a href="https://github.com/ivy3h/7634-s26">repo</a>.
</footer>

<script>
const SECTIONS = __SECTIONS_JSON__;
const EVIDENCE_UNLOCKS = __EVIDENCE_JSON__;
const SUSPECTS = __SUSPECTS_JSON__;
const CORRECT_SUSPECT_ID = "vivienne";

const STORAGE_KEY = "hartley_affair_progress_v1";

function loadProgress() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {read: [], guess: null, unlocked: false};
  } catch (e) {
    return {read: [], guess: null, unlocked: false};
  }
}
function saveProgress(p) { localStorage.setItem(STORAGE_KEY, JSON.stringify(p)); }

let state = loadProgress();
let currentIdx = 0;

// First visible index is always Chapter 1 if locked sections hidden
const firstVisibleIdx = SECTIONS.findIndex(s => s.kind === "chapter");

function isLocked(section) {
  if (section.kind === "chapter") return false;
  return !state.unlocked;
}

function chapterCount() {
  return SECTIONS.filter(s => s.kind === "chapter").length;
}

function renderToc() {
  const ol = document.getElementById("toc");
  ol.innerHTML = "";
  SECTIONS.forEach((sec, i) => {
    const li = document.createElement("li");
    li.textContent = sec.navLabel;
    if (state.read.includes(i)) li.classList.add("read");
    if (isLocked(sec)) li.classList.add("locked");
    if (i === currentIdx) li.classList.add("active");
    li.addEventListener("click", () => {
      if (isLocked(sec)) return;
      showSection(i);
    });
    ol.appendChild(li);
  });
  document.getElementById("read-count").textContent = state.read.filter(i => SECTIONS[i] && SECTIONS[i].kind === "chapter").length;
  document.getElementById("chapter-total").textContent = chapterCount();
}

function renderEvidence() {
  const ul = document.getElementById("evidence");
  ul.innerHTML = "";
  EVIDENCE_UNLOCKS.forEach(ev => {
    const li = document.createElement("li");
    li.textContent = ev.label;
    const unlockAt = SECTIONS.findIndex(s => s.navLabel === ev.unlockedAfter);
    if (unlockAt !== -1 && state.read.includes(unlockAt)) {
      li.classList.add("discovered");
    }
    ul.appendChild(li);
  });
  const sl = document.getElementById("suspect-list");
  sl.innerHTML = "";
  SUSPECTS.forEach(s => {
    const li = document.createElement("li");
    li.textContent = s.name;
    if (state.guess && state.unlocked && s.id === CORRECT_SUSPECT_ID) li.classList.add("discovered");
    sl.appendChild(li);
  });
}

function showSection(i) {
  currentIdx = i;
  const sec = SECTIONS[i];
  const reader = document.getElementById("reader");
  reader.innerHTML = `<h2>${sec.title}</h2>
    <p class="ch-meta">${sec.navLabel}${sec.wordCount ? ' &middot; ' + sec.wordCount + ' words' : ''}</p>
    ${sec.html}`;
  reader.scrollTop = 0;
  document.querySelector("main").scrollTop = 0;
  // Mark as read
  if (sec.kind === "chapter" && !state.read.includes(i)) {
    state.read.push(i);
    saveProgress(state);
  }
  renderToc();
  renderEvidence();
  updateControls();
  updateGuessVisibility();
}

function updateControls() {
  const prev = document.getElementById("prev-btn");
  const next = document.getElementById("next-btn");
  // Allow prev/next to step only among unlocked sections.
  const order = SECTIONS.map((s, i) => [s, i]).filter(([s]) => !isLocked(s)).map(([, i]) => i);
  const pos = order.indexOf(currentIdx);
  prev.disabled = pos <= 0;
  next.disabled = pos >= order.length - 1;
  prev.onclick = () => { if (pos > 0) showSection(order[pos - 1]); };
  next.onclick = () => { if (pos < order.length - 1) showSection(order[pos + 1]); };
}

function updateGuessVisibility() {
  const guess = document.getElementById("guess");
  const readChapters = state.read.filter(i => SECTIONS[i] && SECTIONS[i].kind === "chapter").length;
  const allRead = readChapters >= chapterCount();
  if (allRead && !state.unlocked) {
    guess.classList.remove("hidden");
  } else {
    guess.classList.add("hidden");
  }
}

function renderGuessGrid() {
  const grid = document.getElementById("suspect-grid");
  grid.innerHTML = "";
  SUSPECTS.forEach(s => {
    const btn = document.createElement("button");
    btn.textContent = s.name;
    btn.dataset.id = s.id;
    btn.addEventListener("click", () => handleGuess(s));
    grid.appendChild(btn);
  });
}

function handleGuess(suspect) {
  if (state.unlocked) return;
  state.guess = suspect.id;
  state.unlocked = true;
  saveProgress(state);
  const correct = suspect.id === CORRECT_SUSPECT_ID;
  const grid = document.getElementById("suspect-grid");
  Array.from(grid.children).forEach(btn => {
    btn.disabled = true;
    if (btn.dataset.id === suspect.id) {
      btn.classList.add("chosen");
      btn.classList.add(correct ? "correct" : "wrong");
    }
  });
  const result = document.getElementById("guess-result");
  if (correct) {
    result.className = "shown correct";
    result.innerHTML = `<strong>Correct.</strong> Vivienne Ashford poisoned Edmund Hartley. The hollow ring, the staged toast, the three conspirators &mdash; you pieced it together. The Resolution and Epilogue are now unlocked in the chapter list.`;
  } else {
    result.className = "shown wrong";
    result.innerHTML = `<strong>Not quite.</strong> You accused ${suspect.name}. Read the Resolution and Epilogue to see what actually happened the night of the toast. They are now unlocked in the chapter list.`;
  }
  renderToc();
  renderEvidence();
  updateControls();
}

document.getElementById("reset-btn").addEventListener("click", () => {
  if (!confirm("Reset reading progress and guess?")) return;
  localStorage.removeItem(STORAGE_KEY);
  state = loadProgress();
  currentIdx = firstVisibleIdx;
  document.getElementById("guess-result").className = "";
  document.getElementById("guess-result").innerHTML = "";
  renderGuessGrid();
  updateGuessVisibility();
  showSection(currentIdx);
});

// Initial render.
renderGuessGrid();
currentIdx = firstVisibleIdx;
showSection(currentIdx);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Evidence & suspects (narrative-grounded, tied to the actual story content)
# ---------------------------------------------------------------------------
EVIDENCE = [
    {"label": "Gold hollow ring with flower engraving", "unlockedAfter": "Chapter 1"},
    {"label": "Ring's hidden arsenic chamber", "unlockedAfter": "Chapter 2"},
    {"label": "Sutherland's gentlemen's-club ledger", "unlockedAfter": "Chapter 3"},
    {"label": "Lady Beatrice's letter signed 'B'", "unlockedAfter": "Chapter 4"},
    {"label": "Isabella Rossi's Grand Imperial alibi", "unlockedAfter": "Chapter 5"},
    {"label": "Phone records: Geoffrey's 6:52 PM call", "unlockedAfter": "Chapter 6"},
    {"label": "Blackwood's threatening-email dossier", "unlockedAfter": "Chapter 7"},
    {"label": "Burned Belgian shipping manifest", "unlockedAfter": "Chapter 8"},
    {"label": "Scopolamine toxicology finding", "unlockedAfter": "Chapter 9"},
    {"label": "Red-ink ledger: 'G.H. - usual arrangement - 15%'", "unlockedAfter": "Chapter 10"},
    {"label": "Jeweler's receipt 'discovered' by Charlotte", "unlockedAfter": "Chapter 11"},
]

SUSPECTS = [
    {"id": "geoffrey",  "name": "Geoffrey Hartley"},
    {"id": "pemberton", "name": "Dr. Marcus Pemberton"},
    {"id": "vivienne",  "name": "Vivienne Ashford"},
    {"id": "charlotte", "name": "Charlotte Devereaux"},
    {"id": "blackwood", "name": "Thomas Blackwood"},
]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build() -> None:
    md_text = STORY.read_text(encoding="utf-8")
    raw_sections = parse_story(md_text)

    sections: list[dict] = []
    for sec in raw_sections:
        kind, nav_label = classify(sec["title"])
        # A rough word count for the reader chip.
        text_only = re.sub(r"<[^>]+>", " ", sec["html"])
        word_count = len(re.findall(r"\b\w+\b", text_only))
        sections.append({
            "title": sec["title"],
            "kind": kind,
            "navLabel": nav_label,
            "html": sec["html"],
            "wordCount": word_count,
        })

    html_out = (
        HTML_TEMPLATE
        .replace("__SECTIONS_JSON__", json.dumps(sections, ensure_ascii=False))
        .replace("__EVIDENCE_JSON__", json.dumps(EVIDENCE, ensure_ascii=False))
        .replace("__SUSPECTS_JSON__", json.dumps(SUSPECTS, ensure_ascii=False))
    )
    OUT.write_text(html_out, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT}  ({len(sections)} sections, {size_kb:.1f} KB)")


if __name__ == "__main__":
    build()
