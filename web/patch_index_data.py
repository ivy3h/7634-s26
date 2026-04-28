"""Replace the pre-baked const DATA = {...} in index.html with fresh data from build_game_data()."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.build_game import build_game_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "index.html")

data = build_game_data()
fresh_data_str = json.dumps(data, ensure_ascii=False)

# Verify victim injection
start_loc = data["starting_location"]
chars_here = data["locations"].get(start_loc, {}).get("characters", [])
print(f"Starting location: {start_loc}")
print(f"Characters there: {chars_here}")

with open(INDEX_PATH, encoding="utf-8") as f:
    html = f.read()

marker = "const DATA = "
start_marker = html.index(marker)
start_brace = start_marker + len(marker)
assert html[start_brace] == "{", f"Expected {{ at {start_brace}, got {html[start_brace]!r}"

# Walk forward counting braces (JSON-aware: skip string contents)
depth = 0
i = start_brace
in_string = False
escape_next = False
while i < len(html):
    ch = html[i]
    if escape_next:
        escape_next = False
        i += 1
        continue
    if in_string:
        if ch == "\\":
            escape_next = True
        elif ch == '"':
            in_string = False
    else:
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
    i += 1

end_brace = i
print(f"Old DATA: chars {start_brace}–{end_brace} ({end_brace - start_brace + 1} chars)")
print(f"New DATA: {len(fresh_data_str)} chars")

new_html = html[:start_brace] + fresh_data_str + html[end_brace + 1:]
print(f"New index.html: {len(new_html.encode('utf-8'))} bytes")

with open(INDEX_PATH, "w", encoding="utf-8") as f:
    f.write(new_html)
print("index.html updated successfully.")
