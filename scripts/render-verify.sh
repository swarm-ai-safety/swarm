#!/usr/bin/env bash
# render-verify.sh — verify a DEPLOYED page actually renders, not just that it
# returns HTTP 200.
#
# Why this exists: curl and WebFetch fetch raw HTML and DO NOT execute
# JavaScript. A deploy can report READY and return 200 while the page is
# visually broken — e.g. a client-side chart that never draws, a canvas that
# resizes itself into oblivion, or metrics stuck on their placeholder values.
# This script renders the page in headless Chrome (which runs the JS), then
# makes assertions against the *rendered* DOM and saves a screenshot.
#
# The canonical bug this guards against: SWARM-Gitlawb dashboard, May 2026.
# drawBarChart set canvas.height = offsetHeight*2 with no CSS cap, so the
# canvas doubled on every redraw and blew past the browser's max bitmap size,
# rendering as a blank white box. HTTP was 200; the page was broken. The
# --max-canvas check below catches exactly this class of failure.
#
# Usage:
#   scripts/render-verify.sh <url> [options]
#
# Options:
#   --expect-text "STR"        Assert STR appears in the rendered DOM (repeatable)
#   --reject-text "STR"        Assert STR does NOT appear in rendered DOM (repeatable)
#   --id-nonempty ID           Assert element #ID has non-placeholder text (repeatable)
#                              (fails if text is empty, "0", "--", "Connecting...")
#   --max-canvas N             Fail if any <canvas> backing width/height > N (default 8000)
#   --screenshot PATH          Where to save the screenshot (default /tmp/render-verify.png)
#   --budget MS                Virtual time budget for async JS (default 6000)
#   --window WxH               Headless window size (default 1200,2000)
#
# Exit: 0 if all checks pass, 1 otherwise. Always prints a summary and the
# screenshot path so a human can eyeball the result.
set -uo pipefail

URL=""
declare -a EXPECT_TEXT=()
declare -a REJECT_TEXT=()
declare -a ID_NONEMPTY=()
MAX_CANVAS=8000
SHOT="/tmp/render-verify.png"
BUDGET=6000
WINDOW="1200,2000"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --expect-text) EXPECT_TEXT+=("$2"); shift 2 ;;
    --reject-text) REJECT_TEXT+=("$2"); shift 2 ;;
    --id-nonempty) ID_NONEMPTY+=("$2"); shift 2 ;;
    --max-canvas)  MAX_CANVAS="$2"; shift 2 ;;
    --screenshot)  SHOT="$2"; shift 2 ;;
    --budget)      BUDGET="$2"; shift 2 ;;
    --window)      WINDOW="$2"; shift 2 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *)  URL="$1"; shift ;;
  esac
done

if [[ -z "$URL" ]]; then
  echo "usage: scripts/render-verify.sh <url> [options]" >&2
  exit 2
fi

# Locate a Chrome/Chromium binary.
CHROME=""
for c in \
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  "/Applications/Chromium.app/Contents/MacOS/Chromium" \
  "$(command -v google-chrome 2>/dev/null)" \
  "$(command -v chromium 2>/dev/null)" \
  "$(command -v chromium-browser 2>/dev/null)"; do
  if [[ -n "$c" && -x "$c" ]]; then CHROME="$c"; break; fi
done
if [[ -z "$CHROME" ]]; then
  echo "FAIL: no headless Chrome/Chromium found (install Chrome or set PATH)" >&2
  exit 1
fi

FLAGS=(--headless --disable-gpu --no-sandbox "--virtual-time-budget=$BUDGET")

DOM="$(mktemp)"
trap 'rm -f "$DOM"' EXIT

echo "render-verify: $URL"
echo "  chrome: $CHROME"

# 1. Render DOM (executes JS) and capture a screenshot.
"$CHROME" "${FLAGS[@]}" --dump-dom "$URL" >"$DOM" 2>/dev/null
"$CHROME" "${FLAGS[@]}" "--screenshot=$SHOT" "--window-size=$WINDOW" "$URL" >/dev/null 2>&1

fails=0
note() { printf '  %-4s %s\n' "$1" "$2"; }

if [[ ! -s "$DOM" ]]; then
  note "FAIL" "headless render produced empty DOM (page failed to load?)"
  exit 1
fi

# 2. Runaway-canvas check — the lesson from the Gitlawb bug.
RUNAWAY="$(python3 - "$DOM" "$MAX_CANVAS" <<'PY'
import re, sys
dom = open(sys.argv[1]).read()
cap = int(sys.argv[2])
bad = []
for m in re.finditer(r'<canvas\b[^>]*>', dom):
    tag = m.group(0)
    w = re.search(r'\bwidth="(\d+)"', tag)
    h = re.search(r'\bheight="(\d+)"', tag)
    cid = re.search(r'\bid="([^"]+)"', tag)
    wv = int(w.group(1)) if w else 0
    hv = int(h.group(1)) if h else 0
    if wv > cap or hv > cap:
        bad.append(f"{cid.group(1) if cid else '<canvas>'} ({wv}x{hv})")
print("\n".join(bad))
PY
)"
if [[ -n "$RUNAWAY" ]]; then
  while IFS= read -r line; do
    note "FAIL" "runaway canvas (backing > ${MAX_CANVAS}px): $line"
    fails=$((fails+1))
  done <<< "$RUNAWAY"
else
  note "ok" "no runaway canvas (all backing dims <= ${MAX_CANVAS}px)"
fi

# 3. expect-text assertions
for t in "${EXPECT_TEXT[@]:-}"; do
  [[ -z "$t" ]] && continue
  if grep -qF -- "$t" "$DOM"; then
    note "ok" "found expected text: \"$t\""
  else
    note "FAIL" "missing expected text: \"$t\""; fails=$((fails+1))
  fi
done

# 4. reject-text assertions
for t in "${REJECT_TEXT[@]:-}"; do
  [[ -z "$t" ]] && continue
  if grep -qF -- "$t" "$DOM"; then
    note "FAIL" "found rejected text: \"$t\""; fails=$((fails+1))
  else
    note "ok" "absent (as required): \"$t\""
  fi
done

# 5. id-nonempty assertions — element text must not be empty/placeholder.
for id in "${ID_NONEMPTY[@]:-}"; do
  [[ -z "$id" ]] && continue
  VAL="$(python3 - "$DOM" "$id" <<'PY'
import re, sys, html
dom = open(sys.argv[1]).read(); eid = sys.argv[2]
m = re.search(r'id="'+re.escape(eid)+r'"[^>]*>(.*?)<', dom, re.S)
print(html.unescape((m.group(1) if m else "").strip()))
PY
)"
  case "$VAL" in
    ""|"0"|"--"|"Connecting...")
      note "FAIL" "#$id still shows placeholder: \"$VAL\" (JS did not populate it)"; fails=$((fails+1)) ;;
    *)
      note "ok" "#$id populated: \"$VAL\"" ;;
  esac
done

echo "  screenshot: $SHOT"
if [[ $fails -gt 0 ]]; then
  echo "render-verify: FAIL ($fails issue(s)) — inspect the screenshot above"
  exit 1
fi
echo "render-verify: PASS"
