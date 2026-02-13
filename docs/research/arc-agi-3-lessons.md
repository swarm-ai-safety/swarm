# Lessons from ARC-AGI-3 Agent Development

## Overview

ARC-AGI-3 is the first interactive reasoning benchmark — video-game-like environments on a 64x64 pixel grid where agents explore, learn rules, and solve puzzles. We built a Claude Sonnet 4.5-powered agent (`ClaudeAgent`) that uses vision, hypothesis-driven reasoning, and tool-use to compete. This document captures key lessons from 11 iterations (V1-V11) of agent development.

## Key Architectural Decisions

### Game Type Detection is Critical

ARC-AGI-3 environments are not homogeneous. We identified three distinct game types from the `available_actions` field:

| Game Type | Actions Available | Example | Strategy |
|-----------|------------------|---------|----------|
| **Movement** | [1,2,3,4] (directional) | ls20 | Interactive puzzle with switches, pattern matching |
| **ARC Puzzle** | [1,2,3,4,5,6] (move+confirm+click) | ft09 | Classic input/output transformation, click to edit cells |
| **Click Only** | [6] (click) | vc33 | Pure click-based puzzle solving |

**Lesson: A single prompt strategy fails across game types.** The agent must detect the game type from available actions and dispatch to game-type-specific system prompts. Our initial maze-focused prompt caused the ARC puzzle agent to repeatedly reset instead of clicking, and the click-only agent to click random coordinates.

### The Timer Bar Breaks Naive Change Detection

Every game has a progress bar at row 63 that ticks down 2 pixels per action (~32-40 actions per cycle). This means:

- `frame_hash` changes every action regardless of meaningful grid changes
- `stuck_counter` stays at 0 even when the agent is completely stuck
- The agent never receives "you're stuck" guidance from the prompt

**Fix: `content_hash`** — hash only rows 0-62, ignoring the timer bar. This made stuck detection functional and immediately improved behavior (the agent started resetting and trying new approaches when truly stuck).

### MazeNavigator is Wrong for Interactive Puzzles

V4-V6 introduced a `MazeNavigator` (persistent DFS with graph building across timer resets) designed for movement games. It was fast (instant, no API calls) and could efficiently explore maze corridors. However:

- **Interactive objects are invisible to DFS.** The navigator treats every passable cell identically. Walking onto a rotation switch has the same weight as walking onto empty floor.
- **DFS explores exhaustively when it should be purposeful.** The ls20 puzzle can be solved in ~13 moves (touch switch once, navigate to target). MazeNavigator used 40+ actions exploring the entire grid without ever understanding the puzzle.
- **Early handoff kills reasoning.** With `MAZE_MODE_AFTER=2`, Claude only got 2 reasoning calls before MazeNavigator took over. Not enough to observe the puzzle mechanics.

**Lesson: For interactive puzzles, let the LLM reason about every move.** The cost (5-8 seconds per API call, ~7K tokens per turn) is high but necessary. Blind programmatic exploration cannot solve puzzles that require understanding cause-and-effect relationships.

### The `navigate_to` Virtual Tool: LLM-Directed Programmatic Navigation

V10 introduced a middle ground between full LLM reasoning per move and blind programmatic exploration: a `navigate_to(x, y)` virtual tool. Claude specifies a destination, and the agent executes a greedy Manhattan-distance path programmatically (no API calls during transit).

**Benefits:**
- Reduced API calls from 200 (every move) to ~20-30 (only at decision points)
- 10-20x faster navigation through open areas
- Claude retains strategic control: it decides WHERE to go, the agent handles HOW

**Challenges:**
- **Greedy paths can't handle walls.** A simple "move toward target" path hits walls in corridor-heavy games. The agent needs wall detection, retry logic, and progress-based abort.
- **Wall retry oscillation.** When blocked going left, the agent goes up/down to get around, then tries left again, creating infinite loops. Fixed with progress-based abort: if Manhattan distance to target doesn't decrease after 12 steps, abort and return to Claude.
- **Content change detection is noisy.** We tried interrupting navigation when the grid changed (to detect puzzle events like switch activation). But the timer bar and moving sprites cause 50+ pixel changes per step, drowning out real events. Abandoned in favor of letting Claude observe at arrival.
- **Arrival precision.** With 5-pixel movement steps, the player can only land on positions modulo 5 from the start. A "within 3 pixels" arrival threshold may not mean the player is ON the target.

### Recording Analysis is the Most Important Debugging Tool

The single biggest breakthrough came from analyzing JSONL recordings frame-by-frame:

- **V9:** Discovered the timer bar (rows 62-63) changes every action, breaking stuck detection. Led to `content_hash` which immediately improved ft09 from 125 to 21 actions for level 1.
- **V11:** Discovered that ls20 has NO switch — the "blue object" was the player sprite. The pattern rotation is autonomous, not player-triggered. Every previous iteration was operating on a fundamentally wrong model of the game mechanics.

**Lesson: Never iterate on prompts without understanding the ground truth.** 6 versions (V4-V10) were spent optimizing for a game mechanic that didn't exist. One recording analysis session revealed the real mechanics and required a complete prompt rewrite.

## Prompt Engineering Insights

### Confirm Spam is a Real Failure Mode

When the agent successfully modifies some cells in the ARC puzzle, it often enters a "confirm loop" — submitting the same incorrect answer 20+ times consecutively. This happens because:

1. The frame changes after each confirm (timer ticks), so the agent doesn't detect stuckness
2. The prompt doesn't explicitly warn against repeated confirms
3. Claude's reasoning gets anchored on "I think my answer is correct" and doesn't re-evaluate

**Mitigations:**
- Hard cap: after 3 consecutive confirms without level advancement, force a reset
- Prompt warning: "If confirm doesn't advance to the next level, your answer is WRONG"
- Content-hash-based stuck detection that ignores the timer

### Objects List > Vision for Coordinates

Claude's vision on 512x512 upscaled images is good for understanding layout but imprecise for exact pixel coordinates. The `extract_objects()` function provides:

```
- orange (10px): x=[39-43] y=[44-48], center (41,46)
- blue (6px): x=[19-23] y=[30-34], center (21,32)
```

This structured data is more useful than vision alone for navigation. The movement prompt should direct Claude to "use the objects list coordinates to plan direct paths" rather than trying to visually estimate positions.

### System Prompt Size Matters for Cost

Each Claude call includes the full system prompt. With prompt caching (`cache_control: {"type": "ephemeral"}`), repeated calls within 5 minutes reuse cached tokens. But the system prompt still contributes to context window pressure.

Key optimizations:
- Keep system prompt under 500 tokens
- Use sliding window message history (6 turns = 18 messages max)
- Send images selectively (every 3rd turn for movement, every turn for ARC)
- MAX_TOKENS=512 for responses (tool calls are compact)

## Game-Specific Findings

### ft09 (ARC Puzzle)

- **Grid structure:** 4 quadrants — top-left (example input), top-right (example output), bottom-left (test input), bottom-right (editable test output)
- **Cell size:** Each logical cell is a 6x6 pixel block in the 64x64 grid
- **Click behavior:** Each click cycles a cell to the next color
- **Frame layers:** Initially 5 layers (blinking cursor animation), collapses to 1 after first action
- **Transformation type (level 1):** Makes 3x3 grids 4-fold symmetric
- **V8 result:** 1 level completed (score 2.0) in 125 actions (baseline: 15). First-ever level completion.
- **V9 result:** 1 level completed (score **11.36**) in 21 actions (baseline: 15, ratio: 1.4x). 5.7x score improvement over V8.
- **V9 improvement:** Content-hash stuck detection + confirm spam cap → level 1 solved in 21 actions (was 125 in V8)

### ls20 (Movement/Pattern Puzzle)

- **Actual mechanics (discovered V11 via recording analysis):**
  - A **reference pattern box** (upper area, rows 8-16, cols 32-40) shows the target pattern
  - An **answer box** (bottom-left, rows 52-63, cols 0-11) displays a pattern that **rotates automatically**
  - A **moving indicator** (pink+maroon sprite) autonomously cycles through green corridors
  - Each time the indicator reaches the reference box, the answer box pattern rotates 90 degrees
  - Pattern cycle: 4 states (original → H-flip → 180-rot → V-flip)
  - The player must navigate to the answer box when its pattern matches the reference
  - **There is no switch.** What we identified as a "blue switch" was actually the player sprite's blue accent pixels
- **Grid structure:** Green (3) walls form corridors and box borders. Yellow (4) and gray (5) are walkable. The player starts at ~(41,45) in an open area surrounded by corridors.
- **Timer:** ~40 actions per life, 3 lives total, 7 levels to complete
- **V4-V8 failure:** MazeNavigator explores blindly, never understands puzzle mechanics
- **V9 change:** Disabled MazeNavigator, full Claude reasoning per move — still wandered aimlessly
- **V10-V10.3:** Added `navigate_to` virtual tool for efficient multi-step navigation. Fixed wall detection bug, added progress-based abort for infinite loops. Claude correctly navigated to key locations but didn't complete levels.
- **V11:** Completely rewrote prompt after recording analysis revealed true game mechanics. Previous prompts told Claude to find a non-existent "switch" and navigate to the wrong target box.

### vc33 (Click Only)

- **Grid:** Green left half, black right half, with maroon/grey/teal objects
- **Timer:** Cycles every ~8 frames (fast)
- **7 levels to complete**
- **Grid never changed across 111 frames** in V6 run — suggests clicks weren't landing on interactive targets
- Not yet tested with game-type-aware V8/V9 agent

## Cost Analysis

| Version | Game | Actions | Input Tokens | Output Tokens | Levels | Cost (est) |
|---------|------|---------|--------------|---------------|--------|------------|
| V6 | ft09 | 201 | ~50K | ~3K | 0 | ~$0.15 |
| V8 | ft09 | 201 | ~600K | ~10K | 1 | ~$2.00 |
| V8 | ls20 | 201 | ~16K | ~1K | 0 | ~$0.05 |
| V9 | ft09 | 201 | ~1.6M | ~22K | 1 (score 11.36) | ~$5.00 |
| V9 | ls20 | 201 | ~1.8M | ~14K | 0 | ~$6.00 |
| V10 | ls20 | 201 | ~500K | ~7K | 0 | ~$1.50 |
| V10.4 | ls20 | 201 | ~570K | ~7K | 0 | ~$1.70 |

The key cost driver is images. Each 512x512 PNG is ~1500-3000 tokens. Sending images every turn for 200 actions adds ~400K-600K tokens. For movement games, sending every 3rd turn saves ~70% of image cost.

The `navigate_to` tool (V10+) significantly reduced token usage for movement games by replacing per-step API calls with programmatic navigation, cutting total cost from ~$6 (V9) to ~$1.50 (V10).

## Meta-Lessons

1. **Analyze recordings before iterating.** Every breakthrough came from studying JSONL recordings frame-by-frame (grid diffs, action sequences, frame counts). Understanding the game mechanics precisely was worth 10x more than prompt tweaks. We wasted 6 versions optimizing for non-existent game mechanics because we hadn't studied the recordings carefully enough.

2. **Game-type detection should happen as early as possible.** The first frame's `available_actions` field contains enough information to select the right strategy. Don't waste actions figuring out what kind of game you're playing.

3. **Programmatic components complement but don't replace LLM reasoning.** The MazeNavigator is fast and cheap but blind to semantics. The LLM is slow and expensive but understands intent. The ideal agent uses programmatic execution of LLM-generated plans — the `navigate_to` tool is this pattern: Claude decides the destination, the agent executes the path.

4. **Stuck detection requires ignoring "noise" changes.** Timer bars, animation frames, blinking cursors — all change the frame hash without indicating real progress. Content-aware hashing is essential. But even "content-aware" hashing (skipping timer rows) isn't enough — moving sprites, indicators, and autonomous animations cause 50+ pixel changes per step. True stuck detection may require tracking only player-relevant metrics (position, level completion).

5. **Hard caps prevent catastrophic action waste.** Without a confirm-spam cap, the agent burned 22 consecutive actions on futile confirms. Without a progress-based nav abort, the agent burned 25 consecutive actions oscillating between two wall positions. Simple guardrails save the budget for useful exploration.

6. **Multi-layer frames are common and require deduplication.** ft09 starts with 5 layers (cursor blink animation). Rendering all 5 as separate images wastes tokens. MD5 deduplication across layers reduces this to 1-2 unique images.

7. **Don't hardcode game mechanics you haven't verified.** The movement prompt's "SPATIAL LAYOUT" section hardcoded specific coordinates (switch at y~30-35, target at y<25). This was based on an incorrect model of the game. When the model was wrong, Claude was given precise but incorrect instructions, worse than no instructions at all. Game-specific hints should only be added after frame-level verification from recordings.

8. **Greedy navigation fails in complex environments.** A greedy Manhattan-distance path works for open areas but fails in corridor-heavy environments with walls. The agent needs proper pathfinding (BFS/A*) or at minimum, progress-based abort with fallback to exploratory individual moves. Wall retry + perpendicular approach creates oscillation; progress tracking catches it.

9. **The objects list can be misleading.** `extract_objects()` identifies colored clusters by pixel count, but can't distinguish game-relevant entities from decorative elements. The "blue switch" at (21,32) was actually the player sprite's accent pixels. Objects need semantic context (what role they play) not just spatial data (where they are).

10. **LLM vision is better at layout comprehension than coordinate extraction.** Claude can identify "there are two pattern boxes and a corridor structure" from the image better than it can read exact pixel coordinates. But for navigation, it needs the objects list's precise coordinates. The combination is essential: vision for understanding, structured data for action.

## Bug Taxonomy

A catalog of bugs encountered during development, useful for anyone building similar agents:

| Bug | Version | Impact | Root Cause | Fix |
|-----|---------|--------|------------|-----|
| Timer bar breaks stuck detection | V8→V9 | Agent never detects stuckness, stuck_counter=0 always | `frame_hash` includes timer bar pixels that change every action | `content_hash` that hashes only rows 0-62 |
| Confirm spam | V8→V9 | 22+ actions wasted confirming wrong answers | No cap on consecutive confirms; timer changes mask stuckness | Hard cap: 3 consecutive confirms → force RESET |
| `_prev_player_pos` update ordering | V10→V10.1 | Every nav step looks like a wall hit, navigate_to aborts after 3 steps | `_prev_player_pos` updated before wall detection comparison | Save `old_player_pos` before update, use it in comparison |
| Claude navigates to wrong target | V10.1→V10.2 | Goes to (59,61) instead of target box | No spatial guidance in prompt; Claude guesses wrong box | Added SPATIAL LAYOUT section (later found to be wrong itself) |
| Content change interruption fires every step | V10.3 | navigate_to interrupted on every step, defeating its purpose | Player movement changes grid (2 px), timer changes grid (49+ px), threshold too low | Changed to pixel magnitude threshold (>5), then discovered 51px noise floor from timer |
| Nav queue infinite oscillation | V10.3b→V10.4 | 25+ actions wasted bouncing between two positions | Wall retry counter resets on successful perpendicular moves | Progress-based abort: if no distance improvement after 12 steps, abort |
| Wrong game model (entire paradigm) | V4→V11 | 6 versions optimizing for non-existent mechanics | Never analyzed recordings to verify switch/rotation hypothesis | Frame-by-frame recording analysis revealed autonomous indicator, no switch |

## Version History

| Version | Key Changes | ft09 Score | ls20 Score |
|---------|------------|------------|------------|
| V1-V3 | Basic vision + tool calling | 0 | 0 |
| V4-V6 | MazeNavigator (DFS), multi-layer dedup | 0 | 0 |
| V7-V8 | Game type detection, game-specific prompts | 2.0 | 0 |
| V9 | content_hash, confirm spam cap, disable MazeNav | **11.36** | 0 |
| V10 | navigate_to virtual tool | — | 0 |
| V10.1 | Wall detection fix | — | 0 |
| V10.2 | Spatial layout hints | — | 0 |
| V10.3-10.4 | Content change detection, progress abort | — | 0 |
| V11 | Corrected game mechanics from recording analysis | — | TBD |
