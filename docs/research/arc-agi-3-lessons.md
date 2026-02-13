# Lessons from ARC-AGI-3 Agent Development

## Overview

ARC-AGI-3 is the first interactive reasoning benchmark — video-game-like environments on a 64x64 pixel grid where agents explore, learn rules, and solve puzzles. We built a Claude Sonnet 4.5-powered agent (`ClaudeAgent`) that uses vision, hypothesis-driven reasoning, and tool-use to compete. This document captures key lessons from 9 iterations (V1-V9) of agent development.

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

### ls20 (Movement/Rotation Puzzle)

- **Puzzle mechanics:** Two bordered boxes display pixel patterns. One is a fixed TARGET, one ROTATES when you step on a switch object. Match them, then enter the target box.
- **Switch:** Blue/teal object at ~(21,32). Each touch rotates the movable pattern 90 degrees CW.
- **Pattern cycle:** 4 states (A->B->C->D->A). State B matches the target for level 1.
- **Timer:** ~40 actions per life, 3 lives total
- **Optimal solution:** ~13 moves (1 switch touch + navigate to target)
- **Failure mode (V4-V8):** MazeNavigator explores blindly, never understands puzzle mechanics
- **V9 change:** Disabled MazeNavigator, full Claude reasoning per move

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

The key cost driver is images. Each 512x512 PNG is ~1500-3000 tokens. Sending images every turn for 200 actions adds ~400K-600K tokens. For movement games, sending every 3rd turn saves ~70% of image cost.

## Meta-Lessons

1. **Analyze recordings before iterating.** Every breakthrough came from studying JSONL recordings frame-by-frame (grid diffs, action sequences, frame counts). Understanding the game mechanics precisely was worth 10x more than prompt tweaks.

2. **Game-type detection should happen as early as possible.** The first frame's `available_actions` field contains enough information to select the right strategy. Don't waste actions figuring out what kind of game you're playing.

3. **Programmatic components complement but don't replace LLM reasoning.** The MazeNavigator is fast and cheap but blind to semantics. The LLM is slow and expensive but understands intent. The ideal agent uses programmatic execution of LLM-generated plans.

4. **Stuck detection requires ignoring "noise" changes.** Timer bars, animation frames, blinking cursors — all change the frame hash without indicating real progress. Content-aware hashing is essential.

5. **Hard caps prevent catastrophic action waste.** Without a confirm-spam cap, the agent burned 22 consecutive actions on futile confirms. Simple guardrails (max N consecutive same-action without progress) save the budget for useful exploration.

6. **Multi-layer frames are common and require deduplication.** ft09 starts with 5 layers (cursor blink animation). Rendering all 5 as separate images wastes tokens. MD5 deduplication across layers reduces this to 1-2 unique images.
