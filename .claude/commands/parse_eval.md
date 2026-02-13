# /parse_eval

Parse raw `prime eval run` output into structured metrics tables.

## Usage

`/parse_eval <eval-output-file-or-job-id>`

Examples:
- `/parse_eval /path/to/eval_output.txt`
- `/parse_eval swarm_economy_openai_gpt_4.1_mini_20260212_205322_b3c5c09f`

If given a file path, read it directly. If given a job ID, look for the most recent matching output in the tool results cache or ask the user to provide the output.

## Behavior

### 1) Extract header metadata

From the `--- Evaluation ---` block, extract:
- **Environment**: name
- **Model**: provider/model
- **Provider**: inference endpoint URL
- **Examples**: count
- **Rollouts per example**: count

### 2) Extract reward metrics

From the `--- All ---` / `Rewards:` section, parse each metric line:

```
metric_name: avg - X.XXX, std - X.XXX
```

Build a summary table:

| Metric | Avg | Std | Min | Max |
|--------|-----|-----|-----|-----|
| reward | 0.830 | 0.092 | 0.690 | 1.028 |
| payoff_reward | 0.701 | 0.101 | 0.540 | 0.916 |
| ... | ... | ... | ... | ... |

For each metric, also parse the per-rollout arrays (r1, r2, r3, r4, ...) to compute min/max across all rollouts.

### 3) Extract behavioral metrics

Parse tool call count metrics (lines matching `*_calls: avg - X.XXX, std - X.XXX`):

| Action | Avg | Std | Total |
|--------|-----|-----|-------|
| claim_task | 6.9 | 3.2 | 549 |
| submit_work | 3.1 | 1.5 | 251 |
| ... | ... | ... | ... |

Total = avg * (examples * rollouts_per_example).

### 4) Extract info metrics

Parse:
- `is_truncated`: avg
- `stop_conditions`: breakdown (e.g. `no_tools_called: 0.625, max_turns_reached: 0.375`)
- `num_turns`: avg, std
- `total_tool_calls`: avg, std

### 5) Extract timing

Parse the `Timing:` section:
- generation: min, mean, max
- scoring: min, mean, max
- total: min, mean, max

### 6) Extract usage

Parse `Usage:` section:
- input_tokens (avg)
- output_tokens (avg)

### 7) Output

Print the complete structured summary as markdown. Also output a JSON block with all parsed metrics for programmatic use:

```json
{
  "environment": "swarm-economy",
  "model": "openai/gpt-4.1-mini",
  "examples": 20,
  "rollouts_per_example": 4,
  "rewards": {
    "reward": {"avg": 0.830, "std": 0.092, "min": 0.690, "max": 1.028},
    ...
  },
  "behavioral": {
    "claim_task_calls": {"avg": 6.862, "std": 3.197},
    ...
  },
  "stop_conditions": {"no_tools_called": 0.625, "max_turns_reached": 0.375},
  "timing": {"generation_mean_s": 28, ...},
  "usage": {"input_tokens_avg": 70731, "output_tokens_avg": 553}
}
```

### 8) Example rollout (optional)

If the output includes a `--- Example ---` section with a prompt/completion table, extract the first example rollout as a readable narrative summary:
- What actions the model took in order
- Key outcomes (payoffs, reputation changes, audit results)
- How the rollout ended (idle, max turns, circuit breaker)

## Constraints

- Handle ANSI escape codes in raw terminal output (strip them before parsing).
- Handle both complete output (from file) and truncated output (from tool result preview).
- If a metric is missing, omit it from the table rather than guessing.
- Do NOT modify any files — this is a read-only analysis command.
