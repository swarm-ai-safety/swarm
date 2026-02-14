# /eval_writeup

End-to-end pipeline: parse a Prime Intellect eval run, read the environment source, and generate a blog post.

## Usage

`/eval_writeup <eval-output-or-job-id> [--title "Custom Title"]`

Examples:
- `/eval_writeup /path/to/eval_output.txt`
- `/eval_writeup swarm_economy_openai_gpt_4.1_mini_20260212_205322_b3c5c09f --title "GPT-4.1 Mini Plays the SWARM Economy"`

## Behavior

### Phase 1: Parse eval output

Run `/parse_eval` logic on the provided eval output to extract:
- Header metadata (environment, model, examples, rollouts)
- Reward metrics table (avg, std, min, max per metric)
- Behavioral metrics table (tool call counts)
- Stop conditions, timing, token usage
- Example rollout narrative (if available)

### Phase 1.5: Environment change detection

Before writing analysis — especially when comparing against a previous run — check whether the environment code changed between runs:

1. Extract the run timestamp from the job ID (format: `YYYYMMDD_HHMMSS` after the model name).
2. Check for a previous run of the same environment. If one exists, extract its timestamp too.
3. Run `git log --after="<earlier_time>" --before="<later_time>" -- environments/<env_name>/` in the lab directory (`~/dev/my-lab/`).
4. **If commits exist between runs:**
   - Run `git diff <before_sha>..<after_sha> -- environments/<env_name>/` to get the exact changes.
   - Categorize changes: stop conditions, reward weights, scoring logic, bot behavior, system prompt, governance parameters.
   - The blog post **MUST** attribute performance differences to the code changes, not sampling variance.
   - Include before/after code snippets from the diff.
   - Structure the post around "what changed and why it mattered" rather than "what the model did."
5. **If no commits exist between runs:**
   - Safe to attribute differences to sampling variance, model behavior, or scenario randomness.
6. **If only one run exists (no comparison):**
   - Skip this phase; proceed to Phase 2.

This check prevents publishing incorrect causal narratives (e.g., attributing a 41% improvement to sampling variance when the environment code changed).

### Phase 2: Read environment source

Locate the environment source code. Search in order:
1. `~/dev/my-lab/environments/<env_name>/<env_name>.py`
2. `~/dev/my-lab/environments/<env_name>/`
3. Ask the user for the path

From the source, extract:
- **System prompt**: The instructions given to the LLM agent
- **Agent types**: Programmatic bot classes and their strategies
- **Reward functions**: How composite reward is computed (weights, normalization)
- **Governance mechanics**: Tax, audit, circuit breaker, reputation decay
- **Tool definitions**: Available actions and their effects
- **Dataset generation**: Difficulty presets, scenario variation logic

### Phase 3: Cross-reference and analyze

Combine parsed metrics with environment knowledge to identify:

1. **Dominant strategy**: Which tools drove the most reward? (correlate tool call frequency with payoff_reward)
2. **Underused capabilities**: Tools available but rarely/never called (e.g. reply, vote, propose_trade)
3. **Failure modes**: High claim_task with low submit_work = task starvation. High reject_proposal = trade aversion.
4. **Metric anomalies**: If interaction_quality is low but payoff_reward is high, explain the denominator dilution.
5. **Governance interaction**: Did any rollout hit the circuit breaker? How did tax/audit affect outcomes?
6. **Connections to SWARM theory**: Map observations to concepts from the distributional safety framework (adverse selection, soft labels, phase transitions, etc.)

### Phase 4: Generate blog post

Write a blog post in the style of existing SWARM blog posts (see `docs/blog/` for reference). Structure:

1. **Title + subtitle** — model name + environment + hook
2. **The environment** — what the LLM faces (agents, tools, governance, soft labels)
3. **Results table** — aggregate metrics
4. **Behavioral analysis** — what the model actually does, with tool call breakdown
5. **Key observations** — 3-5 named findings with analysis
6. **Connections to SWARM** — how observations relate to governance theory
7. **Next steps** — what to test next (RL training, adversarial pressure, multi-model comparison)
8. **Footer** — eval job ID, model, environment, timing metadata

### Phase 5: Publish via /add_post

Use the `/add_post` workflow:
1. Derive slug from title
2. Create `docs/blog/<slug>.md`
3. Update `docs/blog/index.md`
4. Update `mkdocs.yml` nav
5. Verify mkdocs build

Do NOT commit — let the user `/ship` when ready.

## Constraints

- Always read the environment source before writing — don't guess at mechanics.
- Cross-reference behavioral stats against the actual tool implementations to ensure accuracy.
- Use the same tone as existing blog posts: technical, data-driven, no hype, concrete numbers.
- Include the eval metadata footer with job ID, model, provider, timing.
- If the eval used multiple models, generate a comparison post instead of a single-model analysis.
- Do NOT fabricate metrics — if a number isn't in the eval output, don't invent it.
- When comparing runs, ALWAYS run Phase 1.5 before writing. Never attribute performance differences to sampling variance without first ruling out environment code changes.
