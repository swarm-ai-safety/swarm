---
name: synthesize
description: Synthesize a SWARM run into vault notes. Reads run.yaml and summary.json, generates an experiment note, checks existing claims for new evidence, updates the master index. Triggers on "/synthesize", "/synthesize [run_id]", "synthesize this run", "generate notes for run".
version: "1.0"
generated_from: "swarm-research-os-v0.1"
user-invocable: true
allowed-tools: Read, Write, Edit, Glob, Grep, Bash
context: fork
model: opus
argument-hint: "[run_id] — the run folder name to synthesize (e.g. 20260213-202050_baseline_governance_v2)"
---

## EXECUTE NOW

**Target: $ARGUMENTS**

Parse immediately:
- If target is a run_id: synthesize that specific run
- If target is empty: list recent runs without experiment notes and ask which to synthesize
- If target is `--all`: synthesize all runs that lack experiment notes
- If target is `--batch N`: synthesize the N most recent unsynthesized runs

**Execute steps 0 through 6 in order. Do not ask for confirmation between steps.**

---

## Step 0: Validate Target

1. Check that `runs/{run_id}/` exists
2. Check that `runs/{run_id}/run.yaml` exists — if not, run `python scripts/backfill-run-yaml.py --run-id {run_id}` to generate it
3. Check if `vault/experiments/{run_id}.md` already exists — if so, report "Already synthesized" and stop (unless `--force`)

---

## Step 1: Read Run Metadata

Read `runs/{run_id}/run.yaml` and extract:
- `slug`, `created_utc`
- `experiment.type`, `experiment.hypothesis`, `experiment.swept_parameters`, `experiment.seeds`, `experiment.total_runs`
- `results.status`, `results.primary_metric`, `results.primary_result`, `results.significant_findings`
- `artifacts.*` — what files are available
- `tags`
- `links.claims` — any pre-linked claims

If `experiment.type` is `sweep` or `study`, also read the summary JSON file referenced in `artifacts.summary`.

---

## Step 2: Read Summary Data

Based on `experiment.type`:

**For `sweep`:** Read `summary.json`. Extract:
- `total_runs`, `total_hypotheses`
- `n_bonferroni_significant` or `bonferroni_survivors`
- `swept_parameters`
- Top 3 most significant results (by effect size)
- `p_hacking_audit`

**For `redteam`:** Read `report.json`. Extract:
- `robustness_score`, `grade`
- `attacks_tested`, `attacks_successful`
- `vulnerabilities` (critical and high severity)

**For `study`:** Read `analysis/summary.json` or `summary.json`. Extract:
- `descriptive` statistics per condition
- `pairwise_tests` — significant comparisons
- `bonferroni_survivors`

**For `single`:** Read `history.json`. Extract:
- Final epoch metrics: welfare, toxicity, acceptance rate
- Agent-type breakdown if available

**For `calibration`:** Read `summary.json` and `recommendation.json`. Extract:
- Parameter values tested
- Recommended configuration

---

## Step 3: Generate Experiment Note

Create `vault/experiments/{run_id}.md` using the template at `vault/templates/experiment-note.md`.

**Critical rules:**
- **description** must be ≤200 chars, must add info beyond the title, no trailing period
- **Title (H1)** must be a prose proposition describing the experiment's purpose and outcome
- **Claims affected** section: scan `vault/claims/` for claims whose topics overlap with this run's tags (≥2 tag overlap). List them with relationship context.
- **Key results** section: include effect sizes, p-values, and correction methods. Use connective words.
- **Reproduction** section: construct the CLI command from run.yaml metadata

**Quality gate:** Can you complete "This experiment showed that [title]"?

---

## Step 4: Check Claims for New Evidence

For each claim file in `vault/claims/`:

1. Read the claim's frontmatter (`domain`, `evidence`, topics footer)
2. Check tag overlap between the claim's topics and the run's tags
3. If overlap ≥ 2 tags:
   a. Read the run's summary data
   b. Determine if the run's results **support** or **weaken** the claim
   c. If the run provides new evidence not already listed in `evidence.supporting` or `evidence.weakening`:
      - Report the finding but **do not auto-edit the claim**
      - Instead, output a structured recommendation:

```
### Claim update recommendation: {claim_id}

**Action**: Add supporting evidence / Add weakening evidence / Add boundary condition
**Run**: {run_id}
**Detail**: {effect size, p-value, correction}
**Suggested edit**: Add to evidence.supporting: ...
```

**Critical:** Never automatically modify claim files. Claims require human judgment. Only recommend updates.

---

## Step 5: Rebuild Index

1. Run `python scripts/index-runs.py` to update `run-index.yaml`
2. Regenerate `vault/_index.md`:
   - Re-scan all vault subdirectories for notes
   - Update counts and listings
   - Preserve any manually-added content in the index

---

## Step 6: Report

Print a structured summary:

```
## Synthesis Complete: {run_id}

### Experiment note
✓ vault/experiments/{run_id}.md
  Type: {experiment_type}
  Key finding: {one-line summary}

### Claims checked
- {N} claims scanned for tag overlap
- {M} claims have overlapping evidence:
  {list of claim_id + recommended action}

### Index
✓ run-index.yaml rebuilt ({total} runs)
✓ vault/_index.md updated

### Next steps
- Review claim update recommendations above
- Edit claim cards manually if evidence warrants it
- Run /verify to check vault health
```

---

## Batch Mode (--all or --batch N)

When processing multiple runs:

1. Glob `runs/*/run.yaml` to find all runs
2. Glob `vault/experiments/*.md` to find already-synthesized runs
3. Compute the unsynthesized set
4. Sort by date (newest first)
5. If `--batch N`, take only the first N
6. For each run, execute Steps 1–4 (skip Step 5 until all runs are done)
7. Run Step 5 once at the end
8. Print a batch summary:

```
## Batch Synthesis Complete

- {N} runs synthesized
- {M} claim update recommendations
- {K} runs skipped (already synthesized)
```

---

## Critical Constraints

1. **Never auto-edit claims.** Claims require human judgment. Only recommend updates.

2. **description field is mandatory** on every generated note. It must add info beyond the title. Max 200 chars. No trailing period.

3. **Topics footer is mandatory** on every generated note. Link to `[[_index]]` at minimum.

4. **Prose-as-title convention.** The H1 heading is a complete proposition, not a label.

5. **Effect sizes and correction methods** must be included in Key Results. Do not report raw p-values without specifying the correction method (Bonferroni, Holm, BH).

6. **Wiki-links inline as prose.** Example: "This replicates the finding from [[baseline governance v2 sweep]] with finer granularity."

7. **Idempotent.** Running /synthesize twice on the same run should produce no changes (unless --force).
