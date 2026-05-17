---
name: verify
description: Run all vault integrity checks — schema validation, evidence provenance, wiki-links, index consistency, and claim health. Triggers on "/verify", "/verify --claims", "/verify --quick", "check vault health", "verify the vault".
version: "1.0"
generated_from: "swarm-research-os-v0.1"
user-invocable: true
allowed-tools: Read, Glob, Grep, Bash
context: fork
model: opus
argument-hint: "[--claims | --quick | --full]"
---

## EXECUTE NOW

**Target: $ARGUMENTS**

Parse immediately:
- `--claims` → only check claim cards (Step 2)
- `--quick` → schema validation only, skip wiki-links and health (Steps 1–2)
- `--full` or empty → run everything (Steps 1–6)

**Execute all applicable steps without asking for confirmation.**

---

## Step 1: Validate Run Schemas

Run the validation script:

```bash
python scripts/validate-run.py --all
```

Report:
- Total runs validated
- Number of errors
- List any errors with run_id and field

---

## Step 2: Validate Vault Notes

Run the vault validation script:

```bash
python scripts/validate-vault.py --all
```

This checks:
- Claim cards: schema compliance, description length, confidence validity, evidence provenance, topics footer
- Experiment notes: required fields, description constraints, run references, topics footer
- Index consistency: claims referenced in _index.md, experiments directory referenced

Report each category separately.

---

## Step 3: Vault Health Audit

Run the comprehensive health audit:

```bash
python scripts/vault-health.py
```

This checks:
- Claim status distribution (active, weakened, superseded, retracted)
- Confidence distribution (high, medium, low, contested)
- Stale claims (not updated in >30 days)
- Broken run references in evidence
- Broken wiki-links (excluding templates)
- Orphaned notes not in index

---

## Step 4: Check Run Index Freshness

```bash
cp run-index.yaml run-index.yaml.bak
python scripts/index-runs.py
diff run-index.yaml.bak run-index.yaml
rm run-index.yaml.bak
```

If the index changed, report that it was stale and has been rebuilt.
If no changes, report the index is up to date.

---

## Step 5: Check Unsynthesized Runs

```bash
python scripts/generate-note.py
```

(No arguments = lists unsynthesized runs)

Report how many runs lack experiment notes.

---

## Step 6: Generate Dashboard

Compile all results into a single dashboard:

```
## Vault Verification Report

### Schema validation
- Runs: {N} validated, {E} errors
- Claims: {N} validated, {E} errors
- Experiments: {N} validated, {E} errors

### Health
- Claims: {total} total ({active} active, {weakened} weakened)
- Confidence: {high} high, {medium} medium, {low} low, {contested} contested
- Stale (>30d): {stale}
- Broken references: {broken_refs}
- Broken wiki-links: {broken_links}

### Index
- Run index: {status}
- Unsynthesized runs: {count}

### Overall
{PASS if 0 errors, WARN if only warnings, FAIL if errors}
```

---

## Critical Constraints

1. **Never modify vault files.** This skill is read-only (except for index rebuild check).
2. **Report all errors explicitly.** Do not summarize away individual errors.
3. **Distinguish errors from warnings.** Broken references are errors. Stale claims are warnings.
4. **Always restore state.** If the index freshness check modifies run-index.yaml, restore the original if the user didn't ask for a rebuild.
