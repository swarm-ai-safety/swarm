---
name: run-query
description: Query the run index and vault for experiment history. Search by tag, date, type, or claim. Returns run metadata and pointers without loading raw data.
version: "1.0"
metadata:
  author: swarm-research-os
  generated_from: research-os-v0.1
allowed-tools: Read Glob Grep Bash
---

## EXECUTE NOW

**Query: $ARGUMENTS**

Parse the query type:
- `tag <tag>` — find runs by tag
- `date <YYYY-MM-DD>` or `date <YYYY-MM>` — find runs by date
- `type <single|sweep|redteam|study|calibration>` — find runs by type
- `claim <claim-id>` — find runs linked to a claim
- `recent [N]` — show N most recent runs (default 10)
- `stats` — show run index statistics
- Free text — search run-index.yaml and vault for matches

---

## Execution

### Source files
- Run index: `/Users/raelisavitt/swarm-artifacts/run-index.yaml`
- Vault claims: `/Users/raelisavitt/swarm-artifacts/vault/claims/`
- Vault experiments: `/Users/raelisavitt/swarm-artifacts/vault/experiments/`
- Run metadata: `/Users/raelisavitt/swarm-artifacts/runs/<run_id>/run.yaml`

### For tag queries
Search `run-index.yaml` for entries where `tags` contains the search term.

### For claim queries
1. Read the claim file at `vault/claims/<claim-id>.md`
2. Extract run references from `evidence.supporting` and `evidence.weakening`
3. Look up each run in `run-index.yaml`
4. Return a summary table

### Output format

```
## Run Query Results: {query}

| Date | Run ID | Type | Tags | Key Finding |
|------|--------|------|------|-------------|
| ... | ... | ... | ... | ... |

{N} runs matched.
```

For individual run details, also show:
- Hypothesis
- Parameters swept
- Primary result
- Claims affected
- Reproduction command
