# /review_external_pr

Check out a PR branch (typically from Copilot or an external contributor), run full quality gates, auto-fix what's possible, push fixes, and report remaining issues.

## Usage

`/review_external_pr <pr-number>`

Examples:
- `/review_external_pr 220`
- `/review_external_pr 215`

## Behavior

### 1. Fetch PR metadata

```bash
gh pr view <pr-number> --json number,title,headRefName,baseRefName,author,state,statusCheckRollup
```

If the PR is already merged or closed, abort with a message.

### 2. Check out the PR branch locally

```bash
gh pr checkout <pr-number>
```

### 3. Run quality gates (collect all results before reporting)

Run all checks, do NOT stop on first failure:

a) **Syntax check** — `python -m py_compile` on all `.py` files touched by the PR:
```bash
gh pr diff <pr-number> --name-only | grep '\.py$' | while read f; do
    python -m py_compile "$f" 2>&1 || echo "SYNTAX ERROR: $f"
done
```

b) **Ruff lint with auto-fix** — run `ruff check --fix` on PR-touched files, then re-check:
```bash
pr_files=$(gh pr diff <pr-number> --name-only | grep '\.py$' | tr '\n' ' ')
ruff check --fix $pr_files
ruff check $pr_files
```

c) **Mypy** — only for files under `swarm/`:
```bash
swarm_files=$(gh pr diff <pr-number> --name-only | grep '^swarm/.*\.py$' | tr '\n' ' ')
if [ -n "$swarm_files" ]; then
    mypy --follow-imports=skip $swarm_files
fi
```

d) **Pytest** — full test suite:
```bash
python -m pytest tests/ -x -q --tb=short
```

### 4. Report results

Print a summary table identical to `/preflight`:
```
PR #<number> Quality Review: <title>
Author: <author>   Branch: <headRefName>
─────────────────────────────
  Syntax check:   PASS / FAIL (N files with errors)
  Ruff lint:      PASS / FAIL (N issues, M auto-fixed)
  Mypy:           PASS / FAIL / SKIP (N errors)
  Tests:          PASS / FAIL (N passed, M failed)
─────────────────────────────
  Verdict:        CLEAN / N issues remain
```

### 5. If auto-fixes were applied

If `ruff check --fix` modified any files:
- Stage the fixed files: `git add <fixed-files>`
- Commit with message: `Fix lint issues from automated review`
- Append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`
- Push to the PR branch: `git push`
- Report what was fixed

### 6. If unfixable issues remain

For each category of remaining issues, provide:
- The specific errors with file:line references
- Suggested fixes (if obvious)
- Whether the issue is a blocker or advisory

Ask the user: "Should I attempt to fix the remaining issues, or leave a review comment on the PR?"

### 7. Optional: Leave a PR review

If the user chooses to leave a review comment, use `gh pr review` with:
- `--approve` if all gates pass
- `--request-changes` if blockers remain, with the summary table in the body
- `--comment` if only advisory issues remain

### 8. Return to previous branch

```bash
git checkout -    # Return to whatever branch was checked out before
```

## Constraints

- Never force-push to the PR branch.
- Never merge the PR automatically.
- If the PR branch is from a fork and you can't push, skip the auto-fix push step and report the fixes as suggestions instead.
- Do not modify files outside the PR's changed file set.
- Always install dependencies first: `python -m pip install -e ".[dev,runtime]"` if not already installed (check with `python -c "import swarm"` first).
