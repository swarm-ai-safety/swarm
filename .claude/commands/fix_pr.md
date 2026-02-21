# /fix_pr

Two modes:

- **No PR reference** (default): Create a PR from uncommitted local changes.
- **With PR reference**: Resolve merge conflicts on an existing PR and optionally merge it.

## Usage

```
/fix_pr [branch-name]                    # Mode A: create PR from local changes
/fix_pr <pr-number-or-url> [fix|merge]   # Mode B: resolve conflicts / merge existing PR
```

Examples:
- `/fix_pr` (auto-generates branch name from changes)
- `/fix_pr fix/circuit-breaker-reset`
- `/fix_pr 239` (resolve conflicts on PR #239)
- `/fix_pr https://github.com/swarm-ai-safety/swarm/pull/239 merge` (resolve conflicts and merge)
- `/fix_pr 239 merge` (resolve + merge)

---

## Mode A: Create PR from Local Changes

Unlike `/pr` which expects you to describe what changed, this mode is for when you've already made edits and just want to ship them on a clean branch.

### Behavior

1. **Sanity check**:
   - Run `git status` to identify modified and untracked files.
   - If there are no changes, abort with a message.

2. **Identify relevant files**:
   - Show the user the list of modified/untracked files.
   - Ask which files to include if it's ambiguous (e.g. mix of unrelated changes).
   - Exclude: `.DS_Store`, `*.db`, `.env*`, `credentials*`.

3. **Create a clean branch**:
   - Stash all changes: `git stash push --include-untracked -m "fix_pr: temp stash"`.
   - Ensure main is up to date: `git checkout main && git pull origin main` (skip if in a session worktree — use `git fetch origin main` and branch from `origin/main` instead).
   - Create the feature branch: `git checkout -b <branch-name>` (or `git checkout -b <branch-name> origin/main` in a worktree).
   - Pop the stash: `git stash pop`.

4. **Stage and commit**:
   - Stage only the relevant files (not `git add -A`).
   - Analyze the diff and draft a concise commit message.
   - Always append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.
   - Commit. If the pre-commit hook fails on a lint issue introduced by the changes, fix it and re-commit. If it fails on a pre-existing/flaky test, note it and use `--no-verify` with an explanation.

5. **Push and open PR**:
   - `git push -u origin <branch-name>`
   - `gh pr create --head <branch-name> --base main --title "..." --body "..."`
   - Body format: `## Summary` (bullet points), `## Test plan` (checklist), Claude Code footer.

6. **Return to main**:
   - `git checkout main` (leave the feature branch pushed).
   - Print the PR URL.

---

## Mode B: Resolve Conflicts / Merge Existing PR

Triggered when the first argument is a PR number or GitHub PR URL.

### Behavior

1. **Parse PR reference**:
   - Extract `owner`, `repo`, and `pr_number` from the argument.
   - If just a number (e.g. `239`), infer from current repo's `origin` remote.
   - If a URL (e.g. `https://github.com/org/repo/pull/239`), parse all three.

2. **Fetch PR metadata**:
   - `gh pr view <pr_number> --repo <owner>/<repo> --json title,headRefName,baseRefName,state,mergeable`
   - If the PR is already merged or closed, abort.
   - If `mergeable` is not `CONFLICTING` and no `merge` flag, report "no conflicts" and skip to step 6.

3. **Clone or reuse**:
   - If the PR is on the **current repo** (same `origin`): fetch and checkout the branch directly.
   - If the PR is on a **different repo**: clone to `/tmp/fix-pr-<pr_number>` via HTTPS (fall back from SSH if it fails).
   - Checkout the PR's head branch.

4. **Merge base branch to surface conflicts**:
   - `git fetch origin <baseRefName>` then `git merge origin/<baseRefName>`.
   - If no conflicts, auto-commit the merge and skip to step 5.
   - If conflicts exist:
     a. Show the conflicting files.
     b. For each file, read the conflict regions and resolve them:
        - If both sides added non-overlapping content (e.g. both appended tests), keep both.
        - If changes overlap, analyze intent from both sides and produce a correct merge.
        - If the resolution is ambiguous, show both versions and ask the user.
     c. Verify no conflict markers remain (`<<<<<<< `, `>>>>>>> `).
     d. Stage resolved files and commit: `git add <files> && git commit` with a merge message.

5. **Push the resolution**:
   - `git push origin <headRefName>`.

6. **Merge (if requested)**:
   - Only if the user passed the `merge` keyword or explicitly asks.
   - **Step 6a — Update branch**: If branch protection requires "up to date":
     - `gh api repos/<owner>/<repo>/pulls/<pr_number>/update-branch -X PUT -f expected_head_sha=<full-sha>`
     - Wait 5 seconds for GitHub to process.
   - **Step 6b — Attempt merge**:
     - `gh pr merge <pr_number> --repo <owner>/<repo> --squash`
   - **Step 6c — Handle branch protection failures**:
     - If merge fails because checks are pending, ask the user: `--admin` (merge now) or `--auto` (merge when checks pass).
     - If `--auto` fails because auto-merge is disabled on the repo:
       - Enable it: `gh api repos/<owner>/<repo> -X PATCH -f allow_auto_merge=true`
       - Retry with `--auto`.
     - Report final status.

7. **Cleanup**:
   - If a `/tmp/fix-pr-*` clone was created, remove it: `rm -rf /tmp/fix-pr-<pr_number>`.
   - If working in the current repo, return to the previous branch: `git checkout -`.

### Constraints (Mode B)

- Never force-push.
- Never auto-resolve conflicts that are ambiguous — ask the user.
- Always verify no conflict markers remain before committing.
- Do not merge unless the user explicitly requests it (via `merge` keyword or in conversation).
- Prefer HTTPS for cloning external repos (SSH keys may not be configured).

---

## Shared Constraints

- Never force-push.
- Never commit `.DS_Store`, `*.db`, or files likely containing secrets.
- If tests fail during pre-commit, stop and report unless the failure is clearly pre-existing.
- If the branch name already exists on the remote (Mode A), abort and ask for a different name.
