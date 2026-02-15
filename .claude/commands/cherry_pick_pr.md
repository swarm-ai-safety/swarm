# /cherry_pick_pr

Cherry-pick one or more commits onto a new branch from main and open a PR. Useful when a commit landed on the wrong branch or you want to split a multi-commit branch into separate PRs.

## Usage

`/cherry_pick_pr <commit-sha> [commit-sha...] [branch-name]`

Examples:
- `/cherry_pick_pr abc1234` (cherry-pick one commit, auto-generate branch name)
- `/cherry_pick_pr abc1234 def5678 fix/gastown-bugs` (cherry-pick two commits onto a named branch)

## Behavior

1. **Validate inputs**:
   - Verify each commit SHA exists: `git cat-file -t <sha>`.
   - If any SHA is invalid, abort with an error.
   - Show the commit messages for confirmation: `git log --oneline -1 <sha>` for each.

2. **Create a clean branch from main**:
   - `git fetch origin main`
   - `git checkout -b <branch-name> origin/main`
   - If no branch name provided, derive one from the first commit's message (e.g. `fix/circuit-breaker-reset`).

3. **Cherry-pick commits**:
   - For each SHA in order: `git cherry-pick <sha> --no-commit`, then `git commit --no-verify` with the original message plus `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.
   - If a cherry-pick has conflicts, stop and report the conflicts. Do not auto-resolve.

4. **Push and open PR**:
   - `git push -u origin <branch-name>`
   - `gh pr create --head <branch-name> --base main --title "..." --body "..."`
   - Title: derived from commit message(s). If multiple commits, summarize.
   - Body format: `## Summary` (bullet points from each commit), `## Test plan`, Claude Code footer.

5. **Clean up source branch** (optional):
   - If the commits came from a feature branch (not main), ask the user if they want to remove those commits from the source branch.
   - If yes: `git checkout <source-branch> && git rebase --onto <sha>~1 <sha> <source-branch>` (for single commit) or interactive rebase guidance for multiple.
   - If no: leave the source branch as-is.
   - Never force-push the source branch without explicit confirmation.

6. **Return to previous branch**:
   - `git checkout <original-branch>` (wherever the user was before).
   - Print the PR URL.

## Constraints

- Never force-push without explicit user confirmation.
- Never auto-resolve cherry-pick conflicts.
- If the branch name already exists on the remote, abort and ask for a different name.
- Do not merge the PR automatically.
