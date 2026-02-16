# /close_and_ship

Close beads issues, commit, sync, and push in one step. Use at end of a work session when you've finished one or more beads tasks.

## Usage

`/close_and_ship <bead-ids> [commit message]`

Examples:
- `/close_and_ship distributional-agi-safety-b6t`
- `/close_and_ship distributional-agi-safety-b6t distributional-agi-safety-swv "Add live integration tests"`
- `/close_and_ship` (with no args: detect in_progress beads and ask which to close)

## Behavior

1) **Identify beads to close**:
   - If `$ARGUMENTS` contains bead IDs (tokens matching `distributional-agi-safety-*` or similar short IDs), collect them.
   - If no IDs are provided, run `bd list --status=in_progress` and show the user what's in progress. Ask which to close.
   - Any token in `$ARGUMENTS` that does NOT look like a bead ID is treated as the commit message.

2) **Close beads**:
   - Run `bd close <id1> <id2> ...` to close all specified issues at once.
   - If any close fails, report the error but continue with the rest.

3) **Sync beads**:
   - Run `bd sync` (or `bd --sandbox sync` if in a session worktree — detect via `$IS_SESSION_WORKTREE` env var or by comparing `git rev-parse --git-dir` vs `--git-common-dir`).

4) **Stage and commit**:
   - Run `git status` to see what changed.
   - Stage the `.beads/issues.jsonl` file plus any other modified/untracked files relevant to the work.
   - Do NOT stage `.DS_Store`, `*.db`, `.env`, or credential files.
   - If a commit message was provided, use it. Otherwise auto-generate from the closed bead titles (e.g. "Close b6t: AWM live integration tests").
   - Append `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.

5) **Push**:
   - Push to the current branch's remote.
   - If push fails with non-fast-forward:
     1. Stash any uncommitted changes.
     2. `git pull --rebase`.
     3. Pop stash if needed.
     4. Retry push once.
   - Never force-push.

6) **Verify**:
   - Run `bd sync --status` to confirm beads are synced.
   - Print final status:
     ```
     Shipped:
       Closed:  <bead-id-1>, <bead-id-2>
       Commit:  <sha> <message>
       Branch:  <branch>
       Beads:   synced
     ```

## Constraints

- Never force-push.
- Never commit files likely containing secrets.
- If `bd close` fails for a bead, report it but don't abort the whole flow.
- If there are no code changes (only beads changes), that's fine — still commit and push the beads sync.
- If pre-commit hooks fail, stop and report — do not bypass.
