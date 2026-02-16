# /branch_guard

Check that you're on the expected git branch before starting work. Use at the beginning of a session or before creating/editing files.

## Usage

`/branch_guard [expected-branch]`

Examples:
- `/branch_guard feat/isometric-viz`
- `/branch_guard main`
- `/branch_guard` (just show current branch and recent context)

## Behavior

1) **Check current branch**:
   - Run `git branch --show-current`.
   - Run `git status --short` to show working tree state.

2) **Compare against expected**:
   - If `$ARGUMENTS` specifies an expected branch and the current branch doesn't match:
     - Warn clearly: "WARNING: On branch `<current>`, expected `<expected>`"
     - Show uncommitted changes on current branch (if any).
     - Ask: "Switch to `<expected>`?" before doing anything.
   - If they match, confirm: "On correct branch: `<expected>`"

3) **If no expected branch given**:
   - Show the current branch, its tracking remote, and how many commits ahead/behind.
   - Show any in-progress beads (`bd list --status=in_progress`) to remind what work is active.
   - This gives enough context to decide if you're in the right place.

4) **Stale branch detection**:
   - If the current branch has no commits ahead of its upstream AND there are no local changes, note: "Branch is clean and up-to-date — nothing in progress here."

## Constraints

- Never switch branches automatically without asking.
- If there are uncommitted changes, warn before any branch switch.
- This is a read-only check — it never modifies files or git state.
