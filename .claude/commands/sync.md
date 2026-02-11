# /sync

Atomic fetch-rebase-push. Resolves the common "push rejected because remote advanced" problem in one step.

## Usage

`/sync [branch]`

Examples:
- `/sync` (syncs current branch)
- `/sync main` (syncs the specified branch)

## Behavior

1) **Stash if dirty**: If the working tree has uncommitted changes, run `git stash --include-untracked`. Note stash was created so it can be restored at the end.

2) **Fetch**: Run `git fetch origin`.

3) **Check divergence**: Run `git rev-list --left-right --count HEAD...origin/<branch>` to see how far ahead/behind we are.
   - If already up to date (0 behind, 0 ahead or only ahead): skip rebase, go to push.
   - If behind: rebase onto origin/<branch>.
   - If diverged (both ahead and behind): rebase onto origin/<branch>.

4) **Rebase**: Run `git rebase origin/<branch>`.
   - If rebase succeeds: continue to push.
   - If rebase has conflicts: abort the rebase (`git rebase --abort`), restore stash if created, and report the conflicting files. Do NOT attempt to resolve conflicts automatically.

5) **Push**: Run `git push origin <branch>`.
   - If no upstream is set, use `git push -u origin <branch>`.

6) **Restore stash**: If step 1 stashed changes, run `git stash pop`.

7) **Report**:
   ```
   Synced:
     Branch:  <branch>
     Rebased: <N> commits from origin/<branch>
     Pushed:  <M> local commits
     Status:  clean | stash restored
   ```

## Constraints

- Never force-push. If the push still fails after rebase, report the error and stop.
- Never resolve merge conflicts automatically — abort and report.
- Never modify git config.
- If the branch has no remote tracking branch and no `origin/<branch>` exists, report the error and stop.
- Safe to run repeatedly — if already synced, it's a no-op.
