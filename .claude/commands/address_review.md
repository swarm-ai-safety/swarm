# /address-review

Fetch review comments on a GitHub PR, apply fixes, push, and reply to each comment thread.

## Usage

`/address-review [pr-number] [--repo owner/repo]`

Examples:
- `/address-review` (auto-detect from current branch or most recent open PR by `--author=@me`)
- `/address-review 11`
- `/address-review 11 --repo The-Vibe-Company/companion`

## Behavior

### 1) Find the PR

- If `<pr-number>` and `--repo` are provided, use them directly.
- If only `<pr-number>` is provided, use the current repo's remote.
- If neither is provided:
  - Check if the current branch tracks a PR: `gh pr view --json number,url`
  - If not, search across all repos: `gh search prs --author=@me --state=open --limit 10`
  - Present matches and let the user pick.

### 2) Fetch review context

Run in parallel:
- `get_pull_request` — title, body, head/base refs, head SHA
- `get_pull_request_reviews` — review summaries
- `get_pull_request_comments` — inline review comments (the actionable items)
- `get_pull_request_files` — changed files list

### 3) Present comment summary

Display a table of unresolved review comments:

```
| # | File | Line | Author | Issue |
|---|------|------|--------|-------|
| 1 | src/api/bridge.ts | 90 | copilot | requestIndex memory leak |
| 2 | src/api/bridge.ts | 190 | copilot | ensureController race condition |
```

If there are no review comments to address, report that and stop.

### 4) Clone or locate the source

- Determine the head repo (fork) from the PR's `head.repo.clone_url`.
- If the head repo matches the current working directory's remote, work in place.
- Otherwise, clone to `/tmp/<repo-name>` and checkout the PR branch.
- If already cloned from a previous run, `git fetch && git checkout <branch> && git pull`.

### 5) Apply fixes

For each review comment:
- Read the relevant file and surrounding context.
- Determine the fix. If a comment includes a code suggestion block, prefer adopting it.
- Apply the edit.
- Track: `{ comment_id, file, description_of_fix }` for the reply step.

If a comment is informational or not actionable (e.g. a compliment, a question needing user input), skip it and note it in the summary.

### 6) Commit and push

- Stage only the files that were modified to address review comments.
- Commit with message:
  ```
  fix: address review comments

  - <one-line summary per fix>

  Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
  ```
- Push to the PR branch: `git push origin <branch>`

### 7) Reply to comment threads

For each addressed comment, post a reply via:
```
gh api repos/<owner>/<repo>/pulls/comments/<comment_id>/replies \
  -f body="Fixed in <short-sha> — <description of fix>."
```

Run replies in parallel where possible.

### 8) Print summary

```
Addressed 4/4 review comments on PR #11
  Commit: abc1234
  Replies posted: 4
  Skipped (not actionable): 0
  PR: https://github.com/owner/repo/pull/11
```

## Constraints

- Never force-push.
- Never modify files outside the scope of the review comments.
- If a review comment requires a design decision or is ambiguous, ask the user rather than guessing.
- If the PR branch has diverged from the base and there are merge conflicts, report them and stop rather than attempting resolution.
- Do not merge the PR after addressing comments.
- Do not dismiss reviews — let the reviewer re-review.
- If working in a cloned fork under `/tmp/`, remind the user that local changes live there (not in the main working directory).
