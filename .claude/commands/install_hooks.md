# /install_hooks

Install repo-provided git hooks that enforce research hygiene. Also use this after editing any hook in `.claude/hooks/`.

## Usage

`/install_hooks`

## Behavior

1) Discover all hook scripts in `.claude/hooks/` (e.g. `pre-commit`, `pre-push`).

2) For each hook found:
- If a hook already exists in `.git/hooks/` that **differs** from the source, create a `.bak` copy first.
- Copy from `.claude/hooks/<hook>` to `.git/hooks/<hook>`.
- Ensure it is executable (`chmod +x`).

3) Print a summary:
- Which hooks were installed or updated.
- Which hooks were unchanged (already in sync).
- How to bypass (only for emergencies): `SKIP_SWARM_HOOKS=1 git commit ...`

4) Verify the installed hooks match the source (diff check).

## When to run

- After cloning the repo for the first time.
- **After any edit to a file in `.claude/hooks/`** — always re-run `/install_hooks` to sync the change into `.git/hooks/`. The source of truth is `.claude/hooks/`, never `.git/hooks/` directly.
- After pulling changes that modified `.claude/hooks/`.

## Safety

- Never modify hooks outside this repository.
- Do not delete existing user hooks; if a hook exists and differs, create a `.bak` copy first.
- Never install hooks from outside `.claude/hooks/`.
