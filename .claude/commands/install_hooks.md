# /install_hooks

Install repo-provided git hooks that enforce research hygiene.

## Usage

`/install_hooks`

## Behavior

1) Copy hook scripts from `.claude/hooks/` into `.git/hooks/`:
- `pre-commit`
- `pre-push`

2) Ensure they are executable.

3) Print what each hook runs and how to bypass (only for emergencies):
- `SKIP_SWARM_HOOKS=1 git commit ...`

## Safety

- Never modify hooks outside this repository.
- Do not delete existing user hooks; if a hook exists, create a `.bak` copy first.

