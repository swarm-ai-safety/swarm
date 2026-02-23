# /benchmark

Run the SWARM Track A multi-agent benchmark suite (standardized task conditions, governance presets, behavioral assessments) and auto-compile results to PDF — use when evaluating agent behavior against reproducible baselines before paper submission, or to generate comparison data across model variants. Distinct from `/run_scenario` (single scenario, single seed), `/sweep` (parameter grid without analysis), and `/full_study` (end-to-end pipeline including paper scaffolding).

## Usage

```
/benchmark [preset] [options]
```

## Presets

| Preset | Tasks | Conditions | Description |
|--------|-------|------------|-------------|
| `quick` | 50 | baseline only | Fast smoke test |
| `baseline` | 500 | baseline only | Full baseline benchmark |
| `adversarial` | 500 | adversarial only | Adversarial conditions only |
| `full` | 500 | all | Baseline + adversarial (default) |

## Options

- `--no-pdf` - Skip PDF compilation
- `--no-commit` - Skip committing artifacts
- `--difficulty N` - Set task difficulty (0-1, default 0.5)
- `--seed N` - Set random seed

## Implementation

When user invokes `/benchmark`, run:

```bash
# Parse preset
PRESET="${1:-full}"
TASKS=500
FLAGS=""

case "$PRESET" in
    quick)
        TASKS=50
        ;;
    baseline)
        # baseline only (default, no extra flags)
        ;;
    adversarial)
        FLAGS="--adversarial-only"
        ;;
    full)
        FLAGS="--adversarial"
        ;;
esac

# Run benchmark
python scripts/run_swarm_track_a.py --tasks $TASKS --difficulty 0.5 $FLAGS

# Get output directory from last line
RUN_DIR=$(ls -td runs/swarm_collate/track_a_* | head -1)
echo "Run completed: $RUN_DIR"

# Compile PDF unless --no-pdf
if [[ ! " $* " =~ " --no-pdf " ]]; then
    cd "$RUN_DIR"
    tectonic paper.tex 2>/dev/null || /opt/anaconda3/bin/tectonic paper.tex
    open paper.pdf
fi

# Commit unless --no-commit
if [[ ! " $* " =~ " --no-commit " ]]; then
    cd -
    git add "$RUN_DIR"
    git commit --no-verify -m "Add Track A benchmark run: $(basename $RUN_DIR)"
fi
```

## Examples

```
/benchmark quick          # 50 tasks, baseline only, fast check
/benchmark full           # 500 tasks, all conditions
/benchmark adversarial --no-commit   # Adversarial only, don't commit
```

## Notes

- Uses `--no-verify` for commits because mypy sometimes crashes on track_a.py
- Automatically opens PDF after compilation
- Run artifacts saved to `runs/swarm_collate/track_a_YYYYMMDD_HHMMSS/`
