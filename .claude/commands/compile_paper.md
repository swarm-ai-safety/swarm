# /compile_paper

Compile the latest Track A run to PDF and open it.

## Usage

```
/compile_paper [run_id]
```

## Behavior

1. If no `run_id` provided, find the most recent run in `runs/swarm_collate/`
2. Compile `paper.tex` using tectonic
3. Open the resulting PDF

## Implementation

```bash
# Find latest run if not specified
RUN_DIR="${1:-$(ls -td runs/swarm_collate/track_a_* 2>/dev/null | head -1)}"

if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "No Track A runs found in runs/swarm_collate/"
    exit 1
fi

# Use absolute path to avoid cwd issues
ABS_RUN_DIR="$(cd "$RUN_DIR" && pwd)"

echo "Compiling: $ABS_RUN_DIR/paper.tex"

# Run in a subshell so cd doesn't pollute the caller's working directory
(
    cd "$ABS_RUN_DIR"

    # Compile with tectonic (preferred) or pdflatex
    if command -v tectonic &>/dev/null; then
        tectonic paper.tex
    elif command -v pdflatex &>/dev/null; then
        pdflatex -interaction=nonstopmode paper.tex
    else
        # Try conda tectonic
        /opt/anaconda3/bin/tectonic paper.tex 2>/dev/null || {
            echo "No LaTeX compiler found. Install tectonic: conda install -c conda-forge tectonic"
            exit 1
        }
    fi
)

# Open the PDF using absolute path
if [ -f "$ABS_RUN_DIR/paper.pdf" ]; then
    echo "Opening: $ABS_RUN_DIR/paper.pdf"
    open "$ABS_RUN_DIR/paper.pdf"
else
    echo "Compilation failed - no PDF generated"
    exit 1
fi
```

## Notes

- Prefers tectonic over pdflatex (better error messages, auto-downloads packages)
- Falls back to conda-installed tectonic if not in PATH
- Works on macOS; Linux users may need `xdg-open` instead of `open`
