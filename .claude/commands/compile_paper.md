# /compile_paper

Convert a SWARM paper from markdown to LaTeX (if needed), compile to PDF, and open it.

## Usage

```
/compile_paper <slug>
```

Examples:
- `/compile_paper kernel_market_v2`
- `/compile_paper collusion_governance`

## Arguments

- `slug`: Paper identifier. Resolves to `docs/papers/<slug>.md` (source) and `docs/papers/<slug>.tex` (output).

## Behavior

### Step 1: Locate source

Look for files in this order:
1. `docs/papers/<slug>.tex` — if it exists and is newer than the `.md`, skip conversion
2. `docs/papers/<slug>.md` — convert to LaTeX using the template

### Step 2: Convert markdown to LaTeX (if needed)

If the `.tex` file does not exist or the `.md` is newer:

1. Read `docs/papers/template.tex` for the standard preamble
2. Read `docs/papers/<slug>.md` for content
3. Convert markdown to LaTeX:
   - `# Title` → `\title{}`
   - `## Section` → `\section{}`
   - `### Subsection` → `\subsection{}`
   - `**bold**` → `\textbf{}`
   - `*italic*` → `\textit{}`
   - `` `code` `` → `\texttt{}`
   - `$math$` → pass through
   - Tables → `\begin{table}[H]` with `booktabs`
   - Lists → `\begin{itemize}` / `\begin{enumerate}`
   - Code blocks → `\begin{verbatim}`
   - `%` → `\%`, `_` in prose → `\_`
4. Write `docs/papers/<slug>.tex`

Use the template preamble exactly. Replace `%TITLE%`, `%AUTHOR%`, `%ABSTRACT%`, and `%BODY%` placeholders with converted content.

For `%AUTHOR%`, resolve in order:
1. `$SWARM_AUTHOR` environment variable
2. `git config user.name`
3. Ask the user (do not guess from OS username)

### Step 3: Compile

```bash
cd docs/papers/ && tectonic <slug>.tex
```

If tectonic is not in PATH, try `/opt/anaconda3/bin/tectonic`.

### Step 4: Write submission-ready copy to `research/papers/`

After successful compilation, copy the `.tex` to `research/papers/<slug>.tex` with section name normalization for ClawXiv/AgentXiv submission compatibility:

1. Read the compiled `docs/papers/<slug>.tex`
2. Apply section renames:
   - `\section{Experimental Setup}` → `\section{Methods}`
   - `\section{Experimental Methods}` → `\section{Methods}`
3. Write to `research/papers/<slug>.tex`

This eliminates the manual copy step before `/submit_paper`. The `docs/papers/` copy remains the canonical source; `research/papers/` is the submission-ready derivative.

### Step 5: Open

```bash
open docs/papers/<slug>.pdf
```

Report file size.

## Template

The standard template lives at `docs/papers/template.tex`. All SWARM papers share the same preamble (geometry, booktabs, graphicx, hyperref, amsmath, amssymb, caption, array, longtable, float, enumitem, verbatim).

## Notes

- Prefers tectonic over pdflatex (better error messages, auto-downloads packages)
- Falls back to conda-installed tectonic if not in PATH
- Works on macOS; Linux users may need `xdg-open` instead of `open`
- If tectonic fails, show the error and do NOT retry — the user should fix the LaTeX
