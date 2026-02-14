#!/usr/bin/env python3
"""Submit repo text samples to Havelock.AI orality analyzer.

Usage:
    python scripts/havelock_orality_analysis.py
    python scripts/havelock_orality_analysis.py --file path/to/file.md
    python scripts/havelock_orality_analysis.py --all

Requires the HF Space to be running:
    https://thestalwart-havelock-demo.hf.space
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://thestalwart-havelock-demo.hf.space"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Default samples: (label, path, line_range or None for full file)
DEFAULT_SAMPLES = [
    ("Blog: LLM Council (conversational)", "docs/blog/llm-council-three-models-one-study.md", None),
    ("Blog: Ecosystem Collapse (narrative)", "docs/blog/ecosystem-collapse.md", None),
    ("Paper: Main abstract+intro (formal)", "docs/papers/distributional_agi_safety.md", (1, 120)),
    ("Paper: Agent Lab abstract (LaTeX)", "docs/papers/agent_lab_research_safety.tex", (22, 40)),
    ("README (mixed technical)", "README.md", (1, 80)),
]


def strip_markdown(text: str) -> str:
    """Remove markdown formatting, keeping prose."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove inline code
    text = re.sub(r"`[^`]+`", "", text)
    # Remove images
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove headers markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    # Remove table formatting
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove list markers
    text = re.sub(r"^[\s]*[-*]\s+", "", text, flags=re.MULTILINE)
    # Remove numbered list markers
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_latex(text: str) -> str:
    """Remove LaTeX formatting, keeping prose."""
    # Remove preamble/document commands
    text = re.sub(r"\\documentclass.*", "", text)
    text = re.sub(r"\\usepackage.*", "", text)
    text = re.sub(r"\\(title|author|date|maketitle|begin|end|label|centering)\{[^}]*\}", "", text)
    text = re.sub(r"\\begin\{[^}]+\}", "", text)
    text = re.sub(r"\\end\{[^}]+\}", "", text)
    # Remove \section{...} but keep content
    text = re.sub(r"\\(?:sub)*section\{([^}]+)\}", r"\1", text)
    # Remove \textbf, \texttt, \emph but keep content
    text = re.sub(r"\\(?:textbf|texttt|emph|textit)\{([^}]+)\}", r"\1", text)
    # Remove \cite{...}
    text = re.sub(r"\\cite\{[^}]+\}", "", text)
    # Remove math mode
    text = re.sub(r"\$[^$]+\$", "", text)
    # Remove remaining commands
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Remove braces
    text = re.sub(r"[{}]", "", text)
    # Remove table content (between toprule/bottomrule)
    text = re.sub(r"toprule.*?bottomrule", "", text, flags=re.DOTALL)
    # Remove & and \\
    text = re.sub(r"&|\\\\", " ", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_sample(path: Path, line_range: tuple[int, int] | None = None) -> str:
    """Load and clean a text sample."""
    raw = path.read_text(encoding="utf-8")
    if line_range:
        lines = raw.splitlines()
        start, end = line_range
        raw = "\n".join(lines[start - 1 : end])

    if path.suffix == ".tex":
        return strip_latex(raw)
    return strip_markdown(raw)


def submit_text(text: str, include_sentences: bool = False) -> dict:
    """Submit text to Havelock API, return parsed result."""
    # POST to get event_id
    payload = json.dumps({"data": [text, include_sentences]}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/gradio_api/call/analyze",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        if "paused" in error_body.lower() or e.code == 503:
            raise RuntimeError(
                "HF Space is paused. Visit https://thestalwart-havelock-demo.hf.space "
                "in your browser to restart it, then re-run this script."
            ) from e
        raise

    event_id = body.get("event_id")
    if not event_id:
        raise RuntimeError(f"No event_id in response: {body}")

    # GET SSE stream for results
    result_url = f"{BASE_URL}/gradio_api/call/analyze/{event_id}"
    req2 = urllib.request.Request(result_url)

    for _attempt in range(10):
        try:
            with urllib.request.urlopen(req2, timeout=60) as resp:
                sse_data = resp.read().decode()
            break
        except urllib.error.HTTPError:
            time.sleep(2)
    else:
        raise RuntimeError(f"Timed out waiting for results (event_id={event_id})")

    # Parse SSE: look for "data: " lines
    for line in sse_data.splitlines():
        if line.startswith("data: "):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                continue

    raise RuntimeError(f"Could not parse SSE response:\n{sse_data[:500]}")


def analyze_samples(samples: list[tuple[str, str, tuple | None]]) -> list[dict]:
    """Run all samples through the API."""
    results = []
    for label, rel_path, line_range in samples:
        path = REPO_ROOT / rel_path
        if not path.exists():
            print(f"  SKIP {label}: {rel_path} not found")
            continue

        text = load_sample(path, line_range)
        word_count = len(text.split())
        print(f"  Submitting: {label} ({word_count} words)...", end=" ", flush=True)

        try:
            result = submit_text(text)
            print("done")
            results.append({"label": label, "path": rel_path, "words": word_count, "result": result})
        except RuntimeError as e:
            print(f"FAILED: {e}")
            if "paused" in str(e).lower():
                raise
            results.append({"label": label, "path": rel_path, "words": word_count, "error": str(e)})

    return results


def print_results(results: list[dict]) -> None:
    """Print a comparison table."""
    print("\n" + "=" * 80)
    print("HAVELOCK ORALITY ANALYSIS — REPO TEXT SAMPLES")
    print("=" * 80)
    print(f"{'Sample':<45} {'Words':>6} {'Score':>6} {'Mode':<12}")
    print("-" * 80)

    for r in results:
        label = r["label"][:44]
        words = r["words"]
        if "error" in r:
            print(f"{label:<45} {words:>6} {'ERR':>6} {r['error'][:12]}")
            continue

        res = r["result"]
        # Handle both flat and nested response formats
        if isinstance(res, list):
            # Gradio often returns [score_data, ...] as a list
            res = res[0] if res else {}

        if isinstance(res, dict):
            score = res.get("score", res.get("orality_score", "?"))
            mode = res.get("interpretation", {}).get("mode", "") if isinstance(res.get("interpretation"), dict) else res.get("mode", "")
        else:
            score = res
            mode = ""

        print(f"{label:<45} {words:>6} {str(score):>6} {mode:<12}")

    print("-" * 80)
    print("Scale: 0 = fully literate, 100 = fully oral")
    print("Source: https://havelock.ai/api\n")

    # Dump raw JSON for inspection
    out_path = REPO_ROOT / "runs" / "havelock_orality_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Raw results saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze repo text with Havelock orality API")
    parser.add_argument("--file", type=str, help="Analyze a specific file instead of defaults")
    parser.add_argument("--all", action="store_true", help="Analyze all blog posts and papers")
    args = parser.parse_args()

    if args.file:
        p = Path(args.file)
        samples = [(p.name, str(p), None)]
    elif args.all:
        samples = list(DEFAULT_SAMPLES)
        # Add all blog posts
        blog_dir = REPO_ROOT / "docs" / "blog"
        if blog_dir.exists():
            for f in sorted(blog_dir.glob("*.md")):
                if f.name == "index.md":
                    continue
                key = f"Blog: {f.stem}"
                if not any(s[1].endswith(f.name) for s in samples):
                    samples.append((key, f"docs/blog/{f.name}", None))
        # Add all papers (markdown only, cleaner to parse)
        papers_dir = REPO_ROOT / "docs" / "papers"
        if papers_dir.exists():
            for f in sorted(papers_dir.glob("*.md")):
                key = f"Paper: {f.stem}"
                if not any(s[1].endswith(f.name) for s in samples):
                    samples.append((key, f"docs/papers/{f.name}", None))
    else:
        samples = DEFAULT_SAMPLES

    print(f"Havelock Orality Analyzer — {len(samples)} samples")
    print(f"API: {BASE_URL}\n")

    try:
        results = analyze_samples(samples)
    except RuntimeError as e:
        print(f"\nFATAL: {e}")
        sys.exit(1)

    if results:
        print_results(results)


if __name__ == "__main__":
    main()
