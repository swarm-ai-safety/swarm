#!/usr/bin/env python
"""Validate a paper before submission to clawxiv.

Usage:
    python scripts/validate_paper.py path/to/paper.tex
    python scripts/validate_paper.py path/to/paper.tex --submit
    python scripts/validate_paper.py path/to/paper.tex --update clawxiv.2602.00044
"""

import argparse
import sys
from pathlib import Path

from swarm.research.platforms import ClawxivClient, Paper
from swarm.research.submission import (
    SubmissionValidator,
    submit_with_validation,
    update_with_validation,
)


def load_paper_from_tex(tex_path: Path) -> Paper:
    """Load a paper from a .tex file.

    Extracts title and abstract from LaTeX source.
    """
    import re

    source = tex_path.read_text()

    # Extract title
    title_match = re.search(r"\\title\{([^}]+)\}", source)
    title = title_match.group(1) if title_match else tex_path.stem

    # Extract abstract
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        source,
        re.DOTALL,
    )
    abstract = abstract_match.group(1).strip() if abstract_match else ""

    # Default categories
    categories = ["cs.MA", "cs.AI"]

    return Paper(
        title=title,
        abstract=abstract,
        source=source,
        categories=categories,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate a paper before submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate only
    python scripts/validate_paper.py research/papers/my_paper.tex

    # Validate and submit (dry run)
    python scripts/validate_paper.py research/papers/my_paper.tex --submit --dry-run

    # Validate and submit for real
    python scripts/validate_paper.py research/papers/my_paper.tex --submit

    # Update existing paper
    python scripts/validate_paper.py research/papers/my_paper.tex --update clawxiv.2602.00044
""",
    )
    parser.add_argument("tex_file", type=Path, help="Path to .tex file")
    parser.add_argument("--submit", action="store_true", help="Submit after validation")
    parser.add_argument("--update", type=str, help="Update existing paper ID")
    parser.add_argument("--dry-run", action="store_true", help="Validate without submitting")
    parser.add_argument("--min-score", type=float, default=60.0, help="Minimum quality score (0-100)")
    parser.add_argument("--api-key", type=str, help="clawxiv API key (or set CLAWXIV_API_KEY)")

    args = parser.parse_args()

    # Load paper
    if not args.tex_file.exists():
        print(f"Error: File not found: {args.tex_file}")
        sys.exit(1)

    print(f"Loading paper from: {args.tex_file}")
    paper = load_paper_from_tex(args.tex_file)
    print(f"Title: {paper.title}")
    print(f"Source: {len(paper.source)} chars")
    print()

    # Just validate
    if not args.submit and not args.update:
        validator = SubmissionValidator()
        result = validator.validate(paper)
        print(result.report())
        sys.exit(0 if result.passed else 1)

    # Submit or update
    client = ClawxivClient(api_key=args.api_key)

    if args.update:
        success, validation, submission = update_with_validation(
            client,
            args.update,
            paper,
            dry_run=args.dry_run,
            min_score=args.min_score,
        )
    else:
        success, validation, submission = submit_with_validation(
            client,
            paper,
            dry_run=args.dry_run,
            min_score=args.min_score,
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
