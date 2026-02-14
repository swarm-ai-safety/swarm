#!/usr/bin/env python
"""Generate phylogeny animation for a simulation run.

Usage::

    python examples/generate_phylogeny.py runs/<run_id> [--output custom.html]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm.analysis.phylogeny import generate_phylogeny  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate animated agent phylogeny visualization",
    )
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Output HTML path"
    )
    args = parser.parse_args()

    output = generate_phylogeny(args.run_dir, args.output)
    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
