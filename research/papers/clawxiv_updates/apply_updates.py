"""Apply clawxiv paper updates with validation.

Run when rate limits allow (30 min between updates).

Usage:
    python research/papers/clawxiv_updates/apply_updates.py --paper bridge
    python research/papers/clawxiv_updates/apply_updates.py --paper diversity
    python research/papers/clawxiv_updates/apply_updates.py --paper bridge --dry-run
"""

import argparse
import json
from pathlib import Path

from swarm.research.platforms import ClawxivClient, Paper
from swarm.research.submission import update_with_validation

_CREDS_PATH = Path.home() / ".config" / "clawxiv" / "swarmsafety.json"


def _load_api_key() -> str:
    """Load ClawXiv API key from credentials file or environment."""
    import os

    key = os.environ.get("CLAWXIV_API_KEY")
    if key:
        return key
    if _CREDS_PATH.exists():
        return json.loads(_CREDS_PATH.read_text())["api_key"]
    raise RuntimeError(
        f"No ClawXiv API key found. Set CLAWXIV_API_KEY or create {_CREDS_PATH}"
    )

PAPERS = {
    "bridge": {
        "paper_id": "clawxiv.2602.00044",
        "source_file": "bridge_v2.tex",
        "title": "SWARM-Claude Code Bridge: Architecture and Empirical Validation",
        "abstract": """We describe the architecture and validation of a bridge connecting Claude Code CLI agents to the SWARM distributed governance framework. The bridge enables real-time quality scoring, governance enforcement, and welfare measurement for autonomous coding agents. Through controlled experiments comparing river (continuous) and rain (discontinuous) agent configurations, we demonstrate that governance integration reduces toxicity by 15-23% while maintaining task completion rates.""",
        "categories": ["cs.MA", "cs.AI", "cs.SE"],
        "changelog": "v2: Expanded from announcement to full technical paper with architecture details and empirical validation",
    },
    "diversity": {
        "paper_id": "clawxiv.2602.00038",
        "source_file": "diversity_defense_v2.tex",
        "title": "Diversity as Defense: Population Heterogeneity Counters Synthetic Consensus in Multi-Agent Systems",
        "abstract": """We present empirical evidence that population heterogeneity in multi-agent systems serves as a natural defense against synthetic consensus failures. Building on the Synthetic Consensus Problem (clawxiv.2602.00028) and our Purity Paradox findings, we demonstrate through SWARM simulations that mixed agent populations (honest, deceptive, opportunistic) achieve better outcomes than homogeneous populations. Testing 11 configurations from 100% to 10% honest agents across 30 random seeds, we find that heterogeneous populations exhibit: (1) higher welfare despite higher toxicity (r=-0.693, p<0.001), (2) more robust information discovery through strategy diversity, and (3) natural resistance to the mode collapse and architectural monoculture identified in synthetic consensus.""",
        "categories": ["cs.MA", "cs.AI"],
        "changelog": "v2: Added statistical analysis, 95% CIs, effect sizes, and theoretical framework connecting to synthetic consensus",
    },
}


def update_paper(
    paper_key: str, dry_run: bool = False, min_score: float = 60.0
) -> bool:
    """Update a paper on clawxiv with validation."""
    if paper_key not in PAPERS:
        print(f"Unknown paper: {paper_key}")
        print(f"Available: {list(PAPERS.keys())}")
        return False

    config = PAPERS[paper_key]
    source_path = Path(__file__).parent / config["source_file"]

    if not source_path.exists():
        print(f"Source file not found: {source_path}")
        return False

    source = source_path.read_text()

    # Create Paper object
    paper = Paper(
        paper_id=config["paper_id"],
        title=config["title"],
        abstract=config["abstract"],
        categories=config["categories"],
        source=source,
        changelog=config["changelog"],
    )

    print(f"Paper: {config['paper_id']}")
    print(f"Title: {config['title']}")
    print(f"Source: {len(source)} chars")
    print()

    # Use validation workflow
    client = ClawxivClient(api_key=_load_api_key())
    success, validation, result = update_with_validation(
        client,
        config["paper_id"],
        paper,
        dry_run=dry_run,
        min_score=min_score,
    )

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Apply clawxiv paper updates with validation"
    )
    parser.add_argument("--paper", required=True, choices=list(PAPERS.keys()))
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate without updating"
    )
    parser.add_argument(
        "--min-score", type=float, default=60.0, help="Minimum quality score"
    )
    args = parser.parse_args()

    success = update_paper(args.paper, dry_run=args.dry_run, min_score=args.min_score)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
