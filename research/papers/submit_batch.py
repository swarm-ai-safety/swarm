#!/usr/bin/env python3
"""Submit papers to ClawXiv and AgentXiv in batch.

Submits three papers to both platforms:
  1. Distributional AGI Safety (main paper)
  2. Governance Mechanisms (empirical study)
  3. Collusion Dynamics & Network Resilience

Usage:
    python research/papers/submit_batch.py --dry-run          # validate only
    python research/papers/submit_batch.py                    # submit to both
    python research/papers/submit_batch.py --platform clawxiv # clawxiv only
    python research/papers/submit_batch.py --platform agentxiv # agentxiv only
"""

import argparse
import base64
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from swarm.research.platforms import (
    AgentxivClient,
    ClawxivClient,
    Paper,
    SubmissionResult,
)
from swarm.research.submission import SubmissionValidator


@dataclass
class PaperSpec:
    """Specification for a paper to submit."""

    name: str
    tex_path: Path
    md_path: Path
    title: str
    abstract: str
    categories_clawxiv: list[str]
    category_agentxiv: str
    figures_dir: Path | None = None


ROOT = Path(__file__).resolve().parent.parent.parent
PAPERS_DIR = ROOT / "research" / "papers"
DOCS_DIR = ROOT / "docs" / "papers"
FIGURES_DIR = DOCS_DIR / "figures"


def extract_abstract_from_tex(tex_path: Path) -> str:
    """Extract abstract from LaTeX source."""
    source = tex_path.read_text()
    match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        source,
        re.DOTALL,
    )
    if match:
        abstract = match.group(1).strip()
        # Strip LaTeX commands for plain-text abstract
        abstract = re.sub(r"\\emph\{([^}]+)\}", r"\1", abstract)
        abstract = re.sub(r"\\textbf\{([^}]+)\}", r"\1", abstract)
        abstract = re.sub(r"\\citet?\{[^}]+\}", "", abstract)
        abstract = re.sub(r"~", " ", abstract)
        abstract = re.sub(r"---", "—", abstract)
        abstract = re.sub(r"--", "–", abstract)
        return abstract
    return ""


def extract_title_from_tex(tex_path: Path) -> str:
    """Extract title from LaTeX source."""
    source = tex_path.read_text()
    match = re.search(r"\\title\{([^}]+)\}", source)
    if match:
        title = match.group(1)
        title = title.replace("\\\\", " ").replace("\\", "")
        return title.strip()
    return tex_path.stem


def extract_abstract_from_md(md_path: Path) -> str:
    """Extract abstract from Markdown source."""
    text = md_path.read_text()
    match = re.search(
        r"## Abstract\s*\n+(.*?)(?=\n## |\n---)",
        text,
        re.DOTALL,
    )
    if match:
        abstract = match.group(1).strip()
        # Clean markdown formatting
        abstract = re.sub(r"\*\*([^*]+)\*\*", r"\1", abstract)
        abstract = re.sub(r"\*([^*]+)\*", r"\1", abstract)
        abstract = re.sub(r"\$([^$]+)\$", r"\1", abstract)
        return abstract
    return ""


def build_paper_specs() -> list[PaperSpec]:
    """Build specifications for all papers to submit."""
    specs = []

    # Paper 1: Distributional AGI Safety
    tex1 = PAPERS_DIR / "distributional_agi_safety.tex"
    md1 = DOCS_DIR / "distributional_agi_safety.md"
    specs.append(
        PaperSpec(
            name="distributional_agi_safety",
            tex_path=tex1,
            md_path=md1,
            title=extract_title_from_tex(tex1) if tex1.exists() else (
                "Distributional AGI Safety: Governance Trade-offs in "
                "Multi-Agent Systems Under Adversarial Pressure"
            ),
            abstract=extract_abstract_from_tex(tex1) if tex1.exists() else (
                extract_abstract_from_md(md1)
            ),
            categories_clawxiv=["cs.MA", "cs.AI"],
            category_agentxiv="multi-agent-systems",
            figures_dir=FIGURES_DIR,
        )
    )

    # Paper 2: Governance Mechanisms
    tex2 = PAPERS_DIR / "governance_mechanisms.tex"
    md2 = DOCS_DIR / "governance_mechanisms_multi_agent_safety.md"
    specs.append(
        PaperSpec(
            name="governance_mechanisms",
            tex_path=tex2,
            md_path=md2,
            title=(
                "Governance Mechanisms for Distributional Safety in "
                "Multi-Agent Systems: An Empirical Study Across "
                "Scenario Archetypes"
            ),
            abstract=extract_abstract_from_md(md2) if md2.exists() else "",
            categories_clawxiv=["cs.MA", "cs.AI"],
            category_agentxiv="multi-agent-systems",
        )
    )

    # Paper 3: Collusion Dynamics
    tex3 = PAPERS_DIR / "collusion_dynamics.tex"
    md3 = DOCS_DIR / "collusion_dynamics_network_resilience.md"
    specs.append(
        PaperSpec(
            name="collusion_dynamics",
            tex_path=tex3,
            md_path=md3,
            title=(
                "Progressive Decline vs. Sustained Operation: How Network "
                "Topology and Collusion Detection Shape Multi-Agent "
                "Safety Dynamics"
            ),
            abstract=extract_abstract_from_md(md3) if md3.exists() else "",
            categories_clawxiv=["cs.MA", "cs.AI"],
            category_agentxiv="multi-agent-systems",
        )
    )

    return specs


def load_credentials(platform: str) -> str:
    """Load API key for a platform."""
    if platform == "clawxiv":
        creds_path = Path.home() / ".config" / "clawxiv" / "swarmsafety.json"
        if creds_path.exists():
            return json.loads(creds_path.read_text())["api_key"]
        # Fallback to .credentials
        creds_md = ROOT / ".credentials" / "agent_platforms.md"
        if creds_md.exists():
            text = creds_md.read_text()
            match = re.search(r"clx_[0-9a-f]{32}", text)
            if match:
                return match.group(0)
        raise FileNotFoundError("No clawxiv credentials found")

    elif platform == "agentxiv":
        creds_path = Path.home() / ".config" / "agentxiv" / "credentials.json"
        if creds_path.exists():
            creds = json.loads(creds_path.read_text())
            return creds.get("current", {}).get("api_key", "")
        raise FileNotFoundError("No agentxiv credentials found")

    raise ValueError(f"Unknown platform: {platform}")


def submit_to_clawxiv(
    spec: PaperSpec, client: ClawxivClient, dry_run: bool
) -> SubmissionResult | None:
    """Submit a paper to ClawXiv (LaTeX format)."""
    if not spec.tex_path.exists():
        print(f"  [SKIP] No .tex file: {spec.tex_path}")
        return None

    source = spec.tex_path.read_text()

    # Collect referenced images as base64
    images: dict[str, str] = {}
    if spec.figures_dir and spec.figures_dir.is_dir():
        for match in re.finditer(r"\\includegraphics[^{]*\{([^}]+)\}", source):
            fig_name = match.group(1)
            fig_path = spec.figures_dir / fig_name
            if fig_path.exists():
                images[fig_name] = base64.b64encode(
                    fig_path.read_bytes()
                ).decode()

    paper = Paper(
        title=spec.title,
        abstract=spec.abstract,
        source=source,
        categories=spec.categories_clawxiv,
        images=images if images else None,
    )

    # Validate first
    validator = SubmissionValidator()
    validation = validator.validate(paper)
    print(f"  Validation: score={validation.quality_score:.0f}/100, "
          f"passed={validation.passed}")

    if not validation.passed:
        print("  [FAIL] Validation failed:")
        for issue in validation.errors():
            print(f"    - [{issue.code}] {issue.message}")
        return None

    if dry_run:
        print(f"  [DRY-RUN] Would submit to ClawXiv ({len(source)} chars)")
        return SubmissionResult(
            success=True,
            paper_id="dry-run",
            message="Dry run - not submitted",
        )

    result = client.submit(paper)
    print(f"  Result: success={result.success}, id={result.paper_id}")
    if not result.success:
        print(f"  Error: {result.message}")
    return result


def submit_to_agentxiv(
    spec: PaperSpec, client: AgentxivClient, dry_run: bool
) -> SubmissionResult | None:
    """Submit a paper to AgentXiv (Markdown format)."""
    if not spec.md_path.exists():
        print(f"  [SKIP] No .md file: {spec.md_path}")
        return None

    md_content = spec.md_path.read_text()
    paper = Paper(
        title=spec.title,
        abstract=spec.abstract,
        source=md_content,
        categories=[spec.category_agentxiv],
    )

    if dry_run:
        print(f"  [DRY-RUN] Would submit to AgentXiv ({len(md_content)} chars)")
        return SubmissionResult(
            success=True,
            paper_id="dry-run",
            message="Dry run - not submitted",
        )

    result = client.submit(paper)
    print(f"  Result: success={result.success}, id={result.paper_id}")
    if not result.success:
        print(f"  Error: {result.message}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch-submit papers to ClawXiv and AgentXiv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview without submitting",
    )
    parser.add_argument(
        "--platform",
        choices=["clawxiv", "agentxiv", "both"],
        default="both",
        help="Which platform(s) to submit to (default: both)",
    )
    args = parser.parse_args()

    specs = build_paper_specs()
    do_clawxiv = args.platform in ("clawxiv", "both")
    do_agentxiv = args.platform in ("agentxiv", "both")

    # Load clients
    clx_client = None
    ax_client = None

    if do_clawxiv and not args.dry_run:
        try:
            api_key = load_credentials("clawxiv")
            clx_client = ClawxivClient(api_key=api_key)
            print(f"ClawXiv client ready (key: {api_key[:8]}...)")
        except FileNotFoundError as e:
            print(f"WARNING: {e} — skipping ClawXiv submissions")
            do_clawxiv = False

    if do_agentxiv and not args.dry_run:
        try:
            api_key = load_credentials("agentxiv")
            ax_client = AgentxivClient(api_key=api_key)
            print(f"AgentXiv client ready (key: {api_key[:8]}...)")
        except FileNotFoundError as e:
            print(f"WARNING: {e} — skipping AgentXiv submissions")
            do_agentxiv = False

    print()
    print("=" * 60)
    print("Batch Paper Submission")
    print("=" * 60)
    print(f"Papers: {len(specs)}")
    print(f"Platforms: {'ClawXiv' if do_clawxiv else ''} "
          f"{'AgentXiv' if do_agentxiv else ''}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    results: dict[str, dict[str, SubmissionResult | None]] = {}

    for spec in specs:
        print(f"--- {spec.name} ---")
        print(f"  Title: {spec.title[:70]}...")
        print(f"  .tex: {'EXISTS' if spec.tex_path.exists() else 'MISSING'} "
              f"({spec.tex_path.name})")
        print(f"  .md:  {'EXISTS' if spec.md_path.exists() else 'MISSING'} "
              f"({spec.md_path.name})")
        results[spec.name] = {}

        if do_clawxiv:
            print("  [ClawXiv]")
            results[spec.name]["clawxiv"] = submit_to_clawxiv(
                spec,
                clx_client or ClawxivClient(api_key="dry-run"),
                args.dry_run,
            )

        if do_agentxiv:
            print("  [AgentXiv]")
            results[spec.name]["agentxiv"] = submit_to_agentxiv(
                spec,
                ax_client or AgentxivClient(api_key="dry-run"),
                args.dry_run,
            )

        print()

    # Summary
    print("=" * 60)
    print("SUBMISSION SUMMARY")
    print("=" * 60)
    for name, platforms in results.items():
        for platform, result in platforms.items():
            status = "N/A"
            paper_id = ""
            if result is None:
                status = "SKIPPED"
            elif result.success:
                status = "OK"
                paper_id = f" → {result.paper_id}"
            else:
                status = "FAILED"
                paper_id = f" ({result.message})"
            print(f"  {name:40s} {platform:10s} [{status}]{paper_id}")


if __name__ == "__main__":
    main()
