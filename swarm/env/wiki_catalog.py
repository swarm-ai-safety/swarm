"""Research-themed wiki page catalog for Moltipedia seeding."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, List

from swarm.env.wiki import PageStatus, WikiPage

if TYPE_CHECKING:
    from swarm.env.wiki import WikiTaskPool

WIKI_CATALOG: List[dict] = [
    # --- Distributional Safety ---
    {
        "title": "Distributional AGI Safety",
        "content": (
            "Distributional AGI safety studies how risk and benefit are allocated "
            "across populations of interacting agents rather than focusing on a "
            "single principal-agent pair. The framework uses soft (probabilistic) "
            "labels p in [0,1] to express uncertainty about whether an interaction "
            "is beneficial, avoiding brittle binary classifications."
        ),
        "quality_score": 0.55,
    },
    {
        "title": "Soft Labels in Safety Evaluation",
        "content": (
            "Soft labels assign a probability p = P(v=+1) to each interaction "
            "rather than a hard good/bad label. This enables calibrated risk "
            "metrics such as expected toxicity E[1-p | accepted] and quality gap "
            "E[p | accepted] - E[p | rejected]."
        ),
        "quality_score": 0.50,
    },
    {
        "title": "Proxy Computation",
        "content": (
            "The ProxyComputer converts observable signals (task_progress, "
            "rework_count, verifier_rejections, engagement) into v_hat in [-1,+1] "
            "via a weighted combination, then applies a calibrated sigmoid to "
            "produce p = P(v=+1). Proxy calibration is critical for downstream "
            "payoff accuracy."
        ),
        "quality_score": 0.45,
    },
    {
        "title": "Expected Surplus",
        "content": (
            "Expected surplus S_soft = p * s_plus - (1-p) * s_minus captures the "
            "probabilistic value of an interaction. When p is high the surplus is "
            "dominated by the positive payoff s_plus; when p is low the expected "
            "loss (1-p)*s_minus dominates."
        ),
        "quality_score": 0.40,
    },
    {
        "title": "Expected Harm Externality",
        "content": (
            "The expected harm externality E_soft = (1-p) * h quantifies the "
            "ecosystem-level cost of accepting a potentially harmful interaction. "
            "Externality internalization parameters rho_a and rho_b control how "
            "much each party bears this cost."
        ),
        "quality_score": 0.42,
    },
    # --- Governance Mechanisms ---
    {
        "title": "Transaction Tax",
        "content": (
            "A transaction tax levied on each accepted interaction creates a "
            "common fund for ecosystem maintenance. The tax rate and split ratio "
            "between initiator and counterparty are governance parameters."
        ),
        "quality_score": 0.38,
    },
    {
        "title": "Circuit Breaker Mechanism",
        "content": (
            "Circuit breakers freeze agent participation when toxicity exceeds a "
            "threshold or policy violations accumulate. The freeze duration and "
            "threshold are tunable governance levers designed to prevent cascading "
            "ecosystem harm."
        ),
        "quality_score": 0.52,
    },
    {
        "title": "Staking and Slashing",
        "content": (
            "Agents deposit a stake before participating. Misbehavior triggers "
            "slashing at a configurable rate. Staking deters hit-and-run adversarial "
            "strategies by making participation costly for bad actors."
        ),
        "quality_score": 0.44,
    },
    {
        "title": "Reputation Decay",
        "content": (
            "Reputation scores decay over time at a configurable rate, preventing "
            "agents from coasting on historical good behavior while engaging in "
            "current exploitation. Decay rate is a governance parameter."
        ),
        "quality_score": 0.35,
    },
    {
        "title": "Vote Normalization",
        "content": (
            "Vote normalization caps the maximum voting weight any single agent "
            "can exert, preventing plutocratic capture of governance decisions by "
            "high-reputation agents."
        ),
        "quality_score": 0.37,
    },
    {
        "title": "Random Audit Mechanism",
        "content": (
            "Random audits probabilistically inspect interactions post-hoc. "
            "Detected violations incur a multiplied penalty. The audit probability "
            "and penalty multiplier trade off oversight cost against deterrence."
        ),
        "quality_score": 0.41,
    },
    {
        "title": "Bandwidth Cap",
        "content": (
            "Bandwidth caps limit the number of interactions an agent can initiate "
            "per time step. This prevents spam and resource exhaustion attacks "
            "while allowing honest agents sufficient throughput."
        ),
        "quality_score": 0.33,
    },
    # --- Collusion Dynamics ---
    {
        "title": "Collusion Detection",
        "content": (
            "Collusion detection identifies pairs of agents whose interaction "
            "frequency and payoff correlation exceed statistical thresholds. The "
            "collusion score combines frequency ratio and Pearson correlation of "
            "outcomes."
        ),
        "quality_score": 0.48,
    },
    {
        "title": "Collusion Dynamics and Network Resilience",
        "content": (
            "Collusion dynamics study how cooperative exploitation strategies "
            "emerge and persist in multi-agent networks. Network topology strongly "
            "influences collusion resilience: scale-free networks are more "
            "vulnerable than Erdos-Renyi random graphs."
        ),
        "quality_score": 0.53,
    },
    {
        "title": "Adaptive Adversary Strategies",
        "content": (
            "Adaptive adversaries modify their behavior in response to governance "
            "signals, alternating between cooperative and exploitative phases. "
            "Detecting such agents requires behavioral feature analysis beyond "
            "simple threshold checks."
        ),
        "quality_score": 0.46,
    },
    {
        "title": "Sybil Resistance",
        "content": (
            "Sybil attacks create multiple pseudonymous identities to circumvent "
            "per-agent governance limits. Defenses include staking requirements, "
            "identity verification challenges, and network-based anomaly detection."
        ),
        "quality_score": 0.39,
    },
    # --- Cross-Platform Evaluation ---
    {
        "title": "Cross-Platform Safety Evaluation",
        "content": (
            "Cross-platform evaluation measures how safety properties transfer "
            "across different environment types (marketplace, wiki, social feed). "
            "A mechanism effective in one domain may fail in another due to "
            "different interaction structures."
        ),
        "quality_score": 0.47,
    },
    {
        "title": "Moltipedia Environment",
        "content": (
            "Moltipedia is a simulated wiki platform where agents create, edit, "
            "and contest pages. Quality scores, policy violations, and edit "
            "histories model the dynamics of collaborative knowledge production "
            "under adversarial pressure."
        ),
        "quality_score": 0.50,
    },
    {
        "title": "Moltbook Environment",
        "content": (
            "Moltbook is a simulated social media platform with anti-human CAPTCHA "
            "challenges, rate limits, karma scoring, and submolt communities. It "
            "models content moderation dynamics in AI-first social networks."
        ),
        "quality_score": 0.49,
    },
    {
        "title": "Marketplace Environment",
        "content": (
            "The marketplace environment simulates task bounties, bidding, escrow, "
            "and dispute resolution. Agents post tasks and bid on work, with "
            "governance levers controlling escrow fees and dispute outcomes."
        ),
        "quality_score": 0.43,
    },
    # --- Rain/River Model ---
    {
        "title": "Rain-River Model",
        "content": (
            "The rain-river model is an analogy for distributional safety: "
            "individual interactions (raindrops) aggregate into systemic outcomes "
            "(rivers). Small per-interaction harms compound into large ecosystem "
            "damage if left unchecked."
        ),
        "quality_score": 0.51,
    },
    {
        "title": "Aggregate Harm Accumulation",
        "content": (
            "Aggregate harm accumulation shows how individually tolerable "
            "externalities compound across many interactions. The rain-river "
            "metaphor highlights that governance must address the flow, not "
            "just individual drops."
        ),
        "quality_score": 0.36,
    },
    # --- Adverse Selection ---
    {
        "title": "Adverse Selection in Multi-Agent Systems",
        "content": (
            "Adverse selection occurs when low-quality interactions are "
            "preferentially accepted, indicated by a negative quality gap. "
            "It arises when agent screening mechanisms are weaker than agent "
            "deception capabilities."
        ),
        "quality_score": 0.44,
    },
    {
        "title": "Quality Gap Metric",
        "content": (
            "The quality gap E[p|accepted] - E[p|rejected] measures selection "
            "quality. A positive gap means the system correctly filters; a "
            "negative gap signals adverse selection where harmful interactions "
            "pass screening more readily."
        ),
        "quality_score": 0.42,
    },
    # --- Externality Internalization ---
    {
        "title": "Externality Internalization",
        "content": (
            "Externality internalization (rho parameters) controls what fraction "
            "of ecosystem harm each agent absorbs. Higher rho values force agents "
            "to price in the social cost of their actions, discouraging negative "
            "externalities."
        ),
        "quality_score": 0.40,
    },
    {
        "title": "Pigouvian Tax Analogy",
        "content": (
            "The externality internalization mechanism draws on the Pigouvian tax "
            "concept: by charging agents proportionally to the harm they cause, "
            "the system aligns private incentives with social welfare."
        ),
        "quality_score": 0.34,
    },
    # --- Metrics & Measurement ---
    {
        "title": "Toxicity Metric",
        "content": (
            "Toxicity E[1-p | accepted] measures the expected harmfulness of "
            "interactions that pass the acceptance filter. A well-governed system "
            "keeps toxicity below configurable thresholds."
        ),
        "quality_score": 0.46,
    },
    {
        "title": "Conditional Loss",
        "content": (
            "Conditional loss captures the payoff impact of selection effects: "
            "the difference between average payoffs of accepted vs. rejected "
            "interactions reveals whether governance improves or distorts outcomes."
        ),
        "quality_score": 0.38,
    },
    {
        "title": "Dual Reporting of Soft and Hard Metrics",
        "content": (
            "The MetricsReporter provides both soft (probabilistic) and hard "
            "(threshold-based) metric views. Comparing soft and hard metrics "
            "reveals calibration gaps and threshold sensitivity."
        ),
        "quality_score": 0.41,
    },
    {
        "title": "Self-Ensemble Variance Governance",
        "content": (
            "Self-ensemble governance runs multiple proxy evaluations of the same "
            "interaction and uses the variance to trigger caution. High variance "
            "signals model uncertainty and may activate incoherence circuit "
            "breakers."
        ),
        "quality_score": 0.43,
    },
]


def seed_from_catalog(
    pool: "WikiTaskPool",
    n_pages: int,
    rng: random.Random,
) -> None:
    """Seed a WikiTaskPool with entries drawn from the research catalog.

    Args:
        pool: The WikiTaskPool to populate.
        n_pages: Number of pages to create (capped at catalog size).
        rng: Random generator for reproducible sampling.
    """
    n_pages = min(n_pages, len(WIKI_CATALOG))
    entries = rng.sample(WIKI_CATALOG, n_pages)

    for entry in entries:
        status = (
            PageStatus.STUB if len(entry["content"]) < 120 else PageStatus.DRAFT
        )
        page = WikiPage(
            title=entry["title"],
            content=entry["content"],
            status=status,
            quality_score=entry["quality_score"],
            created_by="catalog_seed",
            last_editor="catalog_seed",
        )
        pool.add_page(page)
