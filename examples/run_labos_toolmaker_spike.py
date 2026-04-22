#!/usr/bin/env python
"""LabOS Toolmaker→Critic screening spike (SWA-68).

Research question
-----------------
LabOS (Stanford/Princeton AI co-scientist, arxiv 2510.14861) has a
Toolmaker agent that manufactures new Python tools at runtime and a
Critic agent that accepts/rejects them. Under asymmetric information
(the Toolmaker sees the tool's innards, the Critic only sees the
proposed interface + claims), is the Critic's implicit acceptance
probability well-calibrated against *downstream* tool quality?

If yes, the Critic is already computing something like SWARM's
``p = P(v = +1)`` — which makes the Toolmaker→Critic loop a natural
host for distributional-safety screening. If no, we have an adverse-
selection story: low-quality tools get accepted disproportionately,
and governance levers (staking, ρ-internalization, audits) can be
studied on the same event stream.

Approach
--------
This spike does NOT import ``labos``. Instead it synthesizes a
Toolmaker→Critic event stream with structure that matches LabOS's
dry-lab loop and a known ground-truth quality label per proposed tool.
The stream is logged as an append-only JSONL via
:class:`swarm.logging.event_log.EventLog`, then run through
:class:`swarm.core.proxy.ProxyComputer` to derive ``v_hat`` and ``p``.
We compare ``p`` against the ground-truth to produce a calibration
plot and to compute the adverse-selection signal
``quality_gap = E[p|accepted] - E[p|rejected]``.

This matches the spike scope in SWA-68:
    1. scenarios/labos_toolmaker_screening.yaml exists.
    2. Toolmaker→Critic instrumented with accept/reject, proxy-
       compatible observables, and a downstream quality signal.
    3. Events logged through swarm/logging/event_log.py.
    4. One calibration plot: Critic's p vs. ground-truth outcome.
    5. A ~1-page scoping note at docs/bridges/labos.md (companion).

Usage
-----
    python examples/run_labos_toolmaker_spike.py
    python examples/run_labos_toolmaker_spike.py --seed 7 --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm.core.proxy import ProxyComputer, ProxyObservables, ProxyWeights
from swarm.logging.event_log import EventLog
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.events import Event, EventType
from swarm.models.interaction import InteractionType, SoftInteraction

# ---------------------------------------------------------------------------
# Toolmaker profiles
# ---------------------------------------------------------------------------

@dataclass
class ToolmakerProfile:
    """Synthetic behavioral profile for a LabOS-style Toolmaker.

    Each field controls how the Toolmaker's proposed tool looks to the
    Critic (observables) vs. how the tool actually performs on the
    downstream Wet-Lab-Error-Detection-style task (ground truth).
    """

    agent_id: str
    agent_type: str  # "honest" | "opportunistic" | "careless"

    # --- Ground-truth quality distribution ---
    # mean/std of `v_true ∈ [0, 1]`, the tool's actual success probability
    # when used by downstream agents. Anchored at the high-calibre end for
    # honest, wide+low for opportunistic/careless.
    v_true_mean: float
    v_true_std: float

    # --- Observable feature noise / inflation ---
    # doc_completeness, test_coverage, interface_coherence are what the
    # Critic can see at accept-time. `signal_gain` > 1 means the Toolmaker
    # inflates those signals relative to v_true (opportunistic pattern).
    # `signal_noise` is the per-feature Gaussian noise.
    signal_gain: float
    signal_noise: float

    # --- Claim inflation ---
    # `claimed_usefulness` is self-reported. Opportunistic Toolmakers
    # claim more than they can deliver.
    claim_inflation: float


HONEST = ToolmakerProfile(
    agent_id="",
    agent_type="honest",
    v_true_mean=0.78,
    v_true_std=0.10,
    signal_gain=1.0,
    signal_noise=0.07,
    claim_inflation=0.02,
)

OPPORTUNISTIC = ToolmakerProfile(
    agent_id="",
    agent_type="opportunistic",
    v_true_mean=0.42,
    v_true_std=0.15,
    signal_gain=1.6,        # pads observables above true quality
    signal_noise=0.05,
    claim_inflation=0.30,
)

CARELESS = ToolmakerProfile(
    agent_id="",
    agent_type="careless",
    v_true_mean=0.55,
    v_true_std=0.18,
    signal_gain=0.9,        # honest-ish signals
    signal_noise=0.15,      # but very noisy — thin docs, spotty tests
    claim_inflation=0.05,
)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def make_toolmakers(n_honest: int, n_opportunistic: int, n_careless: int) -> List[ToolmakerProfile]:
    """Construct Toolmaker roster with unique agent IDs."""
    roster: List[ToolmakerProfile] = []
    for i in range(n_honest):
        roster.append(ToolmakerProfile(**{**HONEST.__dict__, "agent_id": f"honest-{i}"}))
    for i in range(n_opportunistic):
        roster.append(ToolmakerProfile(**{**OPPORTUNISTIC.__dict__, "agent_id": f"opportunistic-{i}"}))
    for i in range(n_careless):
        roster.append(ToolmakerProfile(**{**CARELESS.__dict__, "agent_id": f"careless-{i}"}))
    return roster


# ---------------------------------------------------------------------------
# Synthetic event generation
# ---------------------------------------------------------------------------

@dataclass
class ToolProposal:
    """One Toolmaker→Critic proposal with latent and observable fields."""

    tool_id: str
    toolmaker: str
    toolmaker_type: str
    epoch: int
    step: int

    # Ground truth — what we are calibrating against
    v_true: float

    # Observables visible to the Critic at accept-time
    doc_completeness: float
    test_coverage: float
    interface_coherence: float
    claimed_usefulness: float

    # Downstream signals produced post-acceptance
    runtime_errors: int
    downstream_retries: int
    downstream_success_rate: float


def _propose_tool(
    tm: ToolmakerProfile,
    tool_idx: int,
    epoch: int,
    step: int,
    rng: random.Random,
) -> ToolProposal:
    """Draw one synthetic tool proposal for a Toolmaker."""
    v_true = _clamp01(rng.gauss(tm.v_true_mean, tm.v_true_std))

    def _signal(base: float) -> float:
        return _clamp01(rng.gauss(base * tm.signal_gain, tm.signal_noise))

    doc = _signal(v_true)
    tests = _signal(v_true)
    coh = _signal(v_true)
    claim = _clamp01(v_true + tm.claim_inflation + rng.gauss(0.0, 0.05))

    # Downstream signals are a noisy function of v_true — this is the
    # Wet-Lab-Error-Detection-style quality signal that appears *after*
    # the Critic has already committed. Runtime errors scale inversely
    # with v_true; retries scale similarly; success rate is bounded by v_true.
    runtime_errors = max(0, int(round(rng.gauss((1 - v_true) * 4.0, 0.6))))
    downstream_retries = max(0, int(round(rng.gauss((1 - v_true) * 2.5, 0.5))))
    success_rate = _clamp01(rng.gauss(v_true, 0.08))

    return ToolProposal(
        tool_id=f"{tm.agent_id}-tool-{tool_idx:03d}",
        toolmaker=tm.agent_id,
        toolmaker_type=tm.agent_type,
        epoch=epoch,
        step=step,
        v_true=v_true,
        doc_completeness=doc,
        test_coverage=tests,
        interface_coherence=coh,
        claimed_usefulness=claim,
        runtime_errors=runtime_errors,
        downstream_retries=downstream_retries,
        downstream_success_rate=success_rate,
    )


def _proposal_to_observables(p: ToolProposal) -> ProxyObservables:
    """Map a ToolProposal to ProxyObservables.

    Mapping rationale (mirrors what a real LabOS instrumentation would
    yield, but computable from synthetic fields):

      task_progress_delta   — downstream_success_rate rescaled to [-1, 1]
      rework_count          — downstream_retries
      verifier_rejections   — runtime_errors (hard failures at call-time)
      tool_misuse_flags     — 1 if the Toolmaker's claim exceeds
                              signalled quality by a large margin (proxy
                              for 'manifest-reality gap')
      counterparty_engagement_delta — documentation + tests + coherence
                                       combined, rescaled to [-1, 1]
    """
    structural = (p.doc_completeness + p.test_coverage + p.interface_coherence) / 3.0
    claim_gap = p.claimed_usefulness - structural
    misuse = 1 if claim_gap > 0.25 else 0

    return ProxyObservables(
        task_progress_delta=2.0 * p.downstream_success_rate - 1.0,
        rework_count=p.downstream_retries,
        verifier_rejections=p.runtime_errors,
        tool_misuse_flags=misuse,
        counterparty_engagement_delta=2.0 * structural - 1.0,
    )


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

def _log_proposal_events(
    log: EventLog,
    proposal: ToolProposal,
    v_hat: float,
    p: float,
    accepted: bool,
    scenario_id: str,
    seed: int,
    interaction_id: str,
) -> None:
    """Emit a proposed → (accepted | rejected) → proxy_computed triplet."""
    ts = datetime.now()

    # 1) Proposed
    log.append(Event(
        event_type=EventType.INTERACTION_PROPOSED,
        timestamp=ts,
        interaction_id=interaction_id,
        agent_id=proposal.toolmaker,
        initiator_id=proposal.toolmaker,
        counterparty_id="critic",
        epoch=proposal.epoch,
        step=proposal.step,
        scenario_id=scenario_id,
        seed=seed,
        payload={
            "bridge": "labos_toolmaker",
            "interaction_type": InteractionType.COLLABORATION.value,
            "tool_id": proposal.tool_id,
            "toolmaker_type": proposal.toolmaker_type,
            "v_hat": v_hat,
            "p": p,
            # Observables the Critic saw
            "doc_completeness": proposal.doc_completeness,
            "test_coverage": proposal.test_coverage,
            "interface_coherence": proposal.interface_coherence,
            "claimed_usefulness": proposal.claimed_usefulness,
            # Ground truth — logged for calibration, not used by the Critic
            "v_true": proposal.v_true,
        },
    ))

    # 2) Accepted or Rejected
    decision_type = (
        EventType.INTERACTION_ACCEPTED if accepted else EventType.INTERACTION_REJECTED
    )
    log.append(Event(
        event_type=decision_type,
        timestamp=ts,
        interaction_id=interaction_id,
        agent_id="critic",
        initiator_id=proposal.toolmaker,
        counterparty_id="critic",
        epoch=proposal.epoch,
        step=proposal.step,
        scenario_id=scenario_id,
        seed=seed,
        payload={
            "bridge": "labos_toolmaker",
            "tool_id": proposal.tool_id,
            "threshold": 0.5,
            "p": p,
        },
    ))

    # 3) Proxy computed (for downstream replay fidelity)
    log.append(Event(
        event_type=EventType.PROXY_COMPUTED,
        timestamp=ts,
        interaction_id=interaction_id,
        agent_id=proposal.toolmaker,
        initiator_id=proposal.toolmaker,
        counterparty_id="critic",
        epoch=proposal.epoch,
        step=proposal.step,
        scenario_id=scenario_id,
        seed=seed,
        payload={
            "bridge": "labos_toolmaker",
            "tool_id": proposal.tool_id,
            "v_hat": v_hat,
            "p": p,
            "runtime_errors": proposal.runtime_errors,
            "downstream_retries": proposal.downstream_retries,
            "downstream_success_rate": proposal.downstream_success_rate,
            "v_true": proposal.v_true,
        },
    ))


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation. Returns 0.0 if undefined."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0
    return cov / denom


def _binned_calibration(
    ps: List[float],
    ys: List[float],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Reliability-diagram style binning of p vs. ground-truth v_true.

    For each [k/n_bins, (k+1)/n_bins) bin of p, compute the mean p and
    the mean ground-truth outcome.

    Returns a list of dicts (one per bin) with at least: bin_lo, bin_hi,
    n, mean_p, mean_y. Empty bins are omitted.
    """
    if not ps:
        return []
    bins: List[List[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    for p, y in zip(ps, ys, strict=True):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))

    rows: List[Dict[str, float]] = []
    for k, pts in enumerate(bins):
        if not pts:
            continue
        mean_p = sum(pt[0] for pt in pts) / len(pts)
        mean_y = sum(pt[1] for pt in pts) / len(pts)
        rows.append({
            "bin_lo": k / n_bins,
            "bin_hi": (k + 1) / n_bins,
            "n": len(pts),
            "mean_p": mean_p,
            "mean_y": mean_y,
        })
    return rows


def _ece(bin_rows: List[Dict[str, float]], n_total: int) -> float:
    """Expected calibration error (L1). Weighted by bin count."""
    if n_total == 0:
        return 0.0
    return sum(
        (r["n"] / n_total) * abs(r["mean_p"] - r["mean_y"])
        for r in bin_rows
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _save_calibration_plot(
    bin_rows: List[Dict[str, float]],
    per_type_bins: Dict[str, List[Dict[str, float]]],
    ece: float,
    pearson: float,
    quality_gap: float,
    out_path: Path,
) -> None:
    """Save the calibration plot (Critic p vs ground-truth v_true)."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 6.0))

    # Identity reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")

    # Aggregate curve with bin counts scaling marker size
    if bin_rows:
        xs = [r["mean_p"] for r in bin_rows]
        ys = [r["mean_y"] for r in bin_rows]
        sizes = [40 + 12 * r["n"] ** 0.5 for r in bin_rows]
        ax.plot(xs, ys, color="black", linewidth=1.6, label="all Toolmakers")
        ax.scatter(xs, ys, s=sizes, color="black", zorder=3)

    # Per-type curves
    colors = {"honest": "#2c7fb8", "opportunistic": "#d95f02", "careless": "#7570b3"}
    for tm_type, rows in per_type_bins.items():
        if not rows:
            continue
        xs = [r["mean_p"] for r in rows]
        ys = [r["mean_y"] for r in rows]
        ax.plot(xs, ys, color=colors.get(tm_type, "#444"), linewidth=1.2, alpha=0.85,
                label=tm_type)
        ax.scatter(xs, ys, s=32, color=colors.get(tm_type, "#444"), alpha=0.85, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Critic's implicit acceptance probability  $p$")
    ax.set_ylabel(r"Downstream ground-truth quality  $\bar{v}_{\mathrm{true}}$")
    title = (
        "LabOS Toolmaker→Critic calibration\n"
        rf"ECE = {ece:.3f}   "
        rf"$\rho(p, v_{{\mathrm{{true}}}})$ = {pearson:.3f}   "
        rf"quality_gap = {quality_gap:+.3f}"
    )
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main spike
# ---------------------------------------------------------------------------

def _load_scenario(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _counts_from_scenario(scenario: dict) -> Tuple[int, int, int]:
    """Extract (n_honest, n_opportunistic, n_careless) from the YAML."""
    n_h, n_o, n_c = 0, 0, 0
    for row in scenario.get("agents", []):
        t = row.get("type", "")
        c = int(row.get("count", 0))
        if t == "honest":
            n_h += c
        elif t == "opportunistic":
            n_o += c
        else:
            # deceptive is the roster-slot used for the careless archetype
            # (the scenario YAML is constrained to registered agent types).
            n_c += c
    return n_h, n_o, n_c


def run_spike(
    scenario_path: Path,
    seed: int,
    output_root: Path,
) -> Path:
    """Run the spike end-to-end and return the run directory."""
    scenario = _load_scenario(scenario_path)
    scenario_id = scenario.get("scenario_id", "labos_toolmaker_screening")

    domain = scenario.get("domain", {}) or {}
    proxy_cfg = domain.get("proxy", {}) or {}
    threshold = float(domain.get("acceptance_threshold", 0.5))
    proposals_per = int(domain.get("proposals_per_toolmaker_per_epoch", 3))
    n_bins = int(domain.get("calibration_bins", 10))

    sim = scenario.get("simulation", {}) or {}
    n_epochs = int(sim.get("n_epochs", 10))
    steps_per_epoch = int(sim.get("steps_per_epoch", 10))

    n_h, n_o, n_c = _counts_from_scenario(scenario)
    toolmakers = make_toolmakers(n_h, n_o, n_c)

    # Prepare run directory.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{ts}_labos_toolmaker_screening_seed{seed}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "csv").mkdir(parents=True, exist_ok=True)

    # Proxy + logger.
    weights = ProxyWeights(
        task_progress=float(proxy_cfg.get("task_progress_weight", 0.4)),
        rework_penalty=float(proxy_cfg.get("rework_weight", 0.2)),
        verifier_penalty=float(proxy_cfg.get("verifier_weight", 0.2)),
        engagement_signal=float(proxy_cfg.get("engagement_weight", 0.2)),
    )
    proxy = ProxyComputer(weights=weights)
    log = EventLog(run_dir / "event_log.jsonl")

    # Generate, score, log.
    rng = random.Random(seed)
    interactions: List[SoftInteraction] = []
    rows: List[Dict[str, float]] = []
    proposal_counter: Dict[str, int] = {tm.agent_id: 0 for tm in toolmakers}

    for epoch in range(n_epochs):
        for step in range(steps_per_epoch):
            # Each Toolmaker proposes every `steps_per_epoch/proposals_per`
            # steps. Simplest: trigger `proposals_per` proposals per
            # Toolmaker in the first `proposals_per` steps of each epoch.
            if step >= proposals_per:
                continue
            for tm in toolmakers:
                proposal_counter[tm.agent_id] += 1
                proposal = _propose_tool(
                    tm, proposal_counter[tm.agent_id], epoch, step, rng,
                )
                obs = _proposal_to_observables(proposal)
                v_hat, p = proxy.compute_labels(obs)
                accepted = p >= threshold

                interaction_id = f"{proposal.tool_id}-ix"
                _log_proposal_events(
                    log, proposal, v_hat, p, accepted,
                    scenario_id, seed, interaction_id,
                )

                interactions.append(SoftInteraction(
                    interaction_id=interaction_id,
                    initiator=proposal.toolmaker,
                    counterparty="critic",
                    interaction_type=InteractionType.COLLABORATION,
                    accepted=accepted,
                    task_progress_delta=obs.task_progress_delta,
                    rework_count=obs.rework_count,
                    verifier_rejections=obs.verifier_rejections,
                    tool_misuse_flags=obs.tool_misuse_flags,
                    counterparty_engagement_delta=obs.counterparty_engagement_delta,
                    v_hat=v_hat,
                    p=p,
                    metadata={
                        "bridge": "labos_toolmaker",
                        "tool_id": proposal.tool_id,
                        "toolmaker_type": proposal.toolmaker_type,
                        "v_true": proposal.v_true,
                        "epoch": proposal.epoch,
                        "step": proposal.step,
                    },
                ))
                rows.append({
                    "tool_id": proposal.tool_id,
                    "toolmaker": proposal.toolmaker,
                    "toolmaker_type": proposal.toolmaker_type,
                    "epoch": proposal.epoch,
                    "step": proposal.step,
                    "doc_completeness": proposal.doc_completeness,
                    "test_coverage": proposal.test_coverage,
                    "interface_coherence": proposal.interface_coherence,
                    "claimed_usefulness": proposal.claimed_usefulness,
                    "runtime_errors": proposal.runtime_errors,
                    "downstream_retries": proposal.downstream_retries,
                    "downstream_success_rate": proposal.downstream_success_rate,
                    "v_true": proposal.v_true,
                    "v_hat": v_hat,
                    "p": p,
                    "accepted": int(accepted),
                })

    # --- Metrics ------------------------------------------------------
    metrics = SoftMetrics()
    toxicity = metrics.toxicity_rate(interactions)
    quality_gap = metrics.quality_gap(interactions)
    mean_p = statistics.mean(i.p for i in interactions) if interactions else 0.0

    ps = [i.p for i in interactions]
    vs = [float(i.metadata.get("v_true", 0.0)) for i in interactions]
    pearson = _pearson(ps, vs)

    bin_rows = _binned_calibration(ps, vs, n_bins=n_bins)
    ece = _ece(bin_rows, len(ps))

    # Per-type binning for the plot
    per_type_bins: Dict[str, List[Dict[str, float]]] = {}
    for tm_type in ("honest", "opportunistic", "careless"):
        ps_t = [
            i.p for i in interactions
            if i.metadata.get("toolmaker_type") == tm_type
        ]
        vs_t = [
            float(i.metadata.get("v_true", 0.0)) for i in interactions
            if i.metadata.get("toolmaker_type") == tm_type
        ]
        per_type_bins[tm_type] = _binned_calibration(ps_t, vs_t, n_bins=n_bins)

    # Per-Toolmaker-type summary
    type_summary: Dict[str, Dict[str, float]] = {}
    for tm_type in ("honest", "opportunistic", "careless"):
        type_rows = [r for r in rows if r["toolmaker_type"] == tm_type]
        if not type_rows:
            continue
        type_summary[tm_type] = {
            "n_tools": len(type_rows),
            "mean_v_true": statistics.mean(r["v_true"] for r in type_rows),
            "mean_p": statistics.mean(r["p"] for r in type_rows),
            "acceptance_rate": statistics.mean(r["accepted"] for r in type_rows),
            "mean_claim_inflation": statistics.mean(
                r["claimed_usefulness"] - r["v_true"] for r in type_rows
            ),
        }

    # --- Artifacts ----------------------------------------------------
    # proposals.csv (one row per proposal)
    csv_path = run_dir / "csv" / "proposals.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # calibration.csv (one row per bin)
    cal_path = run_dir / "csv" / "calibration.csv"
    with open(cal_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bin_lo", "bin_hi", "n", "mean_p", "mean_y"])
        writer.writeheader()
        writer.writerows(bin_rows)

    # Plot
    plot_path = run_dir / "plots" / "calibration_critic_p_vs_v_true.png"
    _save_calibration_plot(
        bin_rows, per_type_bins, ece, pearson, quality_gap, plot_path,
    )

    # summary.json
    summary = {
        "scenario_id": scenario_id,
        "seed": seed,
        "n_epochs": n_epochs,
        "steps_per_epoch": steps_per_epoch,
        "n_toolmakers": {"honest": n_h, "opportunistic": n_o, "careless": n_c},
        "n_proposals": len(interactions),
        "n_accepted": sum(1 for i in interactions if i.accepted),
        "n_rejected": sum(1 for i in interactions if not i.accepted),
        "metrics": {
            "mean_p": mean_p,
            "toxicity_rate": toxicity,
            "quality_gap": quality_gap,
            "adverse_selection_rate": max(0.0, -quality_gap),
            "pearson_p_vtrue": pearson,
            "ece_binned": ece,
        },
        "per_type": type_summary,
        "paths": {
            "event_log": str((run_dir / "event_log.jsonl").relative_to(run_dir)),
            "proposals_csv": str(csv_path.relative_to(run_dir)),
            "calibration_csv": str(cal_path.relative_to(run_dir)),
            "plot": str(plot_path.relative_to(run_dir)),
        },
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print verdict ----------------------------------------------
    print("=" * 70)
    print(f"LabOS Toolmaker spike — seed={seed}, {len(interactions)} proposals")
    print("=" * 70)
    print(f"  mean_p            = {mean_p:.3f}")
    print(f"  toxicity          = {toxicity:.3f}")
    print(f"  quality_gap       = {quality_gap:+.3f}  "
          f"{'(adverse selection)' if quality_gap < 0 else '(separating)'}")
    print(f"  Pearson(p,v_true) = {pearson:+.3f}")
    print(f"  ECE (binned)      = {ece:.3f}")
    print()
    print("Per-Toolmaker-type:")
    for tm_type, s in type_summary.items():
        print(
            f"  {tm_type:>14s}: n={s['n_tools']:>3d}  "
            f"v_true={s['mean_v_true']:.3f}  "
            f"p={s['mean_p']:.3f}  "
            f"accept={s['acceptance_rate']:.3f}  "
            f"claim_inflation={s['mean_claim_inflation']:+.3f}"
        )
    print()
    print(f"Run directory: {run_dir}")
    print(f"Plot:          {plot_path}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LabOS Toolmaker→Critic screening spike (SWA-68)",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("scenarios/labos_toolmaker_screening.yaml"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override simulation.n_epochs from scenario.")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()

    if args.epochs is not None:
        scen = _load_scenario(args.scenario)
        scen.setdefault("simulation", {})["n_epochs"] = args.epochs
        # write into a temp path so we don't mutate the committed scenario
        tmp = args.scenario.with_suffix(".override.yaml")
        with open(tmp, "w") as f:
            yaml.safe_dump(scen, f)
        scenario_path = tmp
    else:
        scenario_path = args.scenario

    run_spike(scenario_path, args.seed, args.output_root)


if __name__ == "__main__":
    main()
