"""Formatting helpers for the demo: KPI cards, colors, descriptions."""

from html import escape as _esc
from typing import Any, Dict, List

AGENT_TYPE_COLORS = {
    "honest": "#28a745",
    "opportunistic": "#ffc107",
    "deceptive": "#6c757d",
    "adversarial": "#dc3545",
    "adaptive_adversary": "#e83e8c",
}

AGENT_TYPE_LABELS = {
    "honest": "Honest",
    "opportunistic": "Opportunistic",
    "deceptive": "Deceptive",
    "adversarial": "Adversarial",
    "adaptive_adversary": "Adaptive Adversary",
}


def kpi_card_css() -> str:
    """Return CSS for KPI metric cards."""
    return """
    <style>
    .kpi-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        flex: 1;
        text-align: center;
        border-left: 4px solid #0d6efd;
    }
    .kpi-card.danger { border-left-color: #dc3545; }
    .kpi-card.success { border-left-color: #28a745; }
    .kpi-card.warning { border-left-color: #ffc107; }
    .kpi-card.info { border-left-color: #0d6efd; }
    .kpi-label {
        font-size: 0.85rem;
        color: #6c757d;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #212529;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #adb5bd;
    }
    </style>
    """


def kpi_card(label: str, value: str, style: str = "info", subtitle: str = "") -> str:
    """Generate HTML for a single KPI card."""
    sub_html = f'<div class="kpi-sub">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="kpi-card {style}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """


def kpi_row(cards: List[str]) -> str:
    """Wrap cards in a flex row."""
    inner = "".join(cards)
    return f'<div class="kpi-row">{inner}</div>'


def format_epoch_metrics_kpis(
    epoch_metrics: list,
    incoherence_series: List[float] | None = None,
) -> str:
    """Generate KPI cards from the final epoch metrics."""
    if not epoch_metrics:
        return "<p>No data</p>"

    last = epoch_metrics[-1]
    toxicity = last.toxicity_rate
    quality_gap = last.quality_gap
    welfare = last.total_welfare
    accepted = last.accepted_interactions
    total = last.total_interactions
    acc_rate = accepted / total if total > 0 else 0
    incoherence = incoherence_series[-1] if incoherence_series else 0.0

    tox_style = (
        "success" if toxicity < 0.2 else ("warning" if toxicity < 0.4 else "danger")
    )
    qg_style = "success" if quality_gap > 0 else "danger"
    incoh_style = (
        "success"
        if incoherence < 0.3
        else ("warning" if incoherence < 0.6 else "danger")
    )

    cards = [
        kpi_card("Toxicity Rate", f"{toxicity:.3f}", tox_style, "lower is better"),
        kpi_card(
            "Quality Gap", f"{quality_gap:.3f}", qg_style, "positive = good selection"
        ),
        kpi_card("Incoherence", f"{incoherence:.3f}", incoh_style, "lower is better"),
        kpi_card("Total Welfare", f"{welfare:.1f}", "info"),
        kpi_card(
            "Acceptance Rate",
            f"{acc_rate:.1%}",
            "info",
            f"{accepted}/{total} interactions",
        ),
    ]
    return kpi_card_css() + kpi_row(cards)


def agent_type_badge(agent_type: str) -> str:
    """Return colored HTML badge for agent type."""
    color = AGENT_TYPE_COLORS.get(agent_type, "#999")
    label = AGENT_TYPE_LABELS.get(agent_type, _esc(agent_type))
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.8rem;">{_esc(label)}</span>'
    )


def scenario_description_card(result: Dict[str, Any]) -> str:
    """Render a description card for a scenario result."""
    sid = _esc(str(result.get("scenario_id", "Unknown")))
    desc = _esc(str(result.get("description", "")))
    n_agents = _esc(str(result.get("n_agents", "?")))
    n_epochs = _esc(str(result.get("n_epochs", "?")))
    return f"""
    <div style="background:#f0f4f8;padding:1rem;border-radius:8px;margin-bottom:1rem;">
        <strong>{sid}</strong>
        <br/><span style="color:#6c757d;">{desc}</span>
        <br/><small>{n_agents} agents &middot;
        {n_epochs} epochs</small>
    </div>
    """
