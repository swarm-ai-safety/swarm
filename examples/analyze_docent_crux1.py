"""Decompose the Docent CRUX-1 "Publish Breathe Easy" agent run by stage category.

Reads the chunked-telemetry CSV exported from Transluce's Docent dashboard
(``examples/data/docent_crux1_agent_runs.csv``) and produces:

1. A per-stage decomposition table (events, messages, minutes) printed to stdout.
2. A two-panel figure (event-rate per chunk + event-share stacked bar) written to
   ``docs/papers/figures/crux1_event_decomposition.png``.

This is a read-only analysis of third-party metadata — no swarm simulation is run.
The case study it supports lives at ``docs/research/docent-crux1-polling-vs-surplus.md``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "examples" / "data" / "docent_crux1_agent_runs.csv"
FIG_PATH = REPO_ROOT / "docs" / "research" / "figures" / "crux1_event_decomposition.png"

STAGE_COLOR = {
    "bootstrap": "#2a9d8f",
    "submission": "#2a9d8f",
    "waiting_for_review": "#e9c46a",
    "in_review": "#f4a261",
    "observability_gap": "#e63946",
    "release": "#264653",
}

PRODUCTIVE = {"bootstrap", "submission", "release"}
POLLING = {"waiting_for_review", "in_review"}


@dataclass
class Chunk:
    index: int
    slug: str
    stage: str
    events: int
    messages: int
    start: datetime
    end: datetime

    @property
    def minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60

    @property
    def events_per_min(self) -> float:
        return self.events / self.minutes if self.minutes else 0.0


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            chunks.append(
                Chunk(
                    index=int(row["metadata.section_index"]),
                    slug=row["metadata.section_slug"],
                    stage=row["metadata.stage"],
                    events=int(row["metadata.event_count"]),
                    messages=int(row["metadata.transcript_message_count"]),
                    start=datetime.fromisoformat(row["metadata.window_start"]),
                    end=datetime.fromisoformat(row["metadata.window_end"]),
                )
            )
    chunks.sort(key=lambda c: c.index)
    return chunks


def category(stage: str) -> str:
    if stage in PRODUCTIVE:
        return "productive"
    if stage in POLLING:
        return "polling"
    return "observability_gap"


def print_report(chunks: list[Chunk]) -> dict[str, dict[str, float]]:
    print(f"{'idx':>3} {'stage':<22} {'events':>7} {'msgs':>6} {'dur_min':>8}  slug")
    for c in chunks:
        print(
            f"{c.index:>3} {c.stage:<22} {c.events:>7} {c.messages:>6} "
            f"{c.minutes:>8.1f}  {c.slug}"
        )

    by_cat: dict[str, dict[str, float]] = defaultdict(
        lambda: {"events": 0, "messages": 0, "minutes": 0.0}
    )
    for c in chunks:
        cat = category(c.stage)
        by_cat[cat]["events"] += c.events
        by_cat[cat]["messages"] += c.messages
        by_cat[cat]["minutes"] += c.minutes

    tot_ev = sum(v["events"] for v in by_cat.values())
    tot_min = sum(v["minutes"] for v in by_cat.values())
    print(f"\nTotal: events={tot_ev} minutes={tot_min:.0f} ({tot_min/60/24:.2f} days)\n")
    print(f"{'category':<18} {'events':>7} {'%events':>8} {'minutes':>9} {'%time':>7}")
    for cat in ("productive", "polling", "observability_gap"):
        d = by_cat[cat]
        print(
            f"{cat:<18} {d['events']:>7.0f} "
            f"{100*d['events']/tot_ev:>7.1f}% "
            f"{d['minutes']:>9.1f} "
            f"{100*d['minutes']/tot_min:>6.1f}%"
        )
    return by_cat


def plot(chunks: list[Chunk], by_cat: dict[str, dict[str, float]], out_path: Path) -> None:
    t0 = min(c.start for c in chunks)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}
    )

    for c in chunks:
        start_h = (c.start - t0).total_seconds() / 3600
        dur_h = c.minutes / 60
        ax1.bar(
            start_h,
            c.events_per_min,
            width=dur_h,
            align="edge",
            color=STAGE_COLOR[c.stage],
            edgecolor="white",
            linewidth=0.5,
        )
        ax1.text(
            start_h + dur_h / 2,
            c.events_per_min + 0.4,
            f"{c.events:,}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax1.set_ylabel("events / minute")
    ax1.set_xlabel("hours from run start")
    ax1.set_title(
        'CRUX-1 "Publish Breathe Easy" — event rate by stage chunk\n'
        "agent is productive by count, not by surplus: 97% of events during polling stages"
    )
    ax1.set_ylim(0, 17)
    ax1.legend(
        handles=[mpatches.Patch(color=c, label=s) for s, c in STAGE_COLOR.items()],
        loc="upper right",
        fontsize=8,
        frameon=False,
    )

    tot_ev = sum(v["events"] for v in by_cat.values())
    p_prod = 100 * by_cat["productive"]["events"] / tot_ev
    p_poll = 100 * by_cat["polling"]["events"] / tot_ev
    p_gap = 100 * by_cat["observability_gap"]["events"] / tot_ev

    ax2.barh(0, p_prod, color="#2a9d8f")
    ax2.barh(0, p_poll, left=p_prod, color="#f4a261")
    ax2.barh(0, p_gap, left=p_prod + p_poll, color="#e63946")
    ax2.text(p_prod / 2, 0, f"{p_prod:.1f}%", ha="center", va="center", color="white",
             fontsize=9, fontweight="bold")
    ax2.text(p_prod + p_poll / 2, 0, f"{p_poll:.1f}% (polling)", ha="center", va="center",
             color="white", fontsize=9, fontweight="bold")
    ax2.text(p_prod + p_poll + p_gap / 2, 0, f"{p_gap:.1f}%", ha="center", va="center",
             color="white", fontsize=8)
    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_xlabel(f"% of total events (n={tot_ev:,.0f})")
    ax2.set_title("Event-share decomposition", fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    chunks = load_chunks(CSV_PATH)
    by_cat = print_report(chunks)
    plot(chunks, by_cat, FIG_PATH)


if __name__ == "__main__":
    main()
