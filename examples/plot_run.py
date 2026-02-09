#!/usr/bin/env python
"""
Generate standard plots from a SWARM run folder.

Usage:
    python examples/plot_run.py runs/<run_id>
    python examples/plot_run.py runs/<run_id> --metric toxicity_rate

Expected inputs (either):
    - <run_dir>/history.json
    - <run_dir>/csv/*_epochs.csv (and optionally *_agents.csv)

Outputs:
    - <run_dir>/plots/*.png (if matplotlib available)
    - <run_dir>/plots/README.txt (fallback if deps missing)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple


def _load_history(run_dir: Path):
    from swarm.analysis.export import load_from_csv, load_from_json

    history_json = run_dir / "history.json"
    if history_json.exists():
        return load_from_json(history_json)

    csv_dir = run_dir / "csv"
    candidates = list(csv_dir.glob("*_epochs.csv")) if csv_dir.is_dir() else []
    if not candidates:
        candidates = list(run_dir.glob("*_epochs.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No history found. Expected {history_json} or *_epochs.csv under {csv_dir}/"
        )

    epochs_path = sorted(candidates)[0]
    agents_path = None
    agent_candidates = list(epochs_path.parent.glob("*_agents.csv"))
    if agent_candidates:
        agents_path = sorted(agent_candidates)[0]
    return load_from_csv(epochs_path=epochs_path, agents_path=agents_path)


def _maybe_import_matplotlib(run_dir: Path):
    mpl_config_dir = run_dir / ".mplconfig"
    xdg_cache_dir = run_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]

        return plt
    except Exception:
        return None


def _write_fallback_readme(plots_dir: Path, message: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / "README.txt"
    path.write_text(
        "\n".join(
            [
                "Plots not generated (missing optional dependencies).",
                "",
                message.strip(),
                "",
                "Install:",
                "  python -m pip install -e '.[analysis]'",
                "",
                "Then re-run:",
                f"  python examples/plot_run.py {plots_dir.parent}",
                "",
            ]
        )
        + "\n"
    )


def _iter_series(history, metric: str) -> Tuple[Iterable[int], Iterable[float]]:
    epochs = [s.epoch for s in history.epoch_snapshots]
    values = []
    for s in history.epoch_snapshots:
        v = getattr(s, metric, None)
        values.append(float(v) if v is not None else 0.0)
    return epochs, values


def _acceptance_rate(history) -> Tuple[Iterable[int], Iterable[float]]:
    epochs = [s.epoch for s in history.epoch_snapshots]
    values = []
    for s in history.epoch_snapshots:
        denom = float(s.total_interactions) if s.total_interactions else 0.0
        values.append(float(s.accepted_interactions) / denom if denom > 0 else 0.0)
    return epochs, values


def _plot_series(plt, plots_dir: Path, history, metric: str) -> Path:
    x, y = _iter_series(history, metric)

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(list(x), list(y), marker="o", linewidth=2)
    ax.set_title(metric)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)

    out = plots_dir / f"{metric}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _plot_acceptance_rate(plt, plots_dir: Path, history) -> Path:
    x, y = _acceptance_rate(history)

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(list(x), list(y), marker="o", linewidth=2)
    ax.set_title("acceptance_rate")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accepted / total")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    out = plots_dir / "acceptance_rate.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate plots for a SWARM run folder"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run folder (contains history.json or csv/*_epochs.csv)",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Plot only a specific metric (default: standard set)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    plots_dir = run_dir / "plots"

    try:
        history = _load_history(run_dir)
    except Exception as err:
        _write_fallback_readme(plots_dir, f"Could not load history: {err}")
        print(f"Wrote {plots_dir / 'README.txt'}")
        return 1

    plt = _maybe_import_matplotlib(run_dir)
    if plt is None:
        _write_fallback_readme(
            plots_dir,
            "matplotlib not available in this environment.",
        )
        print(f"Wrote {plots_dir / 'README.txt'}")
        return 0

    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = (
        [args.metric]
        if args.metric
        else ["toxicity_rate", "quality_gap", "total_welfare"]
    )

    written = []
    for metric in metrics:
        try:
            written.append(_plot_series(plt, plots_dir, history, metric))
        except Exception as err:
            _write_fallback_readme(plots_dir, f"Failed plotting {metric}: {err}")
            print(f"Wrote {plots_dir / 'README.txt'}")
            return 1

    try:
        written.append(_plot_acceptance_rate(plt, plots_dir, history))
    except Exception:
        pass

    (plots_dir / "README.txt").write_text(
        "\n".join(
            [
                "Generated plots:",
                *(f"- {p.name}" for p in written),
                "",
            ]
        )
    )

    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
