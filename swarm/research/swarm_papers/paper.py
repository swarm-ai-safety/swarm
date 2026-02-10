"""Paper synthesis for SWARM Track A runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from swarm.research.platforms import Paper
from swarm.research.swarm_papers.memory import MemoryArtifact


@dataclass
class PaperContext:
    title: str
    abstract: str
    run_id: str
    task_count: int
    conditions: list[dict]
    related_work: list[str]
    memory_items: list[MemoryArtifact]
    critique_summary: "CritiqueSummary"
    family_metrics: dict[str, dict[str, float]]
    figures: list["PaperFigure"] = field(default_factory=list)
    images: dict[str, str] = field(default_factory=dict)


@dataclass
class CritiqueSummary:
    total_flags: int
    flag_rate: float
    top_reasons: list[str]


@dataclass
class PaperFigure:
    filename: str
    caption: str


class PaperBuilder:
    """Builds a LaTeX paper from Track A run summaries."""

    def build(self, context: PaperContext) -> Paper:
        latex = self._render_latex(context)
        return Paper(
            title=context.title,
            abstract=context.abstract,
            source=latex,
            images=context.images,
            authors=["SWARM Research Agents"],
            categories=["swarm", "track-a", "agentrxiv"],
        )

    def _render_latex(self, context: PaperContext) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        table_rows = self._render_table_rows(context.conditions)
        related_work = self._render_related(context.related_work)
        memory_section = self._render_memory(context.memory_items)
        critique_section = self._render_critique(context.critique_summary)
        family_table = self._render_family_table(context.family_metrics, context.conditions)
        figures_section = self._render_figures(context.figures)

        return """\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{graphicx}
\\usepackage{geometry}
\\geometry{margin=1in}
\\title{%s}
\\date{%s}
\\begin{document}
\\maketitle
\\begin{abstract}
%s
\\end{abstract}

\\section{Introduction}
We evaluate SWARM-style coordination mechanisms on Track A (verifiable reasoning),
using controlled arithmetic and word-problem tasks with deterministic checks.
Each condition corresponds to a coordination policy (divergence, critique,
reconciliation, memory). This paper summarizes one full run (ID: %s).

\\section{Methods}
Tasks: %d total, generated with fixed random seed and difficulty calibration.

\\subsection{Conditions}
\\begin{tabular}{llll}
\\toprule
Condition & Accuracy & Avg. Tokens & Notes \\
\\midrule
%s
\\bottomrule
\\end{tabular}

\\section{Results}
Across conditions, we report accuracy (correct/total), disagreement rate when
multiple solvers are active, and reconciliation frequency when enabled.

%s

\\section{Related Work (AgentRxiv)}
\\begin{itemize}
%s
\\end{itemize}

%s

\\section{Limitations}
We treat confidence as a reported scalar and rely on simple divergence heuristics.
Future runs should incorporate stronger validators and richer task suites.

\\end{document}
""" % (
            _escape_latex(context.title),
            _escape_latex(now),
            _escape_latex(context.abstract),
            _escape_latex(context.run_id),
            context.task_count,
            table_rows,
            critique_section
            + ("\n\n" + family_table if family_table else "")
            + ("\n\n" + figures_section if figures_section else ""),
            related_work,
            memory_section,
        )

    def _render_table_rows(self, conditions: Iterable[dict]) -> str:
        rows = []
        for cond in conditions:
            name = _escape_latex(cond.get("name", ""))
            acc = cond.get("accuracy", 0.0)
            tokens = cond.get("avg_tokens", 0.0)
            note = _escape_latex(cond.get("note", ""))
            rows.append(f"{name} & {acc:.3f} & {tokens:.1f} & {note} \\")
        return "\n".join(rows) if rows else "--"

    def _render_memory(self, items: list[MemoryArtifact]) -> str:
        if not items:
            return "\\section{Memory Artifacts}\nNo memory artifacts were accepted in this run."
        lines = ["\\section{Memory Artifacts}", "\\begin{itemize}"]
        for item in items:
            text = f"{item.title}: {item.summary}"
            lines.append(f"\\item {_escape_latex(text)}")
        lines.append("\\end{itemize}")
        return "\n".join(lines)

    def _render_related(self, items: Iterable[str]) -> str:
        items = list(items)
        if not items:
            return "\\item None."
        return "\n".join(f"\\item {_escape_latex(item)}" for item in items)

    def _render_figures(self, figures: list[PaperFigure]) -> str:
        if not figures:
            return ""
        lines = ["\\section{Figures}"]
        for fig in figures:
            lines.append("\\begin{figure}[h]")
            lines.append("\\centering")
            lines.append(f"\\includegraphics[width=0.85\\linewidth]{{{_escape_latex(fig.filename)}}}")
            lines.append(f"\\caption{{{_escape_latex(fig.caption)}}}")
            lines.append("\\end{figure}")
        return "\n".join(lines)

    def _render_critique(self, summary: CritiqueSummary) -> str:
        if summary.total_flags == 0:
            return "\\paragraph{Critique Summary} No critic flags were recorded."
        lines = [
            "\\paragraph{Critique Summary}",
            f"Critic flags: {_escape_latex(str(summary.total_flags))} "
            f"({summary.flag_rate:.1%} of episodes).",
        ]
        if summary.top_reasons:
            lines.append("\\begin{itemize}")
            for reason in summary.top_reasons:
                lines.append(f"\\item {_escape_latex(reason)}")
            lines.append("\\end{itemize}")
        return "\n".join(lines)

    def _render_family_table(
        self, family_metrics: dict[str, dict[str, float]], conditions: list[dict]
    ) -> str:
        if not family_metrics:
            return ""

        condition_names = [cond.get("name", "") for cond in conditions]
        families = _order_families(family_metrics)
        cols = "l" + "r" * len(condition_names)
        lines = ["\\subsection{Per-Family Accuracy}", "\\begin{tabular}{" + cols + "}"]
        header = "Family & " + " & ".join(_escape_latex(name) for name in condition_names) + " \\\\"
        lines.append("\\toprule")
        lines.append(header)
        lines.append("\\midrule")
        for family in families:
            row = [_escape_latex(_family_label(family))]
            for name in condition_names:
                value = family_metrics.get(name, {}).get(family, 0.0)
                row.append(f"{value:.2f}")
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        return "\n".join(lines)


def _escape_latex(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(c, c) for c in text)


def _order_families(family_metrics: dict[str, dict[str, float]]) -> list[str]:
    preferred = ["arithmetic", "algebra", "logic_grid", "symbolic", "word"]
    families = set()
    for metrics in family_metrics.values():
        families.update(metrics.keys())
    ordered = [fam for fam in preferred if fam in families]
    remainder = sorted(fam for fam in families if fam not in ordered)
    return ordered + remainder


def _family_label(family: str) -> str:
    return "logic" if family == "logic_grid" else family
