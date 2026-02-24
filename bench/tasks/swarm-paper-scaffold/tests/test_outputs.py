"""Verification tests for swarm-paper-scaffold task."""

import os
import re

OUTPUT_DIR = "/root/output"


def _read_paper():
    path = os.path.join(OUTPUT_DIR, "paper.md")
    with open(path) as f:
        return f.read()


def test_paper_exists():
    assert os.path.isfile(os.path.join(OUTPUT_DIR, "paper.md"))


def test_has_abstract():
    paper = _read_paper()
    assert "## Abstract" in paper or "# Abstract" in paper, "Missing Abstract section"


def test_has_experimental_setup():
    paper = _read_paper()
    assert "Experimental Setup" in paper or "Methods" in paper, \
        "Missing Experimental Setup/Methods section"


def test_has_results():
    paper = _read_paper()
    assert "## Results" in paper or "# Results" in paper, "Missing Results section"


def test_has_conclusion():
    paper = _read_paper()
    assert "## Conclusion" in paper or "# Conclusion" in paper, "Missing Conclusion section"
    # Conclusion must be non-empty
    conclusion_idx = paper.lower().rfind("conclusion")
    after_conclusion = paper[conclusion_idx:]
    # Strip the heading itself and check there\'s content
    lines = after_conclusion.split("\n")
    content_lines = [line for line in lines[1:] if line.strip() and not line.startswith("#")]
    assert len(content_lines) > 0, "Conclusion section is empty"


def test_scenarios_table_has_5_rows():
    paper = _read_paper()
    # Find the scenarios table (after "Scenarios" heading)
    scenarios_section = paper.split("Scenarios")[1] if "Scenarios" in paper else paper
    # Count table rows (lines starting with |, excluding header separator)
    table_rows = [line for line in scenarios_section.split("\n")
                  if line.strip().startswith("|") and "---" not in line and "Scenario" not in line]
    # Take only up to the next section or table
    data_rows = []
    for row in table_rows:
        if row.count("|") >= 3:  # Valid table row
            data_rows.append(row)
        else:
            break
    assert len(data_rows) >= 5, f"Scenarios table should have 5 rows, found {len(data_rows)}"


def test_results_table_has_numeric_values():
    paper = _read_paper()
    results_section = paper.split("Results")[1] if "Results" in paper else ""
    # Look for numeric values in table rows
    numbers = re.findall(r"\d+\.\d+", results_section)
    assert len(numbers) >= 5, \
        f"Results table should have numeric values, found only {len(numbers)} numbers"
