"""Tests for the CLI entry point (src/__main__.py)."""

import sys
from pathlib import Path

import pytest

from swarm.__main__ import main

BASELINE_SCENARIO = "scenarios/baseline.yaml"
FAST_FLAGS = ["--seed", "42", "--epochs", "2", "--steps", "2"]


class TestMainNoArgs:
    """Test main() with no arguments."""

    def test_no_args_prints_help_and_returns_zero(self, monkeypatch, capsys):
        """main() with no args should print help and return 0."""
        monkeypatch.setattr(sys, "argv", ["python -m src"])
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "usage:" in captured.err.lower()


class TestListSubcommand:
    """Tests for the 'list' subcommand."""

    def test_list_shows_scenario_files(self, monkeypatch, capsys):
        """list should display YAML files from the given directory."""
        monkeypatch.setattr(sys, "argv", ["python -m src", "list", "--dir", "scenarios"])
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "baseline.yaml" in captured.out

    def test_list_nonexistent_directory_returns_one(self, monkeypatch, capsys):
        """list with a nonexistent directory should return 1."""
        monkeypatch.setattr(
            sys, "argv", ["python -m src", "list", "--dir", "/no/such/directory"]
        )
        rc = main()
        assert rc == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestRunSubcommand:
    """Tests for the 'run' subcommand."""

    def test_run_nonexistent_scenario_returns_one(self, monkeypatch, capsys):
        """run with a nonexistent scenario file should return 1."""
        monkeypatch.setattr(
            sys, "argv", ["python -m src", "run", "no_such_file.yaml"]
        )
        rc = main()
        assert rc == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_run_baseline_returns_zero(self, monkeypatch, capsys):
        """run with baseline.yaml should complete and return 0."""
        if not Path(BASELINE_SCENARIO).exists():
            pytest.skip("baseline.yaml not found")
        monkeypatch.setattr(
            sys, "argv", ["python -m src", "run", BASELINE_SCENARIO] + FAST_FLAGS
        )
        rc = main()
        assert rc == 0

    def test_run_seed_override(self, monkeypatch, capsys):
        """run with --seed should produce deterministic output."""
        if not Path(BASELINE_SCENARIO).exists():
            pytest.skip("baseline.yaml not found")
        monkeypatch.setattr(
            sys,
            "argv",
            ["python -m src", "run", BASELINE_SCENARIO, "--seed", "99", "--epochs", "2", "--steps", "2"],
        )
        rc = main()
        assert rc == 0
        out_a = capsys.readouterr().out

        monkeypatch.setattr(
            sys,
            "argv",
            ["python -m src", "run", BASELINE_SCENARIO, "--seed", "99", "--epochs", "2", "--steps", "2"],
        )
        rc = main()
        assert rc == 0
        out_b = capsys.readouterr().out

        assert out_a == out_b

    def test_run_epochs_and_steps_overrides(self, monkeypatch, capsys):
        """run with --epochs and --steps should override the scenario defaults."""
        if not Path(BASELINE_SCENARIO).exists():
            pytest.skip("baseline.yaml not found")
        monkeypatch.setattr(
            sys,
            "argv",
            ["python -m src", "run", BASELINE_SCENARIO, "--seed", "42", "--epochs", "2", "--steps", "2"],
        )
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        # The header should reflect the overridden values
        assert "Epochs:      2" in captured.out
        assert "Steps/epoch: 2" in captured.out

    def test_run_quiet_suppresses_output(self, monkeypatch, capsys):
        """run with -q should suppress progress output."""
        if not Path(BASELINE_SCENARIO).exists():
            pytest.skip("baseline.yaml not found")
        monkeypatch.setattr(
            sys,
            "argv",
            ["python -m src", "run", BASELINE_SCENARIO, "-q"] + FAST_FLAGS,
        )
        rc = main()
        assert rc == 0
        captured = capsys.readouterr()
        # Quiet mode should suppress the banner and table
        assert "Distributional AGI Safety Sandbox" not in captured.out
        assert "Running simulation..." not in captured.out

    def test_run_export_json(self, monkeypatch, capsys, tmp_path):
        """run with --export-json should create a JSON file."""
        if not Path(BASELINE_SCENARIO).exists():
            pytest.skip("baseline.yaml not found")
        json_path = tmp_path / "results.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["python -m src", "run", BASELINE_SCENARIO, "-q", "--export-json", str(json_path)]
            + FAST_FLAGS,
        )
        rc = main()
        assert rc == 0
        assert json_path.exists()
        content = json_path.read_text()
        assert len(content) > 0
        # Should be valid JSON
        import json
        data = json.loads(content)
        assert "simulation_id" in data

    def test_run_export_csv(self, monkeypatch, capsys, tmp_path):
        """run with --export-csv should create CSV files in the given directory."""
        pytest.importorskip("pandas", reason="pandas required for CSV export")
        if not Path(BASELINE_SCENARIO).exists():
            pytest.skip("baseline.yaml not found")
        csv_dir = tmp_path / "csv_output"
        monkeypatch.setattr(
            sys,
            "argv",
            ["python -m src", "run", BASELINE_SCENARIO, "-q", "--export-csv", str(csv_dir)]
            + FAST_FLAGS,
        )
        rc = main()
        assert rc == 0
        assert csv_dir.exists()
        csv_files = list(csv_dir.glob("*.csv"))
        assert len(csv_files) >= 1

    def test_run_prompt_audit_flag_is_accepted(self, monkeypatch, tmp_path):
        """run with --prompt-audit should be accepted by argparse."""
        if not Path(BASELINE_SCENARIO).exists():
            pytest.skip("baseline.yaml not found")
        audit_path = tmp_path / "audit.jsonl"
        monkeypatch.setattr(
            sys,
            "argv",
            ["python -m src", "run", BASELINE_SCENARIO, "-q", "--prompt-audit", str(audit_path)]
            + FAST_FLAGS,
        )
        rc = main()
        assert rc == 0
