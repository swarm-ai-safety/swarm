"""Tests for the AgentGit MVP provenance layer."""

import json
import subprocess
from pathlib import Path

import pytest

from swarm.agentgit.__main__ import _parse_checks, main
from swarm.agentgit.bundle import build_bundle, verify_bundle
from swarm.agentgit.policy import AgentGitPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "agentgit@example.com")
    _git(repo, "config", "user.name", "AgentGit Test")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_build_bundle_passes_for_allowed_diff(tmp_path):
    repo = _init_repo(tmp_path)
    src = repo / "swarm" / "agentgit"
    src.mkdir(parents=True)
    (src / "mvp.py").write_text("print('ok')\n")

    policy = AgentGitPolicy(
        allowed_paths=["swarm/agentgit/**"],
        denied_paths=[".env*"],
        max_changed_files=1,
        max_added_lines=5,
        required_checks=["pytest"],
    )

    bundle = build_bundle(
        repo=repo,
        task_id="issue-1",
        agent_id="codex",
        policy=policy,
        check_results={"pytest": True},
    )

    assert bundle["schema_version"] == "agentgit.provenance.v0"
    assert bundle["policy"]["passed"] is True
    assert bundle["git"]["totals"]["changed_files"] == 1
    assert bundle["receipt"]["status"] == "sealed"
    assert bundle["receipt"]["payload_hash"]


def test_build_bundle_fails_for_denied_path(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / ".env").write_text("SECRET=bad\n")

    policy = AgentGitPolicy(
        allowed_paths=["swarm/**"],
        denied_paths=[".env*"],
        required_checks=["pytest"],
    )

    bundle = build_bundle(
        repo=repo,
        task_id="issue-2",
        agent_id="codex",
        policy=policy,
        check_results={"pytest": True},
    )

    decisions = {
        decision["policy_id"]: decision for decision in bundle["policy"]["decisions"]
    }
    assert bundle["policy"]["passed"] is False
    assert decisions["allowed-paths"]["details"]["violations"] == [".env"]
    assert decisions["denied-paths"]["details"]["violations"] == [".env"]
    assert bundle["receipt"]["policy_results"][0]["passed"] is False


def test_cli_attest_writes_bundle_and_returns_failure_on_policy_fail(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs" / "note.md").write_text("outside scope\n")
    policy_path = tmp_path / "agentgit.policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "allowed_paths:",
                "  - swarm/**",
                "max_changed_files: 1",
                "required_checks:",
                "  - pytest",
            ]
        )
        + "\n"
    )
    output_path = tmp_path / "bundle.json"

    result = subprocess.run(
        [
            "python",
            "-m",
            "swarm.agentgit",
            "attest",
            "--repo",
            str(repo),
            "--task",
            "issue-3",
            "--agent",
            "codex",
            "--policy",
            str(policy_path),
            "--output",
            str(output_path),
            "--check",
            "pytest=pass",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 1
    assert "FAIL agentgit attestation" in result.stdout
    bundle = json.loads(output_path.read_text())
    assert bundle["policy"]["passed"] is False


def test_verify_bundle_detects_tampering(tmp_path):
    repo = _init_repo(tmp_path)
    src = repo / "swarm"
    src.mkdir()
    (src / "ok.py").write_text("print('ok')\n")
    policy = AgentGitPolicy(allowed_paths=["swarm/**"])
    bundle = build_bundle(
        repo=repo,
        task_id="issue-4",
        agent_id="codex",
        policy=policy,
    )

    ok, errors = verify_bundle(bundle)
    assert ok
    assert errors == []

    bundle["git"]["changed_files"][0]["path"] = "tampered.py"
    ok, errors = verify_bundle(bundle)
    assert not ok
    assert "receipt payload_hash does not match bundle payload" in errors


def test_worktree_attest_then_agentgit_verify_loop(tmp_path):
    repo = _init_repo(tmp_path)
    sandbox_root = tmp_path / "sandboxes"
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "allowed_paths:",
                "  - feature/**",
                "max_changed_files: 1",
                "max_added_lines: 5",
                "required_checks:",
                "  - pytest",
            ]
        )
        + "\n"
    )
    output_path = tmp_path / "loop-bundle.json"

    create = subprocess.run(
        [
            "python",
            "-m",
            "swarm.bridges.worktree",
            "create",
            "codex",
            "--repo",
            str(repo),
            "--root",
            str(sandbox_root),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert create.returncode == 0, create.stderr

    sandbox_path = sandbox_root / "sandbox-codex"
    (sandbox_path / "feature").mkdir()
    (sandbox_path / "feature" / "change.py").write_text("print('loop')\n")

    attest = subprocess.run(
        [
            "python",
            "-m",
            "swarm.bridges.worktree",
            "attest",
            "codex",
            "--repo",
            str(repo),
            "--root",
            str(sandbox_root),
            "--task",
            "issue-loop",
            "--policy",
            str(policy_path),
            "--output",
            str(output_path),
            "--check",
            "pytest=pass",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert attest.returncode == 0, attest.stderr
    assert output_path.exists()

    verify = subprocess.run(
        [
            "python",
            "-m",
            "swarm.agentgit",
            "verify",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert verify.returncode == 0, verify.stderr
    assert "PASS agentgit verify" in verify.stdout


# --- CLI unit tests (in-process via main(); exercise __main__.py branches) ---


def _write_policy(tmp_path: Path, *lines: str) -> Path:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("\n".join(lines) + "\n")
    return policy_path


def test_parse_checks_normalises_aliases():
    checks = _parse_checks(["pytest=pass", "lint=FAILED", "types=true", "fmt=False"])
    assert checks == {"pytest": True, "lint": False, "types": True, "fmt": False}


def test_parse_checks_rejects_missing_equals():
    with pytest.raises(SystemExit):
        _parse_checks(["pytest"])


def test_parse_checks_rejects_unknown_result():
    with pytest.raises(SystemExit):
        _parse_checks(["pytest=maybe"])


def test_cli_attest_warn_only_returns_success_on_policy_fail(tmp_path, capsys):
    repo = _init_repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs" / "note.md").write_text("outside scope\n")
    policy_path = _write_policy(tmp_path, "allowed_paths:", "  - swarm/**")
    output_path = tmp_path / "bundle.json"

    rc = main(
        [
            "attest",
            "--repo", str(repo),
            "--task", "issue-warn",
            "--agent", "codex",
            "--policy", str(policy_path),
            "--output", str(output_path),
            "--warn-only",
        ]
    )

    assert rc == 0
    assert "FAIL agentgit attestation" in capsys.readouterr().out
    bundle = json.loads(output_path.read_text())
    assert bundle["policy"]["passed"] is False


def test_cli_attest_uses_env_signing_key_and_verifies(tmp_path, monkeypatch, capsys):
    repo = _init_repo(tmp_path)
    src = repo / "swarm"
    src.mkdir()
    (src / "ok.py").write_text("print('ok')\n")
    policy_path = _write_policy(tmp_path, "allowed_paths:", "  - swarm/**")
    output_path = tmp_path / "bundle.json"
    monkeypatch.setenv("AGENTGIT_SIGNING_KEY", "ab" * 32)

    rc = main(
        [
            "attest",
            "--repo", str(repo),
            "--task", "issue-env",
            "--agent", "codex",
            "--policy", str(policy_path),
            "--output", str(output_path),
        ]
    )
    assert rc == 0
    assert "PASS agentgit attestation" in capsys.readouterr().out

    # Same env key must verify; verify reads AGENTGIT_SIGNING_KEY too.
    assert main(["verify", str(output_path)]) == 0
    assert "PASS agentgit verify" in capsys.readouterr().out


def test_cli_verify_allow_policy_fail(tmp_path, capsys):
    repo = _init_repo(tmp_path)
    (repo / "secret.env").write_text("SECRET=bad\n")
    policy_path = _write_policy(
        tmp_path, "allowed_paths:", "  - swarm/**", "denied_paths:", "  - '*.env'"
    )
    output_path = tmp_path / "bundle.json"

    assert main(
        [
            "attest",
            "--repo", str(repo),
            "--task", "issue-fail",
            "--agent", "codex",
            "--policy", str(policy_path),
            "--output", str(output_path),
            "--warn-only",
        ]
    ) == 0
    capsys.readouterr()

    # Default verify rejects a failed-policy bundle...
    assert main(["verify", str(output_path)]) == 1
    assert "FAIL agentgit verify" in capsys.readouterr().out

    # ...but --allow-policy-fail still confirms hash/signature integrity.
    assert main(["verify", str(output_path), "--allow-policy-fail"]) == 0
    assert "PASS agentgit verify" in capsys.readouterr().out


def test_cli_no_subcommand_prints_help(capsys):
    assert main([]) == 0
    assert "attest" in capsys.readouterr().out
