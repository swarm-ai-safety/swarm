"""Tests for OS-level command isolation (7ge5, slice 2)."""

from pathlib import Path

import pytest

from swarm.bridges.worktree.bridge import WorktreeBridge
from swarm.bridges.worktree.config import WorktreeConfig
from swarm.bridges.worktree.events import WorktreeEventType
from swarm.bridges.worktree.sandbox_launch import detect_backend, wrap_command

CMD = ["python", "-c", "print(1)"]


# --- argv construction (pure, runs on every platform) ---


def test_macos_wrap_structure():
    argv = wrap_command(CMD, sandbox_path="/box", backend="sandbox-exec")
    assert argv[0] == "sandbox-exec"
    assert argv[1] == "-p"
    profile = argv[2]
    assert "(deny network*)" in profile
    assert "(deny file-write*)" in profile
    assert '(allow file-write* (subpath "/box"))' in profile
    assert argv[3:] == CMD  # command appended verbatim


def test_macos_allow_network_omits_network_deny():
    profile = wrap_command(
        CMD, sandbox_path="/box", backend="sandbox-exec", allow_network=True
    )[2]
    assert "(deny network*)" not in profile
    assert "(deny file-write*)" in profile  # writes still confined


def test_macos_profile_escapes_path():
    profile = wrap_command(
        CMD, sandbox_path='/weird "quoted" \\path', backend="sandbox-exec"
    )[2]
    assert '/weird \\"quoted\\" \\\\path' in profile


def test_bwrap_wrap_structure():
    argv = wrap_command(CMD, sandbox_path="/box", backend="bwrap")
    assert argv[0] == "bwrap"
    assert "--unshare-net" in argv
    assert "--ro-bind" in argv
    # rw bind on the sandbox subtree
    i = argv.index("--bind")
    assert argv[i + 1] == "/box" and argv[i + 2] == "/box"
    # command after the -- separator
    sep = argv.index("--")
    assert argv[sep + 1 :] == CMD


def test_bwrap_allow_network_omits_unshare():
    argv = wrap_command(CMD, sandbox_path="/box", backend="bwrap", allow_network=True)
    assert "--unshare-net" not in argv


def test_backend_none_returns_command_unchanged():
    assert wrap_command(CMD, sandbox_path="/box", backend="none") == CMD


def test_empty_command_is_passed_through():
    assert wrap_command([], sandbox_path="/box", backend="sandbox-exec") == []


def test_detect_backend_is_valid():
    assert detect_backend() in {"sandbox-exec", "bwrap", "none"}


# --- executor integration ---


def _bridge(tmp_path: Path, **cfg) -> tuple[WorktreeBridge, str]:
    bridge = WorktreeBridge(
        WorktreeConfig(
            repo_path=str(tmp_path / "repo"),
            sandbox_root=str(tmp_path / "boxes"),
            **cfg,
        )
    )
    # A minimal git repo so a worktree sandbox can be created.
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ["init"],
        ["config", "user.email", "a@b.c"],
        ["config", "user.name", "t"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("x\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True
    )
    sandbox_id = bridge.create_agent_sandbox("codex")
    return bridge, sandbox_id


def test_disabled_isolation_runs_unwrapped(tmp_path):
    bridge, sid = _bridge(tmp_path)  # os_isolation_enabled defaults False
    result, _ = bridge._executor.execute(sid, "codex", ["echo", "hi"])
    assert result.allowed
    assert result.isolation == "none"


def test_fail_closed_denies_when_no_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "swarm.bridges.worktree.executor.detect_backend", lambda: "none"
    )
    bridge, sid = _bridge(
        tmp_path, os_isolation_enabled=True, require_os_isolation=True
    )
    result, events = bridge._executor.execute(sid, "codex", ["echo", "hi"])
    assert not result.allowed
    assert "OS isolation required" in result.deny_reason
    assert any(e.event_type == WorktreeEventType.COMMAND_DENIED for e in events)


def test_fail_open_runs_and_labels_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "swarm.bridges.worktree.executor.detect_backend", lambda: "none"
    )
    bridge, sid = _bridge(
        tmp_path, os_isolation_enabled=True, require_os_isolation=False
    )
    result, _ = bridge._executor.execute(sid, "codex", ["echo", "hi"])
    assert result.allowed
    assert result.isolation == "none"


# --- real-backend enforcement (skipped where no backend exists, e.g. CI) ---

_BACKEND = detect_backend()
_needs_backend = pytest.mark.skipif(
    _BACKEND == "none", reason="no OS sandbox backend on this host"
)


@_needs_backend
def test_network_egress_blocked_under_isolation(tmp_path):
    bridge, sid = _bridge(tmp_path, os_isolation_enabled=True)
    code = (
        "import socket,sys\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1',80),2).close()\n"
        "    print('CONNECTED')\n"
        "except Exception:\n"
        "    sys.exit(7)\n"
    )
    result, _ = bridge._executor.execute(sid, "codex", ["python", "-c", code])
    assert result.isolation == _BACKEND
    assert "CONNECTED" not in result.stdout
    assert result.return_code != 0


@_needs_backend
def test_write_outside_sandbox_blocked(tmp_path):
    bridge, sid = _bridge(tmp_path, os_isolation_enabled=True)
    # Target a genuinely non-temp location: the home dir. (pytest's tmp_path
    # lives under a permitted temp subpath, so it can't serve as "outside".)
    # The write must be blocked while writes inside the sandbox succeed.
    target = Path.home() / ".swarm-sandbox-isolation-test-outside"
    code = (
        "import sys\n"
        f"try:\n"
        f"    open({str(target)!r},'w').write('x')\n"
        f"    print('WROTE')\n"
        f"except Exception:\n"
        f"    sys.exit(8)\n"
    )
    try:
        result, _ = bridge._executor.execute(sid, "codex", ["python", "-c", code])
        assert result.isolation == _BACKEND
        assert "WROTE" not in result.stdout
        assert not target.exists()
    finally:
        target.unlink(missing_ok=True)
