"""OS-level sandbox launchers for worktree command execution.

The worktree executor runs allowlisted commands with
``subprocess.run(command, cwd=sandbox_path)``. ``cwd`` is only a working
directory, not a jail: an allowlisted ``python`` can still write anywhere on
disk and open network sockets. This module wraps a command's argv in a real OS
confinement so a granted command cannot **write outside its sandbox** or
**reach the network**.

Two backends are supported, selected by host:

- ``sandbox-exec`` (macOS): an SBPL profile that denies file writes outside the
  sandbox (plus temp/dev) and denies network access.
- ``bwrap`` (Linux, bubblewrap): a read-only root bind with a read-write bind on
  only the sandbox subtree, and a private (empty) network namespace.

This is **write + network** confinement, not a full jail: reads are not
restricted (interpreters need their standard library). The functions here only
build argv — they never execute — so they are pure and easy to test. No
``shell=True`` is ever introduced, preserving the executor's NO-shell invariant.
"""

from __future__ import annotations

import platform
import shutil
from typing import List

Backend = str  # "sandbox-exec" | "bwrap" | "none"

# Paths an otherwise-confined process still legitimately needs to write to.
_MACOS_WRITABLE_SUBPATHS = ("/private/tmp", "/private/var/folders", "/tmp")
_MACOS_WRITABLE_LITERALS = ("/dev/null", "/dev/stdout", "/dev/stderr")


def detect_backend() -> Backend:
    """Return the OS sandbox backend available on this host, or ``"none"``."""

    system = platform.system()
    if system == "Darwin" and shutil.which("sandbox-exec"):
        return "sandbox-exec"
    if system == "Linux" and shutil.which("bwrap"):
        return "bwrap"
    return "none"


def _sbpl_string(path: str) -> str:
    r"""Quote a path as an SBPL string literal (escape ``\`` and ``"``)."""

    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _build_macos_profile(sandbox_path: str, *, allow_network: bool) -> str:
    """Build an SBPL profile: permissive reads, writes confined, network gated."""

    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
        f"(allow file-write* (subpath {_sbpl_string(sandbox_path)}))",
    ]
    for sub in _MACOS_WRITABLE_SUBPATHS:
        lines.append(f"(allow file-write* (subpath {_sbpl_string(sub)}))")
    for lit in _MACOS_WRITABLE_LITERALS:
        lines.append(f"(allow file-write* (literal {_sbpl_string(lit)}))")
    if not allow_network:
        lines.append("(deny network*)")
    return "".join(lines)


def wrap_command(
    command: List[str],
    *,
    sandbox_path: str,
    backend: Backend,
    allow_network: bool = False,
) -> List[str]:
    """Wrap ``command`` argv for ``backend``, confining writes + network.

    ``backend == "none"`` returns the command unchanged (caller is responsible
    for recording that no isolation was applied).
    """

    if not command:
        return list(command)

    if backend == "sandbox-exec":
        profile = _build_macos_profile(sandbox_path, allow_network=allow_network)
        return ["sandbox-exec", "-p", profile, *command]

    if backend == "bwrap":
        argv = [
            "bwrap",
            "--ro-bind", "/", "/",
            "--dev", "/dev",
            "--proc", "/proc",
            "--bind", sandbox_path, sandbox_path,
            "--chdir", sandbox_path,
        ]
        if not allow_network:
            argv.append("--unshare-net")
        argv.append("--")
        argv.extend(command)
        return argv

    return list(command)
