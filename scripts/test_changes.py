#!/usr/bin/env python3
"""Run pytest with pytest-testmon and optional cached metadata.

Default behavior:
- If a cached .testmondata is found, run pytest with test selection.
- If no cache is found, run a full test pass while recording testmon data.

Usage:
  python scripts/test_changes.py -- pytest -q -m "not slow"
  python scripts/test_changes.py --full -- pytest tests/ -v
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_cached_testmon(cache_dir: Path) -> Path | None:
    if not cache_dir.exists():
        return None

    direct = cache_dir / ".testmondata"
    if direct.exists():
        return direct

    try:
        revs = subprocess.run(
            ["git", "rev-list", "--first-parent", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except subprocess.CalledProcessError:
        revs = []

    for sha in revs:
        for candidate in (
            cache_dir / sha / ".testmondata",
            cache_dir / f"testmondata-{sha}" / ".testmondata",
        ):
            if candidate.exists():
                return candidate

    latest = cache_dir / "testmondata-latest" / ".testmondata"
    if latest.exists():
        return latest

    for candidate in cache_dir.rglob(".testmondata"):
        return candidate

    return None


def _ensure_local_testmon(cache_dir: Path) -> bool:
    local = Path(".testmondata")
    if local.exists():
        return True

    cached = _find_cached_testmon(cache_dir)
    if cached is None:
        return False

    shutil.copy2(cached, local)
    return True


def _strip_leading_double_dash(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pytest with pytest-testmon.")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("TESTMON_CACHE_DIR", ".testmon_cache"),
        help="Directory containing cached .testmondata artifacts.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full tests while recording testmon metadata.",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    pytest_args = _strip_leading_double_dash(args.pytest_args)
    if not pytest_args:
        pytest_args = ["pytest"]

    cache_dir = Path(args.cache_dir)
    has_cache = _ensure_local_testmon(cache_dir)

    has_testmon_flag = any(
        flag in pytest_args for flag in ("--testmon", "--testmon-noselect")
    )
    if not has_testmon_flag:
        if args.full or not has_cache:
            pytest_args.append("--testmon-noselect")
        else:
            pytest_args.append("--testmon")

    print("[test_changes] cache_dir=", cache_dir, file=sys.stderr)
    print("[test_changes] has_cache=", has_cache, file=sys.stderr)
    print("[test_changes] args=", " ".join(pytest_args), file=sys.stderr)

    result = subprocess.run(pytest_args)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
