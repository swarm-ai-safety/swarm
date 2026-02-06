"""Shared pytest configuration and environment checks."""

import sys


def pytest_configure(config):
    """Verify test dependencies are importable, with a clear error message."""
    missing = []
    for mod in ("numpy", "pydantic"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    if missing:
        print(
            f"\nERROR: Missing required packages: {', '.join(missing)}\n"
            f"The pytest binary may be running in a different environment than\n"
            f"the project's installed dependencies. Try:\n"
            f"  python -m pytest tests/ -v\n"
            f"Or reinstall with:  pip install -e '.[dev]'\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
