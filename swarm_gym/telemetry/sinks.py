"""Trace output sinks: file, console."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


class FileSink:
    """Write trace data to a JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)


class ConsoleSink:
    """Print trace data to stdout."""

    def write(self, data: Dict[str, Any]) -> None:
        print(json.dumps(data, indent=2), file=sys.stdout)
