#!/usr/bin/env python3
"""Parse CHANGELOG.md and emit JSON array of releases for GitHub Actions."""

import json
import re
import sys
from pathlib import Path


def parse_changelog(path: str) -> list[dict]:
    """Parse a Keep-a-Changelog file into a list of release dicts."""
    text = Path(path).read_text()
    # Match version headers like ## [1.6.0] - 2026-02-15
    header_re = re.compile(
        r"^## \[(?P<version>\d+\.\d+\.\d+)\]\s*-\s*(?P<date>\d{4}-\d{2}-\d{2})",
        re.MULTILINE,
    )

    matches = list(header_re.finditer(text))
    releases = []

    for i, m in enumerate(matches):
        version = m.group("version")
        date = m.group("date")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Build the title
        is_latest = i == 0
        prerelease = int(version.split(".")[0]) < 1

        releases.append(
            {
                "tag": f"v{version}",
                "name": f"v{version}",
                "date": date,
                "body": body,
                "latest": is_latest,
                "prerelease": prerelease,
            }
        )

    return releases


def main() -> None:
    changelog = sys.argv[1] if len(sys.argv) > 1 else "CHANGELOG.md"
    releases = parse_changelog(changelog)
    json.dump(releases, sys.stdout, indent=2)
    print()  # trailing newline


if __name__ == "__main__":
    main()
