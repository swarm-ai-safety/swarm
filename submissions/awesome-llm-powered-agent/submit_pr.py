#!/usr/bin/env python3
"""
Submit SWARM to hyp1231/awesome-llm-powered-agent via GitHub API.

Usage:
    export GITHUB_TOKEN=ghp_...
    python submissions/awesome-llm-powered-agent/submit_pr.py

This script will:
  1. Fork hyp1231/awesome-llm-powered-agent (if not already forked)
  2. Fetch the current README.md
  3. Insert the SWARM entry into the Multi-Agent Simulation Projects section
  4. Commit the change to a new branch
  5. Open a pull request with a descriptive body
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

UPSTREAM_OWNER = "hyp1231"
UPSTREAM_REPO = "awesome-llm-powered-agent"
BRANCH_NAME = "add-swarm-multi-agent-safety"
PR_TITLE = "Add SWARM to Multi-Agent Simulation Projects"

SWARM_ENTRY = (
    '* ![SWARM Stars](https://img.shields.io/github/stars/swarm-ai-safety/swarm) '
    '[SWARM](https://github.com/swarm-ai-safety/swarm) '
    '- A research framework for studying emergent risks and governance in '
    'multi-agent LLM systems, featuring probabilistic safety metrics, '
    'configurable governance levers, and first-class LLM agent support.'
)

PR_BODY_PATH = Path(__file__).with_name("PR_BODY.md")


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

def gh_api(
    method: str,
    path: str,
    *,
    token: str,
    body: dict | None = None,
    accept: str = "application/vnd.github+json",
) -> dict:
    """Minimal GitHub REST API wrapper."""
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", accept)
    if data:
        req.add_header("Content-Type", "application/json")

    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if not data:
                    return {}
                return json.loads(data.decode())
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode() if exc.fp else ""
            if exc.code == 422 and "already exists" in payload.lower():
                # Branch / fork already exists — treat as success
                return json.loads(payload)
            if exc.code in (502, 503) and attempt < 3:
                wait = 2 ** (attempt + 1)
                print(f"  Retrying in {wait}s (HTTP {exc.code})...")
                time.sleep(wait)
                continue
            print(f"HTTP {exc.code}: {payload}", file=sys.stderr)
            raise
    raise RuntimeError("Max retries exceeded")


def get_or_create_fork(token: str) -> str:
    """Fork the upstream repo (no-op if already forked). Returns fork owner."""
    user = gh_api("GET", "/user", token=token)
    fork_owner: str = user["login"]
    print(f"Authenticated as {fork_owner}")

    # Check if fork already exists
    try:
        gh_api("GET", f"/repos/{fork_owner}/{UPSTREAM_REPO}", token=token)
        print(f"Fork already exists: {fork_owner}/{UPSTREAM_REPO}")
    except urllib.error.HTTPError:
        print(f"Forking {UPSTREAM_OWNER}/{UPSTREAM_REPO}...")
        gh_api(
            "POST",
            f"/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/forks",
            token=token,
        )
        # Wait for fork to be ready
        for _ in range(12):
            time.sleep(5)
            try:
                gh_api("GET", f"/repos/{fork_owner}/{UPSTREAM_REPO}", token=token)
                break
            except urllib.error.HTTPError:
                pass
        else:
            raise RuntimeError("Fork not ready after 60 s")
        print("Fork created.")

    return fork_owner


def get_readme(owner: str, token: str) -> tuple[str, str]:
    """Return (decoded content, sha) of README.md on the default branch."""
    data = gh_api(
        "GET",
        f"/repos/{owner}/{UPSTREAM_REPO}/contents/README.md",
        token=token,
    )
    content = base64.b64decode(data["content"]).decode()
    return content, data["sha"]


def patch_readme(content: str) -> str:
    """Insert the SWARM entry into the Multi-Agent Simulation Projects section."""
    # Verify idempotency
    if "swarm-ai-safety/swarm" in content:
        print("SWARM entry already present — skipping patch.")
        return content

    header_match = re.search(
        r"^###\s+Multi-Agent Simulation Projects\s*$",
        content,
        flags=re.MULTILINE,
    )
    if not header_match:
        raise ValueError(
            "Could not locate 'Multi-Agent Simulation Projects' section — "
            "upstream format may have changed."
        )

    header_line_end = content.find("\n", header_match.end())
    if header_line_end == -1:
        header_line_end = len(content)
        section_start = header_line_end
    else:
        section_start = header_line_end + 1

    next_section = re.search(r"^##\s+", content[section_start:], flags=re.MULTILINE)
    section_end = section_start + next_section.start() if next_section else len(content)

    section = content[section_start:section_end]
    lines = section.splitlines(keepends=True)

    insert_at = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("* "):
            insert_at = i + 1

    if insert_at is None:
        insert_at = 0
        while insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1

    lines.insert(insert_at, SWARM_ENTRY + "\n")
    new_section = "".join(lines)

    return content[:section_start] + new_section + content[section_end:]


def create_branch(owner: str, token: str) -> None:
    """Create the submission branch from the default branch HEAD."""
    # Get default branch SHA
    repo = gh_api("GET", f"/repos/{owner}/{UPSTREAM_REPO}", token=token)
    default_branch = repo.get("default_branch", "main")
    ref = gh_api(
        "GET",
        f"/repos/{owner}/{UPSTREAM_REPO}/git/ref/heads/{default_branch}",
        token=token,
    )
    sha = ref["object"]["sha"]

    try:
        gh_api(
            "POST",
            f"/repos/{owner}/{UPSTREAM_REPO}/git/refs",
            token=token,
            body={"ref": f"refs/heads/{BRANCH_NAME}", "sha": sha},
        )
        print(f"Branch '{BRANCH_NAME}' created.")
    except urllib.error.HTTPError as exc:
        if exc.code == 422:
            print(f"Branch '{BRANCH_NAME}' already exists.")
        else:
            raise


def commit_readme(owner: str, new_content: str, sha: str, token: str) -> None:
    """Commit the patched README.md to the submission branch."""
    gh_api(
        "PUT",
        f"/repos/{owner}/{UPSTREAM_REPO}/contents/README.md",
        token=token,
        body={
            "message": "Add SWARM to Multi-Agent Simulation Projects",
            "content": base64.b64encode(new_content.encode()).decode(),
            "sha": sha,
            "branch": BRANCH_NAME,
        },
    )
    print("README.md committed.")


def open_pr(fork_owner: str, token: str) -> str:
    """Open a PR from the fork branch to upstream main. Returns PR URL."""
    pr_body = PR_BODY_PATH.read_text() if PR_BODY_PATH.exists() else ""

    try:
        pr = gh_api(
            "POST",
            f"/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/pulls",
            token=token,
            body={
                "title": PR_TITLE,
                "head": f"{fork_owner}:{BRANCH_NAME}",
                "base": "main",
                "body": pr_body,
                "maintainer_can_modify": True,
            },
        )
        url = pr.get("html_url", pr.get("url", ""))
        print(f"PR opened: {url}")
        return url
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode() if exc.fp else ""
        if exc.code == 422 and "already exists" in payload.lower():
            print("A PR for this branch already exists.")
            return ""
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        try:
            token = subprocess.check_output(
                ["gh", "auth", "token"], text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            token = None

    if not token:
        print(
            "Error: GITHUB_TOKEN environment variable is required or "
            "gh auth token must be available.\n"
            "Create a token at https://github.com/settings/tokens with 'repo' scope "
            "or run `gh auth login`.",
            file=sys.stderr,
        )
        sys.exit(1)

    fork_owner = get_or_create_fork(token)
    readme_content, readme_sha = get_readme(fork_owner, token)

    patched = patch_readme(readme_content)
    if patched == readme_content:
        print("Nothing to do.")
        return

    create_branch(fork_owner, token)
    commit_readme(fork_owner, patched, readme_sha, token)
    open_pr(fork_owner, token)
    print("Done!")


if __name__ == "__main__":
    main()
