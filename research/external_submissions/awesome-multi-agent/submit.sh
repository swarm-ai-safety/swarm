#!/usr/bin/env bash
# Submit SWARM to WeiChengTseng/awesome-multi-agent
#
# Prerequisites:
#   - gh CLI authenticated (gh auth login)
#   - git configured with your name/email
#
# Usage:
#   bash research/external_submissions/awesome-multi-agent/submit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(mktemp -d)"
BRANCH="add-swarm-framework"

echo "==> Forking and cloning WeiChengTseng/awesome-multi-agent..."
pushd "$WORK_DIR" >/dev/null
gh repo fork WeiChengTseng/awesome-multi-agent --clone
popd >/dev/null

cd "$WORK_DIR/awesome-multi-agent"

echo "==> Creating branch: $BRANCH"
git fetch origin "$BRANCH" >/dev/null 2>&1 || true
if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
  git checkout -B "$BRANCH" "origin/$BRANCH"
else
  git checkout -b "$BRANCH"
fi

echo "==> Applying changes to README.md..."
python3 - <<'PYEOF'
import sys

img_row_old = """  <tr>
    <td width="23%">
    <img src="https://vitalab.github.io/article/images/autocurricula/env.png" width="100%"/>
    </td>
  </tr>
"""

img_row_new = """  <tr>
    <td width="23%">
    <img src="https://vitalab.github.io/article/images/autocurricula/env.png" width="100%"/>
    </td>
    <td width="23%">
    <img src="https://github.com/swarm-ai-safety/swarm/raw/main/docs/images/swarm-hero.gif" width="100%"/>
    </td>
  </tr>
"""

link_row_old = """  <tr>
    <th>
    <a href="https://github.com/openai/multi-agent-emergence-environments">Multiagent emergence</a>
    </th>

  </tr>
"""

link_row_new = """  <tr>
    <th>
    <a href="https://github.com/openai/multi-agent-emergence-environments">Multiagent emergence</a>
    </th>
    <th>
    <a href="https://github.com/swarm-ai-safety/swarm">SWARM</a>
    </th>

  </tr>
"""

bullet_block = """- **SWARM: System-Wide Assessment of Risk in Multi-agent systems** [[code]](https://github.com/swarm-ai-safety/swarm) [[install]](https://pypi.org/project/swarm-safety/)
  - A distributional safety simulation framework for studying emergent risks in multi-agent AI systems.
  - Uses soft (probabilistic) labels instead of binary classifications to measure toxicity, adverse selection, and governance effectiveness.
  - Built-in agent archetypes (honest, opportunistic, deceptive, adversarial, LLM-backed) and governance levers (taxes, reputation, circuit breakers, audits).
  - YAML-driven scenarios, parameter sweeps, and full replay support for reproducible multi-agent experiments.
  - R. Savitt, 2026. MIT License.
"""

with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

if "swarm-hero.gif" in content:
    print("README.md already includes SWARM in the environment table. No changes needed.")
    sys.exit(0)

if bullet_block in content:
    content = content.replace(bullet_block, "")
    content = content.replace("## Environment\n\n\n", "## Environment\n\n")

if img_row_old not in content:
    print("ERROR: image row marker not found in README.md")
    sys.exit(1)

if link_row_old not in content:
    print("ERROR: link row marker not found in README.md")
    sys.exit(1)

content = content.replace(img_row_old, img_row_new, 1)
content = content.replace(link_row_old, link_row_new, 1)

with open("README.md", "w", encoding="utf-8") as f:
    f.write(content)

print("README.md updated successfully.")
PYEOF

echo "==> Committing..."
git add README.md
git commit -m "Add SWARM: distributional safety framework for multi-agent systems"

echo "==> Pushing to fork..."
git push -u origin "$BRANCH"

echo "==> Creating pull request..."
if ! gh pr create \
    --repo WeiChengTseng/awesome-multi-agent \
    --title "Add SWARM safety simulation framework" \
    --body-file "$SCRIPT_DIR/pr_body.md"; then
  echo "==> PR already exists or could not be created."
fi

echo "==> Done! PR created."

echo "==> Cleaning up temp dir: $WORK_DIR"
