#!/usr/bin/env bash
# Download pre-generated AWM (Agent World Model) environments from HuggingFace.
#
# Usage:
#   bash scripts/download_awm_envs.sh [TARGET_DIR]
#
# Requires:
#   - Python 3.12+ (for running AWM servers, not for downloading)
#   - huggingface_hub (pip install huggingface_hub)
#
# The environments contain SQLite databases and FastAPI server code
# for various domains (e-commerce, project management, booking, etc.).

set -euo pipefail

TARGET_DIR="${1:-external/awm-envs}"
REPO_ID="Snowflake-Labs/agent-world-model"
REVISION="main"

echo "=== AWM Environment Downloader ==="
echo "Target directory: ${TARGET_DIR}"
echo "HuggingFace repo: ${REPO_ID}"
echo ""

# Check for huggingface_hub
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Error: huggingface_hub not installed."
    echo "Install with: pip install huggingface_hub"
    exit 1
fi

# Create target directory
mkdir -p "${TARGET_DIR}"

# Download using huggingface_hub
echo "Downloading AWM environments..."
python3 -c "
from huggingface_hub import snapshot_download
import os

target = os.path.abspath('${TARGET_DIR}')
print(f'Downloading to: {target}')

path = snapshot_download(
    repo_id='${REPO_ID}',
    revision='${REVISION}',
    local_dir=target,
    repo_type='dataset',
)
print(f'Downloaded to: {path}')
"

echo ""
echo "Done! AWM environments downloaded to: ${TARGET_DIR}"
echo ""
echo "To use in a scenario YAML:"
echo "  awm:"
echo "    enabled: true"
echo "    envs_path: ${TARGET_DIR}"
echo "    environment_id: ecommerce_001"
