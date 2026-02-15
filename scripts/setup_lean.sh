#!/usr/bin/env bash
# Setup script for Lean 4 formal verification environment.
#
# Usage:
#   ./scripts/setup_lean.sh          # Full install (elan + project deps)
#   ./scripts/setup_lean.sh --deps   # Only fetch project deps (elan already installed)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LEAN_DIR="$REPO_ROOT/lean"

install_elan() {
  if command -v elan &>/dev/null; then
    echo "[OK] elan already installed: $(elan --version)"
    return 0
  fi
  echo "[*] Installing elan (Lean version manager)..."
  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
  export PATH="$HOME/.elan/bin:$PATH"
  echo "[OK] elan installed: $(elan --version)"
}

build_project() {
  echo "[*] Building Lean project in $LEAN_DIR ..."
  cd "$LEAN_DIR"
  echo "[*] Fetching dependencies (Mathlib4)..."
  lake update
  echo "[*] Building SwarmProofs..."
  lake build
  echo ""
  echo "========================================"
  echo " Lean 4 setup complete!"
  echo "========================================"
  echo " Project:  $LEAN_DIR"
  echo " Proofs:   $LEAN_DIR/SwarmProofs/"
  echo ""
  echo " Commands:"
  echo "   cd $LEAN_DIR && lake build    # Verify all proofs"
  echo ""
}

main() {
  if [[ "${1:-}" != "--deps" ]]; then
    install_elan
  fi
  build_project
}

main "$@"
