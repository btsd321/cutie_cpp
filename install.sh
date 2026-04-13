#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# All arguments are forwarded to build.sh
# Extract --build-dir if user passed it, to know where cmake artifacts live
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == "--build-dir" ]] && [[ $((i+1)) -lt ${#ARGS[@]} ]]; then
        BUILD_DIR="${ARGS[$((i+1))]}"
    fi
done

# ── Step 1: Build ────────────────────────────────────────────────────
echo ">>> Step 1/2: Building..."
bash "${SCRIPT_DIR}/build.sh" "$@"

# ── Step 2: Install ──────────────────────────────────────────────────
echo ""
echo ">>> Step 2/2: Installing..."
cmake --install "${BUILD_DIR}"

echo ""
echo "Install complete."
