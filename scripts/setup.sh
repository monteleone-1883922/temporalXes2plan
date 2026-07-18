#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOWNWARD_DIR="$PROJECT_ROOT/vendor/downward"

# Init submodule if needed
if [ ! -f "$DOWNWARD_DIR/build.py" ]; then
    echo "[SETUP] Initializing submodule..."
    git -C "$PROJECT_ROOT" submodule update --init --depth 1 vendor/downward
fi

# Build if not already built (check the actual binary, not just the
# builds/release directory -- an interrupted build can leave that directory
# in place without ever producing bin/downward)
if [ ! -f "$DOWNWARD_DIR/builds/release/bin/downward" ]; then
    echo "[BUILD] Compiling Fast Downward..."

    # Check dependencies
    for cmd in cmake g++ python3; do
        if ! command -v $cmd &>/dev/null; then
            echo "[ERROR] '$cmd' not found. Install with:"
            echo "  sudo apt install cmake g++ python3"
            exit 1
        fi
    done

    python3 "$DOWNWARD_DIR/build.py"
    echo "[OK] Build complete."
else
    echo "[OK] Fast Downward already built."
fi