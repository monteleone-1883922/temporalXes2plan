#!/bin/bash
set -e 

while [[ $# -gt 0 ]]; do
    case $1 in
        --only-fast-downward)
            shift
            ;;
        --help)
            echo "Usage: ./setup_submodules.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --only-fast-downward   Setup Fast Downward planner"
            echo "  --help                 Show this help message"
            echo ""
            echo "Without options, Fast Downward will be set up."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "============================================="
echo "Setting up Planners"
echo "============================================="
echo ""

# Check if submodules are initialized
if [ ! -f "src/downward/fast-downward.py" ]; then
    NEED_INIT=true
else
    NEED_INIT=false
fi

if [ "$NEED_INIT" = true ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
    echo "Submodules initialized"
    echo ""
else
    echo "Submodules already initialized"
    echo ""
fi

# Setup Fast Downward
echo "============================================="
echo "Building Fast Downward"
echo "============================================="
cd src/downward
if [ -f "./build.py" ]; then
    ./build.py
    echo "Fast Downward built successfully"
else
    echo "Error: build.py not found in src/downward"
    exit 1
fi
cd ../..
echo ""

echo ""
echo "============================================="
echo "Setup Complete!"
echo "============================================="
echo "Fast Downward is ready. You can use it via call_planner.py:"
echo "  cd src"
echo "  python call_planner.py --search astar_lmcut"
echo ""
echo ""
