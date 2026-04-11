#!/usr/bin/env bash
# One-click setup for xrobotoolkit_sdk (PICO VR hand tracking).
# Usage: bash scripts/setup_xrobotoolkit.sh [pybind_dir]
#
# Default pybind_dir: ./third_party/xrobotoolkit/XRoboToolkit-PC-Service-Pybind

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYBIND_DIR="${1:-$REPO_ROOT/third_party/xrobotoolkit/XRoboToolkit-PC-Service-Pybind}"
PYBIND_DIR="$(mkdir -p "$(dirname "$PYBIND_DIR")" && cd "$(dirname "$PYBIND_DIR")" && pwd)/$(basename "$PYBIND_DIR")"

SDK_REPO="https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git"

echo "==> Using xrobotoolkit_sdk source at: $PYBIND_DIR"

# ---- 1. Ensure pybind submodule exists ----
if [ ! -f "$PYBIND_DIR/setup.py" ]; then
    echo "==> Pybind submodule missing, trying to initialize it..."
    git -C "$REPO_ROOT" submodule update --init --recursive -- "third_party/xrobotoolkit/XRoboToolkit-PC-Service-Pybind" || true
fi

if [ ! -f "$PYBIND_DIR/setup.py" ]; then
    echo "==> ERROR: XRoboToolkit pybind source not found at: $PYBIND_DIR"
    echo "==> Please run: git submodule update --init --recursive"
    exit 1
fi

# ---- 2. Clone & build nested C++ SDK ----
SDK_DIR="$PYBIND_DIR/XRoboToolkit-PC-Service"
if [ -d "$SDK_DIR" ]; then
    echo "==> SDK repo already exists, pulling latest..."
    git -C "$SDK_DIR" pull --ff-only || true
else
    echo "==> Cloning C++ SDK..."
    git clone "$SDK_REPO" "$SDK_DIR"
fi

SDK_BUILD_DIR="$SDK_DIR/RoboticsService/PXREARobotSDK"
echo "==> Building C++ SDK..."
pushd "$SDK_BUILD_DIR" > /dev/null
bash build.sh
popd > /dev/null

# ---- 3. Copy build artifacts ----
echo "==> Copying build artifacts to pybind project..."
mkdir -p "$PYBIND_DIR/lib" "$PYBIND_DIR/include"

SDK_LIB=""
for candidate in \
    "$SDK_BUILD_DIR/build/libPXREARobotSDK.so" \
    "$SDK_DIR/RoboticsService/SDK/linux/64/libPXREARobotSDK.so" \
    "$SDK_DIR/RoboticsService/SDK/linux_aarch64/64/libPXREARobotSDK.so"
do
    if [ -f "$candidate" ]; then
        SDK_LIB="$candidate"
        break
    fi
done

if [ -z "$SDK_LIB" ]; then
    echo "==> ERROR: libPXREARobotSDK.so not found after build."
    find "$SDK_DIR/RoboticsService" -name "libPXREARobotSDK.so" || true
    exit 1
fi

cp "$SDK_LIB" "$PYBIND_DIR/lib/"
cp "$SDK_BUILD_DIR/PXREARobotSDK.h" "$PYBIND_DIR/include/"
mkdir -p "$PYBIND_DIR/include/nlohmann"
cp -r "$SDK_BUILD_DIR/nlohmann/." "$PYBIND_DIR/include/nlohmann/"

# ---- 4. Install pybind11 if missing ----
if ! python -c "import pybind11" 2>/dev/null; then
    echo "==> Installing pybind11..."
    pip install pybind11
fi

# ---- 5. Build & install ----
echo "==> Building and installing xrobotoolkit_sdk..."
pushd "$PYBIND_DIR" > /dev/null
pip install .
popd > /dev/null

# ---- 6. Verify ----
if python -c "import xrobotoolkit_sdk; print('xrobotoolkit_sdk imported successfully')" 2>/dev/null; then
    echo "==> Done! xrobotoolkit_sdk is ready."
else
    echo "==> WARNING: import failed. Check build output above for errors."
    exit 1
fi
