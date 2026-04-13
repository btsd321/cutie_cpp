#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="Release"
JOBS="$(nproc 2>/dev/null || echo 4)"

# Dependency paths (empty = auto-detect)
VCPKG_ROOT="/home/lixinlong/Project/linden_perception/thirdparty/vcpkg"  # if set, use vcpkg installed packages (overrides other paths)
CUDA_ROOT="/usr/local/cuda-12.9"
OPENCV_DIR="./vcpkg/installed/x64-linux"
ONNXRUNTIME_ROOT="./vcpkg/installed/x64-linux"
TENSORRT_ROOT="~/Library/tensorrt/TensorRT-10.13.3.9"
INSTALL_PREFIX="${SCRIPT_DIR}/install"

# Backend toggles
ENABLE_ONNXRUNTIME="ON"
ENABLE_TENSORRT="OFF"
BUILD_EXAMPLES="ON"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Build cutie-cpp library.

Dependency paths:
  --vcpkg-root DIR         vcpkg root (uses installed packages, overrides other paths)
  --cuda-root DIR          CUDA toolkit root        (e.g. /usr/local/cuda)
  --opencv-dir DIR         OpenCV cmake config dir  (e.g. /usr/lib/cmake/opencv4)
  --onnxruntime-root DIR   ONNXRuntime install root  (e.g. /opt/onnxruntime)
  --tensorrt-root DIR      TensorRT install root     (e.g. /opt/TensorRT)

Build options:
  --build-dir DIR          Build directory           (default: ./build)
  --build-type TYPE        Release|Debug|RelWithDebInfo (default: Release)
  --install-prefix DIR     CMAKE_INSTALL_PREFIX      (default: /usr/local)
  --jobs N                 Parallel compile jobs     (default: nproc)
  --enable-tensorrt        Enable TensorRT backend   (default: OFF)
  --disable-onnxruntime    Disable ONNXRuntime       (default: ON)
  --disable-examples       Don't build examples      (default: ON)
  --debug                  Shorthand for --build-type Debug
  --clean                  Remove build dir first
  -h, --help               Show this message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vcpkg-root)        VCPKG_ROOT="$2";        shift 2 ;;
        --cuda-root)         CUDA_ROOT="$2";         shift 2 ;;
        --opencv-dir)        OPENCV_DIR="$2";        shift 2 ;;
        --onnxruntime-root)  ONNXRUNTIME_ROOT="$2";  shift 2 ;;
        --tensorrt-root)     TENSORRT_ROOT="$2";     shift 2 ;;
        --build-dir)         BUILD_DIR="$2";         shift 2 ;;
        --build-type)        BUILD_TYPE="$2";        shift 2 ;;
        --install-prefix)    INSTALL_PREFIX="$2";    shift 2 ;;
        --jobs)              JOBS="$2";              shift 2 ;;
        --enable-tensorrt)   ENABLE_TENSORRT="ON";   shift   ;;
        --disable-onnxruntime) ENABLE_ONNXRUNTIME="OFF"; shift ;;
        --disable-examples)  BUILD_EXAMPLES="OFF";   shift   ;;
        --debug)             BUILD_TYPE="Debug";       shift   ;;
        --clean)
            echo "Cleaning build directory: ${BUILD_DIR}"
            rm -rf "${BUILD_DIR}"
            shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Assemble CMake args ──────────────────────────────────────────────
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
    -DENABLE_ONNXRUNTIME="${ENABLE_ONNXRUNTIME}"
    -DENABLE_TENSORRT="${ENABLE_TENSORRT}"
    -DBUILD_EXAMPLES="${BUILD_EXAMPLES}"
)

if [[ -n "${VCPKG_ROOT}" ]]; then
    VCPKG_TOOLCHAIN="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    if [[ ! -f "${VCPKG_TOOLCHAIN}" ]]; then
        echo "ERROR: vcpkg toolchain not found: ${VCPKG_TOOLCHAIN}" >&2
        exit 1
    fi
    CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="${VCPKG_TOOLCHAIN}")
    # derive package paths from vcpkg installed tree (auto-detect triplet)
    VCPKG_INSTALLED="${VCPKG_ROOT}/installed"
    VCPKG_TRIPLET="$(ls "${VCPKG_INSTALLED}" 2>/dev/null | grep -v vcpkg | head -1)"
    if [[ -n "${VCPKG_TRIPLET}" ]]; then
        VCPKG_PKG_DIR="${VCPKG_INSTALLED}/${VCPKG_TRIPLET}"
        [[ -z "${OPENCV_DIR}" ]]       && OPENCV_DIR="${VCPKG_PKG_DIR}"
        [[ -z "${ONNXRUNTIME_ROOT}" ]] && ONNXRUNTIME_ROOT="${VCPKG_PKG_DIR}"
    fi
fi

[[ -n "${CUDA_ROOT}" ]]        && CMAKE_ARGS+=(-DCUDA_TOOLKIT_ROOT_DIR="${CUDA_ROOT}"
                                                -DCUDAToolkit_ROOT="${CUDA_ROOT}"
                                                -DCUDAToolkit_LIBRARY_DIR="${CUDA_ROOT}/targets/x86_64-linux/lib"
                                                -DCMAKE_CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc")
[[ -n "${OPENCV_DIR}" ]]       && CMAKE_ARGS+=(-DOpenCV_DIR="${OPENCV_DIR}")
[[ -n "${ONNXRUNTIME_ROOT}" ]] && CMAKE_ARGS+=(-DONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT}")
[[ -n "${TENSORRT_ROOT}" ]]    && CMAKE_ARGS+=(-DTENSORRT_ROOT="${TENSORRT_ROOT}")

# ── Configure & Build ────────────────────────────────────────────────
echo "============================================"
echo "  cutie-cpp build"
echo "============================================"
echo "  Source:      ${SCRIPT_DIR}"
echo "  Build:       ${BUILD_DIR}"
echo "  Type:        ${BUILD_TYPE}"
echo "  Install:     ${INSTALL_PREFIX}"
echo "  ORT:         ${ENABLE_ONNXRUNTIME}"
echo "  TRT:         ${ENABLE_TENSORRT}"
echo "  Examples:    ${BUILD_EXAMPLES}"
echo "  Jobs:        ${JOBS}"
[[ -n "${VCPKG_ROOT}" ]]       && echo "  vcpkg:       ${VCPKG_ROOT}"
[[ -n "${CUDA_ROOT}" ]]        && echo "  CUDA:        ${CUDA_ROOT}"
[[ -n "${OPENCV_DIR}" ]]       && echo "  OpenCV:      ${OPENCV_DIR}"
[[ -n "${ONNXRUNTIME_ROOT}" ]] && echo "  ORT root:    ${ONNXRUNTIME_ROOT}"
[[ -n "${TENSORRT_ROOT}" ]]    && echo "  TRT root:    ${TENSORRT_ROOT}"
echo "============================================"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j "${JOBS}"

echo ""
echo "Build complete.  Artifacts in: ${BUILD_DIR}"
echo "Run 'cmake --install ${BUILD_DIR}' or use install.sh to install."
