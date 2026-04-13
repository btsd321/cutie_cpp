#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 自动向上查找包含 src 目录的项目根目录
find_project_root() {
    local dir="${SCRIPT_DIR}"
    while [[ "${dir}" != "/" ]]; do
        if [[ -d "${dir}/src" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done
    return 1
}

PROJECT_ROOT="$(find_project_root)" || {
    echo -e "\033[0;31m[ERROR]\033[0m 未找到包含 src 目录的项目根目录（从 ${SCRIPT_DIR} 向上查找）"
    exit 1
}

VCPKG_DIR="${PROJECT_ROOT}/thirdparty/vcpkg"
VCPKG_BIN="${VCPKG_DIR}/vcpkg"
TRIPLET="x64-linux"
TRT_VERSION="10.13.3.9"
TRT_DIRNAME="TensorRT-${TRT_VERSION}"

echo "正在构建 cutie-cpp 库..."
bash $PROJECT_ROOT/build.sh --vcpkg-root "${VCPKG_DIR}" 
echo "构建完成，正在运行示例程序..."
export LD_LIBRARY_PATH="${PROJECT_ROOT}/build:${VCPKG_DIR}/installed/${TRIPLET}/lib:$LD_LIBRARY_PATH"
"${PROJECT_ROOT}/build/demo_basic" ${PROJECT_ROOT}/examples/example.mp4 ${PROJECT_ROOT}/examples/example_frame0_mask.png
