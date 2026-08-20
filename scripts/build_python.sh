#!/usr/bin/env bash
#
# build_python.sh — 构建自包含的 cutie_cpp Python wheel。
#
# 与 C++ 的 build.sh 用同一套 vcpkg 依赖，区别只在于：
#   - 走 scikit-build-core 产出 wheel，而非直接 cmake --build
#   - 把运行时库（libcutie.so + ORT CUDA provider）打进包目录，
#     靠 RPATH=$ORIGIN 解析，安装后无需设置 LD_LIBRARY_PATH
#
# 关于 provider：ORT 核心被 vcpkg 静态链入 libcutie.so，但 CUDA provider 是
# 运行时 dlopen 的插件，必须随包分发，且必须是**同一份 vcpkg 构建**的版本
# （混用 pip 的 onnxruntime-gpu 会让进程里出现两个 ORT 实例并段错误）。
#
# 用户只需自备 CUDA Toolkit (>= 11.8) 与 cuDNN 9。
#
# 用法:
#   bash scripts/build_python.sh --vcpkg-root /path/to/vcpkg
#
set -euo pipefail

# ─── 路径 ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ─── 默认参数 ────────────────────────────────────────────────────────
VCPKG_ROOT=""
PROVIDER_DIR=""
CUDA_ARCHS="89;120"
PYTHON_EXE="${PROJECT_ROOT}/.venv/bin/python3"
OUTPUT_DIR="${PROJECT_ROOT}/dist"
JOBS="$(nproc)"
DO_CLEAN=0
SKIP_DEPS=0

# ─── 颜色输出 ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()   { error "$*"; exit 1; }

usage() {
    cat <<EOF
构建自包含的 cutie_cpp Python wheel。

必需参数:
  --vcpkg-root DIR      vcpkg 根目录（OpenCV CUDA 模块的来源）

可选参数:
  --provider-dir DIR    ONNXRuntime CUDA provider 所在目录
                        (默认从 vcpkg 的 installed/x64-linux/lib 推导)
  --cuda-archs LIST     CUDA 架构，分号分隔          (默认: ${CUDA_ARCHS})
                        89=Ada/RTX40, 120=Blackwell/RTX50
  --python PATH         Python 解释器                (默认: .venv/bin/python)
  --output DIR          wheel 输出目录               (默认: dist/)
  --jobs N              并行编译任务数               (默认: nproc)
  --clean               构建前清空构建目录
  --skip-deps           跳过 pip 依赖安装
  -h, --help            显示本帮助

示例:
  bash scripts/build_python.sh --vcpkg-root ~/vcpkg
  bash scripts/build_python.sh --vcpkg-root ~/vcpkg --cuda-archs "89"
EOF
}

# ─── 参数解析 ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vcpkg-root)   VCPKG_ROOT="$2";   shift 2 ;;
        --provider-dir) PROVIDER_DIR="$2"; shift 2 ;;
        --cuda-archs)   CUDA_ARCHS="$2";   shift 2 ;;
        --python)       PYTHON_EXE="$2";   shift 2 ;;
        --output)       OUTPUT_DIR="$2";   shift 2 ;;
        --jobs)         JOBS="$2";         shift 2 ;;
        --clean)        DO_CLEAN=1;        shift   ;;
        --skip-deps)    SKIP_DEPS=1;       shift   ;;
        -h|--help)      usage; exit 0 ;;
        *)              die "未知参数: $1（用 --help 查看用法）" ;;
    esac
done

# ─── 1. 校验 vcpkg（OpenCV 来源）─────────────────────────────────────
# 代码用到 cv::cuda::{resize,cvtColor,copyMakeBorder,...}，
# 而发行版的 OpenCV 包通常不含 CUDA 模块，因此必须用 vcpkg 构建的版本。
[[ -n "${VCPKG_ROOT}" ]] || die "必须指定 --vcpkg-root（OpenCV CUDA 模块的来源）"
VCPKG_ROOT="$(cd "${VCPKG_ROOT}" && pwd)" || die "vcpkg 目录不存在: ${VCPKG_ROOT}"

VCPKG_TOOLCHAIN="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
[[ -f "${VCPKG_TOOLCHAIN}" ]] || die "未找到 vcpkg 工具链: ${VCPKG_TOOLCHAIN}"

VCPKG_OPENCV="${VCPKG_ROOT}/installed/x64-linux/share/opencv4"
[[ -d "${VCPKG_OPENCV}" ]] || die "vcpkg 中未安装 OpenCV: ${VCPKG_OPENCV}
请先执行: ${VCPKG_ROOT}/vcpkg install opencv4[core,cuda]"

# ─── 2. 校验 CUDA ────────────────────────────────────────────────────
NVCC=""
for candidate in "${CUDA_HOME:-}/bin/nvcc" "$(command -v nvcc || true)" \
                 /usr/local/cuda/bin/nvcc; do
    [[ -n "${candidate}" && -x "${candidate}" ]] && { NVCC="${candidate}"; break; }
done
[[ -n "${NVCC}" ]] || die "未找到 nvcc。请安装 CUDA Toolkit (>= 11.8)，
或设置 CUDA_HOME 指向其安装目录。"

CUDA_VERSION="$("${NVCC}" --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
info "CUDA ${CUDA_VERSION} (${NVCC})"

# sm_120 (Blackwell) 需要 CUDA >= 12.8，提前拦住以免编译到一半才失败
if [[ "${CUDA_ARCHS}" == *"120"* ]]; then
    if [[ "$(printf '%s\n12.8\n' "${CUDA_VERSION}" | sort -V | head -1)" != "12.8" ]]; then
        die "CUDA ${CUDA_VERSION} 不支持 sm_120 (Blackwell)。
请升级到 CUDA >= 12.8，或改用 --cuda-archs \"89\"。"
    fi
fi

# ─── 3. 准备 Python 环境 ─────────────────────────────────────────────
if [[ ! -x "${PYTHON_EXE}" ]]; then
    if [[ "${PYTHON_EXE}" == "${PROJECT_ROOT}/.venv/bin/python" ]]; then
        info "创建虚拟环境 ${PROJECT_ROOT}/.venv"
        python3 -m venv "${PROJECT_ROOT}/.venv" || die "创建 venv 失败"
    else
        die "Python 解释器不可用: ${PYTHON_EXE}"
    fi
fi
info "Python: $("${PYTHON_EXE}" --version 2>&1) (${PYTHON_EXE})"

if [[ "${SKIP_DEPS}" -eq 0 ]]; then
    info "安装构建依赖..."
    "${PYTHON_EXE}" -m pip install --quiet --upgrade pip \
        || warn "pip 升级失败，继续"
    "${PYTHON_EXE}" -m pip install --quiet \
        "pybind11>=2.12" "scikit-build-core>=0.8" build numpy PyYAML \
        || die "安装构建依赖失败"
fi

PYBIND11_DIR="$("${PYTHON_EXE}" -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
    || die "无法定位 pybind11 的 CMake 目录"

# ─── 4. 校验 ONNXRuntime CUDA provider ───────────────────────────────
# provider 必须与静态链入 libcutie.so 的 ORT 核心同源（同一份 vcpkg 构建），
# 因此默认从 vcpkg 的库目录取。这里提前校验，避免构建到最后才失败。
PROVIDER_CUDA="libonnxruntime_providers_cuda.so"

if [[ -z "${PROVIDER_DIR}" ]]; then
    PROVIDER_DIR="${VCPKG_ROOT}/installed/x64-linux/lib"
fi
PROVIDER_DIR="$(cd "${PROVIDER_DIR}" && pwd)" \
    || die "provider 目录不存在: ${PROVIDER_DIR}"

[[ -f "${PROVIDER_DIR}/${PROVIDER_CUDA}" ]] || die "在 ${PROVIDER_DIR} 中未找到 ${PROVIDER_CUDA}。
wheel 缺少它将无法推理。请确认 vcpkg 安装的 onnxruntime 带 CUDA 支持，
或用 --provider-dir 指定其所在目录。"

PROVIDER_SIZE="$(du -h "${PROVIDER_DIR}/${PROVIDER_CUDA}" | cut -f1)"
info "ONNXRuntime CUDA provider: ${PROVIDER_DIR} (${PROVIDER_SIZE})"

# provider 的 CUDA 架构决定 wheel 能跑在哪些显卡上——它与 libcutie.so 的架构
# 是两套独立的编译产物，只有两者都覆盖目标架构才能真正跑起来。
if command -v cuobjdump >/dev/null 2>&1 || [[ -x "${NVCC%/nvcc}/cuobjdump" ]]; then
    CUOBJDUMP="$(command -v cuobjdump || echo "${NVCC%/nvcc}/cuobjdump")"
    PROVIDER_ARCHS="$("${CUOBJDUMP}" --list-elf "${PROVIDER_DIR}/${PROVIDER_CUDA}" 2>/dev/null \
        | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')"
    [[ -n "${PROVIDER_ARCHS}" ]] && info "provider 支持的架构: ${PROVIDER_ARCHS}"

    # 逐个检查请求的架构是否被 provider 覆盖
    IFS=';' read -ra _requested <<< "${CUDA_ARCHS}"
    for arch in "${_requested[@]}"; do
        if [[ -n "${PROVIDER_ARCHS}" && "${PROVIDER_ARCHS}" != *"sm_${arch} "* ]]; then
            warn "provider 不含 sm_${arch}，该架构的显卡上推理可能失败"
        fi
    done
fi

# ─── 5. 构建 wheel ───────────────────────────────────────────────────
if [[ "${DO_CLEAN}" -eq 1 ]]; then
    info "清空构建目录 build/skbuild"
    rm -rf "${PROJECT_ROOT}/build/skbuild"
fi

mkdir -p "${OUTPUT_DIR}"

# 通过环境变量把机器相关的路径传给 pyproject.toml，
# 避免把绝对路径写进仓库（见 [tool.scikit-build.cmake.define]）。
export CUTIE_VCPKG_TOOLCHAIN="${VCPKG_TOOLCHAIN}"
export CUTIE_ORT_PROVIDER_DIR="${PROVIDER_DIR}"
export CUTIE_CUDA_ARCHS="${CUDA_ARCHS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"

info "开始构建 wheel（CUDA 架构: ${CUDA_ARCHS}，并行度: ${JOBS}）"
info "首次构建需编译 CUDA kernel，可能耗时 10-20 分钟..."

cd "${PROJECT_ROOT}"
"${PYTHON_EXE}" -m build --wheel --no-isolation --outdir "${OUTPUT_DIR}" \
    -C cmake.define.Python_EXECUTABLE="${PYTHON_EXE}" \
    -C cmake.define.pybind11_DIR="${PYBIND11_DIR}" \
    || die "wheel 构建失败"

# ─── 6. 报告产物 ─────────────────────────────────────────────────────
WHEEL_FILE="$(ls -t "${OUTPUT_DIR}"/cutie_cpp-*.whl 2>/dev/null | head -1)"
[[ -n "${WHEEL_FILE}" ]] || die "构建似乎成功但未找到 wheel 文件"

WHEEL_SIZE="$(du -h "${WHEEL_FILE}" | cut -f1)"
info "构建完成: ${WHEEL_FILE} (${WHEEL_SIZE})"

echo
echo "包含的运行时库:"
"${PYTHON_EXE}" - "${WHEEL_FILE}" <<'PYEOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    entries = [i for i in archive.infolist() if ".so" in i.filename]
    for item in sorted(entries, key=lambda i: -i.file_size):
        print(f"  {item.file_size / 1048576:8.1f} MB  {item.filename}")
PYEOF

cat <<EOF

安装方式:
    ${PYTHON_EXE} -m pip install "${WHEEL_FILE}"

安装后自检:
    ${PYTHON_EXE} -m cutie_cpp

外部依赖（未打包，需自备）:
    CUDA Toolkit >= 11.8    https://developer.nvidia.com/cuda-downloads
    cuDNN 9                 pip install nvidia-cudnn-cu12
EOF
