#!/usr/bin/env bash
#
# build_python.sh — 构建自包含的 cutie_cpp Python wheel。
#
# 本脚本独立于 C++ 的 build.sh，两者使用**不同的 ONNX Runtime**：
#   - build.sh    : vcpkg 的静态 ORT，链入 libcutie.so
#   - 本脚本      : 官方预编译 GPU 包的动态 ORT
#
# 为什么必须分开：vcpkg 把 ORT 静态链入 libcutie.so，而 ORT 的 CUDA provider
# 是运行时 dlopen 的插件。静态核心 + 外部 provider 会让进程里出现两个 ORT 实例，
# 导致段错误。官方包提供动态 libonnxruntime.so，provider 能正常 bridge，
# 且体积小得多（provider 351 MB vs vcpkg 919 MB）。
#
# 产出的 wheel 自带 libcutie.so、ORT 核心与 provider；
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
ORT_VERSION="1.23.2"
ORT_ROOT=""
CUDA_ARCHS="89;120"
PYTHON_EXE="${PROJECT_ROOT}/.venv/bin/python"
OUTPUT_DIR="${PROJECT_ROOT}/dist"
JOBS="$(nproc)"
DO_CLEAN=0
SKIP_DEPS=0

# 官方预编译包的下载地址模板
ORT_BASE_URL="https://github.com/microsoft/onnxruntime/releases/download"
ORT_CACHE_DIR="${PROJECT_ROOT}/thirdparty/onnxruntime"

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
  --ort-version VER     ONNX Runtime 版本            (默认: ${ORT_VERSION})
  --ort-root DIR        已解包的官方 ORT 目录，跳过下载
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
        --ort-version)  ORT_VERSION="$2";  shift 2 ;;
        --ort-root)     ORT_ROOT="$2";     shift 2 ;;
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

# ─── 4. 准备官方 ONNX Runtime GPU 包 ─────────────────────────────────
# 官方包同时提供动态 libonnxruntime.so、CUDA provider 插件和头文件，
# 是 wheel 能自包含的前提。
prepare_onnxruntime() {
    local tarball_name="onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
    local extract_dir="${ORT_CACHE_DIR}/onnxruntime-linux-x64-gpu-${ORT_VERSION}"
    local tarball_path="${ORT_CACHE_DIR}/${tarball_name}"

    # 已解包且关键文件齐全则直接复用
    if [[ -f "${extract_dir}/lib/libonnxruntime_providers_cuda.so" ]]; then
        info "复用已解包的 ONNX Runtime: ${extract_dir}"
        ORT_ROOT="${extract_dir}"
        return 0
    fi

    mkdir -p "${ORT_CACHE_DIR}"

    if [[ ! -f "${tarball_path}" ]]; then
        local url="${ORT_BASE_URL}/v${ORT_VERSION}/${tarball_name}"
        info "下载 ONNX Runtime GPU ${ORT_VERSION} (约 230 MB)..."
        # --continue-at - 支持断点续传，避免大文件重下
        curl -fL --retry 3 --continue-at - -o "${tarball_path}.part" "${url}" \
            || die "下载失败: ${url}"
        mv "${tarball_path}.part" "${tarball_path}"
    else
        info "复用已下载的压缩包: ${tarball_path}"
    fi

    info "解包到 ${ORT_CACHE_DIR}"
    tar xzf "${tarball_path}" -C "${ORT_CACHE_DIR}" || die "解包失败: ${tarball_path}"

    [[ -f "${extract_dir}/lib/libonnxruntime_providers_cuda.so" ]] \
        || die "解包后未找到 CUDA provider，压缩包可能损坏: ${tarball_path}
可删除该文件后重新运行本脚本。"

    ORT_ROOT="${extract_dir}"
}

if [[ -n "${ORT_ROOT}" ]]; then
    ORT_ROOT="$(cd "${ORT_ROOT}" && pwd)" || die "ORT 目录不存在: ${ORT_ROOT}"
    [[ -f "${ORT_ROOT}/lib/libonnxruntime_providers_cuda.so" ]] \
        || die "${ORT_ROOT} 中没有 libonnxruntime_providers_cuda.so。
--ort-root 应指向官方 GPU 预编译包的解包目录（含 include/ 与 lib/）。"
    info "使用指定的 ONNX Runtime: ${ORT_ROOT}"
else
    prepare_onnxruntime
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
export CUTIE_ORT_ROOT="${ORT_ROOT}"
export CUTIE_CUDA_ARCHS="${CUDA_ARCHS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"

# 忽略 vcpkg 自带的 onnxruntime config，确保用上面准备的官方包。
# 这是 CUTIE_PREFER_ORT_MODULE=ON 之外的第二重保险。
export CUTIE_IGNORE_PATH="${VCPKG_ROOT}/installed/x64-linux/share/onnxruntime"

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
