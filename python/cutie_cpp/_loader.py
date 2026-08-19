"""C++ 扩展模块的加载与错误诊断。

wheel 随包分发以下运行时库（都在 cutie_cpp 包目录内，靠 RPATH=$ORIGIN 互相解析）：
    libcutie.so                        推理库（OpenCV 已静态链入）
    _core.*.so                         pybind11 扩展
    libonnxruntime.so.1                ONNX Runtime 核心
    libonnxruntime_providers_cuda.so   CUDA execution provider
    libonnxruntime_providers_shared.so provider bridge

**不随包分发**（体积过大，需用户自备）：
    CUDA 运行时  libcudart / libcublas / libcublasLt，合计约 800 MB
    cuDNN 9      6 个子库合计约 600 MB

因此加载失败几乎都源于这两者缺失。本模块负责把裸 ImportError / RuntimeError
转换成指明缺失库名与安装方式的提示。

注意：**不要**尝试借用 `onnxruntime-gpu` pip 包的 provider 插件。ORT 的 provider
需要与其核心库严格配套，混用会让进程里出现两个 ORT 实例，直接段错误。
本包自带配套的 provider，无需也不应从别处加载。
"""

import re
from pathlib import Path


# 从错误消息中提取缺失的库文件名
_MISSING_LIB_PATTERN = re.compile(r"(lib[\w.+-]*\.so[\w.]*)")

# CUDA 运行时库 → 所属组件
_CUDA_LIBS = {
    "libcudart": "CUDA Runtime",
    "libcublas": "cuBLAS",
    "libcublasLt": "cuBLAS",
    "libcufft": "cuFFT",
    "libnvrtc": "NVRTC",
}

# cuDNN 库 → 用途。ORT 1.23 要求 cuDNN 9（不兼容 cuDNN 8）。
_CUDNN_LIBS = {
    "libcudnn": "cuDNN 主库",
    "libcudnn_graph": "cuDNN 图 API",
    "libcudnn_ops": "cuDNN 算子",
    "libcudnn_heuristic": "cuDNN 启发式选择",
    "libcudnn_engines_precompiled": "cuDNN 预编译引擎",
    "libcudnn_engines_runtime_compiled": "cuDNN 运行时编译引擎",
}

_CUDA_HINT = """
缺少 CUDA 运行时库{detail}。

wheel 已包含 libcutie.so 与 ONNX Runtime，但 CUDA 运行时约 800 MB，
需要单独安装。二选一：

  1. 安装 CUDA Toolkit（推荐，需 >= 11.8）
         https://developer.nvidia.com/cuda-downloads
     安装后确保库目录在搜索路径中：
         export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

  2. 只装运行时库（无需完整 Toolkit）
         pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12

排查：ldconfig -p | grep libcudart
"""

_CUDNN_HINT = """
缺少 cuDNN{detail}。

ONNX Runtime 的 CUDA provider 依赖 **cuDNN 9**（不兼容 cuDNN 8）。
cuDNN 约 600 MB，未随 wheel 分发。安装：

    pip install nvidia-cudnn-cu12

或从 NVIDIA 官网安装系统包：
    https://developer.nvidia.com/cudnn-downloads

排查：ldconfig -p | grep libcudnn
"""


def _describe_missing(message, known_libs):
    """从错误消息中提取缺失库的描述。

    Args:
        message (str): 错误消息文本。
        known_libs (dict): 库名前缀 → 组件说明的映射。

    Returns:
        str: 形如 "：libcudart.so.12（CUDA Runtime）" 的描述；
             无法识别时返回空字符串。
    """
    match = _MISSING_LIB_PATTERN.search(message)
    if match is None:
        return ""

    lib_name = match.group(1)
    # 去掉版本号后缀以匹配已知库表，如 libcudart.so.12 → libcudart
    stem = lib_name.split(".so")[0]
    component = known_libs.get(stem)

    if component is None:
        return f"：{lib_name}"
    return f"：{lib_name}（{component}）"


def _preload_pip_nvidia_libs():
    """预加载 pip 安装的 nvidia-* 包中的库。

    通过 pip 安装 nvidia-cudnn-cu12 等包时，库文件位于
    site-packages/nvidia/*/lib，不在动态链接器的默认搜索路径中。
    这里用 RTLD_GLOBAL 预加载，让后续 dlopen 能复用已加载的符号。

    Returns:
        list[str]: 成功处理的库目录列表。
    """
    import ctypes
    import site

    search_roots = []
    for getter in (site.getsitepackages, lambda: [site.getusersitepackages()]):
        try:
            search_roots.extend(getter())
        except (AttributeError, TypeError):
            continue

    loaded_dirs = []
    for site_dir in search_roots:
        nvidia_root = Path(site_dir) / "nvidia"
        if not nvidia_root.is_dir():
            continue

        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            for lib_path in sorted(lib_dir.glob("lib*.so.*")):
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # 单个库加载失败不影响其它库，交由后续导入报错
                    continue
            loaded_dirs.append(str(lib_dir))

    return loaded_dirs


def load_core():
    """导入 C++ 扩展模块，失败时给出可操作的诊断信息。

    Returns:
        module: cutie_cpp._core 扩展模块。

    Raises:
        ImportError: 加载失败。消息中包含缺失的库名与安装建议。
    """
    try:
        from cutie_cpp import _core

        return _core
    except ImportError as exc:
        first_error = str(exc)

        # 可能是 pip 安装的 CUDA/cuDNN 库不在搜索路径中，预加载后重试一次
        if _mentions(first_error, _CUDA_LIBS) or _mentions(first_error, _CUDNN_LIBS):
            if _preload_pip_nvidia_libs():
                try:
                    from cutie_cpp import _core

                    return _core
                except ImportError:
                    pass  # 重试仍失败，走下面的统一报错

        raise ImportError(_diagnose(first_error)) from exc


def _mentions(message, known_libs):
    """判断错误消息是否提到了某组库。

    Args:
        message (str): 错误消息。
        known_libs (dict): 库名前缀 → 组件说明。

    Returns:
        bool: 提到其中任一库时为 True。
    """
    return any(name in message for name in known_libs)


def _diagnose(message):
    """根据错误消息生成诊断文本。

    Args:
        message (str): 原始错误消息。

    Returns:
        str: 含缺失库说明与安装建议的诊断文本。
    """
    # cuDNN 要先判断：其库名以 libcudnn 开头，不会与 _CUDA_LIBS 混淆，
    # 但两组都可能出现在同一条消息里，cuDNN 的提示更具体。
    if _mentions(message, _CUDNN_LIBS):
        return (
            _CUDNN_HINT.format(detail=_describe_missing(message, _CUDNN_LIBS))
            + f"\n原始错误: {message}"
        )

    if _mentions(message, _CUDA_LIBS):
        return (
            _CUDA_HINT.format(detail=_describe_missing(message, _CUDA_LIBS))
            + f"\n原始错误: {message}"
        )

    # 非 CUDA/cuDNN 相关（如 libcutie.so 缺失、ABI 不匹配、wheel 损坏）
    package_dir = Path(__file__).resolve().parent
    return (
        f"加载 cutie_cpp 扩展模块失败: {message}\n"
        f"包目录: {package_dir}\n"
        f"目录内容: {sorted(p.name for p in package_dir.glob('*.so*'))}\n"
        "若从源码构建，请确认已用 scripts/build_python.sh 构建，"
        "且 libcutie.so 与 ONNX Runtime 库位于扩展模块同一目录。"
    )


def runtime_error_help(error):
    """为推理期的运行时错误补充依赖诊断。

    ORT 在创建 CUDA session 时才 dlopen provider，而 provider 又依赖 cuDNN，
    因此这类缺失表现为**构造处理器时失败**，而非导入失败。

    Args:
        error (Exception | str): 捕获到的错误或其消息。

    Returns:
        str: 需要提示时返回以换行开头的诊断文本；无关错误返回空字符串。
    """
    message = str(error)

    if _mentions(message, _CUDNN_LIBS):
        return "\n" + _CUDNN_HINT.format(
            detail=_describe_missing(message, _CUDNN_LIBS)
        )

    if _mentions(message, _CUDA_LIBS):
        return "\n" + _CUDA_HINT.format(detail=_describe_missing(message, _CUDA_LIBS))

    # provider 加载失败但未指明具体缺哪个库时，最常见原因是缺 cuDNN 9
    if "providers_cuda" in message or "providers_shared" in message:
        bundled = bundled_libraries()
        has_provider = any("providers_cuda" in name for name in bundled)
        if has_provider:
            return (
                "\n\nprovider 插件已随包分发但加载失败，最常见原因是缺少 cuDNN 9。"
                "\n可尝试: pip install nvidia-cudnn-cu12"
                "\n详细诊断: python -m cutie_cpp"
            )
        return (
            f"\n\n包目录中未找到 provider 插件（现有: {bundled}）。"
            "\nwheel 可能构建不完整，请用 scripts/build_python.sh 重新构建。"
        )

    return ""


def library_status():
    """检查外部依赖库的可用性，用于安装诊断。

    Returns:
        dict[str, dict[str, str | None]]: 形如
            {"CUDA": {"libcudart": "/path/..." | None}, "cuDNN": {...}}。
    """
    return {
        "CUDA": {name: _find_library(name) for name in _CUDA_LIBS},
        "cuDNN": {name: _find_library(name) for name in _CUDNN_LIBS},
    }


def _find_library(stem):
    """定位指定库的实际路径。

    Args:
        stem (str): 库名前缀，如 "libcudart"。

    Returns:
        str | None: 找到的路径；未找到时为 None。
    """
    import ctypes.util

    # find_library 期望不带 lib 前缀的名字
    found = ctypes.util.find_library(stem[len("lib") :])
    if found is not None:
        return found

    # CUDA 12 的 soname 带主版本号，find_library 可能命中不到，补充常见路径
    for candidate_dir in (
        Path("/usr/local/cuda/lib64"),
        Path("/usr/lib/x86_64-linux-gnu"),
    ):
        matches = sorted(candidate_dir.glob(f"{stem}.so.*"))
        if matches:
            return str(matches[0])

    return None


def bundled_libraries():
    """列出随包分发的运行时库。

    Returns:
        list[str]: 包目录下的共享库文件名（已排序）。
    """
    package_dir = Path(__file__).resolve().parent
    return sorted(p.name for p in package_dir.glob("*.so*"))
