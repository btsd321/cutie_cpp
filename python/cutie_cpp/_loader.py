"""C++ 扩展模块的加载与错误诊断。

wheel 中打包了 libcutie.so（已静态链入 ONNX Runtime 与 OpenCV），但
**不包含 CUDA 运行时**——libcudart / libcublas / libcublasLt 合计约 800 MB，
随包分发不现实，且用户通常已装有 CUDA Toolkit。

因此导入失败最常见的原因是缺少 CUDA 库。本模块把裸 ImportError 转换成
指明缺失库名与安装方式的提示。
"""

import os
import re
import sys
from pathlib import Path


# 从 ImportError 消息中提取缺失的库文件名
_MISSING_LIB_PATTERN = re.compile(r"(lib[\w.+-]*\.so[\w.]*)")

# 已知库名 → 所属组件与安装建议
_CUDA_LIBS = {
    "libcudart": "CUDA Runtime",
    "libcublas": "cuBLAS",
    "libcublasLt": "cuBLAS",
    "libnvrtc": "NVRTC",
}

# ONNX Runtime 的 CUDA execution provider 插件文件名。
# 这两个库由 ORT 在创建 CUDA session 时 dlopen，不是链接期依赖。
_PROVIDER_SHARED = "libonnxruntime_providers_shared.so"
_PROVIDER_CUDA = "libonnxruntime_providers_cuda.so"

_PROVIDER_HINT = """
未找到 ONNX Runtime 的 CUDA execution provider 插件
（libonnxruntime_providers_cuda.so / libonnxruntime_providers_shared.so）。

ONNX Runtime 的核心库已静态链入 libcutie.so，但 CUDA provider 是运行时
按需 dlopen 的插件，体积约 900 MB 且额外依赖 cuDNN 9，因此不随 wheel 分发。

安装方式：

    pip install onnxruntime-gpu

它会同时提供 provider 插件与配套的 cuDNN，本库会自动从其安装目录加载。

若已装在非标准位置，可用环境变量指定所在目录：

    export CUTIE_ORT_PROVIDER_DIR=/path/to/dir/containing/providers
"""

_INSTALL_HINT = """
cutie_cpp 需要 CUDA 运行时库，但未在系统中找到{detail}。

wheel 中已包含 libcutie.so（ONNX Runtime 与 OpenCV 已静态链接），
但 CUDA 运行时体积约 800 MB，需要单独安装。

可选的解决方式：

  1. 安装 CUDA Toolkit（推荐，需 >= 11.8）
         https://developer.nvidia.com/cuda-downloads
     安装后确认库目录在动态链接器搜索路径中：
         export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

  2. 只装运行时库（无需完整 Toolkit）
         pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12

  3. 已装 CUDA 但仍报错时，确认 libcudart.so.12 可被找到：
         ldconfig -p | grep libcudart
"""


def _describe_missing(message):
    """从 ImportError 消息中提取缺失库的描述。

    Args:
        message (str): ImportError 的消息文本。

    Returns:
        str: 形如 "：缺少 libcudart.so.12（CUDA Runtime）" 的描述；
             无法识别时返回空字符串。
    """
    match = _MISSING_LIB_PATTERN.search(message)
    if match is None:
        return ""

    lib_name = match.group(1)
    # 去掉版本号后缀以匹配已知库表，如 libcudart.so.12 → libcudart
    stem = lib_name.split(".so")[0]
    component = _CUDA_LIBS.get(stem)

    if component is None:
        return f"：缺少 {lib_name}"
    return f"：缺少 {lib_name}（{component}）"


def _add_pip_cuda_paths():
    """把 pip 安装的 nvidia-* 包的库目录加入搜索路径。

    通过 pip 安装 nvidia-cuda-runtime-cu12 时，库文件位于
    site-packages/nvidia/*/lib，不在默认搜索路径中。这里在导入扩展前
    预加载这些目录下的库，使动态链接器能解析它们。

    Returns:
        list[str]: 找到并成功预加载的库目录。
    """
    import ctypes

    loaded_dirs = []
    for site_dir in sys.path:
        nvidia_root = Path(site_dir) / "nvidia"
        if not nvidia_root.is_dir():
            continue

        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            # 预加载各库，让后续 dlopen 能直接复用已加载的符号
            for lib_path in sorted(lib_dir.glob("lib*.so.*")):
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # 单个库加载失败不影响其它库，交由后续导入报错
                    continue
            loaded_dirs.append(str(lib_dir))

    return loaded_dirs


def find_provider_dir():
    """定位 ONNX Runtime 的 CUDA provider 插件目录。

    搜索顺序：
        1. CUTIE_ORT_PROVIDER_DIR 环境变量
        2. pip 安装的 onnxruntime / onnxruntime-gpu 包的 capi 目录
        3. 已在 LD_LIBRARY_PATH 中（返回 None，交由动态链接器处理）

    Returns:
        pathlib.Path | None: provider 插件所在目录；找不到时为 None。
    """
    env_dir = os.environ.get("CUTIE_ORT_PROVIDER_DIR")
    if env_dir and (Path(env_dir) / _PROVIDER_SHARED).is_file():
        return Path(env_dir)

    # onnxruntime-gpu 把插件放在 <site-packages>/onnxruntime/capi/
    try:
        import onnxruntime

        capi_dir = Path(onnxruntime.__file__).resolve().parent / "capi"
        if (capi_dir / _PROVIDER_SHARED).is_file():
            return capi_dir
    except ImportError:
        pass

    return None


def _prepare_ort_providers():
    """让 ONNX Runtime 能找到 CUDA provider 插件。

    ONNX Runtime 用 dlopen 按需加载 provider，其查找依赖动态链接器的搜索路径。
    这里把 provider 目录加入 LD_LIBRARY_PATH，并预加载 providers_shared，
    使后续 dlopen 能直接命中。

    Note:
        LD_LIBRARY_PATH 的修改对当前进程的后续 dlopen 生效有限（glibc 在启动时
        缓存该变量），因此关键手段是 RTLD_GLOBAL 预加载，让符号提前进入进程。

    Returns:
        pathlib.Path | None: 实际使用的 provider 目录；未找到时为 None。
    """
    import ctypes

    provider_dir = find_provider_dir()
    if provider_dir is None:
        return None

    # 追加到 LD_LIBRARY_PATH，供 ORT 内部按名称 dlopen 时参考
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if str(provider_dir) not in existing.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{provider_dir}{os.pathsep}{existing}" if existing else str(provider_dir)
        )

    # 用绝对路径 RTLD_GLOBAL 预加载，使 ORT 后续按 soname dlopen 时复用已加载实例
    for lib_name in (_PROVIDER_SHARED, _PROVIDER_CUDA):
        lib_path = provider_dir / lib_name
        if not lib_path.is_file():
            continue
        try:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # 常见原因是缺 cuDNN；此处不报错，留给 ORT 给出更具体的信息
            continue

    return provider_dir


def load_core():
    """导入 C++ 扩展模块，失败时给出可操作的诊断信息。

    导入前会先准备 ONNX Runtime 的 CUDA provider 插件路径
    （见 _prepare_ort_providers），使 wheel 安装的场景无需手动设 LD_LIBRARY_PATH。

    Returns:
        module: cutie_cpp._core 扩展模块。

    Raises:
        ImportError: 扩展模块加载失败。消息中包含缺失的库名与安装建议。
    """
    _prepare_ort_providers()

    try:
        from cutie_cpp import _core

        return _core
    except ImportError as exc:
        first_error = str(exc)

        # 可能是 pip 安装的 CUDA 库不在搜索路径中，预加载后重试一次
        if any(name in first_error for name in _CUDA_LIBS):
            if _add_pip_cuda_paths():
                try:
                    from cutie_cpp import _core

                    return _core
                except ImportError:
                    pass  # 重试仍失败，走下面的统一报错

        if any(name in first_error for name in _CUDA_LIBS):
            detail = _describe_missing(first_error)
            raise ImportError(
                _INSTALL_HINT.format(detail=detail) + f"\n原始错误: {first_error}"
            ) from exc

        # 非 CUDA 相关的导入失败（如 libcutie.so 缺失、ABI 不匹配）
        package_dir = Path(__file__).resolve().parent
        raise ImportError(
            f"加载 cutie_cpp 扩展模块失败: {first_error}\n"
            f"包目录: {package_dir}\n"
            f"目录内容: {sorted(p.name for p in package_dir.glob('*.so'))}\n"
            "若从源码构建，请确认已用 -DENABLE_PYTHON=ON 编译，"
            "且 libcutie.so 与 _core*.so 位于同一目录。"
        ) from exc


def provider_help(error):
    """针对 provider 相关的错误，返回附加的安装提示。

    ONNX Runtime 在创建 CUDA session 时才 dlopen provider 插件，因此缺失它
    表现为处理器构造失败。此函数识别该情形并给出安装指引。

    Args:
        error (Exception | str): 捕获到的错误或其消息。

    Returns:
        str: 需要提示时返回以换行开头的提示文本；否则返回空字符串。
    """
    message = str(error)
    if "providers_shared" not in message and "providers_cuda" not in message:
        return ""

    provider_dir = find_provider_dir()
    if provider_dir is not None:
        # 目录找到了却仍失败，最常见的原因是缺 cuDNN
        return (
            f"\n\n已找到 provider 目录 {provider_dir}，但加载失败。"
            "\n通常是缺少 cuDNN 9，可尝试: pip install nvidia-cudnn-cu12"
        )
    return "\n" + _PROVIDER_HINT


def cuda_library_status():
    """检查 CUDA 依赖库的可用性，用于诊断。

    Returns:
        dict[str, str | None]: 库名 → 解析到的路径；未找到时值为 None。
    """
    import ctypes.util

    status = {}
    for stem in _CUDA_LIBS:
        # CUDA 12 的 soname 带主版本号，find_library 不一定命中，两种都试
        found = ctypes.util.find_library(stem[len("lib") :])
        if found is None:
            candidate = Path(f"/usr/local/cuda/lib64/{stem}.so.12")
            found = str(candidate) if candidate.exists() else None
        status[stem] = found
    return status
