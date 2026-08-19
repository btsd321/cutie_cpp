"""安装诊断入口。

用于快速确认安装是否可用、依赖库是否齐全、模型能否被找到：

    python -m cutie_cpp
"""

import sys
from pathlib import Path


def _print_extension_status():
    """打印扩展模块与随包二进制的状态。

    Returns:
        bool: 扩展模块可正常导入时为 True。
    """
    package_dir = Path(__file__).resolve().parent
    print(f"包目录: {package_dir}")

    shared_objects = sorted(p.name for p in package_dir.glob("*.so"))
    print(f"随包二进制: {shared_objects or '(无)'}")

    try:
        from cutie_cpp._loader import load_core

        core = load_core()
        print(f"扩展模块: 已加载，版本 {core.__version__}")
        return True
    except ImportError as exc:
        print(f"扩展模块: 加载失败\n{exc}")
        return False


def _print_cuda_status():
    """打印 CUDA 依赖库的解析结果。"""
    from cutie_cpp._loader import cuda_library_status

    print("\nCUDA 依赖库:")
    for name, path in cuda_library_status().items():
        print(f"  {name:<14} {path or '未找到'}")


def _print_model_status():
    """打印模型目录的搜索结果。"""
    from cutie_cpp.exceptions import ModelNotFoundError
    from cutie_cpp.model_zoo import ENV_MODEL_DIR, resolve_model_dir

    print("\n模型:")
    try:
        model_dir, prefix = resolve_model_dir()
        print(f"  目录 {model_dir}")
        print(f"  前缀 {prefix}")
    except ModelNotFoundError as exc:
        print(f"  未找到: {exc}")
        print(f"  可设置环境变量 {ENV_MODEL_DIR} 指定目录")


def main():
    """执行诊断并返回退出码。

    Returns:
        int: 0 表示扩展模块可用，1 表示加载失败。
    """
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    if not _print_extension_status():
        _print_cuda_status()
        return 1

    _print_cuda_status()
    _print_model_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
