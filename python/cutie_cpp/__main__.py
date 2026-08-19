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

    from cutie_cpp._loader import bundled_libraries

    print("随包运行时库:")
    libraries = bundled_libraries()
    if not libraries:
        print("  (无)")
    for name in libraries:
        size_mb = (package_dir / name).stat().st_size / 1048576
        print(f"  {size_mb:8.1f} MB  {name}")

    try:
        from cutie_cpp._loader import load_core

        core = load_core()
        print(f"扩展模块: 已加载，版本 {core.__version__}")
        return True
    except ImportError as exc:
        print(f"扩展模块: 加载失败\n{exc}")
        return False


def _print_dependency_status():
    """打印外部依赖库（CUDA、cuDNN）的解析结果。

    Returns:
        bool: 全部依赖齐全时为 True。
    """
    from cutie_cpp._loader import library_status

    all_found = True
    for group, libraries in library_status().items():
        missing = [name for name, path in libraries.items() if path is None]
        status = "齐全" if not missing else f"缺少 {len(missing)} 个"
        print(f"\n{group} 依赖 ({status}):")
        for name, path in libraries.items():
            print(f"  {name:<36} {path or '未找到'}")

        if missing:
            all_found = False
            if group == "cuDNN":
                print("  → pip install nvidia-cudnn-cu12")
            else:
                print("  → 安装 CUDA Toolkit >= 11.8，或 pip install nvidia-cuda-runtime-cu12")

    return all_found


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

    extension_ok = _print_extension_status()
    dependencies_ok = _print_dependency_status()

    if not extension_ok:
        return 1

    _print_model_status()

    if not dependencies_ok:
        print("\n注意: 部分外部依赖未找到，推理时可能失败。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
