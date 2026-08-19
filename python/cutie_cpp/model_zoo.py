"""ONNX 模型目录与前缀的自动发现。

Cutie 被拆成 6 个 ONNX 子模块，文件名格式为 {prefix}_{submodule}.onnx。
本模块负责在常见位置找到这样一组文件，并推导出 model_prefix，
让用户不必手写完整路径。

搜索顺序：
    1. 显式传入的路径
    2. CUTIE_MODEL_DIR 环境变量
    3. 已安装包同级的 share/cutie_cpp/model/
    4. 从当前文件向上查找仓库根的 share/model/
"""

import os
from pathlib import Path

from cutie_cpp.exceptions import ModelNotFoundError
from cutie_cpp.utils.logging_utils import get_logger


logger = get_logger(__name__)

# 6 个 ONNX 子模块的名称，与 share/scripts/export_onnx.py 的导出保持一致
SUBMODULE_NAMES = (
    "pixel_encoder",
    "key_projection",
    "mask_encoder",
    "pixel_fuser",
    "object_transformer",
    "mask_decoder",
)

# 用于识别模型前缀的锚点文件后缀
_ANCHOR_SUFFIX = "_pixel_encoder.onnx"

# 环境变量名
ENV_MODEL_DIR = "CUTIE_MODEL_DIR"


def find_model_prefix(model_dir):
    """扫描目录，从锚点文件推导模型前缀。

    Args:
        model_dir (str | Path): 存放 ONNX 文件的目录。

    Returns:
        str: 模型前缀，如 "cutie-base-mega"。

    Raises:
        ModelNotFoundError: 目录不存在，或找不到 *_pixel_encoder.onnx 文件。
    """
    directory = Path(model_dir)
    if not directory.is_dir():
        raise ModelNotFoundError(f"模型目录不存在: {directory}")

    candidates = sorted(
        path.name[: -len(_ANCHOR_SUFFIX)]
        for path in directory.glob(f"*{_ANCHOR_SUFFIX}")
    )
    if not candidates:
        raise ModelNotFoundError(
            f"目录 {directory} 中没有找到 *{_ANCHOR_SUFFIX} 文件。"
            f"请先用 share/scripts/export_onnx.py 导出 ONNX 子模块。"
        )

    if len(candidates) > 1:
        logger.warning(
            f"目录 {directory} 中存在多套模型 {candidates}，使用第一个: {candidates[0]}"
        )
    return candidates[0]


def missing_submodules(model_dir, model_prefix):
    """检查 6 个 ONNX 子模块中哪些缺失。

    Args:
        model_dir (str | Path): 存放 ONNX 文件的目录。
        model_prefix (str): 模型前缀。

    Returns:
        list[str]: 缺失的文件名列表；全部齐全时返回空列表。
    """
    directory = Path(model_dir)
    missing = []
    for name in SUBMODULE_NAMES:
        filename = f"{model_prefix}_{name}.onnx"
        if not (directory / filename).is_file():
            missing.append(filename)
    return missing


def _candidate_dirs():
    """生成模型目录的候选搜索路径（按优先级排序）。

    Returns:
        list[Path]: 候选目录列表。
    """
    candidates = []

    env_dir = os.environ.get(ENV_MODEL_DIR)
    if env_dir:
        candidates.append(Path(env_dir))

    # 已安装场景：<prefix>/lib/pythonX/site-packages/cutie_cpp → <prefix>/share/cutie_cpp/model
    package_dir = Path(__file__).resolve().parent
    for parent in package_dir.parents:
        candidates.append(parent / "share" / "cutie_cpp" / "model")

    # 开发场景：从当前文件向上找含 share/model 的仓库根
    for parent in package_dir.parents:
        candidates.append(parent / "share" / "model")

    return candidates


def resolve_model_dir(model_dir=None):
    """定位一个包含完整 ONNX 子模块的模型目录。

    Args:
        model_dir (str | Path | None): 显式指定的目录。为 None 时按预设顺序自动搜索。

    Returns:
        tuple[Path, str]: (模型目录, 模型前缀)。

    Raises:
        ModelNotFoundError: 显式目录无效，或所有候选路径都找不到完整模型。
    """
    if model_dir is not None:
        directory = Path(model_dir)
        prefix = find_model_prefix(directory)
        missing = missing_submodules(directory, prefix)
        if missing:
            raise ModelNotFoundError(
                f"模型目录 {directory} 缺少子模块: {', '.join(missing)}"
            )
        logger.debug(f"使用显式指定的模型目录: {directory} (前缀 {prefix})")
        return directory, prefix

    for candidate in _candidate_dirs():
        if not candidate.is_dir():
            continue
        try:
            prefix = find_model_prefix(candidate)
        except ModelNotFoundError:
            continue
        if missing_submodules(candidate, prefix):
            continue
        logger.info(f"自动找到模型目录: {candidate} (前缀 {prefix})")
        return candidate, prefix

    raise ModelNotFoundError(
        "未能自动定位 ONNX 模型目录。请显式传入 model_dir，"
        f"或设置环境变量 {ENV_MODEL_DIR}。"
    )
