"""cutie_cpp —— Cutie 视频目标分割的 Python 接口。

本库把 C++ 推理库 libcutie.so 封装为 Python API，用 numpy 数组完成
视频目标分割（VOS），无需 PyTorch 依赖。推理全程在 GPU 上进行，
每帧只有一次 H2D 上传和一次 D2H 下载。

快速上手：

    import cutie_cpp

    cutie_cpp.setup_logging()

    config = cutie_cpp.CutieConfig.base_default("share/model")
    with cutie_cpp.VideoSegmenter(config) as segmenter:
        # 首帧提供掩码，目标 ID 自动从掩码推导
        result = segmenter.step(first_frame, mask=first_mask)
        # 后续帧自动跟踪
        for frame in frames:
            result = segmenter.step(frame)
            result.index_mask  # (H, W) int32，像素值 = 目标 ID

需要 CUDA GPU 和导出好的 6 个 ONNX 子模块文件，
模型导出见 share/scripts/export_onnx.py。
"""

from cutie_cpp import _core
from cutie_cpp.config import CutieConfig, Device, LongTermConfig, ModelDims, ModelVariant
from cutie_cpp.exceptions import (
    ConfigError,
    CutieError,
    InferenceError,
    ModelNotFoundError,
)
from cutie_cpp.model_zoo import find_model_prefix, resolve_model_dir
from cutie_cpp.results import SegmentationResult
from cutie_cpp.segmenter import VideoSegmenter
from cutie_cpp.utils.logging_utils import get_logger, setup_logging
from cutie_cpp.visualize import mask_to_color, overlay_mask

# 版本号由 C++ 侧的 CUTIE_VERSION 提供，保证与 libcutie.so 一致
__version__ = _core.__version__

__all__ = [
    # 主接口
    "VideoSegmenter",
    "SegmentationResult",
    # 配置
    "CutieConfig",
    "LongTermConfig",
    "ModelDims",
    "Device",
    "ModelVariant",
    # 模型发现
    "resolve_model_dir",
    "find_model_prefix",
    # 可视化
    "overlay_mask",
    "mask_to_color",
    # 日志
    "setup_logging",
    "get_logger",
    # 异常
    "CutieError",
    "ConfigError",
    "ModelNotFoundError",
    "InferenceError",
    "__version__",
]
