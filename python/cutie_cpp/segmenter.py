"""视频目标分割的主入口。

VideoSegmenter 封装 C++ CutieProcessor，负责：
    - 配置校验与模型目录自动发现
    - numpy 数组的规范化（保证 dtype 与 C 连续，满足零拷贝要求）
    - 首帧掩码的目标 ID 推导
    - C++ 异常到 cutie_cpp 异常体系的转换

注意事项：
    - 实例是**有状态**的：内部维护跨帧的内存与目标跟踪状态，
      一个实例对应一路视频流。
    - **非线程安全**：多路并发请为每路创建独立实例。
    - 换视频时调用 reset()，不要复用带有上一段视频内存的实例。
"""

import numpy as np

from cutie_cpp import _core
from cutie_cpp.config import CutieConfig, Device
from cutie_cpp.exceptions import ConfigError, InferenceError
from cutie_cpp.model_zoo import find_model_prefix, resolve_model_dir
from cutie_cpp.results import SegmentationResult
from cutie_cpp.utils.logging_utils import get_logger


logger = get_logger(__name__)

# Python logging 等级 → C++ LogLevel 的映射阈值
_LOG_LEVEL_THRESHOLDS = (
    (10, _core.LogLevel.DEBUG),
    (20, _core.LogLevel.INFO),
    (30, _core.LogLevel.WARN),
)


def _to_native_log_level(level):
    """把 Python logging 等级数值映射到 C++ LogLevel。

    Args:
        level (int): Python logging 等级数值，如 logging.INFO。

    Returns:
        _core.LogLevel: 对应的 C++ 日志等级。
    """
    for threshold, native_level in _LOG_LEVEL_THRESHOLDS:
        if level <= threshold:
            return native_level
    return _core.LogLevel.ERROR


def _prepare_image(image):
    """把输入帧规范化为零拷贝可用的 numpy 数组。

    Args:
        image (np.ndarray): BGR 图像，形状 (H, W, 3)。

    Returns:
        np.ndarray: dtype uint8、C 连续的数组。

    Raises:
        InferenceError: 输入不是 ndarray、维度或通道数不符时抛出。
    """
    if not isinstance(image, np.ndarray):
        raise InferenceError(f"image 必须是 numpy 数组，收到 {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InferenceError(
            f"image 必须是 (H, W, 3) 的 BGR 图像，收到形状 {image.shape}"
        )
    if image.dtype != np.uint8:
        raise InferenceError(f"image 的 dtype 必须是 uint8，收到 {image.dtype}")

    # ascontiguousarray 在已连续时是零开销的直接返回，不会产生额外拷贝
    return np.ascontiguousarray(image)


def _prepare_mask(mask, image_shape):
    """把首帧掩码规范化为 int32、C 连续的数组。

    Args:
        mask (np.ndarray | None): 索引掩码，形状 (H, W)。为 None 时返回空数组。
        image_shape (tuple): 对应图像的形状，用于尺寸一致性校验。

    Returns:
        np.ndarray: dtype int32、C 连续的数组；mask 为 None 时是 (0, 0) 空数组。

    Raises:
        InferenceError: 掩码维度、尺寸不符时抛出。
    """
    if mask is None:
        return np.zeros((0, 0), dtype=np.int32)

    if not isinstance(mask, np.ndarray):
        raise InferenceError(f"mask 必须是 numpy 数组，收到 {type(mask).__name__}")
    if mask.ndim != 2:
        raise InferenceError(f"mask 必须是 (H, W) 的二维数组，收到形状 {mask.shape}")
    if mask.shape != image_shape[:2]:
        raise InferenceError(
            f"mask 尺寸 {mask.shape} 与图像尺寸 {image_shape[:2]} 不一致"
        )

    # 掩码通常是 imread 出来的 uint8，需转 int32 以匹配 C++ 的 CV_32SC1。
    # 这次转换不可避免，但掩码只在首帧传入，不影响逐帧性能。
    return np.ascontiguousarray(mask, dtype=np.int32)


class VideoSegmenter:
    """有状态的视频目标分割器。

    典型用法（首帧给掩码，后续帧自动跟踪）：

        with VideoSegmenter(config) as segmenter:
            result = segmenter.step(first_frame, mask=first_mask)
            for frame in remaining_frames:
                result = segmenter.step(frame)

    Attributes:
        config (CutieConfig): 生效的配置（模型路径已解析为绝对信息）。
        frame_index (int): 已处理的帧数，也是下一帧的序号。
    """

    def __init__(self, config=None, use_native_logger=False, log_level=None):
        """初始化分割器并加载模型。

        Args:
            config (CutieConfig | None): 推理配置。为 None 时用 base 变体默认配置
                并自动搜索模型目录。
            use_native_logger (bool): True 使用 C++ 原生 stdout 日志；
                False（默认）把 C++ 日志转发到 Python logging，与本库日志格式统一。
                高频 debug 场景下转发需反复获取 GIL，此时可设为 True 以降低开销。
            log_level (int | None): C++ 侧日志等级（Python logging 数值）。
                为 None 时沿用 cutie_cpp logger 的当前等级。

        Raises:
            ConfigError: 配置校验失败。
            ModelNotFoundError: 模型目录或 ONNX 子模块缺失。
            InferenceError: C++ 侧模型加载失败。
        """
        self.config = self._resolve_config(config)
        self._frame_index = 0
        self._closed = False

        effective_level = (
            log_level if log_level is not None else logger.getEffectiveLevel()
        )

        logger.info(
            f"初始化分割器: 模型 {self.config.model_prefix} @ {self.config.model_dir}, "
            f"设备 {self.config.device.value}:{self.config.device_id}"
        )

        try:
            self._processor = _core.NativeProcessor(
                self.config.to_native(),
                use_native_logger=use_native_logger,
                log_level=_to_native_log_level(effective_level),
            )
        except RuntimeError as exc:
            # C++ 侧 std::runtime_error 会被 pybind11 转成 RuntimeError，
            # 这里统一包装成本库的异常类型。
            raise InferenceError(f"加载模型失败: {exc}") from exc

        # 复用同一个 options 对象，避免逐帧构造
        self._options = _core.NativeStepOptions()
        logger.info("分割器初始化完成")

    @staticmethod
    def _resolve_config(config):
        """补全并校验配置。

        model_dir 为空时自动搜索，model_prefix 为空时从目录推导，
        最后执行完整校验。

        Args:
            config (CutieConfig | None): 待处理的配置。

        Returns:
            CutieConfig: 补全并校验通过的配置。

        Raises:
            ConfigError: 类型错误或校验失败。
            ModelNotFoundError: 找不到模型。
        """
        if config is None:
            config = CutieConfig.base_default()
        elif not isinstance(config, CutieConfig):
            raise ConfigError(
                f"config 必须是 CutieConfig 实例，收到 {type(config).__name__}"
            )

        if config.model_dir is None:
            model_dir, prefix = resolve_model_dir(None)
            config.model_dir = model_dir
            if not config.model_prefix:
                config.model_prefix = prefix
        elif not config.model_prefix:
            config.model_prefix = find_model_prefix(config.model_dir)

        return config.validate()

    # ─── 属性 ────────────────────────────────────────────────────────

    @property
    def frame_index(self):
        """已处理的帧数。

        Returns:
            int: 下一帧将获得的序号。
        """
        return self._frame_index

    @property
    def object_ids(self):
        """当前活跃的目标 ID 列表。

        Returns:
            list[int]: 目标 ID。
        """
        self._ensure_open()
        return list(self._processor.active_objects())

    @property
    def num_objects(self):
        """当前活跃目标数量。

        Returns:
            int: 目标个数。
        """
        self._ensure_open()
        return self._processor.num_objects()

    # ─── 推理 ────────────────────────────────────────────────────────

    def step(
        self,
        image,
        mask=None,
        object_ids=None,
        return_prob=False,
        end_of_sequence=False,
        force_permanent=False,
    ):
        """处理一帧并返回分割结果。

        首帧需同时提供 image 和 mask；后续帧只传 image。
        object_ids 省略时会从 mask 的非零像素自动推导。

        Args:
            image (np.ndarray): BGR 帧，形状 (H, W, 3)、dtype uint8。
            mask (np.ndarray | None): 首帧索引掩码，形状 (H, W)，像素值为目标 ID，
                0 为背景。后续帧传 None。
            object_ids (list[int] | None): 要跟踪的目标 ID。为 None 时从 mask 推导。
            return_prob (bool): 是否返回概率图。开销较大（1080p/3 目标约 25 MB/帧），
                默认关闭。
            end_of_sequence (bool): 标记为序列末帧，触发内存整合。
            force_permanent (bool): 强制把该帧写入永久内存。

        Returns:
            SegmentationResult: 分割结果。

        Raises:
            InferenceError: 输入不合法，或 C++ 推理层出错。
        """
        self._ensure_open()

        prepared_image = _prepare_image(image)
        prepared_mask = _prepare_mask(mask, prepared_image.shape)

        if object_ids is None:
            ids = []
        else:
            ids = [int(obj_id) for obj_id in object_ids]
            if any(obj_id == 0 for obj_id in ids):
                raise InferenceError("目标 ID 不能为 0，0 是背景的保留值")

        self._options.end = end_of_sequence
        self._options.force_permanent = force_permanent

        try:
            native_result = self._processor.step(
                prepared_image, prepared_mask, ids, self._options, return_prob
            )
        except (RuntimeError, ValueError) as exc:
            raise InferenceError(f"第 {self._frame_index} 帧推理失败: {exc}") from exc

        result = SegmentationResult(
            index_mask=native_result.index_mask,
            object_ids=list(native_result.object_ids),
            prob=native_result.prob if return_prob else None,
            frame_index=self._frame_index,
        )

        logger.debug(
            f"第 {self._frame_index} 帧完成: {result.num_objects} 个目标 "
            f"{result.object_ids}, 掩码 {result.shape}"
        )
        self._frame_index += 1
        return result

    # ─── 目标与内存管理 ──────────────────────────────────────────────

    def delete_objects(self, object_ids):
        """停止跟踪指定目标并释放其内存。

        Args:
            object_ids (list[int]): 要删除的目标 ID。

        Raises:
            InferenceError: C++ 侧操作失败时抛出。
        """
        self._ensure_open()
        ids = [int(obj_id) for obj_id in object_ids]
        try:
            self._processor.delete_objects(ids)
        except RuntimeError as exc:
            raise InferenceError(f"删除目标 {ids} 失败: {exc}") from exc
        logger.info(f"已停止跟踪目标 {ids}，剩余 {self.num_objects} 个")

    def clear_memory(self):
        """清空全部内存（工作 / 长期 / 感知）。

        效果等同于开始一段新视频，但保留已加载的模型。
        """
        self._ensure_open()
        self._processor.clear_memory()
        logger.debug("已清空全部内存")

    def clear_non_permanent_memory(self):
        """清空非永久内存（工作 + 长期），保留感知内存。

        适用于视频内部发生场景切换的情况。
        """
        self._ensure_open()
        self._processor.clear_non_permanent_memory()
        logger.debug("已清空非永久内存")

    def clear_sensory_memory(self):
        """清空感知内存（每个目标的短期视觉上下文）。

        适用于目标外观发生剧烈变化的情况。
        """
        self._ensure_open()
        self._processor.clear_sensory_memory()
        logger.debug("已清空感知内存")

    def reset(self):
        """重置为可处理新视频的初始状态。

        清空全部内存、停止跟踪所有目标并把帧计数归零，模型保持加载状态，
        因此比重建实例快得多。

        Note:
            C++ 的 clear_memory() 只重建内存管理器，不会清空目标列表
            （见 src/core/inference_core.cpp 的 InferenceCore::clear_memory）。
            换视频意味着目标也要重新指定，所以这里额外删除全部目标。
        """
        self._ensure_open()

        active = self._processor.active_objects()
        if active:
            self._processor.delete_objects(list(active))
        self._processor.clear_memory()
        self._frame_index = 0
        logger.info("分割器已重置，可处理新视频")

    # ─── 生命周期 ────────────────────────────────────────────────────

    def close(self):
        """释放 C++ 处理器及其 GPU 资源。

        close 之后任何推理调用都会抛出 InferenceError。可重复调用。
        """
        if self._closed:
            return
        self._processor = None
        self._closed = True
        logger.info(f"分割器已关闭，共处理 {self._frame_index} 帧")

    def _ensure_open(self):
        """检查分割器尚未关闭。

        Raises:
            InferenceError: 分割器已关闭时抛出。
        """
        if self._closed:
            raise InferenceError("分割器已关闭，请重新创建实例")

    def __enter__(self):
        """进入上下文管理器。

        Returns:
            VideoSegmenter: 自身。
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出上下文管理器时释放资源。

        Returns:
            bool: 始终为 False，不吞掉异常。
        """
        self.close()
        return False

    def __repr__(self):
        """返回便于调试的字符串表示。

        Returns:
            str: 含模型、帧数与目标数的描述。
        """
        state = "closed" if self._closed else f"{self.num_objects} objects"
        return (
            f"VideoSegmenter(model={self.config.model_prefix!r}, "
            f"frames={self._frame_index}, {state})"
        )
