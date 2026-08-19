"""推理配置的 dataclass 定义、校验与序列化。

本模块是 C++ CutieConfig 的 Python 镜像，额外提供：
    - 参数校验（validate），在进入 C++ 之前拦截错误配置
    - YAML / dict 双向序列化，便于把配置外置到文件
    - base / small 变体的预设工厂方法

注意：当前推理后端只支持 CUDA。Device.CPU 保留以对齐 C++ 枚举，
但 validate() 会直接拒绝，因为 C++ 侧会在构造时抛异常（见 src/ort/cv/ort_cutie.cpp）。
"""

import dataclasses
import enum
from dataclasses import dataclass, field
from pathlib import Path

from cutie_cpp import _core
from cutie_cpp.exceptions import ConfigError
from cutie_cpp.model_zoo import missing_submodules
from cutie_cpp.utils.logging_utils import get_logger


logger = get_logger(__name__)


class Device(enum.Enum):
    """计算设备。"""

    CPU = "cpu"
    CUDA = "cuda"


class ModelVariant(enum.Enum):
    """模型架构变体。"""

    BASE = "base"
    SMALL = "small"


# Python 枚举 → C++ 枚举的映射
_DEVICE_TO_NATIVE = {
    Device.CPU: _core.Device.CPU,
    Device.CUDA: _core.Device.CUDA,
}

_VARIANT_TO_NATIVE = {
    ModelVariant.BASE: _core.ModelVariant.BASE,
    ModelVariant.SMALL: _core.ModelVariant.SMALL,
}


@dataclass
class LongTermConfig:
    """长期内存整合参数。

    长期内存用基于原型的压缩存储较早的帧，抑制长视频的内存增长。
    仅在 CutieConfig.use_long_term 为 True 时生效。

    Attributes:
        count_usage (bool): 是否统计使用频次用于原型选择。
        max_mem_frames (int): 触发整合前的最大帧数。
        min_mem_frames (int): 整合后保留的最小帧数。
        num_prototypes (int): 原型聚类数量。
        max_num_tokens (int): 长期内存的最大 token 数。
        buffer_tokens (int): 整合时的缓冲 token 数。
    """

    count_usage: bool = True
    max_mem_frames: int = 10
    min_mem_frames: int = 5
    num_prototypes: int = 128
    max_num_tokens: int = 10000
    buffer_tokens: int = 2000


@dataclass
class ModelDims:
    """模型架构维度。

    由 base_default / small_default 自动填充，只有使用自定义权重时才需要改。

    Attributes:
        key_dim (int): key 投影维度。
        value_dim (int): value 维度。
        sensory_dim (int): 感知内存维度。
        pixel_dim (int): 像素特征维度。
        f16_dim (int): 1/16 分辨率特征维度。
        f8_dim (int): 1/8 分辨率特征维度。
        f4_dim (int): 1/4 分辨率特征维度。
        num_queries (int): object transformer 的 query 数量。
    """

    key_dim: int = 64
    value_dim: int = 256
    sensory_dim: int = 256
    pixel_dim: int = 256
    f16_dim: int = 1024
    f8_dim: int = 512
    f4_dim: int = 256
    num_queries: int = 16


@dataclass
class CutieConfig:
    """Cutie 视频目标分割的完整推理配置。

    推荐用 base_default() / small_default() 创建后再按需调整字段，
    而不是手工填充全部参数。

    Attributes:
        model_dir (Path | None): 存放 6 个 ONNX 子模块的目录。为 None 时由
            VideoSegmenter 自动搜索（见 model_zoo.resolve_model_dir）。
        model_prefix (str): ONNX 文件名前缀，如 "cutie-base-mega"。
            为空时从 model_dir 自动推导。
        variant (ModelVariant): 模型变体，影响 ModelDims 默认值。
        device (Device): 计算设备。当前只支持 CUDA。
        device_id (int): GPU 设备序号，多卡时使用。
        single_object (bool): 单目标模式优化开关。
        max_internal_size (int): 动态分辨率下短边像素上限。仅在模型为动态
            分辨率时生效；固定分辨率模型会优先使用模型自身的输入尺寸。
        mem_every (int): 每 N 帧写入一次内存。值越大内存增长越慢。
        top_k (int): 内存读取时的 top-K 亲和度选择。
        chunk_size (int): transformer 分块大小，-1 表示不分块。
        stagger_updates (int): 把内存更新分摊到 N 帧，削平算力峰值。
        max_mem_frames (int): 工作内存的最大帧数（FIFO）。
        use_long_term (bool): 是否启用长期内存。
        long_term (LongTermConfig): 长期内存参数。
        model (ModelDims): 模型架构维度。
    """

    model_dir: Path = None
    model_prefix: str = ""
    variant: ModelVariant = ModelVariant.BASE

    device: Device = Device.CUDA
    device_id: int = 0
    single_object: bool = False

    max_internal_size: int = 480
    mem_every: int = 5
    top_k: int = 30
    chunk_size: int = -1
    stagger_updates: int = 5

    max_mem_frames: int = 5
    use_long_term: bool = False
    long_term: LongTermConfig = field(default_factory=LongTermConfig)
    model: ModelDims = field(default_factory=ModelDims)

    @classmethod
    def base_default(cls, model_dir=None, **overrides):
        """创建 base 变体的默认配置。

        Args:
            model_dir (str | Path | None): ONNX 模型目录。为 None 时后续自动搜索。
            **overrides: 要覆盖的字段，如 mem_every=3。

        Returns:
            CutieConfig: base 变体配置。

        Raises:
            ConfigError: overrides 中包含未知字段时抛出。
        """
        config = cls(
            model_dir=Path(model_dir) if model_dir is not None else None,
            variant=ModelVariant.BASE,
            model=ModelDims(
                key_dim=64,
                value_dim=256,
                sensory_dim=256,
                pixel_dim=256,
                f16_dim=1024,
                f8_dim=512,
                f4_dim=256,
                num_queries=16,
            ),
        )
        config._apply_overrides(overrides)
        return config

    @classmethod
    def small_default(cls, model_dir=None, **overrides):
        """创建 small 变体的默认配置。

        Args:
            model_dir (str | Path | None): ONNX 模型目录。为 None 时后续自动搜索。
            **overrides: 要覆盖的字段。

        Returns:
            CutieConfig: small 变体配置。

        Raises:
            ConfigError: overrides 中包含未知字段时抛出。
        """
        config = cls(
            model_dir=Path(model_dir) if model_dir is not None else None,
            variant=ModelVariant.SMALL,
        )
        config._apply_overrides(overrides)
        return config

    def _apply_overrides(self, overrides):
        """把关键字覆盖项写入配置。

        Args:
            overrides (dict): 字段名 → 取值。

        Raises:
            ConfigError: 存在未知字段时抛出。
        """
        valid_names = {f.name for f in dataclasses.fields(self)}
        for key, value in overrides.items():
            if key not in valid_names:
                raise ConfigError(
                    f"未知配置字段 {key!r}，可用字段: {sorted(valid_names)}"
                )
            setattr(self, key, value)

    # ─── 校验 ────────────────────────────────────────────────────────

    def validate(self, check_files=True):
        """校验配置的合法性。

        在构造 C++ 处理器之前调用，把配置错误拦在 Python 侧，
        以便给出比 C++ 异常更具体的提示。

        Args:
            check_files (bool): 是否检查模型目录与 6 个 ONNX 文件是否存在。
                单元测试中可关掉以免依赖真实模型。

        Returns:
            CutieConfig: 返回自身，便于链式调用。

        Raises:
            ConfigError: 任一参数非法时抛出，错误信息说明具体原因。
        """
        # 当前后端只支持 CUDA：C++ 侧 OrtCutie 构造函数在非 CUDA 时直接抛异常，
        # 这里提前拦住并说明原因。
        if self.device is not Device.CUDA:
            raise ConfigError(
                f"当前推理后端只支持 Device.CUDA，收到 {self.device}。"
                "ONNX Runtime 后端的所有中间张量都在 GPU 上，没有 CPU 实现路径。"
            )
        if self.device_id < 0:
            raise ConfigError(f"device_id 必须 >= 0，收到 {self.device_id}")

        # 数值范围校验：这些参数若为非正值会在 C++ 侧造成越界或死循环
        positive_fields = {
            "max_internal_size": self.max_internal_size,
            "mem_every": self.mem_every,
            "top_k": self.top_k,
            "stagger_updates": self.stagger_updates,
            "max_mem_frames": self.max_mem_frames,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ConfigError(f"{name} 必须为正整数，收到 {value}")

        if self.chunk_size == 0 or self.chunk_size < -1:
            raise ConfigError(
                f"chunk_size 必须为正整数或 -1（不分块），收到 {self.chunk_size}"
            )

        if self.use_long_term:
            long_term = self.long_term
            if long_term.min_mem_frames > long_term.max_mem_frames:
                raise ConfigError(
                    f"long_term.min_mem_frames ({long_term.min_mem_frames}) "
                    f"不能大于 max_mem_frames ({long_term.max_mem_frames})"
                )
            if long_term.buffer_tokens >= long_term.max_num_tokens:
                raise ConfigError(
                    f"long_term.buffer_tokens ({long_term.buffer_tokens}) 必须小于 "
                    f"max_num_tokens ({long_term.max_num_tokens})"
                )
            if long_term.num_prototypes <= 0:
                raise ConfigError(
                    f"long_term.num_prototypes 必须为正整数，收到 {long_term.num_prototypes}"
                )

        if check_files:
            self._validate_model_files()

        return self

    def _validate_model_files(self):
        """检查模型目录与 6 个 ONNX 子模块文件。

        Raises:
            ConfigError: 目录未设置、不存在，或子模块文件缺失时抛出。
        """
        if self.model_dir is None:
            raise ConfigError("model_dir 未设置。请显式指定，或交由 VideoSegmenter 自动搜索")

        model_dir = Path(self.model_dir)
        if not model_dir.is_dir():
            raise ConfigError(f"模型目录不存在: {model_dir}")

        if not self.model_prefix:
            raise ConfigError(
                "model_prefix 未设置。可用 model_zoo.find_model_prefix() 自动推导"
            )

        missing = missing_submodules(model_dir, self.model_prefix)
        if missing:
            raise ConfigError(
                f"模型目录 {model_dir} 缺少以下子模块: {', '.join(missing)}。"
                "请先用 share/scripts/export_onnx.py 导出完整的 6 个 ONNX 文件。"
            )

    # ─── 序列化 ──────────────────────────────────────────────────────

    def to_dict(self):
        """转换为可 YAML/JSON 序列化的字典。

        Path 转为字符串，Enum 转为其 value，嵌套 dataclass 递归展开。

        Returns:
            dict: 配置字典。
        """

        def convert(value):
            if isinstance(value, enum.Enum):
                return value.value
            if isinstance(value, Path):
                return str(value)
            return value

        result = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if dataclasses.is_dataclass(value):
                result[f.name] = {
                    nested.name: convert(getattr(value, nested.name))
                    for nested in dataclasses.fields(value)
                }
            else:
                result[f.name] = convert(value)
        return result

    @classmethod
    def from_dict(cls, data):
        """从字典构造配置。

        Args:
            data (dict): 配置字典，结构与 to_dict() 的输出一致。
                未出现的字段使用默认值。

        Returns:
            CutieConfig: 构造的配置对象。

        Raises:
            ConfigError: 存在未知字段，或枚举取值非法时抛出。
        """
        payload = dict(data)
        valid_names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(payload) - valid_names
        if unknown:
            raise ConfigError(
                f"配置中存在未知字段: {sorted(unknown)}，可用字段: {sorted(valid_names)}"
            )

        if "variant" in payload and not isinstance(payload["variant"], ModelVariant):
            payload["variant"] = _parse_enum(ModelVariant, payload["variant"], "variant")
        if "device" in payload and not isinstance(payload["device"], Device):
            payload["device"] = _parse_enum(Device, payload["device"], "device")
        if payload.get("model_dir") is not None:
            payload["model_dir"] = Path(payload["model_dir"])

        for key, nested_cls in (("long_term", LongTermConfig), ("model", ModelDims)):
            if isinstance(payload.get(key), dict):
                nested_valid = {f.name for f in dataclasses.fields(nested_cls)}
                nested_unknown = set(payload[key]) - nested_valid
                if nested_unknown:
                    raise ConfigError(
                        f"{key} 中存在未知字段: {sorted(nested_unknown)}"
                    )
                payload[key] = nested_cls(**payload[key])

        return cls(**payload)

    @classmethod
    def from_yaml(cls, path):
        """从 YAML 文件加载配置。

        Args:
            path (str | Path): YAML 文件路径。

        Returns:
            CutieConfig: 构造的配置对象。

        Raises:
            ConfigError: 文件不存在、内容不是映射，或字段非法时抛出。
        """
        import yaml

        yaml_path = Path(path)
        if not yaml_path.is_file():
            raise ConfigError(f"配置文件不存在: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError(f"配置文件 {yaml_path} 的顶层结构必须是映射（key: value）")

        logger.debug(f"从 {yaml_path} 加载配置")
        return cls.from_dict(data)

    def to_yaml(self, path):
        """把配置写入 YAML 文件。

        Args:
            path (str | Path): 输出文件路径，父目录会自动创建。
        """
        import yaml

        yaml_path = Path(path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, allow_unicode=True, sort_keys=False)
        logger.info(f"配置已写入 {yaml_path}")

    # ─── 转换到 C++ ──────────────────────────────────────────────────

    def to_native(self):
        """转换为 C++ 侧的 NativeConfig。

        Returns:
            _core.NativeConfig: 供 NativeProcessor 构造使用的配置对象。
        """
        native = _core.NativeConfig()
        native.variant = _VARIANT_TO_NATIVE[self.variant]
        native.model_dir = str(self.model_dir) if self.model_dir is not None else ""
        native.model_prefix = self.model_prefix

        native.device = _DEVICE_TO_NATIVE[self.device]
        native.device_id = self.device_id
        native.single_object = self.single_object

        native.max_internal_size = self.max_internal_size
        native.mem_every = self.mem_every
        native.top_k = self.top_k
        native.chunk_size = self.chunk_size
        native.stagger_updates = self.stagger_updates

        native.max_mem_frames = self.max_mem_frames
        native.use_long_term = self.use_long_term

        # 嵌套结构体在 C++ 侧按值持有，须整体赋回才能生效
        native_long_term = native.long_term
        native_long_term.count_usage = self.long_term.count_usage
        native_long_term.max_mem_frames = self.long_term.max_mem_frames
        native_long_term.min_mem_frames = self.long_term.min_mem_frames
        native_long_term.num_prototypes = self.long_term.num_prototypes
        native_long_term.max_num_tokens = self.long_term.max_num_tokens
        native_long_term.buffer_tokens = self.long_term.buffer_tokens
        native.long_term = native_long_term

        native_model = native.model
        native_model.key_dim = self.model.key_dim
        native_model.value_dim = self.model.value_dim
        native_model.sensory_dim = self.model.sensory_dim
        native_model.pixel_dim = self.model.pixel_dim
        native_model.f16_dim = self.model.f16_dim
        native_model.f8_dim = self.model.f8_dim
        native_model.f4_dim = self.model.f4_dim
        native_model.num_queries = self.model.num_queries
        native.model = native_model

        return native


def _parse_enum(enum_cls, value, field_name):
    """把字符串解析为枚举成员，兼容大小写。

    Args:
        enum_cls (type): 目标枚举类。
        value: 待解析的取值，通常是字符串。
        field_name (str): 字段名，用于错误信息。

    Returns:
        枚举成员。

    Raises:
        ConfigError: 取值不在枚举中时抛出。
    """
    if isinstance(value, str):
        for member in enum_cls:
            if member.value == value.lower() or member.name == value.upper():
                return member
    raise ConfigError(
        f"{field_name} 取值非法: {value!r}，可选: {[m.value for m in enum_cls]}"
    )
