"""配置模块的单元测试。

这些测试不需要 GPU，也不需要真实模型文件（用 check_files=False 或 tmp_path 伪造）。
"""

import dataclasses

import pytest

import cutie_cpp
from cutie_cpp.config import CutieConfig, Device, LongTermConfig, ModelVariant


def make_fake_model_dir(tmp_path, prefix="cutie-base-mega", skip=()):
    """在临时目录中创建伪造的 ONNX 文件。

    Args:
        tmp_path (pathlib.Path): pytest 提供的临时目录。
        prefix (str): 模型前缀。
        skip (tuple[str, ...]): 要故意跳过（不创建）的子模块名。

    Returns:
        pathlib.Path: 伪造的模型目录。
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    for name in cutie_cpp.model_zoo.SUBMODULE_NAMES:
        if name in skip:
            continue
        (model_dir / f"{prefix}_{name}.onnx").write_bytes(b"fake")
    return model_dir


class TestDefaults:
    """预设工厂方法的测试。"""

    def test_base_default_sets_variant_and_dims(self, tmp_path):
        """base_default 应设置 BASE 变体与对应的模型维度。"""
        config = CutieConfig.base_default(tmp_path)
        assert config.variant is ModelVariant.BASE
        assert config.model.f16_dim == 1024
        assert config.device is Device.CUDA

    def test_small_default_sets_variant(self, tmp_path):
        """small_default 应设置 SMALL 变体。"""
        config = CutieConfig.small_default(tmp_path)
        assert config.variant is ModelVariant.SMALL

    def test_overrides_applied(self, tmp_path):
        """关键字参数应覆盖默认值。"""
        config = CutieConfig.base_default(tmp_path, mem_every=3, top_k=10)
        assert config.mem_every == 3
        assert config.top_k == 10

    def test_unknown_override_rejected(self, tmp_path):
        """未知字段应报 ConfigError 而非静默忽略。"""
        with pytest.raises(cutie_cpp.ConfigError, match="未知配置字段"):
            CutieConfig.base_default(tmp_path, no_such_field=1)

    def test_model_dir_none_allowed(self):
        """model_dir 可以为 None，留待后续自动搜索。"""
        assert CutieConfig.base_default().model_dir is None


class TestValidate:
    """校验逻辑的测试。"""

    def test_cpu_device_rejected(self, tmp_path):
        """当前后端只支持 CUDA，CPU 应被明确拒绝。

        C++ 侧 OrtCutie 构造函数在非 CUDA 时会抛异常，
        校验层要提前拦住并说明原因。
        """
        config = CutieConfig.base_default(tmp_path, device=Device.CPU)
        with pytest.raises(cutie_cpp.ConfigError, match="只支持 Device.CUDA"):
            config.validate(check_files=False)

    def test_negative_device_id_rejected(self, tmp_path):
        """device_id 必须非负。"""
        config = CutieConfig.base_default(tmp_path, device_id=-1)
        with pytest.raises(cutie_cpp.ConfigError, match="device_id"):
            config.validate(check_files=False)

    @pytest.mark.parametrize(
        "field",
        ["max_internal_size", "mem_every", "top_k", "stagger_updates", "max_mem_frames"],
    )
    def test_non_positive_fields_rejected(self, tmp_path, field):
        """这些参数为非正值会导致 C++ 侧越界，须拦下。"""
        config = CutieConfig.base_default(tmp_path, **{field: 0})
        with pytest.raises(cutie_cpp.ConfigError, match=field):
            config.validate(check_files=False)

    def test_chunk_size_minus_one_allowed(self, tmp_path):
        """chunk_size = -1 表示不分块，是合法取值。"""
        config = CutieConfig.base_default(tmp_path, chunk_size=-1)
        config.validate(check_files=False)

    @pytest.mark.parametrize("value", [0, -2])
    def test_invalid_chunk_size_rejected(self, tmp_path, value):
        """chunk_size 只能是正整数或 -1。"""
        config = CutieConfig.base_default(tmp_path, chunk_size=value)
        with pytest.raises(cutie_cpp.ConfigError, match="chunk_size"):
            config.validate(check_files=False)

    def test_long_term_frame_bounds_checked(self, tmp_path):
        """long_term 的 min 帧数不应大于 max 帧数。"""
        config = CutieConfig.base_default(
            tmp_path,
            use_long_term=True,
            long_term=LongTermConfig(max_mem_frames=5, min_mem_frames=10),
        )
        with pytest.raises(cutie_cpp.ConfigError, match="min_mem_frames"):
            config.validate(check_files=False)

    def test_long_term_token_bounds_checked(self, tmp_path):
        """buffer_tokens 必须小于 max_num_tokens。"""
        config = CutieConfig.base_default(
            tmp_path,
            use_long_term=True,
            long_term=LongTermConfig(max_num_tokens=1000, buffer_tokens=1000),
        )
        with pytest.raises(cutie_cpp.ConfigError, match="buffer_tokens"):
            config.validate(check_files=False)

    def test_long_term_ignored_when_disabled(self, tmp_path):
        """未启用长期内存时，其参数不合法也不应报错。"""
        config = CutieConfig.base_default(
            tmp_path,
            use_long_term=False,
            long_term=LongTermConfig(max_mem_frames=5, min_mem_frames=10),
        )
        config.validate(check_files=False)

    def test_missing_model_dir_rejected(self):
        """model_dir 为 None 时文件校验应报错。"""
        config = CutieConfig.base_default()
        with pytest.raises(cutie_cpp.ConfigError, match="model_dir 未设置"):
            config.validate(check_files=True)

    def test_nonexistent_model_dir_rejected(self, tmp_path):
        """model_dir 指向不存在的目录时应报错。"""
        config = CutieConfig.base_default(tmp_path / "nope")
        config.model_prefix = "x"
        with pytest.raises(cutie_cpp.ConfigError, match="模型目录不存在"):
            config.validate(check_files=True)

    def test_complete_model_dir_accepted(self, tmp_path):
        """6 个子模块齐全时校验应通过。"""
        model_dir = make_fake_model_dir(tmp_path)
        config = CutieConfig.base_default(model_dir)
        config.model_prefix = "cutie-base-mega"
        assert config.validate(check_files=True) is config

    def test_missing_submodule_reported(self, tmp_path):
        """缺失的子模块文件名应出现在错误信息中。"""
        model_dir = make_fake_model_dir(tmp_path, skip=("mask_decoder",))
        config = CutieConfig.base_default(model_dir)
        config.model_prefix = "cutie-base-mega"
        with pytest.raises(cutie_cpp.ConfigError, match="mask_decoder"):
            config.validate(check_files=True)


class TestSerialization:
    """字典与 YAML 序列化的测试。"""

    def test_to_dict_converts_enums_and_paths(self, tmp_path):
        """枚举转 value、Path 转字符串，才能被 YAML 序列化。"""
        data = CutieConfig.base_default(tmp_path).to_dict()
        assert data["device"] == "cuda"
        assert data["variant"] == "base"
        assert isinstance(data["model_dir"], str)
        assert isinstance(data["long_term"], dict)

    def test_dict_roundtrip_preserves_values(self, tmp_path):
        """to_dict → from_dict 应还原所有字段。"""
        original = CutieConfig.base_default(
            tmp_path, mem_every=7, use_long_term=True, top_k=12
        )
        restored = CutieConfig.from_dict(original.to_dict())
        assert restored == original

    def test_yaml_roundtrip_preserves_values(self, tmp_path):
        """to_yaml → from_yaml 应还原所有字段。"""
        original = CutieConfig.small_default(tmp_path, mem_every=2)
        path = tmp_path / "cfg.yaml"
        original.to_yaml(path)
        assert CutieConfig.from_yaml(path) == original

    def test_from_dict_accepts_partial(self):
        """未出现的字段应使用默认值。"""
        config = CutieConfig.from_dict({"mem_every": 9})
        assert config.mem_every == 9
        assert config.top_k == 30  # 默认值

    def test_from_dict_rejects_unknown_field(self):
        """未知顶层字段应报错，避免配置笔误被静默忽略。"""
        with pytest.raises(cutie_cpp.ConfigError, match="未知字段"):
            CutieConfig.from_dict({"mem_evry": 9})

    def test_from_dict_rejects_unknown_nested_field(self):
        """未知嵌套字段同样应报错。"""
        with pytest.raises(cutie_cpp.ConfigError, match="long_term"):
            CutieConfig.from_dict({"long_term": {"bogus": 1}})

    def test_from_dict_rejects_bad_enum(self):
        """非法枚举取值应报错并列出可选项。"""
        with pytest.raises(cutie_cpp.ConfigError, match="device"):
            CutieConfig.from_dict({"device": "tpu"})

    @pytest.mark.parametrize("value", ["cuda", "CUDA"])
    def test_enum_parsing_is_case_insensitive(self, value):
        """枚举取值应兼容大小写写法。"""
        assert CutieConfig.from_dict({"device": value}).device is Device.CUDA

    def test_from_yaml_missing_file_raises(self, tmp_path):
        """配置文件不存在时应报 ConfigError。"""
        with pytest.raises(cutie_cpp.ConfigError):
            CutieConfig.from_yaml(tmp_path / "missing.yaml")


class TestToNative:
    """转换为 C++ NativeConfig 的测试。"""

    def test_scalar_fields_forwarded(self, tmp_path):
        """标量字段应逐一传递到 C++ 侧。"""
        config = CutieConfig.base_default(
            tmp_path, mem_every=3, top_k=11, max_internal_size=320
        )
        config.model_prefix = "cutie-base-mega"
        native = config.to_native()

        assert native.model_dir == str(tmp_path)
        assert native.model_prefix == "cutie-base-mega"
        assert native.mem_every == 3
        assert native.top_k == 11
        assert native.max_internal_size == 320

    def test_nested_structs_forwarded(self, tmp_path):
        """嵌套结构体在 C++ 侧按值持有，须确认整体赋回生效。"""
        config = CutieConfig.base_default(tmp_path)
        config.long_term.num_prototypes = 77
        config.model.num_queries = 8
        native = config.to_native()

        assert native.long_term.num_prototypes == 77
        assert native.model.num_queries == 8

    def test_enums_mapped(self, tmp_path):
        """Python 枚举应正确映射到 C++ 枚举。"""
        native = CutieConfig.small_default(tmp_path).to_native()
        assert native.variant == cutie_cpp._core.ModelVariant.SMALL
        assert native.device == cutie_cpp._core.Device.CUDA

    def test_all_dataclass_fields_covered(self, tmp_path):
        """守卫测试：新增 dataclass 字段时提醒同步更新 to_native()。"""
        config = CutieConfig.base_default(tmp_path)
        native = config.to_native()
        for field in dataclasses.fields(config):
            assert hasattr(native, field.name), (
                f"NativeConfig 缺少字段 {field.name}，请同步更新 to_native() 与 py_config.cpp"
            )
