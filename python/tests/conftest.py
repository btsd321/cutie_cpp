"""pytest 共享配置与 fixture。

需要 GPU 和真实模型的测试统一打 @pytest.mark.gpu 标记，
在缺少条件的环境中自动跳过，保证 `pytest -m "not gpu"` 能在任何机器上跑通。
"""

import pytest

import cutie_cpp
from cutie_cpp.exceptions import ModelNotFoundError


def pytest_configure(config):
    """注册自定义标记，避免 --strict-markers 下报未知标记。

    Args:
        config (pytest.Config): pytest 配置对象。
    """
    config.addinivalue_line("markers", "gpu: 需要 CUDA GPU 与真实 ONNX 模型的测试")


@pytest.fixture(scope="session")
def model_location():
    """定位可用的 ONNX 模型目录。

    Returns:
        tuple[pathlib.Path, str]: (模型目录, 模型前缀)。

    Raises:
        pytest.skip.Exception: 找不到完整模型时跳过依赖它的测试。
    """
    try:
        return cutie_cpp.resolve_model_dir()
    except ModelNotFoundError as exc:
        pytest.skip(f"未找到 ONNX 模型，跳过 GPU 测试: {exc}")


@pytest.fixture(scope="session")
def segmenter(model_location):
    """创建整个测试会话共用的分割器。

    模型加载需数秒，因此按 session 作用域复用；各测试之间用 reset() 清状态。

    Args:
        model_location (tuple): model_location fixture 的结果。

    Yields:
        cutie_cpp.VideoSegmenter: 已加载模型的分割器。
    """
    model_dir, _ = model_location
    config = cutie_cpp.CutieConfig.base_default(model_dir)

    try:
        instance = cutie_cpp.VideoSegmenter(config)
    except cutie_cpp.CutieError as exc:
        pytest.skip(f"无法初始化分割器（可能没有可用 GPU）: {exc}")

    yield instance
    instance.close()
