"""cutie_cpp 的异常类型定义。

所有本库主动抛出的异常都继承自 CutieError，便于调用方用单个 except 兜住。
C++ 层抛出的 std::runtime_error 会被 pybind11 转成内置 RuntimeError，
由 segmenter 模块捕获后包装成 InferenceError 或 ModelNotFoundError。
"""


class CutieError(Exception):
    """cutie_cpp 所有异常的基类。"""


class ConfigError(CutieError):
    """配置参数非法。

    例如设备设为不支持的 CPU、数值参数超出有效范围、必填字段为空等。
    """


class ModelNotFoundError(CutieError):
    """模型目录或 ONNX 子模块文件缺失。"""


class InferenceError(CutieError):
    """推理过程中出错。

    包括输入数组形状/类型不合法，以及 C++ 推理层抛出的运行时错误。
    """
