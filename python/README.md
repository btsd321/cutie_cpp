# cutie_cpp — Python 接口

Cutie 视频目标分割（VOS）C++ 推理库的 Python 封装。用 numpy 数组完成
逐帧目标分割，无需 PyTorch 依赖。

推理全程在 GPU 上进行，每帧只有**一次 H2D 上传**和**一次 D2H 下载**，
因此性能与 C++ 直接调用相当。

## 环境要求

- CUDA GPU（当前不支持 CPU 推理）
- CUDA Toolkit ≥ 11.8
- Python ≥ 3.8
- 已导出的 6 个 ONNX 子模块文件

## 安装

### 方式一：构建 wheel（推荐）

```bash
bash scripts/build_python.sh --vcpkg-root /path/to/vcpkg
pip install dist/cutie_cpp-*.whl
```

`scripts/build_python.sh` 是 Python 专用的构建链，与 C++ 的 `build.sh` 共用同一套
vcpkg 依赖。它会自动准备 `.venv` 依赖并产出**自包含**的 wheel：`libcutie.so`
与 ONNX Runtime 的 CUDA provider 都打包在内，安装后无需设置 `LD_LIBRARY_PATH`。

常用选项：

```bash
# 只编译单个架构以缩短编译时间（默认 "89;120"）
bash scripts/build_python.sh --vcpkg-root /path/to/vcpkg --cuda-archs "89"

# provider 不在 vcpkg 默认位置时显式指定
bash scripts/build_python.sh --vcpkg-root /path/to/vcpkg --provider-dir /path/to/lib

bash scripts/build_python.sh --help   # 全部选项
```

首次构建需编译 CUDA kernel，约 10-20 分钟。
构建目录固定为 `build/skbuild`，增量重建很快。

### 方式二：CMake 构建后用 PYTHONPATH

适合与 C++ 开发同步迭代：扩展模块会就地输出到 `python/cutie_cpp/`，
不打包运行时库，因此依赖构建目录里的 `libcutie.so`。

```bash
bash build.sh --enable-python --vcpkg-root ./vcpkg/
export PYTHONPATH=$PWD/python
python -c "import cutie_cpp; print(cutie_cpp.__version__)"
```

### 安装后自检

```bash
python -m cutie_cpp
```

会打印随包库清单、CUDA/cuDNN 依赖解析结果和模型目录，便于快速定位问题。

### 关于 wheel 体积

wheel 约 700 MB，主要由两部分构成：

| 文件 | 体积 | 说明 |
|---|---|---|
| `libonnxruntime_providers_cuda.so` | 919 MB | ORT 的 CUDA 算子，覆盖多个 GPU 架构 |
| `libcutie.so` | 234 MB | 推理库，含静态链入的 OpenCV |

provider 必须随包分发：ORT 核心已静态链入 `libcutie.so`，但 provider 是运行时
`dlopen` 的插件，且**必须与核心同源**。用 pip 的 `onnxruntime-gpu` 替代会让进程里
出现两个 ORT 实例，直接段错误。

CUDA 运行时（约 800 MB）与 cuDNN（约 600 MB）不打包，需用户自备。

## 快速上手

```python
import cv2
import numpy as np
import cutie_cpp

cutie_cpp.setup_logging()

# 模型目录省略时自动搜索
config = cutie_cpp.CutieConfig.base_default(mem_every=3)

with cutie_cpp.VideoSegmenter(config) as segmenter:
    capture = cv2.VideoCapture("video.mp4")

    # 首帧提供掩码，目标 ID 从掩码非零像素自动推导
    ok, frame = capture.read()
    mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE).astype(np.int32)
    result = segmenter.step(frame, mask=mask)

    # 后续帧自动跟踪
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = segmenter.step(frame)

        result.index_mask          # (H, W) int32，像素值 = 目标 ID
        result.object_ids          # [1, 2]
        result.binary_mask(1)      # (H, W) bool
        result.object_area(1)      # 该目标的像素数
```

## 核心 API

### VideoSegmenter

有状态的分割器，**一个实例对应一路视频流**，非线程安全。

| 方法 | 说明 |
|---|---|
| `step(image, mask=None, object_ids=None, return_prob=False, ...)` | 处理一帧，返回 `SegmentationResult` |
| `delete_objects(ids)` | 停止跟踪指定目标并释放其内存 |
| `reset()` | 清空内存与目标，用于处理新视频（模型保持加载） |
| `clear_memory()` / `clear_sensory_memory()` | 细粒度内存控制 |
| `close()` | 释放 GPU 资源。支持 `with` 语句自动调用 |

属性：`object_ids`、`num_objects`、`frame_index`、`config`。

### 输入约定

- `image`：`(H, W, 3)` **uint8 BGR**。已是 C 连续时零拷贝传入。
  float 或 4 通道输入会显式报错，不做静默转换。
- `mask`：`(H, W)` 索引掩码，像素值即目标 ID，0 为背景。仅首帧需要。
- 目标 ID 不能为 0（0 是背景保留值）。

### CutieConfig

```python
# 预设 + 覆盖
config = cutie_cpp.CutieConfig.base_default("share/model", mem_every=3, top_k=20)

# 从 YAML 加载（示例见 examples/configs/base.yaml）
config = cutie_cpp.CutieConfig.from_yaml("configs/base.yaml")

# 序列化
config.to_dict()
config.to_yaml("out.yaml")

# 校验（VideoSegmenter 构造时会自动调用）
config.validate()
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_internal_size` | 480 | 动态分辨率下的短边上限，调小可显著提速 |
| `mem_every` | 5 | 每 N 帧写入内存，越大越省显存 |
| `top_k` | 30 | 内存读取的 top-K 亲和度 |
| `max_mem_frames` | 5 | 工作内存帧数 |
| `use_long_term` | False | 长视频（数千帧）建议开启 |

### 异常

全部继承自 `CutieError`：`ConfigError`（配置非法）、
`ModelNotFoundError`（模型缺失）、`InferenceError`（输入非法或推理出错）。

### 日志

统一走标准库 `logging`，C++ 侧日志默认也转发到同一套 handler，
格式一致（含毫秒时间戳、等级、文件名行号）。

```python
cutie_cpp.setup_logging(level=logging.DEBUG, log_dir="logs")
```

高频 debug 场景下日志转发需反复获取 GIL，可改用 C++ 原生输出：

```python
segmenter = cutie_cpp.VideoSegmenter(config, use_native_logger=True)
```

## 示例

`examples/` 下有四个可直接运行的脚本（基础分割、多目标、YAML 配置、性能基准），
详见 [examples/README.md](examples/README.md)。

## 测试

```bash
pip install -e ".[test]"

pytest -m "not gpu"   # 不需要 GPU 与模型
pytest -m gpu         # 端到端，需 GPU 与真实模型
```

## 常见问题

**`ModelNotFoundError`** — 未找到 ONNX 文件。用 `--model-dir` 指定，
或设 `CUTIE_MODEL_DIR` 环境变量。模型导出见 `share/scripts/export_onnx.py`。

**`ConfigError: 当前推理后端只支持 Device.CUDA`** — ONNX Runtime 后端的
所有中间张量都在 GPU 上，没有 CPU 实现路径。

**`ImportError: libcutie.so: cannot open shared object file`** — 用 CMake
构建方式时需让运行时找到 `libcutie.so`。扩展模块已设 RPATH 指向构建目录；
若移动了文件，设 `LD_LIBRARY_PATH` 指向 `libcutie.so` 所在目录。

**推理慢** — 优先调小 `max_internal_size`；确认没有不必要地开 `return_prob`
（1080p / 3 目标每帧多下载约 25 MB）。

## 与 C++ API 的对应

| Python | C++ |
|---|---|
| `cutie_cpp.VideoSegmenter` | `cutie::cv::segmentation::CutieProcessor` |
| `cutie_cpp.CutieConfig` | `cutie::core::CutieConfig` |
| `SegmentationResult` | `cutie::types::CutieMask` |

Python 侧 `step()` 走的是 C++ 的 `step_gpu()` 全 GPU 路径，
只把最终索引掩码下载到 CPU。
