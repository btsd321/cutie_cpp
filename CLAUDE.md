# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指导。

## 项目概述

cutie-cpp 是 [Cutie](https://github.com/hkchengrex/Cutie) 视频目标分割（VOS）模型的 C++17 推理库。它将 PyTorch 推理流程移植为纯 C++ 共享库（`libcutie.so`），无需任何 Python 依赖。项目架构参考 [lite.ai.toolkit](https://github.com/xlite-dev/lite.ai.toolkit) 。

**支持的推理后端：**
- **ONNX Runtime**（默认）：含 CUDA EP，兼容性好
- **TensorRT**（可选）：高性能推理，支持 FP16/INT8 量化，智能引擎缓存

## 构建命令

> **注意**：项目现在要求 CUDA ≥ 11.8，构建时必须确保 CUDA Toolkit 已正确安装并可被 CMake 检测到。

```bash
# 默认构建（ONNX Runtime 后端，Release 模式）
bash build.sh --vcpkg-root ./vcpkg/

# Debug 构建
bash build.sh --debug --vcpkg-root ./vcpkg/

# 清理重建
bash build.sh --clean

# 指定自定义依赖路径
bash build.sh --cuda-root /usr/local/cuda --onnxruntime-root /opt/onnxruntime

# 启用 TensorRT 后端
bash build.sh --enable-tensorrt --vcpkg-root ./vcpkg/

# 同时启用两个后端
bash build.sh --enable-onnxruntime --enable-tensorrt --vcpkg-root ./vcpkg/

# 启用 Python 绑定（扩展模块就地输出到 python/cutie_cpp/）
bash build.sh --enable-python --vcpkg-root ./vcpkg/

# 构建后安装
bash install.sh
# 或：cmake --install build/
```

Python 库有**独立的构建链** `scripts/build_python.sh`，与上面的 `build.sh` 分工不同：

```bash
# 产出自包含 wheel（推荐；自动准备 .venv 依赖 + 下载官方 ONNX Runtime）
bash scripts/build_python.sh --vcpkg-root /path/to/vcpkg
.venv/bin/pip install dist/cutie_cpp-*.whl

# 只编译单架构以缩短耗时（默认 "89;120"）
bash scripts/build_python.sh --vcpkg-root /path/to/vcpkg --cuda-archs "89"

# 开发模式：CMake 就地构建 + PYTHONPATH（不能用于产出可分发 wheel）
bash build.sh --enable-python --vcpkg-root ./vcpkg/
PYTHONPATH=python .venv/bin/python -c "import cutie_cpp"

# 安装诊断（打印随包库、CUDA/cuDNN 解析结果、模型目录）
.venv/bin/python -m cutie_cpp

# 测试（gpu 标记的用例需要 GPU 与真实模型）
.venv/bin/python -m pytest -m "not gpu"
.venv/bin/python -m pytest -m gpu
```

**两条构建链的 ONNX Runtime 不同，且不可混用**：
- `build.sh`（C++）用 vcpkg 的**静态** ORT，核心库链入 `libcutie.so`
- `scripts/build_python.sh` 用官方预编译包的**动态** `libonnxruntime.so`

原因：ORT 的 CUDA provider 是运行时 `dlopen` 的插件，必须与其核心库严格配套。
静态核心 + 外部 provider 会让进程里出现两个 ORT 实例并**段错误**。
所以 wheel 必须走动态链接路径（由 CMake 选项 `CUTIE_PREFER_ORT_MODULE=ON` 控制），
provider 也随包分发。官方包的 provider 还小得多（351 MB vs vcpkg 的 919 MB）。
CUDA 运行时与 cuDNN 体积过大（各约 800/600 MB），不打包，由 `_loader.py` 提示安装。

构建产物默认输出到 `build/` 目录：`build/libcutie.so`、`build/demo_basic`。

项目目前没有测试用例，唯一可运行的示例为：
```bash
./build/demo_basic <video_path> <first_frame_mask.png>
```

## 架构

### 推理流水线

Cutie 模型被拆分为 6 个 ONNX 子模块，每帧按顺序调用：

`pixel_encoder` → `key_projection` → `mask_encoder` → `pixel_fuser` → `object_transformer` → `mask_decoder`

### 内存系统

三种内存类型负责跨帧状态维护：
- **工作内存（Working memory）**：最近 N 帧的 FIFO 缓冲区（KV 缓存）
- **长期内存（Long-term memory）**：可选的基于压缩原型的记忆
- **感知内存（Sensory memory）**：每个目标的 GRU 隐藏状态，每帧更新

### 代码组织

- **`src/common/`** — GPU 公共代码（ORT 和 TRT 后端共享）：
  - `cuda_kernels.cu` — CUDA kernel 实现（concat、slice、sigmoid、aggregate_softmax、get_similarity、top_k_softmax、one_hot、mask_merge、fill_zero、copy_d2d 等张量操作）
  - `gpu_memory.cpp` — GPU 内存分配器（Ort::Value GPU 内存管理、GpuMat 零拷贝转换、CPU↔GPU 数据传输）
  - `gpu_tensor_ops.cpp` — GPU 张量操作原语（相似度计算、softmax、记忆读出 readout、特征聚合 aggregate、sigmoid、stack/split 等，供 KeyValueMemoryStore 和 MemoryManager 调用）
  - `gpu_image_preprocess.cu` — GPU 图像预处理（BGR→RGB、resize、归一化、pad）
  - `gpu_mask_preprocess.cu` — GPU 掩码预处理
  - `gpu_postprocess.cu` — GPU 后处理
- **`src/core/`** — 与平台无关的推理逻辑：`inference_core.cpp`（主循环）、`memory_manager.cpp`、`kv_memory_store.cpp`、`object_manager.cpp`、`processor.cpp`（公共 API 实现）
- **`src/ort/`** — ONNX Runtime 后端：`ort_handler.cpp`（会话管理）、`ort_cutie.cpp`（子模块封装）
- **`src/trt/`** — TensorRT 后端：`trt_engine_builder.cpp`（引擎构建）、`trt_handler.cpp`（引擎管理）、`trt_cutie.cpp`（子模块封装）
- **`src/python/`** — pybind11 绑定层（`ENABLE_PYTHON=ON` 时编译为 `cutie_cpp._core`）：
  - `module.cpp` — 模块入口，调用各注册函数
  - `py_convert.h/.cpp` — numpy ↔ cv::Mat 零拷贝转换、GPU 张量单次 D2H 到 numpy
  - `py_config.cpp` — `CutieConfig` / `StepOptions` / 枚举的薄映射（不做校验）
  - `py_processor.cpp` — `CutieProcessor` 绑定，推理时释放 GIL
  - `py_logger.h/.cpp` — C++ 日志转发到 Python `logging`
- **`python/cutie_cpp/`** — Python 包：`segmenter.py`（`VideoSegmenter` 主类）、
  `config.py`（dataclass 配置 + 校验 + YAML）、`results.py`、`model_zoo.py`（模型自动发现）、
  `visualize.py`、`exceptions.py`、`utils/logging_utils.py`
- **`python/examples/`** — Python 示例，**`python/tests/`** — pytest 测试
- **`include/cutie/`** — 公共 API 头文件，入口为 `cutie.h`
- **`include/cutie/core/processor.h`** — `CutieConfig` 结构体和 `CutieProcessor` 类声明

### 关键设计模式

- **PIMPL**：`CutieProcessor` 使用 `std::unique_ptr<Impl>` 隐藏内部实现
- **命名空间别名**：公共 API 位于 `cutie::cv::segmentation::CutieProcessor`（lite.ai.toolkit 规范）
- **编译期后端选择**：CMake 选项 `ENABLE_ONNXRUNTIME` / `ENABLE_TENSORRT` 控制编译哪套源文件，由 `#ifdef ENABLE_*` 预处理宏保护
- **代码复用**：GPU 公共代码（CUDA kernels、内存管理、张量操作）在 `src/common/` 中共享，ORT 和 TRT 后端均可使用
- **`CutieProcessor` 有状态且非线程安全** — 每路视频流使用独立实例
- **IO Binding（ORT）**：`OrtCutie` 使用 ONNX Runtime IO Binding API，所有子模块的输入输出直接绑定 GPU 内存，避免 CPU↔GPU 数据拷贝
- **智能引擎缓存（TRT）**：`TrtEngineBuilder` 首次构建引擎后序列化到 `.engine` 文件，后续直接加载（节省 10-60 秒启动时间）
- **全 GPU 数据流**：输入图像 CPU→GPU 上传后，所有中间特征和内存数据（f16/f8/f4/pix_feat、KV 记忆、sensory memory、obj_v）保持在 GPU，仅最终 logits 在输出阶段下载到 CPU
- **Python 零拷贝约定**：绑定层每帧只有 1 次 H2D + 1 次 D2H。numpy 输入用
  `cv::Mat(h, w, type, ptr)` 借用缓冲区（不拷贝），输出用 `cudaMemcpy2D` 直接写入
  预分配的 numpy 缓冲区。**不要复用 `GpuCutieMask::download()`**——它会自行分配
  `cv::Mat`，导致二次拷贝。`GpuMat` 是 pitched 内存，D2H 必须用 `cudaMemcpy2D`
- **Python 侧校验前置**：`Device::kCPU` 在 C++ 会直接抛异常（`src/ort/cv/ort_cutie.cpp`），
  故 Python 的 `CutieConfig.validate()` 提前拦截并给出原因。`py_config.cpp` 只做薄映射
- **输入预处理 GPU 化**：`GpuMemoryAllocator::preprocess_image_gpu()` 将 BGR→RGB 转换、resize、ImageNet 归一化、pad 全部在 GPU 上完成（`cv::cuda::cvtColor` / `cv::cuda::resize` / `cv::cuda::copyMakeBorder`）；resize 目标尺寸优先从模型参数读取（`model_h_` / `model_w_`），若模型为动态分辨率（`model_h_` ≤ 0）则回退到 `max_internal_size` 等比缩放逻辑；`InferenceCore` 缓存 GPU 图像张量（`cached_image_gpu_`），避免同一帧重复上传；`ensure_features()` 和 `add_memory()` 不再接收 CPU `image_blob`

### 依赖项

- **linden_logger**：日志库，若系统未安装则从 `.ref_project/linden_logger/` 作为 CMake 子目录加载
- **OpenCV（≥ 4.0）**：图像读写与 cv::Mat 操作
- **ONNX Runtime（≥ 1.16）**：主推理后端
- **CUDA Toolkit（≥ 11.8）**：GPU 加速
- **cuBLAS**：CUDA 矩阵运算库（CUDA Toolkit 自带，用于 GPU 端张量计算）
- **TensorRT（≥ 10.0）**：可选的高性能后端

## 编程规范
1. 所有cpp、hpp文件开头要描述文件作用、注意事项(Doxygen风格没有可不写)
2. 所有函数要配置Doxygen注释
3. 注释使用中文，关键代码块前、头文件中的函数成员变量、关键宏等都要加注释
4. cpp日志使用[专有日志库](.ref_project/linden_logger)，禁止直接用std::cout打印, 关键路径要加debug或info打印，高频用debug，低频用info
5. 代码风格见[格式化文件](./.clang-format)
6. 模块化开发，不要一个文件、一个函数写所有，开发功能之前先规划合适的接口和存放位置
7. c++17风格

## 模型文件

ONNX 模型通过 `share/scripts/export_onnx.py` 从 `.pth` 权重导出（需要 Python + PyTorch）。每套权重生成 6 个 `.onnx` 文件，命名格式为 `{model_prefix}_{submodule}.onnx`，存放于 `share/model/`。C++ 代码通过 `model_prefix`（如 `cutie-base-mega`）定位对应的子模块文件。

## 语言

项目文档、注释及提交信息均使用中文编写。

## 本地参考项目
[Cutie](.ref_project/Cutie) 对应远程仓库https://github.com/hkchengrex/Cutie
[linden_logger](.ref_project/linden_logger)
