# Cutie C++ 项目快速参考

## 项目概述

**Cutie C++** 是 [Cutie](https://github.com/hkchengrex/Cutie) 视频目标分割（VOS）模型的 C++17 推理库。

- **语言**: C++17
- **推理框架**: ONNX Runtime（默认）/ TensorRT（可选）
- **GPU 支持**: NVIDIA CUDA 11.8+，cuBLAS
- **编译**: CMake 3.20+
- **特色**: 全 GPU 数据流，输入预处理 GPU 化，智能引擎缓存（TRT）

## 快速开始

### 1. 构建项目

```bash
cd /home/lixinlong/Project/cutie_cpp

# 默认构建（ONNX Runtime 后端，Release 模式）
bash build.sh --vcpkg-root ./vcpkg/

# 启用 TensorRT 后端（首次构建引擎需 10-60 秒，后续自动加载缓存）
bash build.sh --enable-tensorrt --vcpkg-root ./vcpkg/

# 同时启用两个后端
bash build.sh --enable-onnxruntime --enable-tensorrt --vcpkg-root ./vcpkg/

# Debug 构建
bash build.sh --debug --vcpkg-root ./vcpkg/

# 清理重建
bash build.sh --clean --vcpkg-root ./vcpkg/
```

### 2. 运行演示

```bash
# 基本用法（自动检测模型前缀）
./build/demo_basic <video_path> <first_frame_mask.png>
```

### 3. 生成 Doxygen 文档（可选）

```bash
doxygen Doxyfile
# 输出到 share/docs/html/index.html
```

## 核心 API

### 基本用法

```cpp
#include "cutie/cutie.h"

using namespace cutie::cv::segmentation;

// 1. 创建配置
CutieConfig config = CutieConfig::base_default("./models/");
config.model_prefix = "cutie-base-mega";  // 对应 ONNX 文件前缀
config.max_internal_size = 480;            // 输入图像短边限制
config.use_long_term = true;               // 启用长期记忆（可选）

// 2. 创建处理器（加载模型，初始化 GPU 内存）
CutieProcessor processor(config);

// 3. 处理第一帧（提供初始掩码）
cv::Mat frame = cv::imread("frame_0.jpg");
cv::Mat mask = cv::imread("mask_0.png", cv::IMREAD_GRAYSCALE);
std::vector<ObjectId> objects = {1, 2};  // 追踪 2 个对象

auto result = processor.step(frame, mask, objects);
// result.index_mask: H×W CV_32SC1，像素值 = ObjectId（0 = 背景）
// result.object_ids: {1, 2}

// 4. 处理后续帧（无需掩码，自动传播分割）
//    内部流程：CPU→GPU 上传 → GPU 预处理（resize/归一化/pad）→ 
//             特征提取 → 内存管理 → 分割解码 → GPU→CPU 下载结果
for (int i = 1; i < num_frames; ++i) {
    frame = cv::imread(fmt::format("frame_{}.jpg", i));
    result = processor.step(frame);
    // 处理结果...
}

// 5. 对象管理（可选）
processor.delete_objects({2});             // 删除对象 2
processor.clear_non_permanent_memory();   // 清除非永久记忆
```

> **性能说明**：当前 API 接受 CPU `cv::Mat` 输入，内部自动完成 GPU 上传、预处理、推理、下载。同一帧的图像会被缓存在 GPU，避免重复上传。

## 项目结构

```
include/cutie/
├── core/                    # 核心推理和内存管理
│   ├── processor.h         # 主 API 入口（CutieConfig, CutieProcessor）
│   ├── inference_core.h    # 推理流程编排
│   ├── memory_manager.h    # 三层内存系统
│   ├── kv_memory_store.h   # KV 内存存储
│   └── object_manager.h    # 对象 ID 映射
├── common/                 # GPU 公共头文件
│   ├── gpu_memory.h        # GPU 内存分配器
│   ├── gpu_tensor_ops.h    # GPU 张量操作原语
│   └── cuda_kernels.h      # CUDA kernel 声明
├── ort/                    # ONNX Runtime 后端
│   ├── ort_handler.h       # ORT 会话管理
│   └── ort_cutie.h         # 6 个子模块包装器
├── trt/                    # TensorRT 后端（可选编译）
│   ├── trt_handler.h       # 引擎管理
│   ├── trt_engine_builder.h # 引擎构建
│   └── trt_cutie.h         # 6 个子模块包装器
├── types.h                 # 核心数据类型
└── cutie.h                 # 主头文件

src/
├── common/                 # GPU 公共代码（ORT 和 TRT 共享）
│   ├── cuda_kernels.cu     # CUDA kernel 实现
│   ├── gpu_memory.cpp      # GPU 内存管理
│   ├── gpu_tensor_ops.cpp  # 张量操作（相似度、softmax、readout 等）
│   ├── gpu_image_preprocess.cu   # 图像预处理（BGR→RGB、resize、归一化）
│   ├── gpu_mask_preprocess.cu    # 掩码预处理
│   └── gpu_postprocess.cu        # 后处理
├── core/                   # 平台无关的推理逻辑
│   ├── inference_core.cpp  # 主推理循环
│   ├── memory_manager.cpp  # 内存管理
│   ├── kv_memory_store.cpp # KV 存储
│   ├── object_manager.cpp  # 对象管理
│   └── processor.cpp       # 公共 API 实现
├── ort/                    # ONNX Runtime 后端实现
│   ├── ort_handler.cpp     # 会话管理（IO Binding）
│   └── ort_cutie.cpp       # 子模块封装
└── trt/                    # TensorRT 后端实现
    ├── trt_engine_builder.cpp # 引擎构建与缓存
    ├── trt_handler.cpp        # 引擎管理
    └── trt_cutie.cpp          # 子模块封装
```

## 关键概念

### 推理流程

Cutie 模型被拆分为 **6 个 ONNX 子模块**，每帧按顺序调用：

```
pixel_encoder → key_projection → mask_encoder → 
pixel_fuser → object_transformer → mask_decoder
```

所有中间特征（f16/f8/f4、key/value、sensory、obj_v）保持在 GPU 内存中，仅最终 logits 下载到 CPU。

### 三层内存系统

1. **工作内存** (Working Memory)
   - 最近 N 帧的 FIFO 缓冲区（KV 缓存）
   - 快速访问，用于短期追踪
   - 由 `KeyValueMemoryStore` 管理

2. **长期内存** (Long-Term Memory)
   - 可选的基于压缩原型的记忆
   - 用于长视频序列
   - 通过 `config.use_long_term` 启用

3. **感知内存** (Sensory Memory)
   - 每个对象的 GRU 隐藏状态
   - 每帧更新，用于对象外观建模

所有内存数据存储在 GPU，通过 GPU 张量操作（`gpu_tensor_ops.cpp`）进行相似度计算、softmax、特征聚合等。

### 对象管理

- **ObjectId**: 用户分配的对象标识符（1, 2, 3, ...）
- **0**: 背景（保留）
- 对象 ID 在整个视频序列中保持不变
- 支持中途删除对象：`processor.delete_objects({id})`

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_prefix` | （必填） | ONNX 文件前缀（如 `cutie-base-mega`） |
| `model_dir` | （必填） | ONNX 文件所在目录 |
| `variant` | `kBase` | 模型变体（`kBase` 或 `kSmall`） |
| `device` | `kCUDA` | 推理设备（`kCPU` 或 `kCUDA`） |
| `device_id` | 0 | GPU 设备 ID |
| `max_internal_size` | 480 | 输入图像短边限制（像素），优先从模型参数读取 |
| `mem_every` | 5 | 每 N 帧添加到工作内存 |
| `top_k` | 30 | 内存读取的 Top-K 相似度 |
| `max_mem_frames` | 5 | 工作内存帧数（FIFO） |
| `stagger_updates` | 5 | 内存更新分散帧数 |
| `use_long_term` | false | 启用长期内存 |
| `single_object` | false | 单对象模式（跳过多对象融合） |

## 性能优化

### 1. 选择合适的推理后端
```cpp
// ONNX Runtime（默认）：兼容性好，使用 IO Binding 避免拷贝
bash build.sh --enable-onnxruntime --vcpkg-root ./vcpkg/

// TensorRT：性能更优，首次构建引擎需时间，后续自动加载缓存
bash build.sh --enable-tensorrt --vcpkg-root ./vcpkg/
```

### 2. 调整分辨率
```cpp
// 对于高分辨率视频，降低内部分辨率
config.max_internal_size = 320;  // 更快，精度略低

// 对于低分辨率视频，提高内部分辨率
config.max_internal_size = 640;  // 更慢，精度更高
```

### 3. 调整内存参数
```cpp
// 更频繁地更新内存（精度优先）
config.mem_every = 3;
config.max_mem_frames = 10;

// 减少内存更新频率（速度优先）
config.mem_every = 10;
config.max_mem_frames = 3;
```

### 4. 选择合适的模型变体
```cpp
// base 变体：精度优先，GPU 推理
auto config = CutieConfig::base_default("./models/");

// small 变体：速度/显存受限场景
auto config = CutieConfig::small_default("./models/");
```

## 常见问题

### Q: 如何处理多个对象？
A: 在第一帧提供所有对象的掩码和 ID 列表：
```cpp
std::vector<ObjectId> objects = {1, 2, 3};
auto result = processor.step(frame, mask, objects);
// mask 中像素值 1、2、3 分别对应三个对象
```

### Q: 如何在视频中途删除对象？
A: 使用 `delete_objects()` 方法：
```cpp
processor.delete_objects({2});  // 删除对象 2
```

### Q: 如何在视频中途添加新对象？
A: 目前不支持。需要重新初始化处理器。

### Q: 如何提高精度？
A: 
- 使用 base 模型而不是 small
- 增加 `max_internal_size`
- 减少 `mem_every`（更频繁地更新内存）
- 启用长期记忆：`config.use_long_term = true`

### Q: 如何提高速度？
A:
- 使用 small 模型
- 减少 `max_internal_size`
- 增加 `mem_every`
- 启用 TensorRT 后端（首次构建后显著加速）
- 关闭长期记忆：`config.use_long_term = false`

### Q: TensorRT 引擎缓存在哪里？
A: 引擎文件（`.engine`）缓存在模型目录中，文件名格式为 `{model_prefix}_{submodule}.engine`。首次构建需 10-60 秒，后续直接加载缓存。

## 文件格式

### 输入
- **视频**: MP4, AVI, MOV 等（OpenCV 支持的格式）
- **掩码**: PNG, BMP 等（单通道，像素值 = ObjectId）

### 输出
- **分割掩码**: CV_32SC1（像素值 = ObjectId）
- **概率图**: CV_32FC1（可选）

## 依赖项

- **CMake** ≥ 3.20
- **OpenCV** ≥ 4.0（含 CUDA 模块）
- **CUDA Toolkit** ≥ 11.8
- **cuBLAS**（CUDA Toolkit 自带，用于 GPU 端张量计算）
- **ONNX Runtime** ≥ 1.16（含 CUDA EP）— ONNX 后端
- **TensorRT** ≥ 10.0 — TRT 后端（可选）
- **linden_logger** — 专有日志库（自动从 `.ref_project/linden_logger/` 加载）

## 许可证

参考项目根目录的 LICENSE 文件。

## 参考资源

- [Cutie 原始项目](https://github.com/hkchengrex/Cutie)
- [ONNX Runtime 文档](https://onnxruntime.ai/)
- [OpenCV 文档](https://docs.opencv.org/)
- [CUDA 文档](https://docs.nvidia.com/cuda/)

## 架构亮点

### 全 GPU 数据流

- **输入预处理 GPU 化**：BGR→RGB 转换、resize、ImageNet 归一化、pad 全部在 GPU 完成
- **零拷贝中间结果**：所有特征（f16/f8/f4、key/value、sensory、obj_v）保持在 GPU
- **智能图像缓存**：同一帧的 GPU 图像张量被缓存，避免重复上传
- **IO Binding（ORT）**：所有子模块的输入输出直接绑定 GPU 内存，避免 CPU↔GPU 拷贝

### 代码复用

- **src/common/**：GPU 公共代码（CUDA kernels、内存管理、张量操作）在 ORT 和 TRT 后端之间共享
- **编译期后端选择**：CMake 选项 `ENABLE_ONNXRUNTIME` / `ENABLE_TENSORRT` 控制编译哪套源文件

### 智能引擎缓存（TRT）

- 首次构建 TensorRT 引擎后序列化到 `.engine` 文件
- 后续启动直接加载，节省 10-60 秒启动时间
- 引擎文件与 ONNX 文件放在同一目录

---

**最后更新**: 2026-07-07
