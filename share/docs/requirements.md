# Cutie-CPP 需求文档

> 版本: 0.6.0 | 更新日期: 2026-04-12

## 1. 项目概述

Cutie-CPP 是 [Cutie](https://github.com/hkchengrex/Cutie) 视频物体分割 (VOS) 模型的 C++ 推理库。参考 [lite.ai.toolkit](https://github.com/xlite-dev/lite.ai.toolkit) 的多后端架构模式，提供高性能、可扩展的推理能力。

### 1.1 目标

- 将 Python/PyTorch Cutie 推理流程移植为纯 C++ 动态库 (.so)
- 支持多后端（ONNX Runtime 优先，TensorRT 后续）
- 提供简洁的 C++17 RAII 风格 API
- 支持有状态的多帧视频推理（记忆管理、跟踪）

### 1.2 非目标

- 不含训练功能
- 不含数据集加载/评测管线
- 不含 GUI
- 不含 Python 绑定（初期）

## 2. 功能需求

### FR-01: 视频帧处理

| 项目 | 说明 |
|------|------|
| **输入** | BGR cv::Mat (CV_8UC3)，任意分辨率 |
| **输出** | `CutieMask`: index mask (CV_32SC1, 0=背景) + 活跃对象列表 + 可选概率图 |
| **处理** | BGR→RGB 转换、resize、ImageNet 归一化、pad 至 16 倍数，全部在 GPU 上完成（`GpuMemoryAllocator::preprocess_image_gpu`）；resize 目标尺寸从 ONNX 模型参数自动读取，不依赖 `max_internal_size` |

### FR-02: 首帧初始化

- 用户提供首帧图像 + index mask + 对象 ID 列表
- index mask 中每个像素值为对象 ID（0=背景）
- 对象 ID 为正整数 (int32)，不要求连续
- 支持同一首帧包含多个对象的 mask

### FR-03: 后续帧跟踪

- 仅提供图像，自动传播分割
- 基于记忆匹配机制追踪已注册对象

### FR-04: 中途添加/删除对象

- 中途提供新 mask 可注册新对象
- 通过 `delete_objects()` 删除对象并回收记忆

### FR-05: 记忆管理

| 记忆类型 | 说明 |
|----------|------|
| **工作记忆** | 最近 N 帧 (默认5)，FIFO 淘汰 |
| **长期记忆** | 可选启用，基于使用量的压缩/淘汰 |
| **感知记忆** | per-object，每帧更新的 GRU 隐状态 |

- 提供 `clear_memory()`, `clear_non_permanent_memory()`, `clear_sensory_memory()` 接口

### FR-06: 模型变体

| 变体 | 骨干网络 | ONNX 子模块数 |
|------|----------|---------------|
| **base** | ResNet50 | 6 |
| **small** | ResNet18 | 6 |

### FR-07: 推理后端

| 后端 | 优先级 | 条件编译宏 | 描述 |
|------|--------|-----------|------|
| ONNX Runtime | P0 (首先实现) | `ENABLE_ONNXRUNTIME` | GPU (CUDA EP) + CPU 回退 |
| TensorRT | P1 (后续) | `ENABLE_TENSORRT` | FP16 推理 |

## 3. 非功能需求

### NFR-01: 平台

- Linux x86_64 + NVIDIA GPU (CUDA)
- GCC ≥ 9 / Clang ≥ 10
- CMake ≥ 3.18

### NFR-02: 依赖

| 依赖 | 版本要求 | 必需/可选 |
|------|----------|----------|
| OpenCV | ≥ 4.0 | 必需 |
| ONNX Runtime | ≥ 1.16 | 可选 (ENABLE_ONNXRUNTIME) |
| CUDA Toolkit | ≥ 11.8 | 必需 (GPU 推理基础设施已集成) |
| cuBLAS | 随 CUDA Toolkit | 必需 (GPU 矩阵运算，CUDA Toolkit 自带) |
| TensorRT | ≥ 8.6 | 可选 (ENABLE_TENSORRT) |

### NFR-03: 构建产物

- 动态库: `libcutie.so`
- 头文件: `include/cutie/`
- CMake 导出: `cutieTargets.cmake`

### NFR-04: 线程安全

- 多实例隔离（每个 `CutieProcessor` 独立）
- 单实例非线程安全（不支持并发 `step()`）

### NFR-05: 性能

- 推理延迟与 Python 版对齐或更优
- 零冗余拷贝（GPU tensor 直接传递）

## 4. 架构约束

### AC-01: 参考 lite.ai.toolkit 架构模式

- 每后端独立目录: `cutie/{ort,trt}/`
- 每后端基类 Handler 管理 session 生命周期
- 条件编译 `#ifdef ENABLE_*` 选择后端
- 命名空间别名统一 API: `cutie::cv::segmentation::Cutie`

### AC-02: 有状态层 (Cutie 特有扩展)

- lite.ai.toolkit 是无状态单帧推理
- Cutie 需要在 handler 层之上增加 `core/` 状态管理层
- InferenceCore / MemoryManager / ObjectManager 管理跨帧状态

### AC-03: 模型拆分

- Python CUTIE 模型拆分为 6 个独立 ONNX 子模块
- 各子模块独立推理 session，C++ 层管理调用顺序和中间 tensor 传递
- 子模块列表: pixel_encoder, key_projection, mask_encoder, pixel_fuser, object_transformer, mask_decoder

## 5. ONNX 子模块规格

### 5.1 pixel_encoder.onnx

```
输入: image    [1, 3, H, W] float32 (RGB, ImageNet 归一化)
输出: f16      [1, C16, H/16, W/16] float32
      f8       [1, C8, H/8, W/8] float32
      f4       [1, C4, H/4, W/4] float32
      pix_feat [1, C_pix, H/16, W/16] float32

base:  C16=1024, C8=512, C4=256, C_pix=256
small: C16=512,  C8=256, C4=128, C_pix=256
```

### 5.2 key_projection.onnx

```
输入: f16        [1, C16, H/16, W/16] float32  (包含内部 pix_feat_proj)
输出: key        [1, 64, H/16, W/16] float32
      shrinkage  [1, 1, H/16, W/16] float32
      selection  [1, 64, H/16, W/16] float32
```

### 5.3 mask_encoder.onnx

```
输入: image    [1, 3, H, W] float32
      f16      [1, C16, H/16, W/16] float32
      f8       [1, C8, H/8, W/8] float32
      f4       [1, C4, H/4, W/4] float32
      sensory  [num_objects, C_sensory, H/16, W/16] float32  (C_sensory=base:512, small:256)
      masks    [num_objects, 1, H, W] float32  (soft prob)
输出: mask_value     [1, C_value, num_objects, H/16, W/16] float32 (C_value=base:512, small:256)
      new_sensory    [num_objects, C_sensory, H/16, W/16] float32
      obj_summaries  [num_objects, 1, num_queries, C_pix] float32  (num_queries=16)
```

### 5.4 pixel_fuser.onnx

```
输入: pix_feat   [1, 256, H/16, W/16] float32
      pixel      [1, 256, H/16, W/16] float32  (readout from memory)
      sensory    [1, C_sensory, H/16, W/16] float32
      last_mask  [1, num_objects, H/16, W/16] float32
输出: fused      [1, 256, H/16, W/16] float32
```

### 5.5 object_transformer.onnx

```
输入: pixel_readout [1, 256, H/16, W/16] float32
      obj_memory    [1, num_objects, num_queries, 256] float32  (obj summaries)
输出: updated_readout [1, 256, H/16, W/16] float32
```

### 5.6 mask_decoder.onnx

```
输入: f8               [1, C8, H/8, W/8] float32
      f4               [1, C4, H/4, W/4] float32
      memory_readout   [1, 256, H/16, W/16] float32 (final fused)
      sensory          [num_objects, C_sensory, H/16, W/16] float32
输出: new_sensory [num_objects, C_sensory, H/16, W/16] float32
      logits      [num_objects, 1, H/4, W/4] float32
```

## 6. 实现阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | CMake + config.h.in + types.h + utils.h + ORT 基类 | ✅ 已完成 |
| Phase 2 | ObjectManager + memory_utils + KVMemoryStore + MemoryManager | ✅ 已完成 |
| Phase 3 | ORT Cutie 子模块 (ort_cutie.h/cpp) | ✅ 已完成 |
| Phase 4 | InferenceCore + CutieProcessor + 统一头文件 | ✅ 已完成 |
| Phase 5 | 导出脚本 + 示例 + 测试 | 🔲 未开始 |

## 7. GPU 推理（已完成）

### 7.1 阶段1：GPU 内存管理基础设施（已完成）

本阶段完成了将子模块间中间张量从 CPU 搬移到 GPU 的基础组件，为后续零拷贝 GPU 推理流水线奠定基础。

**新增文件：**

| 文件 | 说明 |
|------|------|
| `include/cutie/ort/core/gpu_memory.h` | `GpuMemoryAllocator` 类声明：GPU 内存分配、CPU↔GPU 传输、`GpuMat↔Ort::Value` 零拷贝转换 |
| `include/cutie/ort/core/cuda_kernels.h` | 10 个 CUDA kernel 的函数声明 |
| `src/ort/core/cuda_kernels.cu` | CUDA kernel 实现（concat、slice、sigmoid、aggregate_softmax、get_similarity、top_k_softmax、one_hot、mask_merge、fill_zero、copy_d2d、add_inplace） |
| `src/ort/core/gpu_memory.cpp` | `GpuMemoryAllocator` 实现 |

**CMakeLists.txt 变更：**
- 添加 CUDA 语言支持（`enable_language(CUDA)`）
- 链接 cuBLAS 库（`-lcublas`）
- 设置目标 GPU 架构：Turing(75)、Ampere(80, 86)、Ada(89)、Hopper(90)

**新增依赖：**
- **cuBLAS**：CUDA 矩阵运算库，随 CUDA Toolkit ≥ 11.8 一同提供

### 7.2 完整 GPU 推理改造进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| GPU 阶段1 | GPU 内存管理基础设施（GpuMemoryAllocator + CUDA kernels） | ✅ 已完成 |
| GPU 阶段2 | OrtCutie 改造为 IO Binding 推理，所有子模块输入输出直接绑定 GPU 内存，消除子模块间 CPU↔GPU 往返拷贝 | ✅ 已完成 |
| GPU 步骤1 | gpu_tensor_ops 模块：相似度/softmax/readout/aggregate/sigmoid/stack/split 全部 GPU 化 | ✅ 已完成 |
| GPU 步骤2 | KeyValueMemoryStore GPU 化：所有 KV 缓冲区从 cv::Mat 迁移为 GPU Ort::Value | ✅ 已完成 |
| GPU 步骤3 | MemoryManager + ImageFeatureStore GPU 化：sensory/obj_v/readout/add_memory 全部 GPU 化 | ✅ 已完成 |
| GPU 步骤4 | InferenceCore GPU 数据流：CPU input → GPU upload → 全 GPU 推理 → GPU download → CPU output | ✅ 已完成 |
| GPU 步骤5 | 输入预处理 GPU 化：BGR→RGB/resize/norm/pad 全部在 GPU 完成；resize 尺寸从模型参数自动读取；缓存 GPU 图像张量避免重复上传 | ✅ 已完成 |

---

## 变更记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-04-09 | 0.1.0 | 初始版本 |
| 2026-04-09 | 0.2.0 | Phase 1-4 代码实现完成; 修正 key_projection 输入为 f16; 修正 ModelDims value_dim/sensory_dim 为 256 |
| 2026-04-12 | 0.3.0 | 新增 GPU 推理章节; GPU 阶段1（内存管理基础设施）完成; 新增 cuBLAS 依赖; CUDA 由可选改为必需 |
| 2026-04-12 | 0.4.0 | GPU 阶段2（OrtCutie IO Binding 改造）完成; 所有子模块输入输出直接绑定 GPU 内存 |
| 2026-04-12 | 0.5.0 | 全 GPU 推理改造完成（步骤1-4）; gpu_tensor_ops 模块、KeyValueMemoryStore/MemoryManager/InferenceCore 全 GPU 化 |
| 2026-04-13 | 0.6.1 | 修复 `resize_channels` 在 `INTER_LINEAR` 模式下因 ORT 连续内存 pitch 不满足 CUDA Texture 对齐要求导致的 `cudaErrorInvalidValue` 崩溃；新增 `bilinear_resize` CUDA kernel，直接对连续 GPU 内存做双线性插值，替代依赖 Texture Object 的 OpenCV 实现，彻底消除 pitch 对齐限制 |
