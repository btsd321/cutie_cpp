# Cutie C++ 项目快速参考

## 项目概述

**Cutie C++** 是 [Cutie](https://github.com/hkchengrex/Cutie) 视频目标分割（VOS）模型的 C++17 推理库。

- **语言**: C++17
- **推理框架**: ONNX Runtime + CUDA
- **GPU 支持**: NVIDIA CUDA 11.8+
- **编译**: CMake 3.20+

## 快速开始

### 1. 构建项目

```bash
cd /home/lixinlong/Project/cutie_cpp

# 默认构建（ONNX Runtime 后端）
bash build.sh --vcpkg-root ./vcpkg/

# 启用 TensorRT 后端
bash build.sh --enable-tensorrt --vcpkg-root ./vcpkg/

# Debug 构建
bash build.sh --debug --vcpkg-root ./vcpkg/
```

### 2. 运行演示

```bash
./build/demo_basic --video input.mp4 --mask first_mask.png [--visualize] [--save-video]
```

### 3. 生成 Doxygen 文档

```bash
doxygen Doxyfile
# 输出到 share/docs/html/index.html
```

## 核心 API

### 基本用法（CPU 路径）

```cpp
#include "cutie/cutie.h"

using namespace cutie::cv::segmentation;

// 1. 创建配置
CutieConfig config = CutieConfig::base_default("./models/");
config.max_internal_size = 480;

// 2. 创建处理器
CutieProcessor processor(config);

// 3. 处理第一帧（提供初始掩码）
cv::Mat frame = cv::imread("frame_0.jpg");
cv::Mat mask = cv::imread("mask_0.png", cv::IMREAD_GRAYSCALE);
std::vector<ObjectId> objects = {1, 2};  // 追踪 2 个对象

auto result = processor.step(frame, mask, objects);
// result.index_mask: H×W 矩阵，像素值 = ObjectId
// result.object_ids: {1, 2}

// 4. 处理后续帧（无需掩码）
for (int i = 1; i < num_frames; ++i) {
    frame = cv::imread(fmt::format("frame_{}.jpg", i));
    result = processor.step(frame);
    // 处理结果...
}
```

### GPU 路径（零拷贝）

```cpp
// 上传帧到 GPU
cv::cuda::GpuMat gpu_frame, gpu_mask;
gpu_frame.upload(frame);
gpu_mask.upload(mask);

// GPU 推理（所有数据留在 GPU）
auto gpu_result = processor.step_gpu(gpu_frame, gpu_mask, objects);

// 按需下载到 CPU
auto cpu_result = gpu_result.download();
```

## 项目结构

```
include/cutie/
├── core/                    # 核心推理和内存管理
│   ├── processor.h         # 主 API 入口
│   ├── inference_core.h    # 推理流程编排
│   ├── memory_manager.h    # 三层内存系统
│   ├── kv_memory_store.h   # KV 内存存储
│   └── object_manager.h    # 对象 ID 映射
├── ort/
│   ├── core/               # ONNX Runtime 和 GPU 操作
│   │   ├── gpu_memory.h    # GPU 内存分配器
│   │   ├── gpu_tensor_ops.h # GPU 张量操作
│   │   └── ort_handler.h   # ORT 会话管理
│   └── cv/
│       └── ort_cutie.h     # 6 个子模块包装器
├── types.h                 # 核心数据类型
└── cutie.h                 # 主头文件

src/
├── core/                   # 核心实现
├── ort/                    # ONNX Runtime 实现
└── utils.cpp               # 工具函数
```

## 关键概念

### 推理流程

6 个 ONNX 子模块按顺序调用：

```
pixel_encoder → key_projection → mask_encoder → 
pixel_fuser → object_transformer → mask_decoder
```

### 三层内存系统

1. **工作内存** (Working Memory)
   - 最近 N 帧的 FIFO 缓冲区
   - 快速访问，用于短期追踪

2. **长期内存** (Long-Term Memory)
   - 可选的基于原型的压缩记忆
   - 用于长视频序列

3. **感知内存** (Sensory Memory)
   - 每个对象的 GRU 隐藏状态
   - 每帧更新

### 对象管理

- **ObjectId**: 用户分配的对象标识符（1, 2, 3, ...）
- **0**: 背景（保留）
- 对象 ID 在整个视频序列中保持不变

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_internal_size` | 480 | 短边限制（像素） |
| `mem_every` | 5 | 每 N 帧添加到内存 |
| `top_k` | 30 | 内存读取的 Top-K 相似度 |
| `max_mem_frames` | 5 | 工作内存帧数 |
| `stagger_updates` | 5 | 内存更新分散帧数 |

## 性能优化

### 1. 使用 GPU 路径
```cpp
// ✅ 推荐：全 GPU 流程
auto gpu_result = processor.step_gpu(gpu_frame);

// ❌ 避免：频繁 CPU↔GPU 转移
auto cpu_result = processor.step(frame);
```

### 2. 批量处理
```cpp
// ✅ 推荐：批量下载结果
std::vector<GpuCutieMask> gpu_results;
for (int i = 0; i < num_frames; ++i) {
    gpu_results.push_back(processor.step_gpu(frames[i]));
}
// 最后统一下载
for (auto& r : gpu_results) {
    auto cpu_r = r.download();
}

// ❌ 避免：逐帧下载
for (int i = 0; i < num_frames; ++i) {
    auto result = processor.step_gpu(frames[i]);
    auto cpu_result = result.download();  // 每帧都转移
}
```

### 3. 调整分辨率
```cpp
// 对于高分辨率视频，降低内部分辨率
config.max_internal_size = 320;  // 更快，精度略低

// 对于低分辨率视频，提高内部分辨率
config.max_internal_size = 640;  // 更慢，精度更高
```

## 常见问题

### Q: 如何处理多个对象？
A: 在第一帧提供所有对象的掩码和 ID 列表：
```cpp
std::vector<ObjectId> objects = {1, 2, 3};
auto result = processor.step(frame, mask, objects);
```

### Q: 如何在视频中途添加新对象？
A: 目前不支持。需要重新初始化处理器。

### Q: 如何提高精度？
A: 
- 使用 base 模型而不是 small
- 增加 `max_internal_size`
- 减少 `mem_every`（更频繁地更新内存）

### Q: 如何提高速度？
A:
- 使用 small 模型
- 减少 `max_internal_size`
- 增加 `mem_every`
- 启用 TensorRT 后端

## 文件格式

### 输入
- **视频**: MP4, AVI, MOV 等（OpenCV 支持的格式）
- **掩码**: PNG, BMP 等（单通道，像素值 = ObjectId）

### 输出
- **分割掩码**: CV_32SC1（像素值 = ObjectId）
- **概率图**: CV_32FC1（可选）

## 依赖项

- **ONNX Runtime** ≥ 1.16
- **OpenCV** ≥ 4.0
- **CUDA Toolkit** ≥ 11.8
- **cuBLAS** (CUDA Toolkit 自带)
- **linden_logger** (子项目)

## 许可证

参考项目根目录的 LICENSE 文件。

## 参考资源

- [Cutie 原始项目](https://github.com/hkchengrex/Cutie)
- [ONNX Runtime 文档](https://onnxruntime.ai/)
- [OpenCV 文档](https://docs.opencv.org/)
- [CUDA 文档](https://docs.nvidia.com/cuda/)

---

**最后更新**: 2026-04-24
