# TensorRT 推理实现规划

## 1. 概述

本文档规划 cutie-cpp 项目的 TensorRT 后端实现，参考现有 ONNX Runtime 实现和 [Cutie 原工程](.ref_project/Cutie)。

### 1.1 目标

- 实现与 ONNX Runtime 后端功能对等的 TensorRT 推理引擎
- 支持 ONNX 模型到 TensorRT Engine 的转换和序列化
- 保持与现有 API 的兼容性（`CutieProcessor` 接口不变）
- 实现全 GPU 数据流，避免 CPU↔GPU 数据拷贝
- 支持动态 batch 和动态分辨率（可选）

### 1.2 技术栈

- **TensorRT ≥ 8.6**：推理引擎
- **CUDA ≥ 11.8**：GPU 加速
- **ONNX Parser**：从 ONNX 模型构建 TensorRT 网络
- **cuBLAS**：矩阵运算（已有依赖）
- **OpenCV CUDA**：图像预处理（已有依赖）

---

## 2. 架构设计

### 2.1 目录结构（已实现）

```
cutie-cpp/
├── include/cutie/
│   ├── common/                     # GPU 公共代码（ORT 和 TRT 共享）✅
│   │   ├── cuda_kernels.h
│   │   ├── gpu_memory.h
│   │   ├── gpu_tensor_ops.h
│   │   ├── gpu_image_preprocess.h
│   │   ├── gpu_mask_preprocess.h
│   │   ├── gpu_postprocess.h
│   │   └── gpu_buffer.h
│   ├── ort/                        # ONNX Runtime 后端
│   │   ├── core/
│   │   │   ├── ort_handler.h
│   │   │   ├── ort_utils.h
│   │   │   └── ort_config.h
│   │   └── cv/ort_cutie.h
│   └── trt/                        # TensorRT 后端 ✅
│       ├── core/
│       │   ├── trt_types.h         ✅
│       │   ├── trt_config.h        ✅
│       │   ├── trt_engine_builder.h ✅
│       │   └── trt_handler.h       ✅
│       └── cv/
│           └── trt_cutie.h         ✅
└── src/
    ├── common/                     # GPU 公共实现 ✅
    │   ├── cuda_kernels.cu
    │   ├── gpu_memory.cpp
    │   ├── gpu_tensor_ops.cpp
    │   ├── gpu_image_preprocess.cu
    │   ├── gpu_mask_preprocess.cu
    │   ├── gpu_postprocess.cu
    │   └── gpu_buffer.cpp
    ├── ort/                        # ONNX Runtime 实现
    │   ├── core/
    │   │   ├── ort_handler.cpp
    │   │   └── ort_utils.cpp
    │   └── cv/ort_cutie.cpp
    └── trt/                        # TensorRT 实现
        ├── core/
        │   ├── trt_engine_builder.cpp ✅
        │   └── trt_handler.cpp        ✅
        └── cv/
            └── trt_cutie.cpp          ⏳ 待实现
```

### 2.2 核心类设计

#### 2.2.1 TrtEngineBuilder（引擎构建器）

负责从 ONNX 模型构建 TensorRT 引擎，支持序列化和反序列化。

**关键功能：**
- ONNX 模型解析（使用 `nvonnxparser::IParser`）
- 引擎优化配置（FP16/INT8、工作空间大小、动态形状）
- 引擎序列化到文件（`.engine` 或 `.trt`）
- 引擎反序列化加载

**接口设计：**
```cpp
class TrtEngineBuilder {
public:
    // 从 ONNX 构建引擎
    std::unique_ptr<nvinfer1::ICudaEngine> build_from_onnx(
        const std::string& onnx_path,
        const BuildConfig& config);
    
    // 序列化引擎到文件
    void serialize_engine(nvinfer1::ICudaEngine* engine, 
                         const std::string& output_path);
    
    // 从文件加载引擎
    std::unique_ptr<nvinfer1::ICudaEngine> deserialize_engine(
        const std::string& engine_path);
    
private:
    std::unique_ptr<nvinfer1::IBuilder> builder_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ILogger> logger_;
};
```

#### 2.2.2 TrtHandler（引擎管理器）

管理单个 TensorRT 引擎的执行上下文，对应 `BasicOrtHandler`。

**关键功能：**
- 引擎加载和执行上下文创建
- 输入输出张量绑定（`setTensorAddress`）
- 异步推理执行（`enqueueV3`）
- 动态形状设置（`setInputShape`）

**接口设计：**
```cpp
class TrtHandler {
public:
    TrtHandler(const std::string& engine_path, int device_id);
    
    // 同步推理（内部管理 CUDA stream）
    void infer(const std::unordered_map<std::string, void*>& input_buffers,
               const std::unordered_map<std::string, void*>& output_buffers);
    
    // 异步推理（用户提供 CUDA stream）
    void infer_async(const std::unordered_map<std::string, void*>& input_buffers,
                     const std::unordered_map<std::string, void*>& output_buffers,
                     cudaStream_t stream);
    
    // 获取输入输出信息
    const std::vector<std::string>& input_names() const;
    const std::vector<std::string>& output_names() const;
    std::vector<int64_t> get_binding_shape(const std::string& name) const;
    
private:
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;
    int device_id_;
};
```

#### 2.2.3 TrtCutie（模型封装）

封装 6 个 TensorRT 子模块，对应 `OrtCutie`。

**关键功能：**
- 管理 6 个 TensorRT 引擎（pixel_encoder、key_projection 等）
- 提供与 `OrtCutie` 相同的接口（`encode_image`、`transform_key` 等）
- GPU 内存管理（复用 `GpuMemoryAllocator` 或适配）
- 张量数据流管理（全 GPU 流水线）

**接口设计：**
```cpp
class TrtCutie {
public:
    struct ImageFeatures {
        void* f16;       // GPU pointer
        void* f8;
        void* f4;
        void* pix_feat;
    };
    
    explicit TrtCutie(const core::CutieConfig& config,
                      std::shared_ptr<linden::log::ILogger> logger = nullptr);
    
    // 与 OrtCutie 相同的接口
    ImageFeatures encode_image(void* image_gpu);
    KeyFeatures transform_key(void* f16_gpu);
    MaskEncoded encode_mask(void* image, void* pix_feat, 
                           void* sensory, void* masks);
    // ... 其他子模块接口
    
    GpuMemoryAllocator& gpu_alloc() { return *gpu_alloc_; }
    
private:
    std::unique_ptr<TrtHandler> pixel_encoder_;
    std::unique_ptr<TrtHandler> key_projection_;
    std::unique_ptr<TrtHandler> mask_encoder_;
    std::unique_ptr<TrtHandler> pixel_fuser_;
    std::unique_ptr<TrtHandler> object_transformer_;
    std::unique_ptr<TrtHandler> mask_decoder_;
    
    std::unique_ptr<GpuMemoryAllocator> gpu_alloc_;
    std::shared_ptr<linden::log::ILogger> logger_;
};
```

### 2.3 GPU 内存管理

**策略：**
- **复用 ONNX Runtime 版本的 `GpuMemoryAllocator`**：该类已实现 GPU 内存分配、CPU↔GPU 传输、张量操作等功能
- **适配层**：TensorRT 使用原始 GPU 指针（`void*`），需要提供 `Ort::Value` ↔ `void*` 的转换接口
- **CUDA Kernels 复用**：`cuda_kernels.cu` 中的所有 kernel 函数可直接复用（concat、slice、sigmoid 等）

**适配接口：**
```cpp
// 在 GpuMemoryAllocator 中添加
class GpuMemoryAllocator {
public:
    // 现有接口（Ort::Value）
    Ort::Value allocate(const std::vector<int64_t>& shape);
    
    // 新增：获取 Ort::Value 的 GPU 指针（用于 TensorRT）
    void* get_device_ptr(Ort::Value& tensor);
    
    // 新增：从 GPU 指针创建 Ort::Value（用于 TensorRT 输出）
    Ort::Value wrap_device_ptr(void* ptr, const std::vector<int64_t>& shape);
};
```

---

## 3. 实现步骤

### 3.1 Phase 1：基础设施（1-2 天）✅ 已完成

**目标：** 搭建 TensorRT 后端的基础框架。

**已完成任务：**
1. ✅ 创建目录结构 `src/trt/` 和 `include/cutie/trt/`
2. ✅ 实现 `TrtEngineBuilder`：
   - ONNX 解析和引擎构建
   - 引擎序列化/反序列化
   - 日志集成（linden_logger）
   - 智能缓存机制（`get_or_build_engine`）
3. ✅ 实现 `TrtHandler`：
   - 引擎加载和执行上下文管理
   - 输入输出绑定（`setTensorAddress`）
   - 同步/异步推理接口（`infer` / `infer_async`）
   - 动态形状支持（`set_input_shape`）
4. ✅ CMake 配置：
   - 更新 `FindTensorRT.cmake`（添加 nvonnxparser 支持）
   - 更新 `CMakeLists.txt`（添加 TensorRT 源文件和链接配置）
5. ✅ 代码重构：
   - 将 GPU 公共代码移动到 `cutie/common` 目录
   - CUDA kernels、GPU 内存管理、GPU 张量操作等可在 ORT 和 TRT 后端间复用

**提交记录：**
```
commit 0aeba7e
refactor: 将 GPU 公共代码移动到 cutie/common 目录
35 files changed, 1865 insertions(+), 44 deletions(-)
```

**验证：**
- ⏳ 待实现：单元测试（加载单个 ONNX 模型，构建引擎，执行简单推理）

### 3.2 Phase 2：模型封装（2-3 天）🔄 进行中

**目标：** 实现 `TrtCutie` 类，封装 6 个子模块。

**当前状态：**
- ✅ `TrtCutie` 头文件已完成（`include/cutie/trt/cv/trt_cutie.h`）
- ⏳ 待实现：`src/trt/cv/trt_cutie.cpp`

**待完成任务：**
1. 实现 `TrtCutie` 构造函数：
   - 加载 6 个 ONNX 模型
   - 使用 `TrtEngineBuilder::get_or_build_engine()` 构建或加载缓存的 TensorRT 引擎
   - 初始化 GPU 内存分配器（复用 `ortcore::GpuMemoryAllocator`）
   - 创建 CUDA stream
2. 实现子模块推理接口：
   - `encode_image()` - 图像编码
   - `transform_key()` - 关键特征投影
   - `encode_mask()` - 掩码编码
   - `pixel_fusion()` - 像素融合
   - `readout_query()` - 对象查询
   - `segment()` - 分割解码
3. GPU 内存管理适配：
   - 使用 `Ort::Value::GetTensorMutableData<float>()` 提取 GPU 指针
   - 通过 `TrtHandler::infer_async()` 传递给 TensorRT
   - 确保张量数据在 GPU 上流转（零拷贝）

**验证：**
- 单帧推理测试：输入单张图像和掩码，验证输出形状和数值正确性

### 3.3 Phase 3：集成到 InferenceCore（1-2 天）

**目标：** 将 TensorRT 后端集成到现有推理流程。

**任务：**
1. 修改 `InferenceCore::Impl`：
   - 添加 `#ifdef ENABLE_TENSORRT` 分支
   - 实例化 `TrtCutie` 替代 `OrtCutie`
2. 确保 `InferenceCore` 的所有接口与 TensorRT 后端兼容
3. 修改 `CutieProcessor` 构造函数：
   - 根据编译选项选择后端

**验证：**
- 运行 `demo_basic`，使用 TensorRT 后端完成完整视频推理

### 3.4 Phase 4：优化和测试（2-3 天）

**目标：** 性能优化和全面测试。

**任务：**
1. **引擎缓存机制**：
   - 首次运行时构建引擎并序列化到 `share/model/*.engine`
   - 后续运行直接加载缓存引擎（跳过 ONNX 解析）
2. **FP16 优化**：
   - 启用 FP16 精度（`setFlag(BuilderFlag::kFP16)`）
   - 验证精度损失可接受
3. **动态形状支持**（可选）：
   - 配置 Optimization Profile（min/opt/max 分辨率）
   - 支持不同分辨率输入
4. **性能对比测试**：
   - TensorRT vs ONNX Runtime 推理速度
   - 内存占用对比
5. **错误处理和日志**：
   - 完善异常处理
   - 添加详细的调试日志

**验证：**
- 性能基准测试（FPS、延迟、内存）
- 多视频、多分辨率测试
- 长时间稳定性测试

---

## 4. 关键技术点

### 4.1 ONNX 到 TensorRT 转换

**流程：**
1. 创建 `IBuilder` 和 `INetworkDefinition`
2. 使用 `nvonnxparser::IParser` 解析 ONNX 文件
3. 配置 `IBuilderConfig`（工作空间、精度、动态形状）
4. 调用 `buildSerializedNetwork()` 生成引擎

**代码示例：**
```cpp
auto builder = std::unique_ptr<nvinfer1::IBuilder>(
    nvinfer1::createInferBuilder(logger));
auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
    builder->createNetworkV2(1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

auto parser = std::unique_ptr<nvonnxparser::IParser>(
    nvonnxparser::createParser(*network, logger));
parser->parseFromFile(onnx_path.c_str(), 
    static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
    builder->createBuilderConfig());
config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 
    1ULL << 30);  // 1GB

auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
    builder->buildSerializedNetwork(*network, *config));
```

### 4.2 推理执行（enqueueV3）

**流程：**
1. 为每个输入/输出张量调用 `setTensorAddress()` 绑定 GPU 指针
2. 如果是动态形状，调用 `setInputShape()` 设置实际形状
3. 调用 `enqueueV3(stream)` 启动异步推理
4. 使用 `cudaStreamSynchronize()` 等待完成（或异步处理）

**代码示例：**
```cpp
// 绑定输入
context->setTensorAddress("image", image_gpu_ptr);
context->setInputShape("image", nvinfer1::Dims4{1, 3, 480, 640});

// 绑定输出
context->setTensorAddress("f16", f16_gpu_ptr);
context->setTensorAddress("f8", f8_gpu_ptr);

// 执行推理
context->enqueueV3(cuda_stream);
cudaStreamSynchronize(cuda_stream);
```

### 4.3 引擎缓存策略

**目标：** 避免每次启动都重新构建引擎（耗时 10-60 秒）。

**实现：**
1. 引擎文件命名：`{model_prefix}_{submodule}.engine`
2. 检查逻辑：
   - 如果 `.engine` 文件存在且时间戳晚于 `.onnx`，直接加载
   - 否则，从 ONNX 构建并序列化
3. 版本校验：
   - 在引擎文件中嵌入 TensorRT 版本和 CUDA 版本
   - 版本不匹配时重新构建

**代码示例：**
```cpp
std::string engine_path = model_prefix + "_pixel_encoder.engine";
std::string onnx_path = model_prefix + "_pixel_encoder.onnx";

if (fs::exists(engine_path) && 
    fs::last_write_time(engine_path) > fs::last_write_time(onnx_path)) {
    engine = builder.deserialize_engine(engine_path);
    logger->info("Loaded cached engine: {}", engine_path);
} else {
    engine = builder.build_from_onnx(onnx_path, config);
    builder.serialize_engine(engine.get(), engine_path);
    logger->info("Built and cached engine: {}", engine_path);
}
```

### 4.4 GPU 内存管理

**挑战：** TensorRT 使用原始 GPU 指针，而现有代码使用 `Ort::Value`。

**解决方案：**
1. **内部统一使用 `Ort::Value`**：
   - `TrtCutie` 内部仍使用 `Ort::Value` 管理 GPU 内存
   - 推理时临时提取 GPU 指针传给 TensorRT
2. **适配层**：
   ```cpp
   void* ptr = tensor.GetTensorMutableData<float>();
   context->setTensorAddress("input", ptr);
   ```
3. **内存生命周期**：
   - 确保 `Ort::Value` 在推理期间保持有效
   - 避免悬空指针

### 4.5 CUDA Kernels 复用

**现有 Kernels（`src/ort/core/cuda_kernels.cu`）：**
- `concat_kernel`、`slice_kernel`
- `sigmoid_kernel`、`aggregate_softmax_kernel`
- `get_similarity_kernel`、`top_k_softmax_kernel`
- `one_hot_kernel`、`mask_merge_kernel`
- `fill_zero_kernel`、`copy_d2d_kernel`

**复用策略：**
- 这些 kernel 函数与推理后端无关，可直接被 TensorRT 后端调用
- 在 `src/trt/core/` 中创建符号链接或共享编译单元
- CMake 配置：
  ```cmake
  if(ENABLE_TENSORRT)
      target_sources(cutie PRIVATE
          src/ort/core/cuda_kernels.cu  # 复用
          src/trt/core/trt_handler.cpp
          src/trt/cv/trt_cutie.cpp
      )
  endif()
  ```

---

## 5. CMake 配置

### 5.1 FindTensorRT.cmake

创建 `cmake/FindTensorRT.cmake` 模块：

```cmake
# FindTensorRT.cmake
find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
    PATH_SUFFIXES include)

find_library(TensorRT_LIBRARY nvinfer
    HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
    PATH_SUFFIXES lib lib64)

find_library(TensorRT_ONNX_PARSER nvonnxparser
    HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
    PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_LIBRARY TensorRT_ONNX_PARSER TensorRT_INCLUDE_DIR)

if(TensorRT_FOUND)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TensorRT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
    
    add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvonnxparser PROPERTIES
        IMPORTED_LOCATION "${TensorRT_ONNX_PARSER}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
endif()
```

### 5.2 主 CMakeLists.txt 修改

```cmake
# 在 CMakeLists.txt 中添加
if(ENABLE_TENSORRT)
    find_package(TensorRT REQUIRED)
    message(STATUS "TensorRT found: ${TensorRT_INCLUDE_DIR}")
    
    target_compile_definitions(cutie PRIVATE ENABLE_TENSORRT)
    target_include_directories(cutie PRIVATE ${TensorRT_INCLUDE_DIR})
    target_link_libraries(cutie PRIVATE 
        TensorRT::nvinfer 
        TensorRT::nvonnxparser)
    
    # 添加 TensorRT 后端源文件
    target_sources(cutie PRIVATE
        src/trt/core/trt_handler.cpp
        src/trt/core/trt_engine_builder.cpp
        src/trt/cv/trt_cutie.cpp
        # 复用 CUDA kernels
        src/ort/core/cuda_kernels.cu
    )
endif()
```

---

## 6. 测试计划

### 6.1 单元测试

**测试项：**
1. **TrtEngineBuilder**：
   - ONNX 解析成功
   - 引擎构建成功
   - 序列化/反序列化一致性
2. **TrtHandler**：
   - 引擎加载成功
   - 输入输出绑定正确
   - 推理结果形状正确
3. **TrtCutie**：
   - 6 个子模块加载成功
   - 单帧推理输出形状正确

### 6.2 集成测试

**测试项：**
1. **demo_basic**：
   - 使用 TensorRT 后端完成完整视频推理
   - 输出掩码与 ONNX Runtime 版本一致（允许小误差）
2. **多分辨率测试**：
   - 480p、720p、1080p 输入
3. **多对象测试**：
   - 1 个对象、2 个对象、N 个对象

### 6.3 性能测试

**指标：**
- **推理速度**：FPS（帧/秒）
- **延迟**：单帧处理时间（毫秒）
- **内存占用**：GPU 显存峰值
- **引擎构建时间**：首次启动耗时

**对比基准：**
- TensorRT vs ONNX Runtime
- FP32 vs FP16

---

## 7. 风险和挑战

### 7.1 动态形状支持

**问题：** Cutie 模型支持动态分辨率输入，TensorRT 需要配置 Optimization Profile。

**解决方案：**
- 定义 min/opt/max 分辨率范围（如 [320, 480, 1920]）
- 在引擎构建时设置 Profile
- 推理时根据实际输入调用 `setInputShape()`

### 7.2 精度损失

**问题：** FP16 模式可能导致精度下降。

**解决方案：**
- 提供 FP32/FP16 配置选项
- 对比测试输出差异（IoU、mAP 等指标）
- 必要时对敏感层禁用 FP16

### 7.3 引擎兼容性

**问题：** TensorRT 引擎与 CUDA/TensorRT 版本强绑定，升级后需重新构建。

**解决方案：**
- 在引擎文件中嵌入版本信息
- 版本不匹配时自动重新构建
- 文档中明确说明兼容性要求

### 7.4 内存管理复杂度

**问题：** `Ort::Value` 和原始 GPU 指针混用可能导致内存泄漏或悬空指针。

**解决方案：**
- 统一使用 `Ort::Value` 管理生命周期
- 仅在推理时临时提取指针
- 添加详细的内存管理日志

---

## 8. 参考资料

### 8.1 官方文档

- [TensorRT C++ API Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html)
- [IExecutionContext Class Reference](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/api/c_api/classnvinfer1_1_1_i_execution_context.html)
- [TensorRT API Migration Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration-guide.html)
- [ONNX Parser API](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/api/c_api/classnvonnxparser_1_1_i_parser.html)

### 8.2 代码参考

- 本项目 ONNX Runtime 实现：`src/ort/`
- Cutie 原工程：`.ref_project/Cutie/`
- lite.ai.toolkit TensorRT 后端示例

---

## 9. 时间估算

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| Phase 1 | 基础设施（TrtEngineBuilder、TrtHandler） | 1-2 天 |
| Phase 2 | 模型封装（TrtCutie） | 2-3 天 |
| Phase 3 | 集成到 InferenceCore | 1-2 天 |
| Phase 4 | 优化和测试 | 2-3 天 |
| **总计** | | **6-10 天** |

---

## 10. 下一步行动

1. **确认技术方案**：与团队讨论本规划的可行性
2. **环境准备**：安装 TensorRT ≥ 8.6
3. **创建分支**：`git checkout -b feature/tensorrt-backend`
4. **开始 Phase 1**：实现 `TrtEngineBuilder` 和 `TrtHandler`

---

**文档版本：** v1.0  
**创建日期：** 2026-04-24  
**作者：** Claude Sonnet 4.6

