# cutie-cpp

[Cutie](https://github.com/hkchengrex/Cutie) 视频物体分割（VOS）模型的 C++ 推理库。参考 [lite.ai.toolkit](https://github.com/xlite-dev/lite.ai.toolkit) 的多后端架构，将 PyTorch 推理流程移植为纯 C++17 动态库，支持有状态多帧视频推理。

---

## 特性

- **纯 C++17**：零 Python 依赖，可直接嵌入生产系统
- **多推理后端**：
  - **ONNX Runtime**（默认）：含 CUDA EP，兼容性好，使用 IO Binding 避免 CPU↔GPU 数据拷贝
  - **TensorRT**（可选）：高性能推理，支持 FP16/INT8 量化，智能引擎缓存（节省 10-60 秒启动时间）
- **全 GPU 数据流**：输入图像 CPU→GPU 上传后，所有中间特征和内存数据保持在 GPU，仅最终结果按需下载
- **GPU 加速预处理**：BGR→RGB 转换、resize、ImageNet 归一化、pad 全部在 GPU 上完成
- **完整的有状态推理**：工作记忆（FIFO KV 缓冲）+ 长期记忆（可选压缩原型）+ 感知记忆（per-object GRU 隐状态）
- **多对象跟踪**：支持同时跟踪多个对象，支持中途删除对象
- **RAII 风格 API**：单头文件 `#include "cutie/cutie.h"` 即可使用

---

## 模型变体

Cutie 提供 **base** 和 **small** 两种变体，每种变体再按训练数据分为三个权重文件，共 6 个 `.pth`：

| 变体 | 骨干网络 | 特征维度 (f16/f8/f4) | 适用场景 |
|------|----------|----------------------|----------|
| **base** | ResNet-50 | 1024 / 512 / 256 | 精度优先，GPU 推理 |
| **small** | ResNet-18 | 256 / 128 / 64 | 速度/显存受限场景 |

### 权重文件说明

| 文件名 | 训练数据 | 说明 |
|--------|----------|------|
| `cutie-base-mega.pth` | MEGA 数据集（最大量） | base 变体最强精度，推荐首选 |
| `cutie-base-nomose.pth` | 不含 MOSE 数据集 | 适用于 MOSE benchmark 评测对照 |
| `cutie-base-wmose.pth` | 含 MOSE 数据集 | 遮挡场景更鲁棒 |
| `cutie-small-mega.pth` | MEGA 数据集 | small 变体最强精度 |
| `cutie-small-nomose.pth` | 不含 MOSE | small 变体，MOSE 评测对照 |
| `cutie-small-wmose.pth` | 含 MOSE | small 变体，遮挡更强 |

> **简单选择原则**：不确定用哪个时，选 `cutie-base-mega`（精度最高）或 `cutie-small-mega`（速度优先）。

---

## 推理架构

Cutie 的推理被拆分为 **6 个 ONNX 子模块**，在每帧推理时按顺序调用：

```
image ──► pixel_encoder ──► f16, f8, f4, pix_feat
              │
          f16 ──► key_projection ──► key, shrinkage, selection
              │
  pix_feat + sensory + masks ──► mask_encoder ──► value, sensory, summaries
              │
  pix_feat + pixel + sensory + last_mask ──► pixel_fuser ──► fused_pixel
              │
  pixel_readout + obj_memory ──► object_transformer ──► updated_readout
              │
  f8 + f4 + readout + sensory ──► mask_decoder ──► new_sensory, logits → mask
```

| 子模块 | 输入 | 输出 | 作用 |
|--------|------|------|------|
| `pixel_encoder` | image `[1,3,H,W]` | f16/f8/f4/pix_feat | 多尺度图像特征提取（ResNet 骨干） |
| `key_projection` | f16 | key, shrinkage, selection | 生成记忆检索用的 key 向量 |
| `mask_encoder` | pix_feat + sensory + masks | value, sensory, summaries | 将 mask 编码进记忆 value |
| `pixel_fuser` | pix_feat + pixel + sensory + last_mask | fused_pixel | 融合像素特征与记忆读出 |
| `object_transformer` | pixel_readout + obj_memory | updated_readout | Transformer 跨对象注意力 |
| `mask_decoder` | f8 + f4 + readout + sensory | sensory, logits | 解码最终分割 logits |

---

## 记忆机制

| 记忆类型 | 说明 | 配置参数 |
|----------|------|----------|
| **工作记忆** | 最近 N 帧，时序 KV 缓冲，FIFO 淘汰 | `mem_every`, `max_mem_frames` |
| **长期记忆** | 可选，基于使用量压缩的原型记忆 | `use_long_term`, `long_term.*` |
| **感知记忆** | per-object GRU 隐状态，每帧更新 | `sensory_dim`（自动按变体设置） |

---

## 编译

### 依赖

| 依赖 | 版本 | 必需 |
|------|------|------|
| CMake | ≥ 3.20 | ✅ |
| GCC/Clang | GCC ≥ 9 | ✅ |
| OpenCV | ≥ 4.0（含 CUDA 模块） | ✅ |
| CUDA Toolkit | ≥ 11.8 | ✅ |
| cuBLAS | （CUDA Toolkit 自带） | ✅ |
| ONNX Runtime | ≥ 1.16（含 CUDA EP） | ONNX 后端 |
| TensorRT | ≥ 10.0 | TRT 后端 |
| linden_logger | （子项目，自动加载） | ✅ |

> **注意**：项目现在要求 CUDA ≥ 11.8，构建时必须确保 CUDA Toolkit 已正确安装并可被 CMake 检测到。

### 编译步骤

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
```

编译产物（默认输出到 `build/` 目录）：
- `build/libcutie.so` — 动态库
- `build/demo_basic` — 示例程序

安装（可选）：
```bash
bash install.sh
# 或
cmake --install build/
```

---

## 导出 ONNX 模型

从 `.pth` 权重文件导出 6 个 ONNX 子模块（需要 Python + PyTorch 环境）：

```bash
cd share/scripts

# 导出 base-mega 权重
python export_onnx.py \
    --variant base \
    --weights ../model/cutie-base-mega.pth \
    --output ../model/

# 导出 small-mega 权重
python export_onnx.py \
    --variant small \
    --weights ../model/cutie-small-mega.pth \
    --output ../model/
```

导出后在 `share/model/` 目录下生成（以 `cutie-base-mega` 为例）：

```
cutie-base-mega_pixel_encoder.onnx
cutie-base-mega_key_projection.onnx
cutie-base-mega_mask_encoder.onnx
cutie-base-mega_pixel_fuser.onnx
cutie-base-mega_object_transformer.onnx
cutie-base-mega_mask_decoder.onnx
```

> ONNX 文件以权重文件名（去扩展名）为前缀，C++ 加载时通过 `model_prefix` 字段定位。

> **注意**：导出脚本对 `downsample_groups` 进行了 monkey-patch，将不兼容 ONNX 导出的 `F.interpolate(mode='area')` 替换为等价的 `F.avg_pool2d`。

---

## 快速使用

```cpp
#include "cutie/cutie.h"

using namespace cutie::cv::segmentation;

// 1. 创建配置（base 变体默认参数）
auto config = CutieConfig::base_default("/path/to/model/dir");
config.model_prefix = "cutie-base-mega";   // 对应 cutie-base-mega_*.onnx
config.use_long_term = true;               // 启用长期记忆（可选）

// 2. 创建处理器（加载 ONNX，初始化推理 session）
CutieProcessor processor(config);

// 3. 首帧：提供图像 + 分割 mask + 对象 ID 列表
cv::Mat frame0 = cv::imread("frame_000.jpg");
cv::Mat mask0  = cv::imread("mask_000.png", cv::IMREAD_GRAYSCALE); // 像素值 = 对象 ID
auto result = processor.step(frame0, mask0, {1, 2, 3});

// 4. 后续帧：仅提供图像，自动传播分割
//    （内部自动完成 GPU 预处理、特征提取、内存管理、分割解码）
for (auto& frame : remaining_frames) {
    auto result = processor.step(frame);
    // result.index_mask — H×W CV_32SC1，像素值为对象 ID（0 = 背景）
    // result.object_ids — 当前活跃对象列表
}

// 5. 中途对象管理（可选）
processor.delete_objects({2});             // 删除对象 2
processor.clear_non_permanent_memory();   // 清除非永久记忆
```

> **性能提示**：内部已全 GPU 化，CPU `cv::Mat` 输入会自动上传到 GPU 并缓存，所有中间计算在 GPU 完成，最终结果下载到 CPU 返回。对于高吞吐场景，可考虑使用 `cv::cuda::GpuMat` 直接传入（需要相应 API 扩展）。

---

## 运行示例

项目目前唯一可运行的示例为：

```bash
# 基本用法
./build/demo_basic <video_path> <first_frame_mask.png>
```

`demo_basic` 会自动扫描模型目录中 `*_pixel_encoder.onnx` 文件推断 `model_prefix`，无需手动指定。示例会逐帧处理视频，输出分割结果到控制台。

---

## 配置参数一览

```cpp
struct CutieConfig {
    ModelVariant variant = ModelVariant::kBase;
    std::string  model_dir;            // ONNX 文件所在目录
    std::string  model_prefix;         // 必填：对应 .pth 文件名（不含扩展名）
    Device       device    = Device::kCUDA;  // kCPU 或 kCUDA
    int          device_id = 0;

    // 推理参数
    int  max_internal_size = 480;      // 长边 resize 目标（像素）
    int  mem_every         = 5;        // 每 N 帧写入工作记忆
    int  top_k             = 30;       // 记忆注意力 top-k
    int  chunk_size        = -1;       // 分块推理（-1 = 不分块）
    int  stagger_updates   = 5;        // 交错更新间隔
    bool single_object     = false;    // 单对象模式（跳过多对象融合）

    // 工作记忆
    int max_mem_frames = 5;

    // 长期记忆（use_long_term = true 时生效）
    bool use_long_term = false;
    struct LongTermConfig {
        bool count_usage     = true;
        int  max_mem_frames  = 10;
        int  min_mem_frames  = 5;
        int  num_prototypes  = 128;
        int  max_num_tokens  = 10000;
        int  buffer_tokens   = 2000;
    } long_term;
};
```

---

## 项目结构

```
cutie-cpp/
├── include/cutie/          # 公共头文件
│   ├── cutie.h             # 统一入口头文件
│   ├── types.h             # CutieMask, ObjectId, 枚举
│   ├── models.h            # 命名空间别名
│   └── core/
│       └── processor.h     # CutieConfig, CutieProcessor 声明
├── src/                    # 实现源码
│   ├── common/             # GPU 公共代码（ORT 和 TRT 后端共享）
│   │   ├── cuda_kernels.cu           # CUDA kernel 实现
│   │   ├── gpu_memory.cpp            # GPU 内存分配器
│   │   ├── gpu_tensor_ops.cpp        # GPU 张量操作原语
│   │   ├── gpu_image_preprocess.cu   # GPU 图像预处理
│   │   ├── gpu_mask_preprocess.cu    # GPU 掩码预处理
│   │   └── gpu_postprocess.cu        # GPU 后处理
│   ├── core/               # 与平台无关的推理逻辑
│   │   ├── inference_core.cpp        # 主推理循环
│   │   ├── memory_manager.cpp        # 三层内存系统
│   │   ├── kv_memory_store.cpp       # KV 内存存储
│   │   ├── object_manager.cpp        # 对象管理
│   │   └── processor.cpp             # 公共 API 实现
│   ├── ort/                # ONNX Runtime 后端
│   │   ├── ort_handler.cpp           # 会话管理
│   │   └── ort_cutie.cpp             # 6 个子模块封装
│   └── trt/                # TensorRT 后端
│       ├── trt_engine_builder.cpp    # 引擎构建
│       ├── trt_handler.cpp           # 引擎管理
│       └── trt_cutie.cpp             # 6 个子模块封装
├── examples/
│   └── demo_basic.cpp      # 基础使用示例
├── share/
│   ├── docs/               # API 参考文档、需求文档
│   ├── model/              # .pth 权重文件 & 导出的 ONNX 文件
│   └── scripts/
│       └── export_onnx.py  # PyTorch → ONNX 导出脚本
├── .ref_project/           # 参考项目（不提交到 Git）
│   ├── Cutie/              # 原始 PyTorch 实现
│   └── linden_logger/      # 专有日志库
├── cmake/                  # CMake 查找模块
├── build.sh                # 一键编译脚本
└── install.sh              # 安装脚本
```

---

## 许可证

见 [LICENSE](LICENSE)。

原始 Cutie 模型版权归 [Ho Kei Cheng](https://github.com/hkchengrex/Cutie) 所有，请遵守其许可证。
