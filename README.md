# cutie-cpp

[Cutie](https://github.com/hkchengrex/Cutie) 视频物体分割（VOS）模型的 C++ 推理库。参考 [lite.ai.toolkit](https://github.com/xlite-dev/lite.ai.toolkit) 的多后端架构，将 PyTorch 推理流程移植为纯 C++17 动态库，支持有状态多帧视频推理。

---

## 特性

- 纯 C++17，零 Python 依赖，可直接嵌入生产系统
- 支持 **ONNX Runtime**（含 CUDA EP）和 **TensorRT**（可选编译）推理后端
- 完整的有状态推理：工作记忆 + 长期记忆 + 感知记忆（GRU 隐状态）
- 支持多对象同时跟踪，支持中途添加/删除对象
- RAII 风格 API，单头文件 `#include "cutie/cutie.h"` 即可使用

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
| CMake | ≥ 3.18 | ✅ |
| GCC/Clang | GCC ≥ 9 | ✅ |
| OpenCV | ≥ 4.0 | ✅ |
| ONNX Runtime | ≥ 1.16 | ONNX 后端 |
| CUDA Toolkit | ≥ 11.8 | GPU 推理 |
| TensorRT | ≥ 10.0 | TRT 后端 |

### 编译步骤

```bash
# 使用默认配置（ONNX Runtime 后端，CUDA 推理）
bash build.sh

# 可选参数
bash build.sh \
    --cuda-root /usr/local/cuda \
    --opencv-dir /path/to/cmake/opencv4 \
    --onnxruntime-root /opt/onnxruntime \
    --install-prefix /usr/local
```

编译产物：
- `build/libcutie.so` — 动态库
- `build/demo_basic` — 示例程序
- `build/include/cutie/config.h` — 生成的配置头文件

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
for (auto& frame : remaining_frames) {
    auto result = processor.step(frame);
    // result.index_mask — H×W CV_32SC1，像素值为对象 ID（0 = 背景）
    // result.object_ids — 当前活跃对象列表
}

// 5. 中途对象管理（可选）
processor.delete_objects({2});             // 删除对象 2
processor.clear_non_permanent_memory();   // 清除非永久记忆
```

---

## 运行示例

```bash
# 自动检测模型目录中的前缀并运行
./build/demo_basic <video_path> <first_frame_mask.png>

# 或指定模型目录
./build/demo_basic /path/to/model/dir <video_path> <first_frame_mask.png>
```

`demo_basic` 会自动扫描模型目录中 `*_pixel_encoder.onnx` 文件推断 `model_prefix`，无需手动指定。

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
│   ├── core/               # 推理核心（记忆、对象管理等）
│   └── ort/                # ONNX Runtime 后端
├── examples/
│   └── demo_basic.cpp      # 基础使用示例
├── share/
│   ├── docs/               # API 参考文档、需求文档
│   ├── model/              # .pth 权重文件 & 导出的 ONNX 文件
│   └── scripts/
│       └── export_onnx.py  # PyTorch → ONNX 导出脚本
├── cmake/                  # CMake 查找模块
├── build.sh                # 一键编译脚本
└── install.sh              # 安装脚本
```

---

## 许可证

见 [LICENSE](LICENSE)。

原始 Cutie 模型版权归 [Ho Kei Cheng](https://github.com/hkchengrex/Cutie) 所有，请遵守其许可证。
