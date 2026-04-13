# Cutie-CPP API 参考文档

> 版本: 0.1.0 | 更新日期: 2026-04-09

## 1. 快速开始

```cpp
#include "cutie/cutie.h"

// 创建配置
auto config = cutie::core::CutieConfig::base_default("./models/cutie-base/");
config.use_long_term = true;

// 创建处理器
cutie::cv::segmentation::CutieProcessor processor(config);

// 首帧: 提供图像 + mask + 对象 ID
cv::Mat frame = cv::imread("frame_000.jpg");
cv::Mat mask = cv::imread("mask_000.png", cv::IMREAD_GRAYSCALE);
auto result = processor.step(frame, mask, {1, 2, 3});

// 后续帧: 仅提供图像
for (auto& f : remaining_frames) {
    result = processor.step(f);
    // result.index_mask: H×W 分割结果 (CV_32SC1)
}
```

---

## 2. 头文件结构

| 头文件 | 用途 |
|--------|------|
| `cutie/cutie.h` | 统一头文件，包含以下所有 |
| `cutie/types.h` | `CutieMask`, `ObjectId`, 枚举类型 |
| `cutie/config.h` | 编译配置宏 (`ENABLE_ONNXRUNTIME` 等) |
| `cutie/models.h` | 命名空间别名 `cutie::cv::segmentation::CutieProcessor` |
| `cutie/utils.h` | 图像预处理工具 |

---

## 3. 命名空间

```
cutie::                           根命名空间
├── types::                       公共类型 (CutieMask 等)
├── core::                        核心实现 (CutieProcessor, CutieConfig 等)
├── cv::segmentation::            公共 API 别名
├── ortcore::                     ORT handler 基类 (内部)
├── ortcv::                       ORT 后端 Cutie 模块 (内部)
├── trtcore::                     TRT handler 基类 (内部)
└── trtcv::                       TRT 后端 Cutie 模块 (内部)
```

---

## 4. 公共类型 — `cutie/types.h`

### 4.1 `cutie::ObjectId`

```cpp
using ObjectId = int32_t;
```

对象标识符。正整数，不要求连续。0 保留为背景。

### 4.2 `cutie::Device`

```cpp
enum class Device { kCPU, kCUDA };
```

### 4.3 `cutie::ModelVariant`

```cpp
enum class ModelVariant { kBase, kSmall };
```

| 变体 | 骨干网络 | 特征维度 |
|------|----------|----------|
| `kBase` | ResNet50 | f16=1024, f8=512, f4=256, sensory=256, value=256 |
| `kSmall` | ResNet18 | f16=256, f8=128, f4=64, sensory=256, value=256 |

### 4.4 `cutie::types::CutieMask`

```cpp
struct CutieMask {
    cv::Mat index_mask;               // H×W, CV_32SC1, 像素值=ObjectId (0=背景)
    std::vector<ObjectId> object_ids; // 当前活跃对象 ID 列表
    cv::Mat prob;                     // [num_objects+1, H, W], CV_32FC1 (可选概率图)
    bool flag = false;                // 结果是否有效
};
```

---

## 5. 配置 — `cutie::core::CutieConfig`

```cpp
struct CutieConfig {
    // === 模型 ===
    ModelVariant variant = ModelVariant::kBase;
    std::string model_dir;          // ONNX 子模块文件目录
    Device device = Device::kCUDA;
    int device_id = 0;
    bool single_object = false;     // true 时跳过多对象聚合

    // === 推理参数 ===
    int max_internal_size = 480;    // 长边 resize 目标
    int mem_every = 5;              // 每 N 帧存入工作记忆
    int top_k = 30;                 // 注意力 top-k
    int chunk_size = -1;            // 分块处理 (-1=不分块)
    int stagger_updates = 5;        // 交错更新间隔

    // === 工作记忆 ===
    int max_mem_frames = 5;         // 工作记忆最大帧数

    // === 长期记忆 ===
    bool use_long_term = false;     // 是否启用长期记忆
    struct LongTermConfig {
        bool count_usage = true;
        int max_mem_frames = 10;
        int min_mem_frames = 5;
        int num_prototypes = 128;
        int max_num_tokens = 10000;
        int buffer_tokens = 2000;
    } long_term;

    // === 工厂方法 ===
    static CutieConfig base_default(const std::string& model_dir);
    static CutieConfig small_default(const std::string& model_dir);
};
```

### 工厂方法

| 方法 | 说明 |
|------|------|
| `base_default(dir)` | base 模型预设 (ResNet50, CUDA, 480px) |
| `small_default(dir)` | small 模型预设 (ResNet18, CUDA, 480px) |

---

## 6. 处理器 — `cutie::core::CutieProcessor`

### 声明

```cpp
class CutieProcessor {
public:
    explicit CutieProcessor(const CutieConfig& config,
                            std::shared_ptr<linden::log::ILogger> logger = nullptr);
    ~CutieProcessor();

    // 不可复制，可移动
    CutieProcessor(const CutieProcessor&) = delete;
    CutieProcessor& operator=(const CutieProcessor&) = delete;
    CutieProcessor(CutieProcessor&&) noexcept;
    CutieProcessor& operator=(CutieProcessor&&) noexcept;

    // 核心推理
    types::CutieMask step(const cv::Mat& image,
                          const cv::Mat& mask = cv::Mat(),
                          const std::vector<ObjectId>& objects = {});

    types::CutieMask step(const cv::Mat& image,
                          const cv::Mat& mask,
                          const std::vector<ObjectId>& objects,
                          const StepOptions& options);

    // 对象管理
    void delete_objects(const std::vector<ObjectId>& objects);
    std::vector<ObjectId> active_objects() const;
    int num_objects() const;

    // 记忆管理
    void clear_memory();
    void clear_non_permanent_memory();
    void clear_sensory_memory();

    // 配置
    void update_config(const CutieConfig& config);
    const CutieConfig& config() const;
};
```

### 6.1 构造函数

```cpp
explicit CutieProcessor(const CutieConfig& config,
                        std::shared_ptr<linden::log::ILogger> logger = nullptr);
```

- 加载 `config.model_dir` 下的 6 个 ONNX 子模块
- `logger` 可选，不传时使用内置默认 logger
- 初始化推理 session（ORT/TRT 由编译配置决定）
- 抛出 `std::runtime_error` 如果模型文件缺失或后端不可用

### 6.2 `step()` — 处理一帧

```cpp
types::CutieMask step(const cv::Mat& image,
                      const cv::Mat& mask = cv::Mat(),
                      const std::vector<ObjectId>& objects = {});
```

**参数:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `image` | `const cv::Mat&` | BGR 图像 (CV_8UC3)，必需 |
| `mask` | `const cv::Mat&` | index mask (CV_8UC1/CV_32SC1)，像素值=ObjectId，0=背景。空=仅跟踪 |
| `objects` | `const std::vector<ObjectId>&` | mask 中的对象 ID 列表。空时自动从 mask 推断 |

**返回:** `types::CutieMask` — 分割结果

**使用场景:**

```cpp
// 场景 1: 首帧初始化
auto result = processor.step(frame, mask, {1, 2, 3});

// 场景 2: 后续帧跟踪
auto result = processor.step(frame);

// 场景 3: 中途追加对象
auto result = processor.step(frame, new_mask, {4, 5});
```

### 6.3 `step()` — 带选项

```cpp
types::CutieMask step(const cv::Mat& image,
                      const cv::Mat& mask,
                      const std::vector<ObjectId>& objects,
                      const StepOptions& options);
```

**StepOptions:**

```cpp
struct StepOptions {
    bool idx_mask = true;     // true: mask 是 index mask; false: 二值 mask
    bool end = false;         // true: 视频结束，触发最终记忆整理
    bool force_permanent = false; // true: 本帧记忆标记为永久
};
```

### 6.4 对象管理

```cpp
void delete_objects(const std::vector<ObjectId>& objects);
```
删除指定对象，释放其记忆。后续帧不再跟踪这些对象。

```cpp
std::vector<ObjectId> active_objects() const;
```
返回当前活跃的对象 ID 列表。

```cpp
int num_objects() const;
```
返回当前活跃对象数量。

### 6.5 记忆管理

| 方法 | 说明 |
|------|------|
| `clear_memory()` | 清除所有记忆（工作+长期+感知），回到初始状态 |
| `clear_non_permanent_memory()` | 清除非永久记忆，保留 `force_permanent` 的帧 |
| `clear_sensory_memory()` | 仅清除感知记忆 (GRU 隐状态) |

### 6.6 配置

```cpp
void update_config(const CutieConfig& config);
```
运行时更新配置（如 `mem_every`, `top_k`, `use_long_term` 等）。不可更改 `variant` 和 `model_dir`。

---

## 7. 别名 — `cutie/models.h`

```cpp
namespace cutie {
namespace cv {
namespace segmentation {
    typedef core::CutieProcessor Cutie;
}
}
}
```

使用方式:
```cpp
cutie::cv::segmentation::Cutie processor(config);
```

---

## 8. 工具函数 — `cutie/utils.h`

```cpp
namespace cutie {
namespace utils {

/// BGR cv::Mat → RGB float32 tensor [1,3,H,W], ImageNet 归一化
/// 内部 resize 到 max_internal_size，padding 到 16 倍数
std::pair<cv::Mat, std::array<int, 4>>
preprocess_image(const cv::Mat& bgr_image, int max_internal_size);

/// index mask (CV_8UC1/CV_32SC1) → per-object one-hot [num_objects,1,H,W]
cv::Mat index_mask_to_one_hot(const cv::Mat& mask,
                               const std::vector<ObjectId>& objects);

/// 概率图 [num_objects+1, H, W] → index mask [H, W] (argmax)
cv::Mat prob_to_index_mask(const cv::Mat& prob);

} // namespace utils
} // namespace cutie
```

---

## 9. 内部组件 (不属于公共 API)

以下组件由 `CutieProcessor` 内部使用，用户通常不需要直接交互:

| 组件 | 头文件 | 说明 |
|------|--------|------|
| `InferenceCore` | `cutie/core/inference_core.h` | 帧处理管线，管理推理状态 |
| `MemoryManager` | `cutie/core/memory_manager.h` | 三层记忆 (工作/长期/感知) |
| `KeyValueMemoryStore` | `cutie/core/kv_memory_store.h` | Bucket-based KV 存储 |
| `ObjectManager` | `cutie/core/object_manager.h` | 对象 ID ↔ 临时索引映射 |
| `ImageFeatureStore` | `cutie/core/image_feature_store.h` | 编码特征缓存 |
| `BasicOrtHandler` | `cutie/ort/core/ort_handler.h` | ORT session 管理基类 |
| `OrtCutieModules` | `cutie/ort/cv/ort_cutie.h` | 6 个 ONNX 子模块 session |
| `BasicTRTHandler` | `cutie/trt/core/trt_handler.h` | TRT engine 管理基类 |

---

## 10. 编译配置

### CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `ENABLE_ONNXRUNTIME` | ON | 构建 ONNX Runtime 后端 |
| `ENABLE_TENSORRT` | OFF | 构建 TensorRT 后端 |

### 构建示例

```bash
mkdir build && cd build

# ORT 后端 (默认)
cmake .. -DENABLE_ONNXRUNTIME=ON

# TRT 后端
cmake .. -DENABLE_ONNXRUNTIME=OFF -DENABLE_TENSORRT=ON

# 双后端
cmake .. -DENABLE_ONNXRUNTIME=ON -DENABLE_TENSORRT=ON

make -j$(nproc)
```

### 模型目录结构

```
models/cutie-base/
├── pixel_encoder.onnx
├── key_projection.onnx
├── mask_encoder.onnx
├── pixel_fuser.onnx
├── object_transformer.onnx
└── mask_decoder.onnx
```

---

## 变更记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-04-09 | 0.1.0 | 初始版本: CutieProcessor, CutieConfig, CutieMask, StepOptions |
| 2026-04-09 | 0.2.0 | Phase 1-4 全部实现; 修正 ModelVariant 维度表; models.h 使用 using 而非 typedef |
