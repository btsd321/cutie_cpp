# Python 示例

四个示例覆盖从最小闭环到性能测量的典型用法，对应 C++ 侧的 `examples/demo_basic.cpp`。

## 准备

示例需要 OpenCV 读写视频：

```bash
.venv/bin/pip install opencv-python
```

模型目录会自动搜索（`CUTIE_MODEL_DIR` 环境变量 → 安装目录 → 仓库 `share/model/`），
也可用 `--model-dir` 显式指定。

若尚未安装为 wheel，用 `PYTHONPATH` 指向源码包目录运行：

```bash
export PYTHONPATH=/path/to/cutie_cpp/python
```

## 示例列表

| 脚本 | 演示内容 |
|---|---|
| `01_segment_video.py` | 基础闭环：逐帧推理、后台线程保存图片、可选输出叠加视频 |
| `02_multi_object.py` | 多目标跟踪，中途 `delete_objects()` 停止跟踪某个目标 |
| `03_config_from_yaml.py` | YAML 配置加载 + `reset()` 复用实例跨多段视频 |
| `04_benchmark.py` | FPS 与 P50/P90/P99 延迟统计，验证绑定层没有多余拷贝 |

## 运行

```bash
# 01 基础分割（结果与日志写入 output/）
python 01_segment_video.py \
    --video ../../examples/example.mp4 \
    --mask ../../examples/example_frame0_mask.png \
    --save-video

# 02 多目标：把单目标掩码拆成两个，第 20 帧停止跟踪目标 2
python 02_multi_object.py \
    --video ../../examples/example.mp4 \
    --mask ../../examples/example_frame0_mask.png \
    --split-mask --drop-object 2 --drop-at 20

# 03 YAML 配置 + 一个实例处理两段视频
python 03_config_from_yaml.py \
    --config configs/base.yaml \
    --clip ../../examples/example.mp4:../../examples/example_frame0_mask.png \
    --clip ../../examples/example.mp4:../../examples/example_frame0_mask.png

# 查看生效配置而不做推理
python 03_config_from_yaml.py --dump-config --clip a:b

# 04 性能基准
python 04_benchmark.py \
    --video ../../examples/example.mp4 \
    --mask ../../examples/example_frame0_mask.png \
    --warmup 5 --frames 100

# 加上概率图下载，观察额外的 D2H 开销
python 04_benchmark.py --video ... --mask ... --return-prob
```

## 掩码格式

掩码是单通道 PNG，**像素值即目标 ID**，0 为背景。例如两个目标的掩码，
像素值应为 1 和 2。

示例脚本会识别常见的 0/255 二值掩码，把它归一为单目标 ID 1。
仓库的 `share/scripts/binarize_mask.py` 可用于生成这类掩码。

## 性能说明

`04_benchmark.py` 的稳态耗时应与 C++ `demo_basic` 相当。绑定层每帧只有
一次 H2D 上传和一次 D2H 下载，与 C++ 全 GPU 路径一致，因此不会有
可测量的额外开销。

调 `--max-internal-size` 是最有效的提速手段（仅对动态分辨率模型生效）；
`--return-prob` 会显著变慢，1080p / 3 目标下每帧要多下载约 25 MB。
