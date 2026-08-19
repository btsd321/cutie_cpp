#!/usr/bin/env python3
"""性能基准测试。

逐帧计时并输出 FPS 与分位数延迟，用于确认 Python 绑定没有引入多余的内存拷贝。
可用 --return-prob 对比下载概率图的额外开销（1080p/3 目标约 25 MB/帧）。

用法：
    python 04_benchmark.py --video FILE --mask FILE [--warmup N] [--frames N]
"""

import argparse
import logging
import statistics
import sys
import time

import cv2
import numpy as np

import cutie_cpp


logger = cutie_cpp.get_logger(__name__)


def parse_args():
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析结果。
    """
    parser = argparse.ArgumentParser(
        description="Cutie Python 绑定性能基准",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", "-i", required=True, help="输入视频文件")
    parser.add_argument("--mask", "-k", required=True, help="首帧掩码")
    parser.add_argument("--model-dir", "-m", default=None, help="ONNX 目录，省略则自动搜索")
    parser.add_argument(
        "--warmup", type=int, default=5, help="预热帧数，不计入统计（含首帧初始化）"
    )
    parser.add_argument("--frames", type=int, default=100, help="计入统计的帧数")
    parser.add_argument(
        "--max-internal-size", type=int, default=480, help="动态分辨率下的短边上限"
    )
    parser.add_argument(
        "--return-prob", action="store_true", help="同时下载概率图，测量其额外开销"
    )
    return parser.parse_args()


def read_frames(video_path, count):
    """预先读入若干帧到内存。

    把视频解码从计时区间里剥离，让测量只反映推理与数据搬运的耗时。

    Args:
        video_path (str): 视频路径。
        count (int): 需要的帧数。视频不够长时循环复用已读帧。

    Returns:
        list[np.ndarray]: BGR 帧列表。

    Raises:
        SystemExit: 视频打开失败或没有可用帧时退出。
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        logger.error(f"打开视频失败: {video_path}")
        sys.exit(1)

    frames = []
    try:
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                break
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        logger.error(f"视频 {video_path} 中没有可读帧")
        sys.exit(1)

    if len(frames) < count:
        logger.warning(
            f"视频只有 {len(frames)} 帧，不足 {count} 帧，将循环复用以补齐"
        )
        source = list(frames)
        while len(frames) < count:
            frames.append(source[len(frames) % len(source)])

    logger.info(f"已预载 {len(frames)} 帧到内存，尺寸 {frames[0].shape}")
    return frames


def load_index_mask(mask_path):
    """读取掩码并转为 int32 索引掩码。

    Args:
        mask_path (str): 掩码文件路径。

    Returns:
        np.ndarray: int32 索引掩码。

    Raises:
        SystemExit: 读取失败时退出。
    """
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        logger.error(f"读取掩码失败: {mask_path}")
        sys.exit(1)

    unique_values = [int(v) for v in np.unique(mask_img) if v != 0]
    if unique_values == [255]:
        return (mask_img > 0).astype(np.int32)
    return mask_img.astype(np.int32)


def report(durations_ms, label):
    """输出一组耗时的统计摘要。

    Args:
        durations_ms (list[float]): 每帧耗时（毫秒）。
        label (str): 统计项标签。
    """
    if not durations_ms:
        logger.warning(f"{label}: 没有采样数据")
        return

    ordered = sorted(durations_ms)
    mean_ms = statistics.fmean(durations_ms)

    def percentile(ratio):
        """取指定分位的耗时。

        Args:
            ratio (float): 分位比例，如 0.5 表示中位数。

        Returns:
            float: 对应分位的耗时（毫秒）。
        """
        index = min(int(len(ordered) * ratio), len(ordered) - 1)
        return ordered[index]

    logger.info(
        f"{label}: {len(durations_ms)} 帧 | 均值 {mean_ms:.1f} ms "
        f"({1000.0 / mean_ms:.1f} FPS) | P50 {percentile(0.5):.1f} ms | "
        f"P90 {percentile(0.9):.1f} ms | P99 {percentile(0.99):.1f} ms | "
        f"最小 {ordered[0]:.1f} ms | 最大 {ordered[-1]:.1f} ms"
    )


def main():
    """基准测试主流程。

    Returns:
        int: 退出码，0 表示成功。
    """
    args = parse_args()
    cutie_cpp.setup_logging(level=logging.INFO)

    total_frames = args.warmup + args.frames
    frames = read_frames(args.video, total_frames)
    index_mask = load_index_mask(args.mask)

    config = cutie_cpp.CutieConfig.base_default(
        args.model_dir, max_internal_size=args.max_internal_size
    )

    logger.info(
        f"配置: max_internal_size={config.max_internal_size} "
        f"mem_every={config.mem_every} return_prob={args.return_prob}"
    )

    try:
        load_start = time.perf_counter()
        with cutie_cpp.VideoSegmenter(config) as segmenter:
            load_ms = (time.perf_counter() - load_start) * 1000.0
            logger.info(f"模型加载耗时 {load_ms:.0f} ms")

            durations = []
            for frame_index, frame in enumerate(frames):
                start = time.perf_counter()
                if frame_index == 0:
                    result = segmenter.step(
                        frame, mask=index_mask, return_prob=args.return_prob
                    )
                else:
                    result = segmenter.step(frame, return_prob=args.return_prob)
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                # 预热帧含 CUDA 上下文初始化与 kernel 自动调优，不计入统计
                if frame_index >= args.warmup:
                    durations.append(elapsed_ms)
                elif frame_index == 0:
                    logger.info(f"首帧（含初始化）{elapsed_ms:.1f} ms")

            report(durations, "稳态推理")
            logger.info(
                f"末帧: 目标 {result.object_ids} 掩码 {result.shape} "
                f"概率图 {'已下载' if result.prob is not None else '未下载'}"
            )
    except cutie_cpp.CutieError as exc:
        logger.error(f"基准测试失败: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
