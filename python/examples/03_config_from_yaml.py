#!/usr/bin/env python3
"""YAML 配置与实例复用示例。

演示三件事：
    1. 从 YAML 文件加载配置，把参数外置到部署环境
    2. 用上下文管理器管理分割器生命周期
    3. 用 reset() 复用同一实例处理多段视频，省去重复加载模型的开销
       （加载 6 个 ONNX 子模块约需数秒）

用法：
    python 03_config_from_yaml.py --config configs/base.yaml \\
        --clip VIDEO1:MASK1 --clip VIDEO2:MASK2
"""

import argparse
import logging
import sys
import time
from pathlib import Path

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
        description="Cutie YAML 配置与实例复用示例",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        default=str(Path(__file__).parent / "configs" / "base.yaml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--clip",
        action="append",
        required=True,
        metavar="VIDEO:MASK",
        help="待处理片段，格式 视频路径:掩码路径。可重复传入多个",
    )
    parser.add_argument("--max-frames", type=int, default=30, help="每段视频最多处理多少帧")
    parser.add_argument("--dump-config", action="store_true", help="打印生效的配置后退出")
    return parser.parse_args()


def parse_clip(spec):
    """解析 "视频:掩码" 形式的片段参数。

    Args:
        spec (str): 形如 "path/to/video.mp4:path/to/mask.png" 的字符串。

    Returns:
        tuple[str, str]: (视频路径, 掩码路径)。

    Raises:
        SystemExit: 格式不正确或文件不存在时退出。
    """
    # 用 rsplit 而非 split，避免 Windows 盘符或路径中的冒号造成误切分
    parts = spec.rsplit(":", 1)
    if len(parts) != 2:
        logger.error(f"--clip 格式应为 视频:掩码，收到 {spec!r}")
        sys.exit(1)

    video_path, mask_path = parts
    for path in (video_path, mask_path):
        if not Path(path).is_file():
            logger.error(f"文件不存在: {path}")
            sys.exit(1)
    return video_path, mask_path


def load_index_mask(mask_path):
    """读取掩码并转为 int32 索引掩码。

    Args:
        mask_path (str): 掩码文件路径。

    Returns:
        np.ndarray: int32 索引掩码，0/255 二值图会被归一为单目标 ID 1。

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


def process_clip(segmenter, video_path, mask_path, max_frames):
    """用给定分割器处理一段视频。

    Args:
        segmenter (cutie_cpp.VideoSegmenter): 分割器实例。
        video_path (str): 视频路径。
        mask_path (str): 首帧掩码路径。
        max_frames (int): 最多处理帧数。

    Returns:
        int: 实际处理的帧数。
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        logger.error(f"打开视频失败: {video_path}")
        return 0

    index_mask = load_index_mask(mask_path)
    processed = 0
    start = time.perf_counter()

    try:
        for frame_index in range(max_frames):
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                break

            if frame_index == 0:
                result = segmenter.step(frame, mask=index_mask)
            else:
                result = segmenter.step(frame)
            processed += 1
    finally:
        capture.release()

    elapsed = time.perf_counter() - start
    if processed > 0:
        logger.info(
            f"{Path(video_path).name}: {processed} 帧, "
            f"{elapsed / processed * 1000:.1f} ms/帧, "
            f"末帧目标 {result.object_ids}"
        )
    return processed


def main():
    """示例主流程。

    Returns:
        int: 退出码，0 表示成功。
    """
    args = parse_args()
    cutie_cpp.setup_logging(level=logging.INFO)

    try:
        config = cutie_cpp.CutieConfig.from_yaml(args.config)
    except (cutie_cpp.ConfigError, FileNotFoundError) as exc:
        logger.error(f"加载配置失败: {exc}")
        return 1

    logger.info(f"配置来自: {args.config}")

    if args.dump_config:
        for key, value in config.to_dict().items():
            logger.info(f"  {key} = {value}")
        return 0

    clips = [parse_clip(spec) for spec in args.clip]

    try:
        # 一个实例处理所有片段：模型只加载一次，每段之间用 reset() 清状态
        with cutie_cpp.VideoSegmenter(config) as segmenter:
            logger.info(f"模型已加载: {segmenter.config.model_prefix}")

            for clip_index, (video_path, mask_path) in enumerate(clips):
                if clip_index > 0:
                    # 换视频必须 reset，否则上一段的内存会污染当前跟踪
                    segmenter.reset()
                logger.info(f"[片段 {clip_index + 1}/{len(clips)}] {video_path}")
                process_clip(segmenter, video_path, mask_path, args.max_frames)
    except cutie_cpp.CutieError as exc:
        logger.error(f"分割失败: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
