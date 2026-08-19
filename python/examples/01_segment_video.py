#!/usr/bin/env python3
"""视频目标分割基础示例（对应 C++ 的 examples/demo_basic.cpp）。

读入视频与首帧掩码，逐帧推理并把结果保存为图片序列，可选输出叠加视频。
保存图片用后台线程，避免磁盘 IO 阻塞推理主循环。

用法：
    python 01_segment_video.py --video FILE --mask FILE [选项]

示例：
    python 01_segment_video.py \\
        --video ../../examples/example.mp4 \\
        --mask ../../examples/example_frame0_mask.png \\
        --save-video
"""

import argparse
import logging
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import cutie_cpp


logger = cutie_cpp.get_logger(__name__)

# 后台保存队列的容量上限，用于形成反压、限制内存占用
MAX_QUEUE_SIZE = 32


def parse_args():
    """解析命令行参数。

    Returns:
        argparse.Namespace: 解析结果。
    """
    parser = argparse.ArgumentParser(
        description="Cutie 视频目标分割示例",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", "-i", required=True, help="输入视频文件")
    parser.add_argument(
        "--mask", "-k", required=True, help="首帧 PNG 掩码（像素值 = 目标 ID）"
    )
    parser.add_argument(
        "--model-dir", "-m", default=None, help="含 6 个 ONNX 子模块的目录，省略则自动搜索"
    )
    parser.add_argument(
        "--output", "-o", default="output", help="结果输出目录（会被清空重建）"
    )
    parser.add_argument(
        "--frame-skip",
        "-s",
        type=int,
        default=0,
        help="每次推理之间跳过 N 帧（0 = 逐帧处理）",
    )
    parser.add_argument(
        "--max-internal-size", type=int, default=480, help="动态分辨率下的短边上限"
    )
    parser.add_argument("--mem-every", type=int, default=5, help="每 N 帧写入一次内存")
    parser.add_argument(
        "--save-video", action="store_true", help="额外保存带掩码叠加的视频"
    )
    parser.add_argument("--visualize", "-v", action="store_true", help="显示实时预览窗口")
    parser.add_argument(
        "--no-save-frames", action="store_true", help="不保存逐帧图片，只统计性能"
    )
    parser.add_argument("--debug", action="store_true", help="打开 debug 级别日志")
    return parser.parse_args()


def load_index_mask(mask_path):
    """读取首帧掩码并转为索引掩码。

    掩码若是 0/255 的二值图，会被归一为单目标 ID 1；否则按原像素值作为目标 ID。

    Args:
        mask_path (str | Path): 掩码文件路径。

    Returns:
        tuple[np.ndarray, list[int]]: (int32 索引掩码, 目标 ID 列表)。

    Raises:
        SystemExit: 文件读取失败，或掩码中没有前景时退出。
    """
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        logger.error(f"读取掩码失败: {mask_path}")
        sys.exit(1)

    index_mask = mask_img.astype(np.int32)
    object_ids = [int(v) for v in np.unique(index_mask) if v != 0]

    if not object_ids:
        logger.error(f"掩码 {mask_path} 中没有非零像素，无法确定跟踪目标")
        sys.exit(1)

    # 常见的 0/255 二值掩码：255 并非有意义的目标 ID，归一为 1
    if object_ids == [255]:
        logger.info("检测到 0/255 二值掩码，归一为单目标 ID 1")
        index_mask = (index_mask > 0).astype(np.int32)
        object_ids = [1]

    logger.info(f"掩码中找到 {len(object_ids)} 个目标: {object_ids}")
    return index_mask, object_ids


def prepare_output_dir(path):
    """清空并重建输出目录。

    Args:
        path (str | Path): 输出目录。

    Returns:
        Path: 已创建好的目录。
    """
    output_dir = Path(path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


def start_saver_thread(output_dir, save_queue):
    """启动后台图片保存线程。

    推理是 GPU 密集型，磁盘写入放到后台线程可避免阻塞主循环。
    线程以 None 作为结束哨兵。

    Args:
        output_dir (Path): 图片输出目录。
        save_queue (queue.Queue): 任务队列，元素为 (帧序号, BGR 图像) 或 None。

    Returns:
        threading.Thread: 已启动的守护线程。
    """

    def worker():
        while True:
            job = save_queue.get()
            try:
                if job is None:  # 结束哨兵
                    break
                frame_index, image = job
                out_path = output_dir / f"frame_{frame_index:06d}.jpg"
                if not cv2.imwrite(str(out_path), image):
                    logger.error(f"写入失败: {out_path}")
            finally:
                save_queue.task_done()

    thread = threading.Thread(target=worker, name="frame-saver", daemon=True)
    thread.start()
    return thread


def build_config(args):
    """根据命令行参数构造推理配置。

    Args:
        args (argparse.Namespace): 命令行参数。

    Returns:
        cutie_cpp.CutieConfig: 配置对象。
    """
    return cutie_cpp.CutieConfig.base_default(
        args.model_dir,
        max_internal_size=args.max_internal_size,
        mem_every=args.mem_every,
    )


def log_frame_stats(result, elapsed_ms):
    """打印单帧的目标面积与耗时统计。

    Args:
        result (cutie_cpp.SegmentationResult): 分割结果。
        elapsed_ms (float): 本帧推理耗时（毫秒）。
    """
    total_pixels = result.index_mask.size
    foreground = 0
    for object_id in result.object_ids:
        area = result.object_area(object_id)
        foreground += area
        logger.debug(f"第 {result.frame_index} 帧 | 目标 {object_id} 像素数={area}")

    ratio = 100.0 * foreground / total_pixels if total_pixels > 0 else 0.0
    logger.debug(
        f"第 {result.frame_index} 帧 | {elapsed_ms:.1f} ms | "
        f"前景 {foreground}/{total_pixels} ({ratio:.1f}%)"
    )


def main():
    """示例主流程。

    Returns:
        int: 退出码，0 表示成功。
    """
    args = parse_args()

    output_dir = prepare_output_dir(args.output)
    # 日志同时写终端和输出目录下的文件，便于事后排查
    cutie_cpp.setup_logging(
        level=logging.DEBUG if args.debug else logging.INFO, log_dir=output_dir
    )

    logger.info("=== Cutie Python 分割示例开始 ===")
    logger.info(f"视频: {args.video}")
    logger.info(f"掩码: {args.mask}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"跳帧: {args.frame_skip}")

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        logger.error(f"打开视频失败: {args.video}")
        return 1

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    logger.info(f"视频信息: fps={source_fps:.1f} 帧数={frame_count:.0f}")

    index_mask, object_ids = load_index_mask(args.mask)

    save_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    saver = None
    if not args.no_save_frames:
        saver = start_saver_thread(output_dir, save_queue)

    video_writer = None
    video_path = output_dir / "output_mask.mp4"

    stride = max(1, args.frame_skip + 1)
    frame_index = 0
    processed = 0
    durations = []

    try:
        with cutie_cpp.VideoSegmenter(build_config(args)) as segmenter:
            logger.info(f"模型: {segmenter.config.model_prefix}")

            while True:
                ok, frame = capture.read()
                # 部分解码器在文件末尾会返回 True 但给出空帧
                if not ok or frame is None or frame.size == 0:
                    break

                # 首帧必须推理以初始化内存，之后按 stride 跳帧
                if frame_index != 0 and frame_index % stride != 0:
                    frame_index += 1
                    continue

                start = time.perf_counter()
                if frame_index == 0:
                    result = segmenter.step(
                        frame, mask=index_mask, object_ids=object_ids
                    )
                else:
                    result = segmenter.step(frame)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                durations.append(elapsed_ms)

                log_frame_stats(result, elapsed_ms)

                visualization = cutie_cpp.overlay_mask(
                    frame, result.index_mask, alpha=0.4
                )

                if args.save_video:
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        fps = source_fps if source_fps > 0 else 25.0
                        height, width = visualization.shape[:2]
                        video_writer = cv2.VideoWriter(
                            str(video_path), fourcc, fps, (width, height)
                        )
                        if not video_writer.isOpened():
                            logger.error(f"打开视频写出失败: {video_path}")
                            video_writer = None
                        else:
                            logger.info(
                                f"写出视频: {video_path} ({width}x{height} @ {fps:.1f} fps)"
                            )
                    if video_writer is not None:
                        # 跳帧时重复写入，保持输出视频与源视频时间轴一致
                        repeat = 1 if frame_index == 0 else stride
                        for _ in range(repeat):
                            video_writer.write(visualization)

                if saver is not None:
                    # 队列满时阻塞，形成反压以限制内存
                    save_queue.put((frame_index, visualization.copy()))

                if args.visualize:
                    cv2.imshow("Cutie Python Demo", visualization)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        logger.info("用户中断")
                        break

                processed += 1
                frame_index += 1
    except cutie_cpp.CutieError as exc:
        logger.error(f"分割失败: {exc}")
        return 1
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()
            logger.info(f"输出视频已保存: {video_path}")
        if saver is not None:
            save_queue.put(None)  # 通知保存线程退出
            saver.join(timeout=30.0)
        if args.visualize:
            cv2.destroyAllWindows()

    if durations:
        # 首帧含模型预热，单独统计以免拉偏平均值
        warmup = durations[0]
        steady = durations[1:] or durations
        logger.info(
            f"完成: 读取 {frame_index} 帧，推理 {processed} 帧 | "
            f"首帧 {warmup:.1f} ms | 稳态均值 {np.mean(steady):.1f} ms "
            f"({1000.0 / np.mean(steady):.1f} FPS)"
        )
    logger.info(f"结果已保存到 {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
