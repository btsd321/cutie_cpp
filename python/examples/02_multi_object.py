#!/usr/bin/env python3
"""多目标跟踪示例，演示中途删除目标。

与 01 的区别：掩码含多个目标 ID，并在指定帧调用 delete_objects()
停止跟踪其中一个，观察后续帧的目标列表变化。

若手头没有多目标掩码，可用 --split-mask 把单目标掩码按垂直中线拆成两个目标，
用于演示接口行为。

用法：
    python 02_multi_object.py --video FILE --mask FILE [--drop-object ID] [--drop-at N]
"""

import argparse
import logging
import sys

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
        description="Cutie 多目标跟踪示例",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", "-i", required=True, help="输入视频文件")
    parser.add_argument("--mask", "-k", required=True, help="首帧掩码（像素值 = 目标 ID）")
    parser.add_argument("--model-dir", "-m", default=None, help="ONNX 目录，省略则自动搜索")
    parser.add_argument(
        "--split-mask",
        action="store_true",
        help="把单目标掩码按垂直中线拆为目标 1 和 2（用于演示）",
    )
    parser.add_argument(
        "--drop-object", type=int, default=None, help="要中途停止跟踪的目标 ID"
    )
    parser.add_argument(
        "--drop-at", type=int, default=20, help="在第 N 帧执行 delete_objects"
    )
    parser.add_argument("--max-frames", type=int, default=60, help="最多处理多少帧")
    parser.add_argument("--output", "-o", default=None, help="可选：叠加视频输出路径")
    return parser.parse_args()


def load_mask(mask_path, split):
    """读取掩码，必要时拆分为两个目标。

    Args:
        mask_path (str): 掩码文件路径。
        split (bool): 是否按垂直中线把前景拆成目标 1 和 2。

    Returns:
        tuple[np.ndarray, list[int]]: (int32 索引掩码, 目标 ID 列表)。

    Raises:
        SystemExit: 掩码读取失败或没有前景时退出。
    """
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        logger.error(f"读取掩码失败: {mask_path}")
        sys.exit(1)

    foreground = mask_img > 0
    if not foreground.any():
        logger.error(f"掩码 {mask_path} 中没有前景像素")
        sys.exit(1)

    index_mask = np.zeros(mask_img.shape, dtype=np.int32)

    if split:
        # 取前景的水平中点，左半为目标 1、右半为目标 2
        columns = np.where(foreground.any(axis=0))[0]
        mid_column = int((columns[0] + columns[-1]) // 2)
        index_mask[foreground] = 1
        index_mask[foreground & (np.arange(mask_img.shape[1]) > mid_column)] = 2
        logger.info(f"掩码已按第 {mid_column} 列拆分为目标 1 / 2")
    else:
        raw_ids = [int(v) for v in np.unique(mask_img) if v != 0]
        if raw_ids == [255]:
            logger.info("检测到 0/255 二值掩码，归一为单目标 ID 1")
            index_mask[foreground] = 1
        else:
            index_mask = mask_img.astype(np.int32)

    object_ids = [int(v) for v in np.unique(index_mask) if v != 0]
    logger.info(f"跟踪 {len(object_ids)} 个目标: {object_ids}")
    return index_mask, object_ids


def main():
    """示例主流程。

    Returns:
        int: 退出码，0 表示成功。
    """
    args = parse_args()
    cutie_cpp.setup_logging(level=logging.INFO)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        logger.error(f"打开视频失败: {args.video}")
        return 1

    index_mask, object_ids = load_mask(args.mask, args.split_mask)

    if args.drop_object is not None and args.drop_object not in object_ids:
        logger.error(f"--drop-object {args.drop_object} 不在掩码目标 {object_ids} 中")
        return 1

    config = cutie_cpp.CutieConfig.base_default(args.model_dir)
    writer = None

    try:
        with cutie_cpp.VideoSegmenter(config) as segmenter:
            for frame_index in range(args.max_frames):
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    break

                # 到达指定帧时停止跟踪某个目标，其内存会被一并释放
                if args.drop_object is not None and frame_index == args.drop_at:
                    logger.info(f"第 {frame_index} 帧: 停止跟踪目标 {args.drop_object}")
                    segmenter.delete_objects([args.drop_object])
                    logger.info(f"剩余目标: {segmenter.object_ids}")

                if frame_index == 0:
                    result = segmenter.step(frame, mask=index_mask, object_ids=object_ids)
                else:
                    result = segmenter.step(frame)

                areas = {
                    object_id: result.object_area(object_id)
                    for object_id in result.object_ids
                }
                logger.info(f"第 {frame_index} 帧: 目标面积 {areas}")

                if args.output is not None:
                    visualization = cutie_cpp.overlay_mask(frame, result.index_mask)
                    if writer is None:
                        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                        height, width = visualization.shape[:2]
                        writer = cv2.VideoWriter(
                            args.output,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (width, height),
                        )
                    writer.write(visualization)
    except cutie_cpp.CutieError as exc:
        logger.error(f"分割失败: {exc}")
        return 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            logger.info(f"输出视频已保存: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
