#!/usr/bin/env python3
"""
将灰度图片二值化为 mask 图（前景=255，背景=0）。

Usage:
    python binarize_mask.py <input_image> [output_image] [--threshold T]

    input_image   - 输入灰度图路径
    output_image  - 输出路径（默认: 同目录下 <name>_binary.png）
    --threshold T - 固定阈值 0-255（默认: 使用 Otsu 自动阈值）
"""

import argparse
import os
import sys
import cv2


def binarize(input_path: str, output_path: str, threshold: int | None) -> None:
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: cannot read '{input_path}'", file=sys.stderr)
        sys.exit(1)

    if threshold is None:
        # Otsu 自动阈值
        t_val, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        print(f"Otsu threshold: {t_val:.1f}")
    else:
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        print(f"Fixed threshold: {threshold}")

    cv2.imwrite(output_path, binary)
    fg_pixels = int((binary > 0).sum())
    total = binary.size
    print(f"Saved: {output_path}  ({img.shape[1]}x{img.shape[0]})")
    print(f"Foreground: {fg_pixels} px ({100*fg_pixels/total:.1f}%)  "
          f"Background: {total - fg_pixels} px ({100*(total-fg_pixels)/total:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="灰度图 → 二值 mask")
    parser.add_argument("input", help="输入灰度图路径")
    parser.add_argument("output", nargs="?", help="输出路径（默认: <name>_binary.png）")
    parser.add_argument("--threshold", "-t", type=int, default=None,
                        help="固定阈值 0-255，不指定则使用 Otsu 自动阈值")
    args = parser.parse_args()

    if args.output is None:
        base, _ = os.path.splitext(args.input)
        args.output = base + "_binary.png"

    binarize(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()
