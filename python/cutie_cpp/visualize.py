"""分割结果的可视化工具。

全部用 numpy 实现，不依赖 opencv-python，因此本库的核心安装无需图像库。
调色板沿用 DAVIS / PASCAL VOC 的经典配色，便于与其它 VOS 工作对照。
"""

import numpy as np


# DAVIS 数据集常用配色（BGR 顺序，与 OpenCV 一致），索引 0 留给背景
_PALETTE_BGR = np.array(
    [
        [0, 0, 0],        # 0 背景（不绘制）
        [50, 50, 220],    # 1 红
        [50, 220, 50],    # 2 绿
        [220, 100, 50],   # 3 蓝
        [50, 200, 220],   # 4 黄
        [220, 50, 200],   # 5 品红
        [220, 200, 50],   # 6 青
        [100, 150, 250],  # 7 橙
        [150, 100, 200],  # 8 紫
    ],
    dtype=np.uint8,
)


def color_for(object_id):
    """获取目标 ID 对应的 BGR 颜色。

    超出调色板范围时按模循环取色，保证任意 ID 都有稳定颜色。

    Args:
        object_id (int): 目标 ID，须为正数。

    Returns:
        np.ndarray: 形状 (3,)、dtype uint8 的 BGR 颜色。
    """
    if object_id <= 0:
        return _PALETTE_BGR[0]

    # 前 8 个 ID 直接对应调色板，之后循环复用（跳过索引 0 的背景色）
    index = (object_id - 1) % (len(_PALETTE_BGR) - 1) + 1
    return _PALETTE_BGR[index]


def overlay_mask(image, index_mask, alpha=0.5, draw_contour=True):
    """把索引掩码以半透明色块叠加到图像上。

    Args:
        image (np.ndarray): BGR 底图，形状 (H, W, 3)、dtype uint8。
        index_mask (np.ndarray): 索引掩码，形状 (H, W)，像素值为目标 ID。
        alpha (float): 掩码不透明度，取值 0.0~1.0。0 为完全透明，1 为完全覆盖。
        draw_contour (bool): 是否用不透明的实色描出目标边缘，提升辨识度。

    Returns:
        np.ndarray: 叠加后的新图像，形状与 image 相同、dtype uint8。原图不被修改。

    Raises:
        ValueError: image 与 index_mask 尺寸不一致，或 alpha 超出范围时抛出。
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image 必须是 (H, W, 3) 的 BGR 图像，收到 {image.shape}")
    if index_mask.shape != image.shape[:2]:
        raise ValueError(
            f"掩码尺寸 {index_mask.shape} 与图像尺寸 {image.shape[:2]} 不一致"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须在 [0, 1] 区间，收到 {alpha}")

    output = image.copy()
    object_ids = np.unique(index_mask)

    for object_id in object_ids:
        if object_id == 0:  # 背景不绘制
            continue

        region = index_mask == object_id
        color = color_for(int(object_id)).astype(np.float32)

        # 只在目标区域做混合，避免整图浮点运算
        blended = output[region].astype(np.float32) * (1.0 - alpha) + color * alpha
        output[region] = blended.astype(np.uint8)

        if draw_contour:
            output[_boundary_of(region)] = color.astype(np.uint8)

    return output


def _boundary_of(region):
    """求二值区域的边界像素。

    用四邻域错位比较代替形态学腐蚀，避免引入 scipy / opencv 依赖。

    Args:
        region (np.ndarray): 形状 (H, W)、dtype bool 的区域掩码。

    Returns:
        np.ndarray: 形状 (H, W)、dtype bool 的边界掩码。
    """
    boundary = np.zeros_like(region)

    # 与四个方向的邻居比较，任一方向不同即为边界
    boundary[:-1, :] |= region[:-1, :] != region[1:, :]
    boundary[1:, :] |= region[1:, :] != region[:-1, :]
    boundary[:, :-1] |= region[:, :-1] != region[:, 1:]
    boundary[:, 1:] |= region[:, 1:] != region[:, :-1]

    # 只保留区域内侧的边界，避免把背景像素染色
    return boundary & region


def mask_to_color(index_mask):
    """把索引掩码渲染为彩色图像。

    Args:
        index_mask (np.ndarray): 索引掩码，形状 (H, W)，像素值为目标 ID。

    Returns:
        np.ndarray: 形状 (H, W, 3)、dtype uint8 的 BGR 彩色掩码，背景为黑。
    """
    height, width = index_mask.shape
    output = np.zeros((height, width, 3), dtype=np.uint8)

    for object_id in np.unique(index_mask):
        if object_id == 0:
            continue
        output[index_mask == object_id] = color_for(int(object_id))

    return output
