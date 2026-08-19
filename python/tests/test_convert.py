"""输入数组规范化与可视化的单元测试。

不需要 GPU。重点验证零拷贝路径的前提条件：
输入已是 uint8 / C 连续时，_prepare_image 不应产生副本。
"""

import numpy as np
import pytest

import cutie_cpp
from cutie_cpp.segmenter import _prepare_image, _prepare_mask


class TestPrepareImage:
    """输入帧规范化的测试。"""

    def test_contiguous_uint8_is_not_copied(self):
        """已连续的 uint8 数组必须原样返回，否则零拷贝路径失效。

        这是绑定层性能的核心前提：np.ascontiguousarray 对已连续数组
        返回同一对象，不产生副本。
        """
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        prepared = _prepare_image(image)
        assert prepared is image, "已连续的数组不应被拷贝"
        # 共享内存的双向验证
        prepared[0, 0, 0] = 42
        assert image[0, 0, 0] == 42

    def test_non_contiguous_input_is_made_contiguous(self):
        """非连续输入（如切片）应被整理为连续数组。"""
        # 通过步长切片构造非连续视图
        source = np.zeros((8, 20, 3), dtype=np.uint8)
        view = source[:, ::2, :]
        assert not view.flags["C_CONTIGUOUS"]

        prepared = _prepare_image(view)
        assert prepared.flags["C_CONTIGUOUS"]
        assert prepared.shape == view.shape

    def test_bgr_channel_order_preserved(self):
        """通道顺序不应被改动，库约定输入为 BGR。"""
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[..., 0] = 1  # B
        image[..., 1] = 2  # G
        image[..., 2] = 3  # R
        prepared = _prepare_image(image)
        assert prepared[0, 0].tolist() == [1, 2, 3]

    def test_non_array_rejected(self):
        """非 ndarray 输入应报 InferenceError。"""
        with pytest.raises(cutie_cpp.InferenceError, match="必须是 numpy 数组"):
            _prepare_image([[1, 2, 3]])

    def test_grayscale_rejected(self):
        """二维灰度图不满足 (H, W, 3) 要求。"""
        with pytest.raises(cutie_cpp.InferenceError, match=r"\(H, W, 3\)"):
            _prepare_image(np.zeros((8, 10), dtype=np.uint8))

    def test_rgba_rejected(self):
        """4 通道图像应被拒绝，避免通道错位产生错误结果。"""
        with pytest.raises(cutie_cpp.InferenceError, match=r"\(H, W, 3\)"):
            _prepare_image(np.zeros((8, 10, 4), dtype=np.uint8))

    def test_float_dtype_rejected(self):
        """float 输入应显式报错，而不是静默转换。

        静默转换会掩盖调用方的归一化错误（例如已归一到 0~1 的图像）。
        """
        with pytest.raises(cutie_cpp.InferenceError, match="uint8"):
            _prepare_image(np.zeros((8, 10, 3), dtype=np.float32))


class TestPrepareMask:
    """掩码规范化的测试。"""

    def test_none_gives_empty_array(self):
        """mask 为 None 时应得到空数组，表示"本帧不提供掩码"。"""
        prepared = _prepare_mask(None, (8, 10, 3))
        assert prepared.size == 0
        assert prepared.dtype == np.int32

    def test_uint8_converted_to_int32(self):
        """uint8 掩码（imread 的常见输出）应转为 int32 以匹配 CV_32SC1。"""
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2, 3] = 7
        prepared = _prepare_mask(mask, (8, 10, 3))
        assert prepared.dtype == np.int32
        assert prepared[2, 3] == 7

    def test_int32_values_preserved(self):
        """已是 int32 时目标 ID 应原样保留。"""
        mask = np.zeros((8, 10), dtype=np.int32)
        mask[1, 1] = 300  # 超出 uint8 范围，验证没有被截断
        prepared = _prepare_mask(mask, (8, 10, 3))
        assert prepared[1, 1] == 300

    def test_shape_mismatch_rejected(self):
        """掩码与图像尺寸不一致应报错，否则 C++ 侧会读到错位数据。"""
        mask = np.zeros((4, 5), dtype=np.int32)
        with pytest.raises(cutie_cpp.InferenceError, match="不一致"):
            _prepare_mask(mask, (8, 10, 3))

    def test_three_dim_mask_rejected(self):
        """掩码必须是二维索引图。"""
        with pytest.raises(cutie_cpp.InferenceError, match=r"\(H, W\)"):
            _prepare_mask(np.zeros((8, 10, 3), dtype=np.int32), (8, 10, 3))


class TestVisualize:
    """可视化工具的测试。"""

    def test_original_image_not_modified(self):
        """overlay_mask 必须返回新图，不能就地改动输入。"""
        image = np.full((6, 6, 3), 100, dtype=np.uint8)
        mask = np.zeros((6, 6), dtype=np.int32)
        mask[1:4, 1:4] = 1

        cutie_cpp.overlay_mask(image, mask)
        assert np.all(image == 100), "输入图像被就地修改了"

    def test_background_pixels_untouched(self):
        """掩码为 0 的位置不应被染色。"""
        image = np.full((6, 6, 3), 100, dtype=np.uint8)
        mask = np.zeros((6, 6), dtype=np.int32)
        mask[2:4, 2:4] = 1

        output = cutie_cpp.overlay_mask(image, mask)
        assert output[0, 0].tolist() == [100, 100, 100]

    def test_alpha_blending_math(self):
        """目标内部像素应按 alpha 与调色板颜色混合。"""
        image = np.full((8, 8, 3), 100, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.int32)
        mask[2:7, 2:7] = 1

        output = cutie_cpp.overlay_mask(image, mask, alpha=0.5, draw_contour=False)
        color = cutie_cpp.visualize.color_for(1).astype(np.float32)
        expected = (100 * 0.5 + color * 0.5).astype(np.uint8)
        # 取区域内部点，避开边界
        assert output[4, 4].tolist() == expected.tolist()

    def test_alpha_zero_keeps_image(self):
        """alpha=0 且不画轮廓时应与原图一致。"""
        image = np.full((6, 6, 3), 100, dtype=np.uint8)
        mask = np.zeros((6, 6), dtype=np.int32)
        mask[1:5, 1:5] = 1

        output = cutie_cpp.overlay_mask(image, mask, alpha=0.0, draw_contour=False)
        assert np.array_equal(output, image)

    def test_distinct_colors_per_object(self):
        """不同目标应得到不同颜色，便于人工辨识。"""
        assert not np.array_equal(
            cutie_cpp.visualize.color_for(1), cutie_cpp.visualize.color_for(2)
        )

    def test_palette_wraps_for_large_ids(self):
        """超出调色板的 ID 应循环取色，且不落到背景黑色。"""
        color = cutie_cpp.visualize.color_for(100)
        assert color.shape == (3,)
        assert color.tolist() != [0, 0, 0]

    def test_shape_mismatch_rejected(self):
        """掩码与图像尺寸不一致应报 ValueError。"""
        with pytest.raises(ValueError, match="不一致"):
            cutie_cpp.overlay_mask(
                np.zeros((6, 6, 3), np.uint8), np.zeros((4, 4), np.int32)
            )

    @pytest.mark.parametrize("alpha", [-0.1, 1.5])
    def test_alpha_out_of_range_rejected(self, alpha):
        """alpha 超出 [0, 1] 应报错。"""
        with pytest.raises(ValueError, match="alpha"):
            cutie_cpp.overlay_mask(
                np.zeros((6, 6, 3), np.uint8), np.zeros((6, 6), np.int32), alpha=alpha
            )

    def test_mask_to_color_maps_ids(self):
        """mask_to_color 应把每个 ID 渲染为其调色板颜色，背景为黑。"""
        mask = np.zeros((4, 4), dtype=np.int32)
        mask[0, 0] = 1
        mask[1, 1] = 2

        colored = cutie_cpp.mask_to_color(mask)
        assert colored.shape == (4, 4, 3)
        assert colored[0, 0].tolist() == cutie_cpp.visualize.color_for(1).tolist()
        assert colored[1, 1].tolist() == cutie_cpp.visualize.color_for(2).tolist()
        assert colored[3, 3].tolist() == [0, 0, 0]
