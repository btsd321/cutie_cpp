"""端到端冒烟测试。

需要 CUDA GPU 与真实 ONNX 模型，全部打 gpu 标记。
用合成图像而非真实视频，让测试不依赖额外数据文件。

运行：
    pytest python/tests -m gpu
"""

import numpy as np
import pytest

import cutie_cpp


pytestmark = pytest.mark.gpu

# 合成帧尺寸。取 16 的倍数以贴合模型的 pad 约束
FRAME_HEIGHT = 240
FRAME_WIDTH = 320


def make_frame(offset=0):
    """生成一帧含矩形前景的合成图像。

    Args:
        offset (int): 前景矩形的水平位移，用于模拟目标运动。

    Returns:
        np.ndarray: (H, W, 3) uint8 的 BGR 图像。
    """
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 40, dtype=np.uint8)
    left = 80 + offset
    # 前景用高亮度色块，与背景形成足够对比，便于模型跟踪
    frame[70:170, left : left + 100] = (200, 180, 160)
    return frame


def make_mask(offset=0):
    """生成与 make_frame 前景对应的索引掩码。

    Args:
        offset (int): 前景矩形的水平位移。

    Returns:
        np.ndarray: (H, W) int32 掩码，前景值为 1。
    """
    mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.int32)
    left = 80 + offset
    mask[70:170, left : left + 100] = 1
    return mask


class TestSingleObject:
    """单目标跟踪的端到端测试。"""

    def test_first_frame_returns_expected_shape(self, segmenter):
        """首帧结果的形状与 dtype 应与输入帧匹配。"""
        segmenter.reset()
        result = segmenter.step(make_frame(), mask=make_mask())

        assert result.index_mask.shape == (FRAME_HEIGHT, FRAME_WIDTH)
        assert result.index_mask.dtype == np.int32
        assert result.object_ids == [1]

    def test_object_ids_inferred_from_mask(self, segmenter):
        """省略 object_ids 时应从掩码非零像素自动推导。"""
        segmenter.reset()
        result = segmenter.step(make_frame(), mask=make_mask())
        assert result.object_ids == [1]

    def test_tracking_persists_across_frames(self, segmenter):
        """后续帧不给掩码也应继续跟踪到目标。"""
        segmenter.reset()
        segmenter.step(make_frame(), mask=make_mask())

        for offset in (5, 10, 15):
            result = segmenter.step(make_frame(offset))
            assert result.object_ids == [1]
            assert result.object_area(1) > 0, "跟踪丢失，前景面积为 0"

    def test_frame_index_increments(self, segmenter):
        """frame_index 应逐帧递增。"""
        segmenter.reset()
        assert segmenter.frame_index == 0

        segmenter.step(make_frame(), mask=make_mask())
        assert segmenter.frame_index == 1

        segmenter.step(make_frame(5))
        assert segmenter.frame_index == 2

    def test_mask_values_are_valid_ids(self, segmenter):
        """掩码像素值只能是 0（背景）或活跃目标 ID。"""
        segmenter.reset()
        result = segmenter.step(make_frame(), mask=make_mask())

        allowed = {0} | set(result.object_ids)
        assert set(np.unique(result.index_mask)).issubset(allowed)


class TestProbabilityMap:
    """概率图下载的测试。"""

    def test_prob_absent_by_default(self, segmenter):
        """默认不下载概率图，以节省每帧的 D2H 带宽。"""
        segmenter.reset()
        result = segmenter.step(make_frame(), mask=make_mask())
        assert result.prob is None

    def test_prob_shape_when_requested(self, segmenter):
        """return_prob=True 时应返回 [N+1, H, W] 的概率图。"""
        segmenter.reset()
        result = segmenter.step(make_frame(), mask=make_mask(), return_prob=True)

        assert result.prob is not None
        assert result.prob.shape == (2, FRAME_HEIGHT, FRAME_WIDTH)
        assert result.prob.dtype == np.float32

    def test_prob_channels_sum_to_one(self, segmenter):
        """各通道概率之和应为 1（softmax 输出的基本性质）。"""
        segmenter.reset()
        result = segmenter.step(make_frame(), mask=make_mask(), return_prob=True)
        channel_sum = result.prob.sum(axis=0)
        assert np.allclose(channel_sum, 1.0, atol=1e-3)


class TestMultiObject:
    """多目标与目标管理的测试。"""

    @staticmethod
    def make_two_object_mask():
        """生成含两个目标的掩码。

        Returns:
            np.ndarray: (H, W) int32 掩码，目标 ID 为 1 和 2。
        """
        mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.int32)
        mask[70:170, 40:130] = 1
        mask[70:170, 180:270] = 2
        return mask

    @staticmethod
    def make_two_object_frame():
        """生成与 make_two_object_mask 对应的合成帧。

        Returns:
            np.ndarray: (H, W, 3) uint8 的 BGR 图像。
        """
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 40, dtype=np.uint8)
        frame[70:170, 40:130] = (200, 60, 60)
        frame[70:170, 180:270] = (60, 200, 60)
        return frame

    def test_tracks_two_objects(self, segmenter):
        """两个目标应同时被跟踪。"""
        segmenter.reset()
        result = segmenter.step(
            self.make_two_object_frame(), mask=self.make_two_object_mask()
        )
        assert result.object_ids == [1, 2]
        assert segmenter.num_objects == 2

    def test_delete_object_removes_it(self, segmenter):
        """delete_objects 后该目标不应再出现在结果中。"""
        segmenter.reset()
        segmenter.step(self.make_two_object_frame(), mask=self.make_two_object_mask())

        segmenter.delete_objects([2])
        assert segmenter.object_ids == [1]

        result = segmenter.step(self.make_two_object_frame())
        assert result.object_ids == [1]
        assert 2 not in np.unique(result.index_mask)


class TestStateManagement:
    """状态与内存管理的测试。"""

    def test_reset_clears_objects_and_counter(self, segmenter):
        """reset 应清空目标与帧计数，使实例可复用于新视频。"""
        segmenter.reset()
        segmenter.step(make_frame(), mask=make_mask())
        assert segmenter.num_objects == 1

        segmenter.reset()
        assert segmenter.num_objects == 0
        assert segmenter.frame_index == 0

    def test_clear_memory_keeps_instance_usable(self, segmenter):
        """clear_memory 后实例仍可接收新的首帧掩码。"""
        segmenter.reset()
        segmenter.step(make_frame(), mask=make_mask())
        segmenter.clear_memory()

        result = segmenter.step(make_frame(), mask=make_mask())
        assert result.object_ids == [1]

    def test_clear_sensory_memory_does_not_drop_objects(self, segmenter):
        """清感知内存不应影响目标列表。"""
        segmenter.reset()
        segmenter.step(make_frame(), mask=make_mask())
        segmenter.clear_sensory_memory()
        assert segmenter.object_ids == [1]


class TestErrorHandling:
    """错误处理的测试。"""

    def test_mask_shape_mismatch_rejected(self, segmenter):
        """掩码与帧尺寸不一致应报 InferenceError。"""
        segmenter.reset()
        bad_mask = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(cutie_cpp.InferenceError, match="不一致"):
            segmenter.step(make_frame(), mask=bad_mask)

    def test_zero_object_id_rejected(self, segmenter):
        """目标 ID 0 是背景保留值，应被拒绝。"""
        segmenter.reset()
        with pytest.raises(cutie_cpp.InferenceError, match="不能为 0"):
            segmenter.step(make_frame(), mask=make_mask(), object_ids=[0])

    def test_use_after_close_rejected(self, model_location):
        """close 之后再调用 step 应报错，而不是访问已释放资源。"""
        model_dir, _ = model_location
        instance = cutie_cpp.VideoSegmenter(
            cutie_cpp.CutieConfig.base_default(model_dir)
        )
        instance.close()

        with pytest.raises(cutie_cpp.CutieError, match="已关闭"):
            instance.step(make_frame(), mask=make_mask())


class TestContextManager:
    """上下文管理器的测试。"""

    def test_closes_on_exit(self, model_location):
        """with 块退出后实例应处于关闭状态。"""
        model_dir, _ = model_location
        config = cutie_cpp.CutieConfig.base_default(model_dir)

        with cutie_cpp.VideoSegmenter(config) as instance:
            result = instance.step(make_frame(), mask=make_mask())
            assert result.object_ids == [1]

        with pytest.raises(cutie_cpp.CutieError, match="已关闭"):
            instance.step(make_frame())
