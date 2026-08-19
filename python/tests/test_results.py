"""SegmentationResult 的单元测试。

不需要 GPU，直接用构造好的 numpy 数组测试结果类的访问器。
"""

import numpy as np
import pytest

from cutie_cpp.results import SegmentationResult


def make_result(with_prob=False):
    """构造一个含两个目标的测试结果。

    Args:
        with_prob (bool): 是否附带概率图。

    Returns:
        SegmentationResult: 构造的结果对象。
    """
    index_mask = np.zeros((6, 8), dtype=np.int32)
    index_mask[1:3, 1:4] = 1  # 6 个像素
    index_mask[4:5, 2:6] = 2  # 4 个像素

    prob = None
    if with_prob:
        # 通道 0 背景，通道 1/2 对应目标 1/2
        prob = np.zeros((3, 6, 8), dtype=np.float32)
        prob[1][index_mask == 1] = 0.9
        prob[2][index_mask == 2] = 0.8

    return SegmentationResult(
        index_mask=index_mask, object_ids=[1, 2], prob=prob, frame_index=7
    )


class TestAccessors:
    """基础属性的测试。"""

    def test_shape_matches_mask(self):
        """shape 应返回掩码的空间尺寸。"""
        assert make_result().shape == (6, 8)

    def test_num_objects_counts_ids(self):
        """num_objects 应等于目标 ID 数量。"""
        assert make_result().num_objects == 2

    def test_frame_index_preserved(self):
        """帧序号应原样保留，便于关联原始视频帧。"""
        assert make_result().frame_index == 7


class TestBinaryMask:
    """二值掩码提取的测试。"""

    def test_selects_only_target_object(self):
        """binary_mask 只应包含指定目标的像素。"""
        result = make_result()
        mask = result.binary_mask(1)
        assert mask.dtype == np.bool_
        assert mask.sum() == 6
        # 目标 2 的位置不应被选中
        assert not mask[4, 3]

    def test_unknown_object_rejected(self):
        """不存在的目标 ID 应报 KeyError。"""
        with pytest.raises(KeyError, match="不在活跃列表"):
            make_result().binary_mask(99)


class TestObjectArea:
    """面积统计的测试。"""

    def test_counts_pixels_per_object(self):
        """object_area 应返回各目标的像素数。"""
        result = make_result()
        assert result.object_area(1) == 6
        assert result.object_area(2) == 4

    def test_returns_python_int(self):
        """返回值应是 Python int，方便直接用于日志与 JSON 序列化。"""
        assert isinstance(make_result().object_area(1), int)


class TestObjectProb:
    """概率图提取的测试。"""

    def test_maps_object_to_correct_channel(self):
        """目标应映射到正确的概率通道（通道 0 是背景）。"""
        result = make_result(with_prob=True)
        assert result.object_prob(1).max() == pytest.approx(0.9)
        assert result.object_prob(2).max() == pytest.approx(0.8)

    def test_shape_matches_mask(self):
        """概率图的空间尺寸应与掩码一致。"""
        assert make_result(with_prob=True).object_prob(1).shape == (6, 8)

    def test_missing_prob_reports_how_to_enable(self):
        """未下载概率图时的报错应提示 return_prob 参数。"""
        with pytest.raises(ValueError, match="return_prob"):
            make_result(with_prob=False).object_prob(1)

    def test_unknown_object_rejected(self):
        """不存在的目标 ID 应报 KeyError。"""
        with pytest.raises(KeyError, match="不在活跃列表"):
            make_result(with_prob=True).object_prob(99)
