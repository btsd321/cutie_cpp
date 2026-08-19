"""分割结果的数据结构。

结果中的数组都已在 CPU（numpy），不持有任何 GPU 资源，
因此可以安全地跨帧长期保存或送入其它线程处理。
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmentationResult:
    """单帧分割结果。

    Attributes:
        index_mask (np.ndarray): 形状 (H, W)、dtype int32 的索引掩码，
            像素值即目标 ID，0 表示背景。尺寸与输入帧一致。
        object_ids (list[int]): 当前活跃的目标 ID 列表。
        prob (np.ndarray | None): 形状 (N+1, H, W)、dtype float32 的概率图，
            通道 0 为背景，通道 1..N 依次对应 object_ids。
            仅在 step(return_prob=True) 时有值，否则为 None。
        frame_index (int): 该结果对应的帧序号，从 0 开始。
    """

    index_mask: np.ndarray
    object_ids: list = field(default_factory=list)
    prob: np.ndarray = None
    frame_index: int = 0

    @property
    def shape(self):
        """掩码的空间尺寸。

        Returns:
            tuple[int, int]: (H, W)。
        """
        return self.index_mask.shape

    @property
    def num_objects(self):
        """活跃目标数量。

        Returns:
            int: 目标个数。
        """
        return len(self.object_ids)

    def binary_mask(self, object_id):
        """提取指定目标的二值掩码。

        Args:
            object_id (int): 目标 ID。

        Returns:
            np.ndarray: 形状 (H, W)、dtype bool 的掩码，该目标所在像素为 True。

        Raises:
            KeyError: object_id 不在当前活跃目标中时抛出。
        """
        if object_id not in self.object_ids:
            raise KeyError(
                f"目标 {object_id} 不在活跃列表 {self.object_ids} 中"
            )
        return self.index_mask == object_id

    def object_area(self, object_id):
        """统计指定目标的像素面积。

        Args:
            object_id (int): 目标 ID。

        Returns:
            int: 该目标占据的像素数。

        Raises:
            KeyError: object_id 不在当前活跃目标中时抛出。
        """
        return int(self.binary_mask(object_id).sum())

    def object_prob(self, object_id):
        """提取指定目标的概率图。

        Args:
            object_id (int): 目标 ID。

        Returns:
            np.ndarray: 形状 (H, W)、dtype float32 的概率图。

        Raises:
            ValueError: 结果中没有概率图（未用 return_prob=True 推理）时抛出。
            KeyError: object_id 不在当前活跃目标中时抛出。
        """
        if self.prob is None or self.prob.size == 0:
            raise ValueError(
                "结果中没有概率图。请在调用 step() 时传入 return_prob=True"
            )
        if object_id not in self.object_ids:
            raise KeyError(
                f"目标 {object_id} 不在活跃列表 {self.object_ids} 中"
            )
        # 通道 0 是背景，目标从通道 1 开始
        channel = self.object_ids.index(object_id) + 1
        return self.prob[channel]
