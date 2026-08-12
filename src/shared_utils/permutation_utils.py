import numpy as np
import pandas as pd

from src.shared_utils.metadata_util import compute_positions


def generate_shuffled_null_distribution(
        metadata: pd.DataFrame,
        model_func,
        n_permutations: int = 1000,
        **func_kwargs
) -> np.ndarray:
    """
    生成无结构随机基线（M2）
    对位置标签随机打乱，重新计算模型RDM，重复n_perm次。
    返回相关系数的数组（长度=n_perm），用于构建零分布。
    """
    if 'position' not in metadata.columns:
        metadata['position'] = compute_positions(metadata)
    original_positions = metadata['position'].values.copy()

    null_rdms = []
    for _ in range(n_permutations):
        shuffled_pos = np.random.permutation(original_positions)
        metadata_shuffled = metadata.copy()
        metadata_shuffled['position'] = shuffled_pos
        rdm = model_func(metadata_shuffled, **func_kwargs)
        null_rdms.append(rdm)

    return np.array(null_rdms)
