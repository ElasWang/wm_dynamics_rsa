import numpy as np
import pandas as pd

from src.shared_utils.simpson_rdm_util import load_simpson_rdm


def visual_rdm(metadata: pd.DataFrame, **kwargs) -> np.ndarray:
    """
    视觉相似性模型 (基于 Simpson 混淆矩阵)
    -根据 metadata 中的字母，构建 NxN RDM。
    """
    visual_similarity_root = kwargs.get('visual_similarity_root')
    full_rdm = load_simpson_rdm(visual_similarity_root)
    letters = metadata['letter'].str.upper().values
    idx = np.array([ord(l) - ord('A') for l in letters], dtype=int)
    n = len(metadata)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = full_rdm[idx[i], idx[j]]
    return rdm
