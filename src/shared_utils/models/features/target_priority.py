import numpy as np
import pandas as pd


def target_priority_feature(metadata: pd.DataFrame, **kwargs) -> np.ndarray:
    df = metadata.sort_values(['run', 'trial', 'position']).reset_index(drop=True)
    n = len(df)
    feat = np.zeros((n, 2), dtype=float)
    for trial_id, group in df.groupby(['run', 'trial']):
        black_indices = group[group['color'] == 'black'].index
        K = len(black_indices)
        if K > 0:
            feat[black_indices, 0] = 1.0
            for idx, rank in zip(black_indices, range(1, K + 1)):
                feat[idx, 1] = float(rank)
    return feat
