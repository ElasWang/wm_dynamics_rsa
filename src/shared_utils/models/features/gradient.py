import numpy as np
import pandas as pd


def gradient_feature(metadata: pd.DataFrame, **kwargs) -> np.ndarray:
    df = metadata.sort_values(['run', 'trial', 'position']).copy()
    n = len(df)
    feat = np.zeros(n, dtype=float)
    for _, group in df.groupby(['run', 'trial']):
        idx = group.index
        if len(idx) == 0:
            continue
        pos = group['position'].values
        max_pos = pos.max()
        if max_pos > 1:
            feat[idx] = (pos - 1) / (max_pos - 1)
        else:
            feat[idx] = 0.0
    return feat.reshape(-1, 1)
