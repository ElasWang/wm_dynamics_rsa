import numpy as np
import pandas as pd


def onehot_feature(metadata: pd.DataFrame, **kwargs) -> np.ndarray:
    df = metadata.sort_values(['run', 'trial', 'position']).copy()
    trial_sizes = df.groupby(['run', 'trial']).size()
    if trial_sizes.empty:
        raise ValueError("没有找到有效的试次数据。")
    max_items = trial_sizes.max()
    df['temp_pos'] = df.groupby(['run', 'trial']).cumcount() + 1
    feat = df['temp_pos'].values.astype(int)
    onehot = np.eye(max_items, dtype=float)[feat - 1]
    onehot = np.round(onehot)
    return onehot
