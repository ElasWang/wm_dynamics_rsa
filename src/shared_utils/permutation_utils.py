from typing import Callable

import numpy as np
import pandas as pd


def generate_shuffled_null_distribution(
        metadata: pd.DataFrame,
        model_func: Callable,
        n_permutations: int = 1000,
        **func_kwargs
) -> np.ndarray:
    df = metadata.copy()
    group_cols = ['run', 'trial']
    null_rdms = []
    for _ in range(n_permutations):
        df_shuffled = df.copy()
        for _, group in df_shuffled.groupby(group_cols):
            idx = group.index
            pos_vals = group['position'].values.copy()
            np.random.shuffle(pos_vals)
            df_shuffled.loc[idx, 'position'] = pos_vals
        rdm = model_func(df_shuffled, **func_kwargs)
        null_rdms.append(rdm)
    return np.array(null_rdms)


def fisher_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))
