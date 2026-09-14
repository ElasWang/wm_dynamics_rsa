import numpy as np
import pandas as pd


def chunking_feature(metadata: pd.DataFrame, **kwargs) -> np.ndarray:

    chunk_pattern = kwargs['chunking_pattern']
    df = metadata.sort_values(['run', 'trial', 'position']).copy()
    n = len(df)
    feat = np.zeros(n, dtype=float)

    for _, group in df.groupby(['run', 'trial']):
        idx = group.index
        if len(idx) == 0:
            continue
        colors = group['color'].values
        black_indices = [i for i, c in enumerate(colors) if c == 'black']
        for i, c in enumerate(colors):
            if c != 'black':
                feat[idx[i]] = -1.0
        n_black = len(black_indices)
        if n_black == 0:
            continue
        group_id = 1
        count_in_group = 0
        pattern_iter = iter(chunk_pattern)
        current_capacity = next(pattern_iter, None)

        for j, original_idx in enumerate(black_indices):
            if current_capacity is None:
                feat[idx[original_idx]] = group_id
                continue
            if count_in_group >= current_capacity:
                try:
                    current_capacity = next(pattern_iter)
                    group_id += 1
                    count_in_group = 0
                except StopIteration:
                    current_capacity = None
                    group_id += 1
            feat[idx[original_idx]] = group_id
            count_in_group += 1

    return feat.reshape(-1, 1)
