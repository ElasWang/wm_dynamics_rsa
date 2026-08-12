

import numpy as np
import pandas as pd

def compute_positions(metadata: pd.DataFrame) -> np.DataFrame:
    """
    为metadata添加position列
    """
    if 'position' in metadata.columns and (metadata['position'] != 0).any():
        return metadata

    df_sorted = metadata.sort_values(['run', 'trial', 'onset']).copy()
    df_sorted['position_calc'] = df_sorted.groupby(['run', 'trial']).cumcount() + 1

    positions = df_sorted.set_index(df_sorted.index)['position_calc']
    metadata['position'] = positions.reindex(metadata.index).fillna(0).astype(int).values
    return metadata
