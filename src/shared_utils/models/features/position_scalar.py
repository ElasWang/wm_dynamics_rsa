import numpy as np
import pandas as pd


def position_scalar_feature(metadata: pd.DataFrame, **kwargs) -> np.ndarray:
    df = metadata.sort_values(['run', 'trial', 'onset']).reset_index(drop=True)
    pos = df['position'].values.astype(float)
    return pos.reshape(-1, 1)
