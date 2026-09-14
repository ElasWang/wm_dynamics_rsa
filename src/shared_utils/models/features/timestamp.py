import numpy as np
import pandas as pd


def timestamp_feature(metadata: pd.DataFrame, **kwargs) -> np.ndarray:
    df = metadata.sort_values(['run', 'trial', 'position']).copy()
    df['trail_start'] = df.groupby(['run', 'trial'])['onset'].transform('min')
    df['t_relative'] = df['onset'] - df['trail_start']
    feat = df[['t_relative']].values.astype(float)
    return feat
