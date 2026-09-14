import numpy as np
import pandas as pd


def content_feature(metadata: pd.DataFrame, **kwargs) -> np.ndarray:
    lambda_param = kwargs['content_lambda_param']
    n = len(metadata)
    feat = np.zeros((n, 27))
    for i, row in metadata.iterrows():
        if row['color'] == 'black':
            idx = ord(row['letter'].upper()) - ord('A')
            if 0 <= idx < 26:
                feat[i, idx] = 1.0
        else:
            idx = ord(row['letter'].upper()) - ord('A')
            if 0 <= idx < 26:
                feat[i, idx] = 1.0 - lambda_param
            feat[i, 26] = lambda_param
    return feat
