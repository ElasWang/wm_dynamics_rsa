import numpy as np
import pandas as pd


def binary_color_feature(metadata: pd.DataFrame, **kwargs) -> np.ndarray:
    """
    颜色控制模型
    """
    colors = metadata['color'].map({'black': 1.0, 'green': 0.0}).fillna(0.0).values
    features = colors.reshape(-1, 1)
    return features
