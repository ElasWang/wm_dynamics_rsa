

import numpy as np
import pandas as pd

from .compute_rdm import features_to_rdm
from ..features.content import content_feature
from ..features.gradient import gradient_feature


def dynamic_mixture_rdm(
        metadata: pd.DataFrame,
        alpha: float = 0.5,
        lambda_param: float = 1.0,
        **kwargs
) -> np.ndarray:
    struct_feat = gradient_feature(metadata)
    rdm_struct = features_to_rdm(struct_feat, metric='euclidean', normalize=True)
    content_feat = content_feature(metadata, lambda_param=lambda_param)
    rdm_content = features_to_rdm(content_feat, metric='correlation', normalize=True)
    mixed = alpha * rdm_struct + (1.0 - alpha) * rdm_content
    np.fill_diagonal(mixed, 0.0)
    if mixed.max() > 0:
        mixed = mixed / mixed.max()
    return mixed
