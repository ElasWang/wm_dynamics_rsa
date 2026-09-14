
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def features_to_rdm(
        features: np.ndarray,
        metric: str = 'correlation',
        normalize: bool = True,
        **kwargs
) -> np.ndarray:
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.shape[1] == 1 and metric == 'correlation':
        metric = 'euclidean'
    if metric == 'correlation':
        rdm = 1 - np.corrcoef(features)
        rdm = np.nan_to_num(rdm, nan=0.0)
    elif metric == 'euclidean':
        rdm = squareform(pdist(features, metric='euclidean'))
    elif metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_distances
        rdm = cosine_distances(features)
    elif metric == 'mahalanobis':
        cov = np.cov(features.T)
        try:
            cov_inv = np.linalg.pinv(cov)
            rdm = squareform(pdist(features, metric='mahalanobis', VI=cov_inv))
        except np.linalg.LinAlgError:
            rdm = squareform(pdist(features, metric='euclidean'))
    elif metric == 'target_priority':
        trial_loads = kwargs['trial_loads']
        n = features.shape[0]
        rdm = np.zeros((n, n))
        task = features[:, 0]
        rank = features[:, 1]
        for i in range(n):
            K_i = trial_loads[i]
            task_i = task[i]
            rank_i = rank[i]
            for j in range(n):
                if i == j:
                    continue
                K_j = trial_loads[j]
                task_j = task[j]
                rank_j = rank[j]
                if task_i == 1 and task_j == 1:
                    rdm[i, j] = abs(rank_i - rank_j)
                elif task_i != task_j:
                    K_max = max(K_i, K_j)
                    rdm[i, j] = float(K_max)
                else:
                    rdm[i, j] = 0.0
    elif metric == 'adjacency_rdm_binary':
        pos = features.flatten()
        diff = np.abs(pos[:, None] - pos[None, :])
        rdm = np.ones_like(diff, dtype=float)
        rdm[diff == 1] = 0.0
        np.fill_diagonal(rdm, 0.0)
    elif metric == 'adjacency_rdm_reciprocal':
        pos = features.flatten()
        diff = np.abs(pos[:, None] - pos[None, :])
        rdm = diff / (1 + diff)
        np.fill_diagonal(rdm, 0.0)
    elif metric == 'manhattan':
        t = features.flatten()
        rdm = np.abs(t[:, None] - t[None, :])
    if normalize:
        if rdm.max() > 0:
            rdm = rdm / rdm.max()
    np.fill_diagonal(rdm, 0)
    return rdm


def compute_rdm_from_metadata(
        metadata: pd.DataFrame,
        feature_extractor: callable,
        metric: str = 'correlation',
        **kwargs
) -> np.ndarray:
    features = feature_extractor(metadata, **kwargs)
    return features_to_rdm(features, metric=metric)
