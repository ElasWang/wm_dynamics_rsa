import numpy as np
import pandas as pd


def aggregate_rdm_to_condition_level(
        rdm_event: np.ndarray,
        metadata: pd.DataFrame,
        condition_col: str = 'position'
) -> np.ndarray:
    positions = metadata[condition_col].values
    unique_positions = np.sort(np.unique(positions))
    n_cond = len(unique_positions)
    rdm_cond = np.zeros((n_cond, n_cond))
    for i, pos_i in enumerate(unique_positions):
        idx_i = np.where(positions == pos_i)[0]
        for j, pos_j in enumerate(unique_positions):
            idx_j = np.where(positions == pos_j)[0]
            block = rdm_event[np.ix_(idx_i, idx_j)]
            rdm_cond[i, j] = np.mean(block)
    np.fill_diagonal(rdm_cond, 0.0)
    return rdm_cond
