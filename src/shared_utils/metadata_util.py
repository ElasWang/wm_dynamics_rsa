
import pandas as pd


def compute_positions(metadata: pd.DataFrame) -> pd.DataFrame:
    if 'position' in metadata.columns and (metadata['position'] != 0).any():
        return metadata
    df_sorted = metadata.sort_values(['run', 'trial', 'onset']).copy()
    df_sorted['position_calc'] = df_sorted.groupby(['run', 'trial']).cumcount() + 1
    positions = df_sorted.set_index(df_sorted.index)['position_calc']
    metadata['position'] = positions.reindex(metadata.index).fillna(0).astype(int).values
    return metadata


def filter_epochs(epochs):
    mask = epochs.metadata['task_role'].isin(['to_remember', 'to_ignore'])
    epochs = epochs[mask]
    return epochs


def filter_metadata(metadatas: pd.DataFrame) -> pd.DataFrame:
    if 'task_role' not in metadatas.columns:
        return metadatas
    mask = metadatas['task_role'].isin(['to_remember', 'to_ignore'])
    return metadatas[mask]