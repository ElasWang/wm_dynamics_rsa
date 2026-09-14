from pathlib import Path

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from src.shared_utils.plotting import set_publication_style, save_fig


def plot_model_rdm_heatmap(rdm, model_name, n_events=64, save_path=None, show=True):

    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    rdm_sub = rdm[:n_events, :n_events]
    im = ax.imshow(rdm_sub, cmap='viridis', origin='upper')
    ax.set_title(f'{model_name} RDM (first {n_events} events)')
    ax.set_xlabel('Events')
    ax.set_ylabel('Events')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        save_fig(fig, save_path)
    plt.close(fig)


def plot_model_rdm_dendrogram(rdm, model_name, n_events=64, save_path=None, show=True):

    set_publication_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    rdm_sub = rdm[:n_events, :n_events]
    dist_vec = squareform(rdm_sub, checks=False)
    Z = linkage(dist_vec, method='average')
    dendrogram(Z, ax=ax, labels=[f'{i + 1}' for i in range(n_events)], leaf_rotation=90, leaf_font_size=8)
    ax.set_title(f'{model_name} RDM Hierarchical Clustering (UPGMA, first {n_events})')
    ax.set_xlabel('Event Index')
    ax.set_ylabel('Distance')
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        save_fig(fig, save_path)
    plt.close(fig)
