
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from src.shared_utils.plotting import set_publication_style, save_fig

log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def plot_erp_overlay(epochs, subject_id, conditions=None,picks=None,
        tmin=None,tmax=None,baseline=(-0.2, 0.0), save_path=None,show=True,):

    set_publication_style()
    if picks is None:
        candidates = ['FZ', 'CZ', 'OZ']
        picks = [ch for ch in candidates if ch in epochs.ch_names]
        if not picks:
            picks = epochs.ch_names[:3]
            logger.info(f"警告: 未找到标准电极，使用前3个通道: {picks}")
    available = [ch for ch in picks if ch in epochs.ch_names]
    if not available:
        raise ValueError(f"所选电极 {picks} 均不在 epochs 中。可用电极示例：{epochs.ch_names[:5]}")
    picks = available
    if conditions is None:
        if 'color' not in epochs.metadata.columns:
            raise ValueError("epochs.metadata 中没有 'color' 列，请提供 conditions 参数")
        colors = epochs.metadata['color'].unique()
        conditions = {c: f"color == '{c}'" for c in colors if c}
    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4), sharey=True)
    if len(picks) == 1:
        axes = [axes]

    for ax, ch in zip(axes, picks):
        for cond_name, query in conditions.items():
            epochs_sub = epochs[query]
            if len(epochs_sub) == 0:
                logger.info(f"警告: 条件 '{cond_name}' 无试次，跳过")
                continue
            evoked = epochs_sub.average(picks=ch)
            if baseline is not None and evoked.baseline is None:
                evoked.apply_baseline(baseline=baseline)
            ax.plot(evoked.times * 1000, evoked.data[0] * 1e6,
                    label=cond_name, linewidth=1.5)
        ax.set_title(ch)
        ax.set_xlabel('Time (ms)')
        ax.axhline(0, linestyle='--', color='gray', linewidth=0.8)
        ax.axvline(0, linestyle=':', color='gray', linewidth=0.8)
        if ax == axes[0]:
            ax.set_ylabel('Amplitude (µV)')
        ax.legend()
    if tmin is not None and tmax is not None:
        ax.set_xlim(tmin * 1000, tmax * 1000)
    else:
        ax.set_xlim(epochs.times[0] * 1000, epochs.times[-1] * 1000)
    fig.suptitle(f'(ERP Overlay by Color Subject {subject_id})', fontsize=14)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


def plot_topomap_at_times(epochs, subject_id,times_ms=[100, 300, 500],condition=None,
        baseline=(-0.2, 0.0),n_epochs_to_avg=None,save_path=None,show=True,):

    set_publication_style()
    if condition is not None:
        epochs_sub = epochs[condition]
        if len(epochs_sub) == 0:
            raise ValueError(f"条件 '{condition}' 无试次")
    else:
        epochs_sub = epochs

    if n_epochs_to_avg is not None and n_epochs_to_avg < len(epochs_sub):
        idx = np.random.choice(len(epochs_sub), n_epochs_to_avg, replace=False)
        epochs_sub = epochs_sub[idx]

    evoked = epochs_sub.average()
    if baseline is not None and evoked.baseline is None:
        evoked.apply_baseline(baseline=baseline)

    times = evoked.times * 1000
    time_indices = [np.argmin(np.abs(times - t)) for t in times_ms]
    fig, axes = plt.subplots(1, len(time_indices), figsize=(4 * len(time_indices), 3.5))
    if len(time_indices) == 1:
        axes = [axes]
    for ax, t_idx, t_ms in zip(axes, time_indices, times_ms):
        data = evoked.data[:, t_idx] * 1e6
        im, _ = mne.viz.plot_topomap(
            data, evoked.info, axes=ax, show=False
        )
        ax.set_title(f'{t_ms} ms')
        if ax == axes[-1]:
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.08, pad=0.02)
            cbar.set_label('µV')
    fig.suptitle(f'Topomaps (condition: {condition or "all"}) - Subject {subject_id}', fontsize=14)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        save_fig(fig, save_path)
    if show:
        plt.show()
    return fig


if __name__ == '__main__':

    pass
