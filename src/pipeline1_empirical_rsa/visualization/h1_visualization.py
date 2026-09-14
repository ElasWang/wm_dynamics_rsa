import logging
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.shared_utils.plotting import set_publication_style, save_fig
from src.shared_utils.plotting.helpers import z_to_r
from src.shared_utils.plotting.styles import COLORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parents[3]
os.chdir(project_root)

def h1_plot_delta_curve( result_dict, model_a,model_b,save_path,):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    times = np.asarray(result_dict['times_ms'], dtype=float)
    mean_delta = np.asarray(result_dict['mean_delta_z'], dtype=float)
    sem_delta = np.asarray(result_dict['sem_delta_z'], dtype=float)
    ax.plot(times, mean_delta,color='red', lw=2,label=f'{model_a} - {model_b}', )
    ax.fill_between( times, mean_delta - sem_delta,mean_delta + sem_delta, color='red', alpha=0.2, )
    significant_drawn = False
    for cluster in result_dict.get('significant_clusters', []):
        if not cluster.get('interpretable_length', False):
            continue
        t0, t1 = cluster['t_win_ms']
        label = 'significant cluster' if not significant_drawn else '_nolegend_'
        ax.axvspan(t0, t1, color='gray', alpha=0.3, zorder=0, label=label)
        significant_drawn = True
    ax.axvline(0, linestyle=':', color='gray', lw=1)
    baseline_left = float(times.min())
    if baseline_left < 0:
        ax.axvspan(baseline_left, 0, color='lightgray', alpha=0.15, zorder=0)
    ax.axhline(0, linestyle='--', color='gray', lw=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Δ Fisher-z Spearman ρ')
    ax.set_title(f'H1: {model_a} vs {model_b}')
    ax.legend(frameon=False)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    plt.close(fig)


_FIG1A_MODELS = [
    ('target_priority_model', 'Target priority',   'target_priority'),
    ('gradient_model',        'Position gradient', 'gradient'),
    ('visual_model',          'Visual',            'visual'),
    ('binary_color_model',    'Color binary',      'content'),
]
def _collect_group_rhos(subject_ids, result_root, suffix=''):
    per_model = {name: [] for name, _, _ in _FIG1A_MODELS}
    common_times = None
    valid_sids = []
    for sid in subject_ids:
        fpath = Path(result_root) / f'sub-{sid}' / f'sub-{sid}_h1_rhos{suffix}.npz'
        if not fpath.exists():
            logger.warning(f"sub-{sid}: 结果缺失，跳过 {fpath}")
            continue
        data = np.load(fpath, allow_pickle=True)
        if 'times_ms' not in data:
            logger.warning(f"sub-{sid}: 缺 times_ms，跳过")
            continue
        t = np.asarray(data['times_ms'], dtype=float)

        if common_times is None:
            common_times = t
        elif len(t) != len(common_times) or not np.allclose(t, common_times, atol=1e-6):
            logger.warning(f"sub-{sid}: 时间轴与其他被试不一致，跳过")
            continue

        if any(name not in data for name, _, _ in _FIG1A_MODELS):
            logger.warning(f"sub-{sid}: 缺模型，跳过")
            continue

        for name, _, _ in _FIG1A_MODELS:
            per_model[name].append(np.asarray(data[name], dtype=float))
        valid_sids.append(str(sid))

    out = {name: np.asarray(per_model[name], dtype=float) for name, _, _ in _FIG1A_MODELS}
    return out, common_times, valid_sids


def h1_all_models(
        subject_ids,
        result_root: str = 'results',
        suffix: str = '',
        output_dir: str = 'figures/paper',
        tmin: float = -100.0,
        tmax: float = 800.0,
        ymin: float = -0.2,
        ymax: float = 0.3,
        show: bool = False,
) -> None:
    rho_stack, times, valid_sids = _collect_group_rhos(subject_ids, result_root, suffix)
    if times is None or len(valid_sids) < 2:
        raise RuntimeError("有效被试数不足，无法画群体曲线")
    n_sub = len(valid_sids)

    time_mask = (times >= tmin) & (times <= tmax)
    times_plot = times[time_mask]
    stats = {}
    for name, _, _ in _FIG1A_MODELS:
        X = np.clip(rho_stack[name][:, time_mask], -0.999999, 0.999999)
        Z = np.arctanh(X)
        Z_mean = np.nanmean(Z, axis=0)
        Z_sem = np.nanstd(Z, axis=0, ddof=1) / np.sqrt(n_sub)
        stats[name] = {
            'mean':  z_to_r(Z_mean),
            'lower': z_to_r(Z_mean - Z_sem),
            'upper': z_to_r(Z_mean + Z_sem),
        }

    set_publication_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, label, color_key in _FIG1A_MODELS:
        s = stats[name]
        c = COLORS[color_key]
        ax.plot(times_plot, s['mean'], color=c, lw=2, label=label)
        ax.fill_between(times_plot, s['lower'], s['upper'],
                        color=c, alpha=0.2, linewidth=0)

    ax.axvline(0, linestyle=':', color='gray', lw=1)
    ax.axhline(0, linestyle='--', color='gray', lw=1)

    ax.set_xlim(tmin, tmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(tmin, tmax + 1, 100))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Spearman ρ')
    ax.set_title(f'H1: model-wise RSA time course (N={n_sub})')
    ax.legend(loc='upper right', frameon=False)

    save_path = Path(output_dir) / f'H1_all_models{suffix}.png'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

    caption_path = save_path.with_name(save_path.stem + '_caption.txt')
    caption_path.write_text(
        f" Model-wise sliding-window RSA time course (H1).\n"
        f"N = {n_sub} subjects. Curves = group mean Spearman rho after Fisher-z;\n"
        f"shaded bands = +/-1 SEM (also computed in Fisher-z space, inverse-transformed).\n"
        f"Vertical dotted line marks stimulus onset (0 ms); horizontal dashed line = rho = 0.\n"
        f"Time axis restricted to {tmin:.0f}-{tmax:.0f} ms.\n",
        encoding='utf-8',
    )
    logger.info(f"Fig 图注已保存: {caption_path}")