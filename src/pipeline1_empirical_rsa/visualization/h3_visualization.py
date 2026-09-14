import logging
import os
from pathlib import Path
from typing import  List, Dict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
import seaborn as sns

from src.shared_utils.plotting import save_fig, set_publication_style
from src.shared_utils.plotting.styles import COLORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parents[3]
os.chdir(project_root)


ENC_WIN = (200.0, 400.0)
MAIN_WIN = (500.0, 700.0)


def _write_caption(png_path: Path, text: str):
    caption_path = png_path.with_name(png_path.stem + '_caption.txt')
    caption_path.write_text(text, encoding='utf-8')
    logger.info(f"图注已保存: {caption_path}")


def _win_to_span(times, tmin, tmax):
    times = np.asarray(times, dtype=float)
    mask = (times >= tmin) & (times <= tmax)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return float(idx[0]) - 0.5, float(idx[-1]) + 0.5


def _draw_generalization_heatmap(
        mean_mat: np.ndarray,
        train_times: np.ndarray,
        test_times: np.ndarray,
        ax,
        cmap: str = 'RdBu_r',
        vmin: float = None,
        vmax: float = None,
        title: str = '',
        annotate_roi: bool = True,
):
    n_train, n_test = mean_mat.shape
    if vmin is None:
        vmin = -np.nanmax(np.abs(mean_mat))
    if vmax is None:
        vmax = np.nanmax(np.abs(mean_mat))

    im = ax.imshow(mean_mat.T, aspect='auto', origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax)

    n_xt = min(5, n_train)
    n_yt = min(5, n_test)
    xt = np.linspace(0, n_train - 1, n_xt).astype(int)
    yt = np.linspace(0, n_test - 1, n_yt).astype(int)
    ax.set_xticks(xt)
    ax.set_xticklabels([f'{train_times[i]:.0f}' for i in xt])
    ax.set_yticks(yt)
    ax.set_yticklabels([f'{test_times[i]:.0f}' for i in yt])
    ax.set_xlabel('Training time (ms)')
    ax.set_ylabel('Test time (ms)')
    ax.set_title(title)

    if annotate_roi:
        enc_span = _win_to_span(train_times, *ENC_WIN)
        main_span = _win_to_span(test_times, *MAIN_WIN)

        if enc_span:
            ax.add_patch(Rectangle(
                (enc_span[0], -0.5), enc_span[1] - enc_span[0], n_test,
                fill=False, ec='limegreen', lw=1.8, ls='--', zorder=5))
        if main_span:
            ax.add_patch(Rectangle(
                (-0.5, main_span[0]), n_train, main_span[1] - main_span[0],
                fill=False, ec='deepskyblue', lw=1.8, ls='--', zorder=5))
        if enc_span and main_span:
            ax.add_patch(Rectangle(
                (enc_span[0], main_span[0]),
                enc_span[1] - enc_span[0], main_span[1] - main_span[0],
                fill=False, ec='black', lw=2.0, zorder=6))

    return im

def h3_group_generalization_heatmap(
        suffix,
        gen_mats: List[np.ndarray],
        train_times_ms: np.ndarray,
        test_times_ms: np.ndarray,
        output_dir: str = 'figures/paper',
        show: bool = False,
        vmin: float = -0.3,
        vmax: float = 0.3,
):
    set_publication_style()
    mean_mat = np.mean([np.asarray(m, dtype=float) for m in gen_mats], axis=0)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = _draw_generalization_heatmap(
        mean_mat, np.asarray(train_times_ms, dtype=float),
        np.asarray(test_times_ms, dtype=float), ax,
        vmin=vmin, vmax=vmax,
        title=f'H3A: Generalization matrix (N={len(gen_mats)})',
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Spearman ρ')

    save_path = Path(output_dir) / f'H3_generalization_structure{suffix}.png'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

    _write_caption(
        save_path,
        " Group-average time-by-time generalization matrix during H3.\n"
        f"N = {len(gen_mats)} subjects. Entry (i, j) = Spearman ρ between the neural "
        "RDM at training time i and test time j.\n"
        "Green dashed box = encoding window (200–400 ms, X-axis).\n"
        "Blue dashed box = maintenance window (500–700 ms, Y-axis).\n"
        "Black solid box = cross-time (encoding × maintenance) region of interest.\n"
    )

def _model_gen_mat(rho_vec: np.ndarray) -> np.ndarray:
    return np.outer(rho_vec, rho_vec)


def h3_generalization_diff(
        suffix,
        struct_by_time: List[np.ndarray],
        control_by_time: Dict[str, List[np.ndarray]],
        test_times_ms: np.ndarray,
        output_dir: str = 'figures/paper',
        show: bool = False,
        vmin: float = -0.05,
        vmax: float = 0.05,
):
    set_publication_style()
    struct_arr = np.asarray(struct_by_time, dtype=float)
    ctrl_names = list(control_by_time.keys())
    ctrl_stack = np.stack(
        [np.asarray(control_by_time[m], dtype=float) for m in ctrl_names],
        axis=0)
    ctrl_mean = np.nanmean(ctrl_stack, axis=0)

    diff_mats = []
    for si in range(struct_arr.shape[0]):
        diff_mats.append(
            _model_gen_mat(struct_arr[si]) - _model_gen_mat(ctrl_mean[si])
        )
    mean_diff = np.mean(diff_mats, axis=0)

    times = np.asarray(test_times_ms, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = _draw_generalization_heatmap(
        mean_diff, times, times, ax,
        vmin=vmin, vmax=vmax, cmap='RdBu_r',
        title=f'H3B: Structure − Control generalization (N={struct_arr.shape[0]})',
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Δρ (structure − control)')

    save_path = Path(output_dir) / f'H3_generalization_diff{suffix}.png'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

    _write_caption(
        save_path,
        "Structure-specific generalization (group average).\n"
        f"N = {struct_arr.shape[0]} subjects. Each entry is the difference between the "
        "structure-model generalization matrix and the average across control models "
        "(visual, content, color).\n"
        "Model generalization matrix = ρ_model[t_i] * ρ_model[t_j], where ρ_model[t] "
        "is the Spearman correlation between the model RDM and the neural RDM at time t.\n"
        "Green / blue / black boxes mark the encoding, maintenance and cross-time "
        "regions of interest respectively.\n"
    )

def h3_behavior_scatter(
        suffix,
        ssi_arr: np.ndarray,
        rt_arr: np.ndarray,
        output_dir: str = 'figures/paper',
        show: bool = False,
):
    set_publication_style()
    ssi_arr = np.asarray(ssi_arr, dtype=float)
    rt_arr = np.asarray(rt_arr, dtype=float)
    mask = np.isfinite(ssi_arr) & np.isfinite(rt_arr)
    if mask.sum() < 3:
        logger.warning("有效被试不足")
        return
    x, y = ssi_arr[mask], rt_arr[mask]

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.regplot(
        x=x, y=y, ax=ax, ci=95,
        scatter_kws={'s': 55, 'alpha': 0.75, 'edgecolor': 'white', 'linewidths': 1.0,
                     'color': COLORS['control']},
        line_kws={'color': COLORS['target_priority'], 'linewidth': 2},
    )
    r, p_two = spearmanr(x, y)
    p_one = p_two / 2 if r < 0 else 1 - p_two / 2
    ax.text(0.05, 0.95, f'ρ = {r:.3f}\np_one = {p_one:.3f}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))

    ax.set_xlabel('SSI (Structure Stability Index)')
    ax.set_ylabel('Residual RT (ms)')
    ax.set_title(f'H3C: SSI vs residual RT (N={mask.sum()})')

    save_path = Path(output_dir) / f'H3_SSI_behavior_scatter{suffix}.png'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

    _write_caption(
        save_path,
        " Correlation between the Structure Stability Index (SSI) and "
        "residual response time across subjects.\n"
        f"N = {mask.sum()} subjects. Each dot = one subject.\n"
        "Red line = OLS regression; shaded band = 95% confidence interval.\n"
        f"Spearman ρ = {r:.3f}, one-sided p = {p_one:.3f} (predicted direction: "
        "higher SSI → shorter residual RT).\n"
    )


