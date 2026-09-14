import logging
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.shared_utils.plotting import save_fig, set_publication_style
from src.shared_utils.plotting.styles import COLORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parents[3]
os.chdir(project_root)


def _p_to_star(p: float) -> str:
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def _place_star(ax, x, mean, sem, p):
    star = _p_to_star(p)
    if not star:
        return
    if mean >= 0:
        y = mean + sem
        ax.text(x, y + 0.01, star, ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    else:
        y = mean - sem
        ax.text(x, y - 0.01, star, ha='center', va='top',
                fontsize=12, fontweight='bold')


def _write_caption(png_path: Path, text: str):
    caption_path = png_path.with_name(png_path.stem + '_caption.txt')
    caption_path.write_text(text, encoding='utf-8')
    logger.info(f"图注已保存: {caption_path}")

def h2_beta_structure_bar(
        suffix,
        h2_group_results: dict,
        output_dir: str = 'figures/paper',
        show: bool = False,
) -> None:
    mean_betas = np.asarray(h2_group_results['mean_betas'], dtype=float)
    sem_betas = np.asarray(h2_group_results['sem_betas'], dtype=float)
    p_values = np.asarray(h2_group_results['p_values'], dtype=float)
    model_names = list(h2_group_results['model_names'])
    n = len(mean_betas)

    set_publication_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(n)

    ax.bar(x, mean_betas, yerr=sem_betas, capsize=5,
           color=COLORS['target_priority'], edgecolor='black', linewidth=0.5)

    for i in range(n):
        _place_star(ax, x[i], mean_betas[i], sem_betas[i], p_values[i])

    ax.axhline(0, linestyle='--', color='gray', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_xlabel('Model')
    ax.set_ylabel('Standardized β (unique contribution)')
    ax.set_title('H2: Structure model β')

    save_path = Path(output_dir) / f'H2_beta_structure_bar{suffix}.png'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

    _write_caption(
        save_path,
        "Standardized regression coefficient (β) of the structure "
        "model in the full H2 model.\n"
        f"Bar = group mean (N={n}); error bar = ±1 SEM across subjects.\n"
        "Dashed horizontal line = β = 0.\n"
        f"Asterisks denote one-sample t-test against 0: * p<.05, ** p<.01, *** p<.001.\n"
    )


def h2_delta_r2_bar(
        suffix,
        h2_group_results: dict,
        output_dir: str = 'figures/paper',
        show: bool = False,
) -> None:
    mean_delta = float(h2_group_results['mean_delta'])
    sem_delta = float(h2_group_results['sem_delta'])
    p_delta = float(h2_group_results['p_delta'])

    set_publication_style()
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(['ΔR²'], [mean_delta], yerr=[sem_delta], capsize=5,
           color=COLORS['target_priority'], edgecolor='black', linewidth=0.5)

    _place_star(ax, 0, mean_delta, sem_delta, p_delta)

    ax.axhline(0, linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel('ΔR²')
    ax.set_ylabel('ΔR² (full − reduced), cross-validated')
    ax.set_title('H2: Predictive gain of structure model')

    save_path = Path(output_dir) / f'H2_delta_r2_bar{suffix}.png'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)

    _write_caption(
        save_path,
        "Cross-validated R² gain of the full H2 model over the reduced "
        "model (with vs. without the structure term).\n"
        "Bar = group mean; error bar = ±1 SEM across subjects.\n"
        "Dashed horizontal line = 0 (no gain).\n"
        f"Asterisks denote one-sample t-test against 0: * p<.05, ** p<.01, *** p<.001.\n"
    )