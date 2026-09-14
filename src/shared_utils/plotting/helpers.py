import logging
import os
import sys

import numpy as np

log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def add_significance_bracket(ax, x1, x2, y, p_value, text='*', offset=0.02):
    ax.plot([x1, x1, x2, x2], [y, y + offset, y + offset, y], lw=1.5, color='k')
    ax.text((x1 + x2) / 2, y + offset + 0.01, text, ha='center', va='bottom', fontsize=12, fontweight='bold')


def shade_significant_cluster(ax, time_indices, y_min=-0.3, y_max=0.6, color='gray', alpha=0.3):
    if time_indices:
        start = time_indices[0] if isinstance(time_indices, list) else time_indices[0][0]
        end = time_indices[-1] if isinstance(time_indices, list) else time_indices[-1][-1]
        ax.axvspan(start, end, ymin=0, ymax=1, color=color, alpha=alpha, zorder=0)


def save_fig(fig, save_path, dpi=300, bbox_inches='tight'):
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
    logger.info(f"图片已保存: {save_path}")


def z_to_r(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)



