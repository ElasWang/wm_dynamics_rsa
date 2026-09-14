import logging
import os
import re
import sys
from pathlib import Path

import mne

from config import CONFIG
from src.pipeline1_empirical_rsa.visualization.qc_summary import generate_qc_summary

log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
def main():
    result_root = Path(CONFIG.get('paths', {}).get('rsa_result', 'results'))
    deriv_root = Path(CONFIG.get('paths', {}).get('deriv_root', '/data/derivatives'))
    figures_group = Path(CONFIG.get('paths', {}).get('figures', '/results/figures'))
    excluded_subjects = CONFIG.get('rsa', {}).get('excluded_subjects', [])
    subject_ids = get_valid_subject_ids(result_root)
    subject_ids = [s for s in subject_ids if s not in excluded_subjects]
    epochs_dict = get_valid_epochs(deriv_root)
    logger.info(f"\n===== 汇总预处理质控指标 =====")
    try:
        qc_summary = generate_qc_summary(epochs_dict,figures_group)
    except Exception as e:
        logger.error(e)

def get_valid_subject_ids(result_root='results'):
    result_root = Path(result_root)
    subject_ids = []
    for sub_dir in result_root.glob('sub-*'):
        h1_event = sub_dir / f"{sub_dir.name}_h1_rhos_event.npz"
        if h1_event.exists():
            subject_ids.append(sub_dir.name[4:])
        else:
            logger.warning(f"跳过 {sub_dir.name}：缺少 H1 event-level 结果")
    return sorted(subject_ids)

def get_valid_epochs(deriv_root='/data/derivatives'):
    result_root = Path(deriv_root)
    epochs_dict = {}
    for fif_name in result_root.glob('sub-*'):
        sub_id = re.match(r'(sub-\d+)', fif_name.name).group(1)
        fif_path = deriv_root / f"{sub_id}_coding_epo.fif"
        if fif_path.exists():
            epochs_dict[sub_id] = mne.read_epochs(fif_path, preload=True)
        else:
            logger.warning(f"跳过 {sub_id.name} ")
    return dict(sorted(epochs_dict.items()))


if __name__ == "__main__":
    main()
