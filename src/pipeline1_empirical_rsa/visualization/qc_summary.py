
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parents[3]
os.chdir(project_root)

def compute_bad_channel_pct(epochs):
    n_ch = len(epochs.ch_names)
    n_bad = len(epochs.info['bads'])
    return 100.0 * n_bad / n_ch if n_ch else np.nan


def compute_ica_bad_comp_pct(epochs):
    ica_exclude = epochs.info.get('temp', {}).get('ica_exclude', None)
    if ica_exclude is None:
        return np.nan
    n_comp = epochs.info.get('temp', {}).get('ica_n_components', np.nan)
    return 100.0 * len(ica_exclude) / n_comp if n_comp else np.nan


def compute_trial_reject_rate(epochs):
    drop_log = epochs.drop_log
    n_total = len(drop_log)
    n_rej = sum(1 for d in drop_log if len(d) > 0)
    return n_rej / n_total if n_total else np.nan


def compute_epochs_peak_uv(epochs):
    data = epochs.get_data() * 1e6
    return float(np.percentile(np.abs(data), 95))


def compute_baseline_drift(epochs):
    data = epochs.get_data() * 1e6
    sfreq = epochs.info['sfreq']
    win = max(1, int(0.1 * sfreq))
    start = data[:, :, :win].mean(axis=-1)
    end = data[:, :, -win:].mean(axis=-1)
    return float(np.mean(np.abs(end - start)))


def compute_split_half_reliability(epochs):
    data = epochs.get_data()
    if data.shape[0] < 4:
        return np.nan
    half_a = data[0::2].mean(axis=0)
    half_b = data[1::2].mean(axis=0)
    a = half_a.mean(axis=0)
    b = half_b.mean(axis=0)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r, _ = pearsonr(a, b)
    return float(r)


def compute_notes(epochs):
    notes = []
    if len(epochs.info['bads']) > 0:
        notes.append(f"bads={epochs.info['bads']}")
    if len(epochs) == 0:
        notes.append("no_epochs")
    return "; ".join(notes)

def generate_qc_summary(epochs_dict,
                        figures_group='/data/figures_group'):
    out_csv = Path("qc_summary.csv")
    figures_group = Path(figures_group)
    figures_group.mkdir(parents=True, exist_ok=True)

    rows = []
    for sub_id, epochs in epochs_dict.items():
        try:
            row = {
                'subject': sub_id,
                'bad_channel_pct': compute_bad_channel_pct(epochs),
                'ica_bad_comp_pct': compute_ica_bad_comp_pct(epochs),
                'trial_reject_rate': compute_trial_reject_rate(epochs),
                'epochs_peak_uv': compute_epochs_peak_uv(epochs),
                'baseline_drift': compute_baseline_drift(epochs),
                'split_half_reliability': compute_split_half_reliability(epochs),
                'notes': compute_notes(epochs),
            }
        except Exception as e:
            logger.exception(f"{sub_id} QC 计算失败: {e}")
            row = {
                'subject': sub_id,
                'bad_channel_pct': np.nan,
                'ica_bad_comp_pct': np.nan,
                'trial_reject_rate': np.nan,
                'epochs_peak_uv': np.nan,
                'baseline_drift': np.nan,
                'split_half_reliability': np.nan,
                'notes': f"error: {e}",
            }
        rows.append(row)

    df = pd.DataFrame(rows, columns=[
        'subject', 'bad_channel_pct', 'ica_bad_comp_pct',
        'trial_reject_rate', 'epochs_peak_uv', 'baseline_drift',
        'split_half_reliability', 'notes',
    ])

    num_cols = df.columns.difference(['subject', 'notes'])
    df[num_cols] = df[num_cols].round(3)

    df.to_csv(out_csv, index=False)
    logger.info(f"QC 汇总已保存: {out_csv.resolve()}")

    df.to_csv(figures_group / 'qc_summary.csv', index=False)
    return df


