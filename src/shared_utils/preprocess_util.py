import logging
from pathlib import Path
from typing import Tuple


import pandas as pd
import mne
import numpy as np
from mne.preprocessing import ICA
from mne.channels import make_standard_montage

# ==================== 日志 ====================
logger = logging.getLogger(__name__)


def resample_data(raw, target_sfreq=250):
    """
    重采样
    """
    if raw.info['sfreq'] != target_sfreq:
        return raw.copy().resample(target_sfreq, npad="auto")
    return raw

def apply_filter(raw, l_freq=0.5, h_freq=40.0):
    """
    带通滤波
    """
    return raw.copy().filter(l_freq, h_freq, fir_design='firwin', verbose=False)

def detect_and_fix_bad_channels(raw: mne.io.Raw, threshold: float = 5.0) -> mne.io.Raw:
    """
    坏道检测与插值
    基于通道方差，超过均值 threshold 倍标准差的标记为坏道
    """
    data, _ = raw[:, :]
    ch_var = np.var(data, axis=1)
    median_var = np.median(ch_var)
    mad_var = np.median(np.abs(ch_var - median_var)) * 1.4826
    bads = []
    for i, ch in enumerate(raw.ch_names):
        if abs(ch_var[i] - median_var) > threshold * mad_var:
            bads.append(ch)
    if bads:
        raw.info['bads'] = bads
        logger.warning(f"  检测到坏道: {bads}，正在插值...")
        raw.interpolate_bads(reset_bads=True)
    else:
        logger.info("  未检测到显著坏道")
    return raw


def standardize_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """标准化通道命名与电极位置"""
    rename_mapping = {
        'FP1': 'Fp1', 'FPZ': 'Fpz', 'FP2': 'Fp2',
        'AFZ': 'AFz', 'FZ': 'Fz', 'FCZ': 'FCz',
        'CZ': 'Cz', 'CPZ': 'CPz', 'PZ': 'Pz',
        'POZ': 'POz', 'OZ': 'Oz'
    }
    raw.rename_channels(rename_mapping)
    raw.set_channel_types({'LEYE': 'eog', 'REYE': 'eog'})
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')
    return raw


def apply_ica(raw: mne.io.Raw, n_components: int = 30, threshold: float = 2.5) -> mne.io.Raw:
    """使用 ICA 自动去除眼电伪迹"""
    eog_channels = [ch for ch in raw.ch_names if 'EYE' in ch]
    if not eog_channels:
        logger.info("  未找到 EOG 通道，跳过 ICA 去眼电")
        return raw
    ica = ICA(n_components=n_components, method='infomax', random_state=97)
    ica.fit(raw, picks=['eeg'], verbose=False)
    eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_channels, threshold=threshold)
    if eog_indices:
        ica.exclude = eog_indices
        raw = ica.apply(raw, verbose=False)
        logger.info(f"  ICA 去噪成功，移除了 {len(eog_indices)} 个眼电成分")
    else:
        logger.info("  未检测到显著眼电成分")
    return raw


def extract_epochs(raw: mne.io.Raw, tmin: float = -0.2, tmax: float = 0.8) -> Tuple[mne.Epochs, np.ndarray]:
    """提取事件并切分 Epochs"""
    events, event_id  = mne.events_from_annotations(raw, verbose=False)
    target_ids = []
    for name, eid in event_id.items():
        if len(name) == 1 and name.isupper():
            target_ids.append(eid)
        elif name.startswith('g') and len(name) == 2 and name[1].isupper():
            target_ids.append(eid)

    mask = np.isin(events[:, 2], target_ids)
    letter_events = events[mask]
    logger.info(f"  总事件数: {len(events)}, 字母事件数: {len(letter_events)}")
    epochs = mne.Epochs(
        raw,
        letter_events,
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0),
        preload=True,
        reject_by_annotation=True,
        verbose=False
    )
    return epochs,mask


def parse_metadata(
    subject_id: str,
    bids_root: Path,
    deriv_root: Path,
    event_mask: np.ndarray = None
) -> pd.DataFrame:
    """从所有 run 的 events.tsv 解析元数据，利用掩码筛选行，并保留 run 编号"""
    all_dfs = []
    for run_num in range(1, 10):
        events_tsv_path = (
            bids_root / f"sub-{subject_id}" / "ses-01" / "eeg"
            / f"sub-{subject_id}_ses-01_task-WorkingMemory_run-{run_num}_events.tsv"
        )
        if not events_tsv_path.exists():
            if run_num == 1:
                logger.warning(f"  run-1 的 events.tsv 不存在: {events_tsv_path}")
            else:
                logger.info(f"  run-{run_num} 的 events.tsv 不存在，停止加载")
                break
        df_run = pd.read_csv(events_tsv_path, sep='\t')
        df_run['run'] = run_num
        all_dfs.append(df_run)
        logger.info(f"  加载 run-{run_num} events.tsv，行数: {len(df_run)}")
    if not all_dfs:
        logger.warning("  未找到任何 events.tsv，返回空 DataFrame")
        return pd.DataFrame()
    df = pd.concat(all_dfs, ignore_index=True)
    if event_mask is not None and len(event_mask) == len(df):
        df = df[event_mask].reset_index(drop=True)
        logger.info(f"  根据事件掩码筛选，metadata 保留 {len(df)} 行")
    else:
        logger.warning("  掩码无效或长度不匹配，保留全量数据")
    df['color'] = df['value'].apply(extract_color)
    df['letter'] = df['letter'] if 'letter' in df.columns else 'X'
    df['load'] = df['memory_cond'] if 'memory_cond' in df.columns else 0
    df['position'] = 0
    meta_save_path = deriv_root / f"sub-{subject_id}_metadata.csv"
    save_cols = ['onset', 'duration', 'position', 'letter', 'color', 'load', 'run','trial']
    existing_cols = [col for col in save_cols if col in df.columns]
    df[existing_cols].to_csv(meta_save_path, index=False)
    logger.info(f"  元数据已保存: {meta_save_path} (共 {len(df)} 行)")
    return df


def extract_color(val: str) -> str:
    """从 value 列提取颜色标识"""
    if isinstance(val, str) and val.startswith('g'):
        return 'green'
    return 'black'