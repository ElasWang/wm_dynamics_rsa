#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工程一数据预处理核心模块 (Shared Tool)

功能:
    将 OpenNeuro ds004117 的原始 BIDS 格式数据，清洗并转换为可用于 RSA 分析的 Epochs。

用法（供上层调度器调用）:
    from src.shared_utils.preprocess_util import process_subject
    epochs, metadata = process_subject(
        subject_id="001",
        bids_root=...,
        deriv_root=...,
        resample_sfreq=250,
        filter_low=0.5,
        filter_high=40.0,
        ica_n_components=30,
        eog_threshold=2.5
    )

作者: wm-dynamics-rsa
版本: 1.0.0
"""

import logging
from pathlib import Path
from typing import Tuple, Union, Optional


import pandas as pd
import mne
import numpy as np
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA
from mne.channels import make_standard_montage

# ==================== 日志 ====================
logger = logging.getLogger(__name__)


# ==================== 核心公共接口 ====================
def process_subject(
    subject_id: str,
    bids_root: Union[str, Path],
    deriv_root: Union[str, Path],
    task: str = "WorkingMemory",
    session: str = "01",
    resample_sfreq: int = 250,
    filter_low: float = 0.5,
    filter_high: float = 40.0,
    ica_n_components: int = 30,
    eog_threshold: float = 2.5,
    epoch_tmin: float = -0.2,
    epoch_tmax: float = 0.8,
) -> Tuple[mne.Epochs, pd.DataFrame]:
    """
    预处理单个被试的 EEG 数据，返回干净的 Epochs 和元数据。

    Args:
        subject_id: 被试编号，如 '001'
        bids_root: BIDS 原始数据根目录 (Path 或 str)
        deriv_root: 预处理输出目录 (Path 或 str)
        task: BIDS 任务名，默认 'WorkingMemory'
        session: BIDS 会话名，默认 '01'
        run: BIDS 运行序号，默认 '1'
        resample_sfreq: 重采样目标频率 (Hz)，默认 250
        filter_low: 高通截止频率 (Hz)，默认 0.5
        filter_high: 低通截止频率 (Hz)，默认 40.0
        ica_n_components: ICA 分解成分数，默认 30
        eog_threshold: 眼电成分检测阈值，默认 2.5
        epoch_tmin: 刺激前基线起始 (秒)，默认 -0.2
        epoch_tmax: 刺激后结束 (秒)，默认 0.8

    Returns:
        epochs: MNE Epochs 对象（已基线校正）
        metadata: DataFrame，包含每个 epoch 的 letter, color, position, load

    Raises:
        FileNotFoundError: 如果原始数据文件不存在
        ValueError: 如果事件筛选后没有剩余有效 epoch
    """
    # ---- 1. 路径标准化 ----
    bids_root = Path(bids_root)
    deriv_root = Path(deriv_root)
    deriv_root.mkdir(parents=True, exist_ok=True)

    # ---- 2. 加载原始 BIDS 数据 ----
    raws = []


    for run_num in range(1, 10):  # 最多探测6个run
        bids_path = BIDSPath(
            subject=subject_id,
            datatype='eeg',
            root=bids_root,
            task=task,
            session=session,
            run=str(run_num)
        )
        # 检查文件是否存在（BIDSPath 的 fpath 属性返回具体文件路径）
        if not bids_path.fpath.exists():
            if run_num == 1:
                raise FileNotFoundError(f"run-1 不存在: {bids_path.fpath}")
            else:
                # 后续 run 不存在，说明已经加载完所有存在的 run
                logger.info(f"  run-{run_num} 不存在，停止加载")
                break
        raw_run = read_raw_bids(bids_path, verbose=False, extra_params={'preload': True})
        raws.append(raw_run)
        logger.info(f"  成功加载 run-{run_num} (采样率: {raw_run.info['sfreq']} Hz)")

    if not raws:
        raise FileNotFoundError(f"被试 {subject_id} 没有任何可用的 run 数据！")
    times = sum([r.n_times for r in raws])
    if len(raws) > 1:
        logger.info(f"  正在拼接 {len(raws)} 个 run...")
        standard_ch_names = raws[0].ch_names
        for raw in raws:
            if raw.ch_names != standard_ch_names:
                raw.reorder_channels(standard_ch_names)
        raw = mne.concatenate_raws(raws,on_mismatch='ignore', preload=True)
    else:
        raw = raws[0]
    logger.info(f"拼接后总时间点: {raw.n_times} (理论值: {times})")
    logger.info(f"  拼接后总时长: {raw.n_times / raw.info['sfreq']:.1f} 秒")
    logger.info(f"  原始采样率: {raw.info['sfreq']} Hz, 通道数: {len(raw.ch_names)}")

    # ---- 3. 重采样至统一频率 ----
    if raw.info['sfreq'] != resample_sfreq:
        logger.info(f"  重采样: {raw.info['sfreq']}Hz -> {resample_sfreq}Hz")
        raw.resample(resample_sfreq, npad="auto")

    # ---- 4. 带通滤波 ----
    raw.filter(filter_low, filter_high, fir_design='firwin', verbose=False)
    logger.info(f"  滤波完成: {filter_low}-{filter_high}Hz")

    # ---- 5. 坏道检测与插值 ----
    raw = _detect_and_fix_bad_channels(raw)

    # ---- 6. 电极标准化 ----
    raw = _standardize_channels(raw)

    # ---- 7. ICA 去除眼电 ----
    raw = _apply_ica(raw, n_components=ica_n_components, threshold=eog_threshold)

    # ---- 8. 提取事件并切分 Epochs ----
    epochs,event_mask  = _extract_epochs(raw, tmin=epoch_tmin, tmax=epoch_tmax)
    logger.info(f"  提取到 {len(epochs)} 个有效 Epochs")

    if len(epochs) == 0:
        raise ValueError(f"被试 {subject_id} 没有有效 epoch，请检查事件标注")

    # ---- 9. 解析元数据 ----
    metadata = _parse_metadata(subject_id, bids_root, deriv_root,event_mask )

    # ---- 10. 保存衍生数据（备份） ----
    epochs_save_path = deriv_root / f"sub-{subject_id}_coding_epo.fif"
    epochs.save(str(epochs_save_path), overwrite=True)
    logger.info(f"  Epochs 已保存: {epochs_save_path}")

    return epochs, metadata


# ==================== 私有辅助函数 ====================

def _detect_and_fix_bad_channels(raw: mne.io.Raw, threshold: float = 5.0) -> mne.io.Raw:
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


def _standardize_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """标准化通道命名与电极位置"""
    # 通道重命名 (适配 ds004117 大写命名)
    rename_mapping = {
        'FP1': 'Fp1', 'FPZ': 'Fpz', 'FP2': 'Fp2',
        'AFZ': 'AFz', 'FZ': 'Fz', 'FCZ': 'FCz',
        'CZ': 'Cz', 'CPZ': 'CPz', 'PZ': 'Pz',
        'POZ': 'POz', 'OZ': 'Oz'
    }
    raw.rename_channels(rename_mapping)

    # 标注眼电通道
    raw.set_channel_types({'LEYE': 'eog', 'REYE': 'eog'})

    # 设置标准 10-5 电极坐标
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')
    return raw


def _apply_ica(raw: mne.io.Raw, n_components: int = 30, threshold: float = 2.5) -> mne.io.Raw:
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


def _extract_epochs(raw: mne.io.Raw, tmin: float = -0.2, tmax: float = 0.8) -> Tuple[mne.Epochs, np.ndarray]:
    """提取事件并切分 Epochs"""
    events, event_id  = mne.events_from_annotations(raw, verbose=False)

    # 筛选字母事件 (ID 1-8 对应 S1-S8)
    # 注意：此逻辑基于 ds004117 的事件编码，如果更换数据集需调整，不包含 'r' 开头的探针
    target_ids = []
    for name, eid in event_id.items():
        if len(name) == 1 and name.isupper():  # 黑色
            target_ids.append(eid)
        elif name.startswith('g') and len(name) == 2 and name[1].isupper():  # 绿色干扰
            target_ids.append(eid)

    mask = np.isin(events[:, 2], target_ids)
    letter_events = events[mask]  # 筛选后事件

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


def _parse_metadata(
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
        df_run['run'] = run_num  # 标记来源
        all_dfs.append(df_run)
        logger.info(f"  加载 run-{run_num} events.tsv，行数: {len(df_run)}")

    if not all_dfs:
        logger.warning("  未找到任何 events.tsv，返回空 DataFrame")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # 应用掩码筛选
    if event_mask is not None and len(event_mask) == len(df):
        df = df[event_mask].reset_index(drop=True)
        logger.info(f"  根据事件掩码筛选，metadata 保留 {len(df)} 行")
    else:
        logger.warning("  掩码无效或长度不匹配，保留全量数据")

    # 后续处理
    df['color'] = df['value'].apply(_extract_color)
    df['letter'] = df['letter'] if 'letter' in df.columns else 'X'
    df['load'] = df['memory_cond'] if 'memory_cond' in df.columns else 0
    df['position'] = 0  # 后续由 RSA 脚本填充

    # ===== 修改点：保存时加入 'run' 列 =====
    meta_save_path = deriv_root / f"sub-{subject_id}_metadata.csv"
    # 确保列存在
    save_cols = ['onset', 'duration', 'position', 'letter', 'color', 'load', 'run','trial']
    # 检查列是否都存在（安全兜底）
    existing_cols = [col for col in save_cols if col in df.columns]
    df[existing_cols].to_csv(meta_save_path, index=False)
    logger.info(f"  元数据已保存: {meta_save_path} (共 {len(df)} 行)")
    return df


def _extract_color(val: str) -> str:
    """从 value 列提取颜色标识"""
    if isinstance(val, str) and val.startswith('g'):
        return 'green'
    return 'black'