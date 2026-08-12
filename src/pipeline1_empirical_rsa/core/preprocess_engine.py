import logging
from pathlib import Path
from typing import Tuple, Union

import mne
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from src.shared_utils.preprocess_util import (
    resample_data,
    apply_filter,
    detect_and_fix_bad_channels,
    standardize_channels,
    apply_ica,
    extract_epochs,
    parse_metadata
)

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
        metadata: DataFrame，包含每个 epoch 的 letter, color, position, load,trail

    Raises:
        FileNotFoundError: 如果原始数据文件不存在
        ValueError: 如果事件筛选后没有剩余有效 epoch
    """
    # ---- 路径标准化 ----
    bids_root = Path(bids_root)
    deriv_root = Path(deriv_root)
    deriv_root.mkdir(parents=True, exist_ok=True)

    # ----加载原始 BIDS 数据 ----
    raws = []

    for run_num in range(1, 10):
        bids_path = BIDSPath(
            subject=subject_id,
            datatype='eeg',
            root=bids_root,
            task=task,
            session=session,
            run=str(run_num)
        )
        if not bids_path.fpath.exists():
            if run_num == 1:
                raise FileNotFoundError(f"run-1 不存在: {bids_path.fpath}")
            else:
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
        raw = mne.concatenate_raws(raws, on_mismatch='ignore', preload=True)
    else:
        raw = raws[0]
    logger.info(f"拼接后总时间点: {raw.n_times} (理论值: {times})")
    logger.info(f"  拼接后总时长: {raw.n_times / raw.info['sfreq']:.1f} 秒")
    logger.info(f"  原始采样率: {raw.info['sfreq']} Hz, 通道数: {len(raw.ch_names)}")

    # ----重采样至统一频率 ----
    if raw.info['sfreq'] != resample_sfreq:
        logger.info(f"  重采样: {raw.info['sfreq']}Hz -> {resample_sfreq}Hz")
        resample_data(raw, resample_sfreq)

    # ---- 带通滤波 ----
    apply_filter(raw,filter_low, filter_high)
    logger.info(f"  滤波完成: {filter_low}-{filter_high}Hz")

    # ---- 坏道检测与插值 ----
    raw = detect_and_fix_bad_channels(raw)

    # ---- 电极标准化 ----
    raw = standardize_channels(raw)

    # ---- ICA 去除眼电 ----
    raw = apply_ica(raw, n_components=ica_n_components, threshold=eog_threshold)

    # ----提取事件并切分 Epochs ----
    epochs, event_mask = extract_epochs(raw, tmin=epoch_tmin, tmax=epoch_tmax)
    logger.info(f"  提取到 {len(epochs)} 个有效 Epochs")

    if len(epochs) == 0:
        raise ValueError(f"被试 {subject_id} 没有有效 epoch，请检查事件标注")

    # ---- 解析元数据 ----
    metadata = parse_metadata(subject_id, bids_root, deriv_root, event_mask)

    # ---- 保存衍生数据----
    epochs_save_path = deriv_root / f"sub-{subject_id}_coding_epo.fif"
    epochs.save(str(epochs_save_path), overwrite=True)
    logger.info(f"  Epochs 已保存: {epochs_save_path}")

    return epochs, metadata
