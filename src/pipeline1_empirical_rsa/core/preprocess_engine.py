import logging
import os
from pathlib import Path
from typing import Tuple

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from mne_bids import BIDSPath, read_raw_bids

from config import CONFIG
from run import PROJECT_ROOT
from src.shared_utils.metadata_util import compute_positions
from src.shared_utils.preprocess_util import (
    resample_data,
    apply_filter,
    detect_and_fix_bad_channels,
    standardize_channels,
    extract_epochs,
    parse_metadata,
    apply_reference,
    drop_bad_trials,
    whitening_manual,
    get_iclabel_bad_components,
    validate_epoch_metadata_alignment,
)


logger = logging.getLogger(__name__)

def process_subject(
        subject_id: str,
) -> Tuple[mne.Epochs, pd.DataFrame]:
    # ---- 从配置文件读取预处理参数 ----
    preproc_cfg = CONFIG.get('preprocessing', {})
    paths_cfg = CONFIG.get('paths', {})

    resample_sfreq = preproc_cfg.get('resample_sfreq', 250)
    filter_low = preproc_cfg.get('filter_low', 0.5)
    filter_high = preproc_cfg.get('filter_high', 50.0)
    epoch_tmin = preproc_cfg.get('epoch_tmin', -0.2)
    epoch_tmax = preproc_cfg.get('epoch_tmax', 0.8)
    baseline = preproc_cfg.get('baseline', (-0.2, 0.0))
    reject = preproc_cfg.get('reject', {'eeg': 0.0001})
    ica_pca_n_components = preproc_cfg.get('ica_pca_n_components', 0.99)
    ica_algorithm = preproc_cfg.get('ica_algorithm', 'runica')
    ica_extended = preproc_cfg.get('ica_extended')
    iclabel_threshold = preproc_cfg.get('iclabel_threshold', 0.8)
    whiten = preproc_cfg.get('whiten', 0.8)

    asr = preproc_cfg.get('asr', {})
    asr_flatline_criterion = asr.get('flatline_criterion', 5)
    asr_channel_criterion = asr.get('channel_criterion', 0.8)
    asr_line_noise_criterion = asr.get('line_noise_criterion', 4)
    asr_burst_criterion = asr.get('burst_criterion', 10)
    asr_highpass = asr.get('highpass', 'off')

    # ----  构造路径（优先使用环境变量，否则基于项目根目录） ----
    bids_root = Path(os.getenv("BIDS_ROOT", PROJECT_ROOT / paths_cfg.get('bids_root', 'data/raw')))
    deriv_root = Path(os.getenv("DERIV_ROOT", PROJECT_ROOT / paths_cfg.get('deriv_root', 'data/derivatives')))

    # ---- 路径标准化 ----
    bids_root = Path(bids_root)
    deriv_root = Path(deriv_root)
    deriv_root.mkdir(parents=True, exist_ok=True)

    electrodes_path = BIDSPath(
        subject=subject_id,
        session="01",
        datatype='eeg',
        root=bids_root,
        task="WorkingMemory",
        suffix='electrodes',
        extension='.tsv',
        run=1
    )
    if not electrodes_path.fpath.exists():
        raise FileNotFoundError(f"电极文件不存在: {electrodes_path.fpath}")
    montage = mne.channels.read_custom_montage(
        electrodes_path.fpath,
        coord_frame='head'
    )

    raws = []
    run_durations = []

    for run_num in range(1, 10):
        bids_path = BIDSPath(
            subject=subject_id,
            datatype="eeg",
            root=bids_root,
            task="WorkingMemory",
            session="01",
            run=str(run_num),
        )

        if not bids_path.fpath.exists():
            if run_num == 1:
                raise FileNotFoundError(f"run-1 不存在: {bids_path.fpath}")
            logger.info(f"run-{run_num} 不存在，停止加载")
            break

        raw_run = read_raw_bids(
            bids_path,
            verbose=False,
            extra_params={"preload": True},
        )
        duration_sec = (raw_run.n_times /raw_run.info["sfreq"])
        raws.append(raw_run)
        run_durations.append(duration_sec)
        logger.info(
            f"成功加载 run-{run_num}: "
            f"{raw_run.n_times} samples, "
            f"{raw_run.info['sfreq']} Hz, "
            f"{duration_sec:.3f}s"
        )
    if not raws:
        raise FileNotFoundError(f"被试 {subject_id} 没有任何可用 run")

    if len(raws) > 1:
        standard_ch_names = raws[0].ch_names
        for raw_run in raws:
            if raw_run.ch_names != standard_ch_names:
                raw_run.reorder_channels(standard_ch_names)
        raw = mne.concatenate_raws(raws,on_mismatch="ignore",preload=True,)
    else:
        raw = raws[0]
    logger.info(
        f"拼接完成: "
        f"{raw.n_times} samples, "
        f"{raw.n_times / raw.info['sfreq']:.3f}s"
    )

    # ---- 电极标准化----
    raw = standardize_channels(raw, montage)

    # ----重采样至统一频率 ----
    if raw.info['sfreq'] != resample_sfreq:
        logger.info(f"  重采样: {raw.info['sfreq']}Hz -> {resample_sfreq}Hz")
        raw = resample_data(raw, resample_sfreq)

    # ---- 带通滤波 ----
    raw = apply_filter(raw, filter_low, filter_high)
    logger.info(f"  滤波完成: {filter_low}-{filter_high}Hz")

    # ---- ASR 校正 ----
    # logger.info("  运行 ASR 校正...")
    # raw = apply_asr(
    #     raw,
    #     flatline_criterion=asr_flatline_criterion,
    #     channel_criterion=asr_channel_criterion,
    #     line_noise_criterion=asr_line_noise_criterion,
    #     burst_criterion=asr_burst_criterion,
    #     highpass=asr_highpass
    # )

    # ---- 坏道检测与插值 ----
    raw = detect_and_fix_bad_channels(raw, threshold=3.0)
    logger.info("  坏道插值完成")

    # ---- PCA保留特征值 > 1e-7 的维度 + ICA (runica, extended) ----
    if ica_pca_n_components is None:
        cov = mne.compute_raw_covariance(raw, picks='eeg')
        eigvals = np.linalg.eigvalsh(cov.data)
        n_pca = np.sum(eigvals > 1e-7)
        if n_pca < 1:
            n_pca = min(raw.info['nchan'], 30)
        logger.info(f"  自动 PCA 维度: {n_pca} (特征值 > 1e-7)")
    else:
        n_pca = ica_pca_n_components

    # ---- ICA：自动去除眼电、心电 ----
    ica = ICA(
        n_components=n_pca,
        method=ica_algorithm,
        fit_params=dict(extended=ica_extended) if ica_algorithm == 'infomax' else {},
        random_state=97
    )
    ica.fit(raw, picks=['eeg'], verbose=False)
    logger.info(f"  ICA 拟合完成，成分数: {ica.n_components_}")

    # ---- ICLabel 自动分类并剔除 ----
    bad_components = get_iclabel_bad_components(ica, raw, threshold=iclabel_threshold)
    if bad_components:
        ica.exclude = bad_components
        raw = ica.apply(raw, verbose=False)
        logger.info(f"  ICLabel 移除了 {len(bad_components)} 个成分 (肌肉/眼电/心电/线噪)")
    else:
        logger.info("  ICLabel 未检测到需剔除的成分")

    # ---- 全脑平均参考 ----
    raw = apply_reference(raw)
    logger.info(f"  重参考完成: {filter_low}-{filter_high}Hz")

    # ---- 解析全部 metadata ----
    metadata = parse_metadata(
        subject_id=subject_id,
        bids_root=bids_root,
        deriv_root=deriv_root,
    )

    run_offsets = {}
    offset = 0.0
    for run_num, duration in enumerate(run_durations,start=1):
        run_offsets[run_num] = offset
        offset += duration
    metadata["run_offset"] = metadata["run"].map(run_offsets)
    metadata["onset_global"] = (
            metadata["run_offset"]
            + metadata["onset"].astype(float)
    )
    letter_meta = metadata[metadata["is_letter"]].copy()
    logger.info(
        f"全部 events: {len(metadata)}, "
        f"letter events: {len(letter_meta)}, "
        f"trials: {metadata['trial'].nunique()}"
    )


    epochs, metadata = extract_epochs(
        raw=raw,
        metadata=metadata,
        tmin=epoch_tmin,
        tmax=epoch_tmax,
        baseline=baseline,
        reject=reject,
    )
    validate_epoch_metadata_alignment(epochs,metadata)
    logger.info(f"Epoch 提取完成: {len(epochs)}" )

    # ---- 白化 ----
    if whiten:
        noise_cov = mne.compute_covariance(
            epochs, tmin=baseline[0], tmax=baseline[1], method='shrunk')
        epochs = whitening_manual(epochs, noise_cov, picks='eeg')
        logger.info("已应用多元噪声归一化")

    # ---- 若某个 Trial 中有任何一个 Epoch 被剔除，则删除该 Trial 的所有 Epoch ----
    epochs, metadata = drop_bad_trials(epochs, metadata)

    # ---- 更新position列 ----
    metadata = compute_positions(metadata)

    # ---- 保存衍生数据----
    epochs.metadata = metadata

    epochs_save_path = deriv_root / f"sub-{subject_id}_coding_epo.fif"
    epochs.save(str(epochs_save_path), overwrite=True)
    logger.info(f"  Epochs 已保存: {epochs_save_path}")
    meta_save_path = deriv_root / f"sub-{subject_id}_metadata.csv"
    metadata.to_csv(str(meta_save_path), index=False)
    logger.info(f"  Metadata 已保存: {meta_save_path}")
    return epochs, metadata
