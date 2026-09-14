import logging
from pathlib import Path
from typing import Tuple

import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA

logger = logging.getLogger(__name__)


def resample_data(raw, target_sfreq=250):
    if raw.info['sfreq'] != target_sfreq:
        return raw.copy().resample(target_sfreq, npad="auto")
    return raw


def apply_filter(raw, l_freq=0.1, h_freq=40.0):
    return raw.copy().filter(l_freq, h_freq, fir_design='firwin', verbose=False)


def apply_reference(raw):
    return raw.copy().set_eeg_reference('average', projection=False)


def detect_and_fix_bad_channels(raw, threshold=3.0):
    flat_channels = []
    for ch in raw.ch_names:
        data_ch = raw.get_data(picks=ch)[0]
        if np.max(data_ch) - np.min(data_ch) < 1e-10:
            flat_channels.append(ch)
    if flat_channels:
        raw.info['bads'].extend(flat_channels)
        logger.warning(f"  检测到平线通道: {flat_channels}")
    data, _ = raw[:, :]
    ch_var = np.var(data, axis=1)
    median_var = np.median(ch_var)
    mad_var = np.median(np.abs(ch_var - median_var)) * 1.4826
    bads = []
    for i, ch in enumerate(raw.ch_names):
        if ch in flat_channels:
            continue
        if abs(ch_var[i] - median_var) > threshold * mad_var:
            bads.append(ch)
    if bads:
        raw.info['bads'].extend(bads)
        logger.warning(f"  方差检测到坏道: {bads}")
    corr_matrix = np.corrcoef(data)
    avg_corr = np.mean(corr_matrix, axis=1)
    median_corr = np.median(avg_corr)
    mad_corr = np.median(np.abs(avg_corr - median_corr)) * 1.4826
    bads_corr = []
    for i, ch in enumerate(raw.ch_names):
        if ch in flat_channels or ch in bads:
            continue
        if (median_corr - avg_corr[i]) > 3.0 * mad_corr:
            bads_corr.append(ch)
    if bads_corr:
        raw.info['bads'].extend(bads_corr)
        logger.warning(f"  相关性检测到坏道: {bads_corr}")
    if raw.info['bads']:
        raw.interpolate_bads(reset_bads=True)
        logger.info(f"  插值完成，共 {len(raw.info['bads'])} 个坏道被修复")
    else:
        logger.info("  未检测到坏道")
    return raw


def standardize_channels(raw: mne.io.Raw, montage) -> mne.io.Raw:
    raw.set_channel_types({'LEYE': 'eog', 'REYE': 'eog'})
    raw.set_montage(montage, on_missing='warn')
    return raw


def apply_ica(raw: mne.io.Raw,
              eog_threshold: float = 2.5,
              ecg_threshold: float = 2.5,
              n_components: float = 0.99,
              ecg_method: str = 'correlation', ) -> mne.io.Raw:
    eog_channels = [ch for ch in raw.ch_names if 'EYE' in ch]
    if eog_channels:
        logger.info(f"  找到 EOG 通道: {eog_channels}")
    else:
        logger.info("  未找到 EOG 通道，将使用 Fp1/Fp2 作为参考进行检测")
    ecg_channels = [ch for ch in raw.ch_names if ch.lower() in ['ecg', 'ekg']]
    if ecg_channels:
        logger.info(f"  找到 ECG 通道: {ecg_channels}")
    else:
        logger.info("  未找到 ECG 通道，将尝试自动检测心电成分")
    ica = ICA(n_components=n_components, method='infomax', random_state=97)
    ica.fit(raw, picks=['eeg'], verbose=False)
    if eog_channels:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_channels, threshold=eog_threshold)
    else:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'], threshold=eog_threshold)
    if ecg_channels:
        ecg_indices, _ = ica.find_bads_ecg(raw, ch_name=ecg_channels, threshold=ecg_threshold)
    else:
        try:
            ecg_indices, _ = ica.find_bads_ecg(raw, method=ecg_method, threshold=ecg_threshold)
        except Exception as e:
            logger.warning(f"  自动检测 ECG 失败: {e}，跳过 ECG 去除")
            ecg_indices = []
    all_bads = list(set(eog_indices + ecg_indices))
    if all_bads:
        ica.exclude = all_bads
        raw = ica.apply(raw, verbose=False)
        logger.info(f"  ICA 去噪成功，移除了 {len(all_bads)}"
                    f" 个成分 (EOG: {len(eog_indices)}, ECG: {len(ecg_indices)})")
    else:
        logger.info("  未检测到显著的伪迹成分")
    return raw


def extract_epochs(
        raw: mne.io.Raw,
        metadata: pd.DataFrame,
        tmin: float = -0.2,
        tmax: float = 0.8,
        baseline: tuple = None,
        reject: dict = None,
        flat: dict = None,
) -> Tuple[mne.Epochs, pd.DataFrame]:
    metadata = metadata.copy()
    if metadata.empty:
        raise ValueError("metadata 为空，无法提取 epochs")
    required_cols = ["run", "onset", "trial", "event_uid"]
    missing = [c for c in required_cols if c not in metadata.columns]
    if missing:
        raise ValueError(f"metadata 缺少必要字段: {missing}")
    target_mask = metadata["is_letter"].astype(bool)
    target_meta = (metadata.loc[target_mask].reset_index(drop=True).copy())
    if len(target_meta) == 0:
        raise ValueError("没有找到任何 letter event")
    if target_meta["event_uid"].duplicated().any():
        duplicated = target_meta.loc[
            target_meta["event_uid"].duplicated(keep=False),
            "event_uid"
        ].tolist()
        raise ValueError(
            f"发现重复 event_uid，共 {len(duplicated)} 个: "
            f"{duplicated[:10]}" )
    if "onset_global" not in target_meta.columns:
        raise ValueError(
            "metadata 缺少 onset_global。"
            "不要直接使用 run 内 onset，因为多个 run 已经拼接。" )
    sample_indices = raw.time_as_index(
        target_meta["onset_global"].to_numpy(),
        use_rounding=True)
    event_codes = np.arange(1,len(target_meta) + 1,dtype=int)
    events = np.column_stack([sample_indices,
        np.zeros(len(target_meta), dtype=int),event_codes])
    event_id = {f"event_{i}": code for i, code in enumerate(event_codes) }
    epochs = mne.Epochs(raw,events, event_id=event_id,tmin=tmin,
        tmax=tmax,baseline=baseline,preload=True,reject=reject,flat=flat,
        reject_by_annotation=True,verbose=False,)
    kept_indices = epochs.selection
    metadata_kept = (target_meta.iloc[kept_indices].reset_index(drop=True))
    if len(epochs) != len(metadata_kept):
        raise RuntimeError(
            f"Epoch / metadata 对齐失败: "
            f"epochs={len(epochs)}, "
            f"metadata={len(metadata_kept)}")
    for i, event in enumerate(epochs.events):
        expected_code = event_codes[kept_indices[i]]
        if event[2] != expected_code:
            raise RuntimeError(
                f"Epoch {i} event code 对不上: "
                f"{event[2]} != {expected_code}")
    logger.info(
        f"letter events: {len(target_meta)}, "
        f"保留 epochs: {len(epochs)}, "
        f"剔除: {len(target_meta) - len(epochs)}")
    return epochs, metadata_kept


def parse_metadata(
        subject_id: str,
        bids_root: Path,
) -> pd.DataFrame:
    all_dfs = []
    cumulative_offset = 0.0
    for run_num in range(1, 10):
        events_tsv_path = (bids_root/ f"sub-{subject_id}"/ "ses-01"/ "eeg"
            / f"sub-{subject_id}_ses-01_task-WorkingMemory_run-{run_num}_events.tsv")
        if not events_tsv_path.exists():
            if run_num == 1:
                raise FileNotFoundError(f"run-1 events.tsv 不存在: {events_tsv_path}")
            logger.info(f"run-{run_num} events.tsv 不存在，停止加载")
            break
        df_run = pd.read_csv(events_tsv_path,sep="\t").copy()
        if "onset" not in df_run.columns:
            raise ValueError(f"run-{run_num} events.tsv 没有 onset 列")
        df_run["run"] = run_num
        df_run["event_index_run"] = np.arange(len(df_run))
        df_run["event_uid"] = [f"sub-{subject_id}_run-{run_num}_event-{i:04d}"
            for i in range(len(df_run))]
        df_run["onset_global"] = (df_run["onset"].astype(float)
                + cumulative_offset )
        if "value" in df_run.columns:
            values = df_run["value"].astype(str)
            is_letter = (values.str.match(r"^[A-Z]$")|
                    values.str.match(r"^[gr][A-Z]$"))
        else:
            raise ValueError( "events.tsv 缺少 value 列，无法判断 letter event")
        df_run["is_letter"] = is_letter
        df_run["color"] = df_run["value"].apply(extract_color)
        if "letter" in df_run.columns:
            df_run["letter"] = (df_run["letter"].astype(str))
        else:
            df_run["letter"] = (df_run["value"].astype(str).str[-1])
        if "memory_cond" in df_run.columns:df_run["load"] = df_run[ "memory_cond" ]
        else:
            df_run["load"] = np.nan
        all_dfs.append(df_run)
        run_duration = (df_run["onset"].astype(float).max() )
        cumulative_offset += run_duration
        logger.info(
            f"加载 run-{run_num}: "
            f"{len(df_run)} events, "
            f"letter={is_letter.sum()}")
    if not all_dfs:
        raise ValueError(f"被试 {subject_id} 没有任何 events.tsv")
    metadata = pd.concat(all_dfs,ignore_index=True)
    if not metadata["event_uid"].is_unique:
        raise RuntimeError( "event_uid 不是唯一的")
    if metadata["onset_global"].isna().any():
        raise RuntimeError( "发现 NaN onset_global")
    return metadata


def extract_color(val: str) -> str:
    if isinstance(val, str) and val.startswith('g'):
        return 'green'
    return 'black'


def drop_bad_trials(
        epochs: mne.Epochs,
        metadata: pd.DataFrame
) -> Tuple[mne.Epochs, pd.DataFrame]:
    if len(epochs) != len(metadata):
        raise RuntimeError(
            f"Epoch / metadata 数量不一致: "
            f"{len(epochs)} vs {len(metadata)}" )
    metadata = metadata.reset_index(drop=True).copy()
    kept_mask = np.zeros(len(metadata),dtype=bool)
    kept_mask[epochs.selection] = True
    metadata["_epoch_kept"] = kept_mask
    trial_quality = (metadata.groupby("trial")["_epoch_kept"].all())
    bad_trial_ids = trial_quality[~trial_quality].index.to_numpy()
    if len(bad_trial_ids) == 0:
        logger.info("没有 trial 因包含坏 epoch 而被整体删除" )
        metadata = metadata.drop(columns=["_epoch_kept"] )
        assert len(epochs) == len(metadata)
        return epochs, metadata
    logger.info(f"发现坏 trial:{len(bad_trial_ids)}")
    keep_mask = ~metadata["trial"].isin(bad_trial_ids)
    metadata_final = (metadata.loc[keep_mask].drop(columns=["_epoch_kept"])
        .reset_index(drop=True) )
    epochs_final = epochs[keep_mask.to_numpy() ]
    if len(epochs_final) != len(metadata_final):
        raise RuntimeError("整体删除 bad trial 后，"
            "epochs 与 metadata 再次发生错位")
    logger.info(
        f"整体剔除后: "
        f"epochs={len(epochs_final)}, "
        f"metadata={len(metadata_final)}" )
    return epochs_final, metadata_final


def whitening_manual(epochs, noise_cov, picks='eeg'):
    if picks == 'eeg':
        picks_idx = mne.pick_types(epochs.info, eeg=True)
    elif picks == 'meg':
        picks_idx = mne.pick_types(epochs.info, meg=True)
    else:
        picks_idx = mne.pick_channels(epochs.ch_names, picks) if isinstance(picks, list) else picks
    epochs = epochs.pick_channels([epochs.ch_names[i] for i in picks_idx])
    data = epochs.get_data()
    cov = noise_cov.data[picks_idx][:, picks_idx]
    n_epochs, n_chans, n_times = data.shape
    reg = 1e-6
    cov_reg = cov + reg * np.eye(n_chans)
    eigvals, eigvecs = np.linalg.eigh(cov_reg)
    eigvals = np.maximum(eigvals, 0)
    whitener = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    data_flat = np.transpose(data, (0, 2, 1)).reshape(-1, n_chans)
    data_whitened_flat = data_flat @ whitener.T
    data_whitened = data_whitened_flat.reshape(n_epochs, n_times, n_chans).transpose(0, 2, 1)
    epochs_whitened = epochs.copy()
    epochs_whitened._data = data_whitened.astype(np.float64)
    logger.info(f"白化矩阵计算完成，通道数: {n_chans},条件数: {eigvals.max() / eigvals.min():.2f}")
    return epochs_whitened


def get_iclabel_bad_components(ica, raw, threshold=0.8):
    try:
        from mne_icalabel import label_components
    except ImportError:
        logger.warning("mne-icalabel 未安装，请安装以使用 ICLabel。")
        return []
    ic_labels = label_components(raw, ica, method='iclabel')
    labels = ic_labels['labels']
    y_prob = ic_labels['y_pred_proba']
    bad_idx = []
    for i, (label, probs) in enumerate(zip(labels, y_prob)):
        if label in ['muscle', 'eye', 'heart', 'line_noise']:
            if probs.max() >= threshold:
                bad_idx.append(i)
    return bad_idx


def validate_epoch_metadata_alignment(
        epochs: mne.Epochs,
        metadata: pd.DataFrame,):
    if len(epochs) != len(metadata):
        raise AssertionError(
            f"数量不一致: "
            f"epochs={len(epochs)}, "
            f"metadata={len(metadata)}" )
    if not metadata["event_uid"].is_unique:
        raise AssertionError("event_uid 存在重复" )
    epoch_times = (epochs.events[:, 0] /epochs.info["sfreq"])
    meta_times = (metadata["onset_global"].to_numpy())
    time_error = np.abs(epoch_times - meta_times)
    max_error = time_error.max()
    logger.info(
        f"最大 event 时间对齐误差: "
        f"{max_error * 1000:.3f} ms" )
    if max_error > (2.0 / epochs.info["sfreq"]):
        raise AssertionError( "Epoch 与 metadata onset 对齐误差过大" )
    logger.info("Epoch ↔ metadata 对齐检查通过" )
