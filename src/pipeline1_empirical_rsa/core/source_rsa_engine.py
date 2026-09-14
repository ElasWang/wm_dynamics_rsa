import logging
from pathlib import Path

import mne

from config import CONFIG

logger = logging.getLogger(__name__)


def _get_forward_solution(info, spacing='ico4'):
    from mne.datasets import fetch_fsaverage
    fs_dir = fetch_fsaverage(verbose=False)
    subjects_dir = fs_dir.parent
    subject = 'fsaverage'
    cache_dir = Path(CONFIG.get('paths', {}).get('source_cache', 'results/cache/source'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    fwd_fname = cache_dir / f'fsaverage-fwd-{spacing}-eeg.fif'
    if fwd_fname.exists():
        fwd = mne.read_forward_solution(fwd_fname, verbose=False)
        src = fwd['src']
        trans = fwd['info']['dev_head_t']
        bem_fname = cache_dir / 'fsaverage-bem-ico4-sol.fif'
        if bem_fname.exists():
            bem_sol = mne.read_bem_solution(bem_fname)
        else:
            bem_model = mne.make_bem_model(subject, ico=4, subjects_dir=subjects_dir,
                                conductivity=(0.3, 0.006, 0.3))
            bem_sol = mne.make_bem_solution(bem_model)
            mne.write_bem_solution(bem_fname, bem_sol)
    else:
        src = mne.setup_source_space(subject, spacing=spacing, subjects_dir=subjects_dir, add_dist=False)
        trans = mne.read_trans(f'{subjects_dir}/{subject}/bem/fsaverage-trans.fif')
        bem_model = mne.make_bem_model(subject, ico=4, subjects_dir=subjects_dir,
                                       conductivity=(0.3, 0.006, 0.3))
        bem_sol = mne.make_bem_solution(bem_model)
        fwd = mne.make_forward_solution(
            info, trans=trans, src=src, bem=bem_sol,
            meg=False, eeg=True, mindist=5.0, n_jobs=4
        )
        mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
        bem_fname = cache_dir / 'fsaverage-bem-ico4-sol.fif'
        mne.write_bem_solution(bem_fname, bem_sol)
    return fwd, src, trans, bem_sol


def _get_inverse_operator(epochs, spacing='ico4', loose=0.2, depth=0.8):
    baseline = CONFIG.get('preprocessing', {}).get('baseline', (-0.2, 0.0))
    noise_cov = mne.compute_covariance(
        epochs, tmin=baseline[0], tmax=baseline[1], method='ledoit_wolf')
    fwd, src, trans, bem_sol = _get_forward_solution(epochs.info, spacing=spacing)
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        epochs.info, fwd, noise_cov, loose=loose, depth=depth)
    return inverse_operator


def cache_cv_source_condition_rdms(subject_id: str,epochs: mne.Epochs,
        tmin: float = -0.2,tmax: float = 0.8,spacing: str = 'ico4',overwrite: bool = False,):
    from pathlib import Path
    import numpy as np
    import logging
    import mne
    from scipy.spatial.distance import pdist, squareform
    from src.shared_utils.models.resolve_model_instances import paths_cfg
    logger = logging.getLogger(__name__)

    cache_root = Path(paths_cfg.get('neural_rdm_root', 'results/rdms/neural'))
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f'sub-{subject_id}_cv_source_rdms.npz'

    if cache_path.exists() and not overwrite:
        logger.info(f"[{subject_id}] 源空间 RDM 缓存已存在，加载: {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        rdms = {int(key[2:]): loaded[key] for key in loaded.files if key.startswith('t_')}
        if "times_ms" not in loaded:
            raise RuntimeError(f"{cache_path} 是旧格式，缺少 times_ms，请重新生成缓存。")
        times_ms = loaded["times_ms"]
        time_indices = loaded["time_indices"]
        return rdms, times_ms, time_indices

    if epochs.metadata is None:
        raise ValueError("Epochs 缺少 metadata")
    if 'position' not in epochs.metadata.columns:
        raise ValueError("metadata 中缺少 'position' 列")
    logger.info(f"[{subject_id}] 计算所有时间点源空间 RDM (spacing={spacing})")
    epochs.set_eeg_reference('average', projection=True)
    inverse_operator = _get_inverse_operator(epochs, spacing=spacing)
    lambda2 = 1.0 / 3.0 ** 2
    n_times = len(epochs.times)
    positions = epochs.metadata['position'].values
    dummy_evoked = epochs.average()
    dummy_stc = mne.minimum_norm.apply_inverse(
        dummy_evoked, inverse_operator, lambda2,
        method="dSPM", pick_ori=None, verbose=False)
    n_vertices = dummy_stc.data.shape[0]
    pos_patterns = {p: None for p in range(1, 9)}
    pos_counts = {p: 0 for p in range(1, 9)}
    for idx, epoch_data in enumerate(epochs.get_data()):
        evoked = mne.EvokedArray(epoch_data, epochs.info, tmin=epochs.times[0])
        stc = mne.minimum_norm.apply_inverse(
            evoked, inverse_operator, lambda2,
            method="dSPM", pick_ori=None, verbose=False)
        p = positions[idx]
        if pos_patterns[p] is None:
            pos_patterns[p] = stc.data.T
        else:
            pos_patterns[p] += stc.data.T
        pos_counts[p] += 1
        if (idx + 1) % 100 == 0:
            logger.debug(f"[{subject_id}] 已处理 {idx + 1}/{len(epochs)} 个试次")
    all_rdms = np.zeros((n_times, 8, 8), dtype=np.float32)
    for t in range(n_times):
        patterns_t = []
        for p in range(1, 9):
            if pos_counts[p] == 0:
                logger.warning(f"[{subject_id}] position {p} 没有试次，使用零向量")
                patterns_t.append(np.zeros(n_vertices))
            else:
                patterns_t.append(pos_patterns[p][t] / pos_counts[p])
        patterns_t = np.array(patterns_t)
        rdm = squareform(pdist(patterns_t, metric='euclidean'))
        np.fill_diagonal(rdm, 0.0)
        all_rdms[t] = rdm.astype(np.float32)
    times = epochs.times
    mask = (times >= tmin) & (times <= tmax)
    time_indices = np.where(mask)[0].tolist()
    if len(time_indices) == 0:
        raise ValueError(f"窗口 {tmin}-{tmax}s 无有效时间点")
    rdms = {t_idx: all_rdms[t_idx] for t_idx in time_indices}
    save_dict = {f't_{t_idx:03d}': rdm for t_idx, rdm in rdms.items()}
    times_s_selected = times[time_indices]
    times_ms_selected = times_s_selected * 1000.0
    save_dict["times_ms"] = times_ms_selected.astype(np.float64)
    save_dict["time_indices"] = np.asarray(time_indices, dtype=np.int32)
    save_dict["sfreq"] = np.asarray(epochs.info['sfreq'], dtype=np.float64)
    save_dict["rsa_tmin"] = np.asarray(tmin, dtype=np.float64)
    save_dict["rsa_tmax"] = np.asarray(tmax, dtype=np.float64)
    np.savez_compressed(cache_path, **save_dict)
    logger.info(
        f"[{subject_id}] 源空间 RDM 缓存完成: {cache_path} (大小: {cache_path.stat().st_size / (1024 * 1024):.1f} MB)")
    return rdms, times_ms_selected, np.asarray(time_indices, dtype=np.int32)
