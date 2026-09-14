import logging
from pathlib import Path

import mne
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from config import CONFIG

logger = logging.getLogger(__name__)


def compute_sensor_rdms(
        epochs: mne.Epochs,
        time_indices: list = None,
        metric: str = None,
) -> dict:
    if metric is None:
        metric = CONFIG.get('rsa', {}).get('dist_metric', 'correlation')
    data = epochs.get_data()
    n_epochs, n_chans, n_times = data.shape
    if time_indices is None:
        time_indices = list(range(n_times))
    rdms = {}
    for t_idx in time_indices:
        patterns = data[:, :, t_idx]
        dist_vec = pdist(patterns, metric=metric)
        rdm = squareform(dist_vec)
        rdms[t_idx] = rdm

    logger.info(f"计算完成，共 {len(rdms)} 个时间点的 RDM (metric={metric})")
    return rdms


def cache_sensor_rdms(
        subject_id: str,
        epochs: mne.Epochs,
        overwrite: bool = False,
        dtype: type = np.float32,
):
    paths_cfg = CONFIG.get('paths', {})
    cache_root = Path(paths_cfg.get('neural_rdm_root', 'results/rdms/neural'))
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f'sub-{subject_id}_rdms.npz'

    if cache_path.exists() and not overwrite:
        logger.info(f"缓存已存在，加载: {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        rdms = {int(key[2:]): loaded[key] for key in loaded.files if key.startswith('t_')}
        if "times_ms" not in loaded:
            raise RuntimeError(f"{cache_path} 是旧格式，缺少 times_ms，请重新生成缓存。")
        times_ms = loaded["times_ms"]
        time_indices = loaded["time_indices"]
        return rdms, times_ms, time_indices

    rsa_cfg = CONFIG.get('rsa', {})
    tmin = rsa_cfg.get('tmin', -0.1)
    tmax = rsa_cfg.get('tmax', 0.8)
    metric = rsa_cfg.get('dist_metric', 'correlation')
    times = epochs.times
    mask = (times >= tmin) & (times <= tmax)
    time_indices = np.where(mask)[0].tolist()
    rdms = compute_sensor_rdms(epochs, time_indices, metric=metric)
    save_dict = {}
    for t_idx, rdm in rdms.items():
        key = f't_{t_idx:03d}'
        save_dict[key] = rdm.astype(dtype)
    times_s_selected = times[time_indices]
    times_ms_selected = times_s_selected * 1000.0
    save_dict["times_ms"] = times_ms_selected.astype(np.float64)
    save_dict["times_s"] = times_s_selected.astype(np.float64)
    save_dict["time_indices"] = np.asarray(time_indices, dtype=np.int32)
    save_dict["sfreq"] = np.asarray(epochs.info['sfreq'], dtype=np.float64)
    save_dict["rsa_tmin"] = np.asarray(tmin, dtype=np.float64)
    save_dict["rsa_tmax"] = np.asarray(tmax, dtype=np.float64)
    np.savez_compressed(cache_path, **save_dict)
    file_size = cache_path.stat().st_size / (1024 * 1024)
    logger.info(f"缓存完成: {cache_path} (大小: {file_size:.1f} MB)")
    logger.info(f"时间轴: {times_ms_selected[0]:.1f} ~ {times_ms_selected[-1]:.1f} ms, 共 {len(times_ms_selected)} 点")

    return rdms, times_ms_selected, np.asarray(time_indices, dtype=np.int32)


def cache_cv_condition_rdms(
        subject_id: str,
        epochs: mne.Epochs,
        condition_col: str = 'position',
        n_repeats: int = 10,
        test_ratio: float = 0.5,
        overwrite: bool = False,
        dtype: type = np.float32,
):

    paths_cfg = CONFIG.get('paths', {})
    cache_root = Path(paths_cfg.get('neural_rdm_root', 'results/rdms/neural'))
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f'sub-{subject_id}_cv_condition_rdms.npz'

    if cache_path.exists() and not overwrite:
        logger.info(f"CV条件级RDM缓存已存在，加载: {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        rdms = {int(key[2:]): loaded[key] for key in loaded.files if key.startswith('t_')}
        if "times_ms" not in loaded:
            raise RuntimeError(f"{cache_path} 是旧格式，缺少 times_ms，请重新生成缓存。")
        times_ms = loaded["times_ms"]
        time_indices = loaded["time_indices"]
        return rdms, times_ms, time_indices

    rsa_cfg = CONFIG.get('rsa', {})
    tmin = rsa_cfg.get('tmin', -0.1)
    tmax = rsa_cfg.get('tmax', 0.8)
    times = epochs.times
    mask = (times >= tmin) & (times <= tmax)
    time_indices = np.where(mask)[0].tolist()

    meta = epochs.metadata
    if meta is None:
        raise ValueError("epochs.metadata 为空，无法进行条件级分析")
    if condition_col not in meta.columns:
        raise ValueError(f"metadata 中缺少列: {condition_col}")

    conditions = sorted(meta[condition_col].unique())
    n_cond = len(conditions)
    data = epochs.get_data()
    rdms = {}

    for t_idx in time_indices:
        patterns = data[:, :, t_idx]
        accum_rdm = np.zeros((n_cond, n_cond))
        valid_reps = 0
        for rep in range(n_repeats):
            train_patterns = {}
            test_patterns = {}
            skip_rep = False
            for cond in conditions:
                idx = np.where(meta[condition_col] == cond)[0]
                if len(idx) < 2:
                    skip_rep = True
                    break
                idx_shuffled = np.random.permutation(idx)
                split = int(len(idx_shuffled) * test_ratio)
                test_idx = idx_shuffled[:split]
                train_idx = idx_shuffled[split:]
                if len(train_idx) == 0 or len(test_idx) == 0:
                    skip_rep = True
                    break
                train_patterns[cond] = np.mean(patterns[train_idx, :], axis=0)
                test_patterns[cond] = np.mean(patterns[test_idx, :], axis=0)
            if skip_rep:
                continue
            valid_reps += 1
            for i, ci in enumerate(conditions):
                for j, cj in enumerate(conditions):
                    if i >= j:
                        continue
                    d_train = np.linalg.norm(train_patterns[ci] - train_patterns[cj])
                    d_test = np.linalg.norm(test_patterns[ci] - test_patterns[cj])
                    d = (d_train + d_test) / 2.0
                    accum_rdm[i, j] += d
                    accum_rdm[j, i] += d
        if valid_reps == 0:
            logger.warning(f"时间点 {t_idx} 没有任何有效的拆分重复，跳过")
            continue
        rdm = accum_rdm / valid_reps
        np.fill_diagonal(rdm, 0.0)
        rdms[t_idx] = rdm.astype(dtype)
    save_dict = {f't_{t_idx:03d}': rdm for t_idx, rdm in rdms.items()}
    times_s_selected = times[time_indices]
    times_ms_selected = times_s_selected * 1000.0
    save_dict["times_ms"] = times_ms_selected.astype(np.float64)
    save_dict["times_s"] = times_s_selected.astype(np.float64)
    save_dict["time_indices"] = np.asarray(time_indices, dtype=np.int32)
    save_dict["sfreq"] = np.asarray(epochs.info['sfreq'], dtype=np.float64)
    save_dict["rsa_tmin"] = np.asarray(tmin, dtype=np.float64)
    save_dict["rsa_tmax"] = np.asarray(tmax, dtype=np.float64)
    np.savez_compressed(cache_path, **save_dict)
    logger.info(f"CV条件级RDM缓存完成: {cache_path} (大小: {cache_path.stat().st_size / (1024 * 1024):.1f} MB)")
    return rdms, times_ms_selected, np.asarray(time_indices, dtype=np.int32)


def load_all_neural_rdms(subject_id: str, cache_root: str):
    cache_path = Path(cache_root) / f'sub-{subject_id}_rdms.npz'
    if not cache_path.exists():
        raise FileNotFoundError(f"缓存不存在: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    rdms = {}
    for key in data.files:
        if key.startswith('t_'):
            rdms[int(key[2:])] = data[key]
    if "times_ms" not in data:
        raise RuntimeError(f"{cache_path} 是旧格式，没有保存 times_ms。")
    times_ms = data["times_ms"]
    time_indices = data["time_indices"]
    if len(rdms) != len(times_ms):
        raise ValueError(f"RDM数与时间点数不一致: RDM={len(rdms)}, times={len(times_ms)}")
    return rdms, times_ms, time_indices


def extract_upper_triangular(rdm: np.ndarray) -> np.ndarray:
    rdm = np.asarray(rdm)
    if rdm.ndim == 1:
        return rdm
    if rdm.ndim == 2:
        return rdm[np.triu_indices_from(rdm, k=1)]


def compute_spearman_correlation(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    vec_a = extract_upper_triangular(rdm_a)
    vec_b = extract_upper_triangular(rdm_b)
    if np.std(vec_a) == 0 or np.std(vec_b) == 0:
        return np.nan
    rho, _ = spearmanr(vec_a, vec_b)
    return float(rho)


def compute_generalization_matrix(
        neural_rdms: dict,
        train_indices: list,
        test_indices: list,
) -> np.ndarray:
    first_rdm = next(iter(neural_rdms.values()))
    triu_idx = np.triu_indices_from(first_rdm, k=1)
    all_indices = set(train_indices) | set(test_indices)
    vecs = {t: neural_rdms[t][triu_idx] for t in all_indices}
    matrix = np.zeros((len(train_indices), len(test_indices)))
    for i, tr in enumerate(train_indices):
        for j, te in enumerate(test_indices):
            matrix[i, j] = compute_spearman_correlation(vecs[tr], vecs[te])
    return matrix


def fit_regression(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import RidgeCV
    reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    reg.fit(X, y)
    return reg.coef_, reg.score(X, y)

def cache_cv_sequence_rdms(
        subject_id: str,
        epochs: mne.Epochs,
        overwrite: bool = False,
        dtype: type = np.float32,):

    paths_cfg = CONFIG.get('paths', {})
    cache_root = Path(paths_cfg.get( 'neural_rdm_root', 'results/rdms/neural' ) )
    cache_root.mkdir( parents=True, exist_ok=True  )
    cache_path = ( cache_root /f'sub-{subject_id}_cv_sequence_rdms.npz')

    if cache_path.exists() and not overwrite:
        logger.info( f"CV序列级RDM缓存已存在，加载: {cache_path}" )
        loaded = np.load( cache_path, allow_pickle=True)
        rdms = {
            int(key[2:]): loaded[key]
            for key in loaded.files
            if key.startswith('t_') }
        if "times_ms" not in loaded:
            raise RuntimeError( f"{cache_path} 是旧格式，缺少 times_ms，请重新生成缓存。" )
        times_ms = loaded["times_ms"]
        time_indices = loaded["time_indices"]
        return ( rdms,times_ms, time_indices )
    rsa_cfg = CONFIG.get(  'rsa', {} )
    tmin = rsa_cfg.get( 'tmin',-0.1 )
    tmax = rsa_cfg.get('tmax',0.8 )
    times = epochs.times
    mask = ((times >= tmin) & (times <= tmax) )
    time_indices = np.where( mask )[0].tolist()
    meta = epochs.metadata
    if meta is None:
        raise ValueError( "epochs.metadata 为空，无法进行 sequence-level RDM 分析")
    sequence_keys = list(zip( meta['run'], meta['trial']) )
    position_values = meta['position'].values
    sequence_ids = list( dict.fromkeys(sequence_keys))
    n_sequences = len( sequence_ids)
    logger.info(
        f"开始计算 sequence-level RDM:subject={subject_id}, Nsequence={n_sequences}" )
    sequence_position_indices = {}
    for seq_id in sequence_ids:
        run_value, trial_value = seq_id
        idx_seq = np.where(
            (meta['run'].values == run_value) & (meta['trial'].values == trial_value))[0]
        positions_seq = position_values[idx_seq ]
        unique_positions = list(dict.fromkeys( positions_seq.tolist()) )
        if len(unique_positions) != 8:
            raise ValueError( f"sequence={seq_id} 不是8个position，"
                f"实际={len(unique_positions)}，positions={unique_positions}")
        position_map = {}
        for pos in unique_positions:
            idx_pos = idx_seq[ positions_seq == pos ]
            if len(idx_pos) != 1:
                raise ValueError( f"sequence={seq_id},position={pos} 有 {len(idx_pos)} 个epoch。"
                    f"\n sequence-level RDM要求一个sequence中每个position对应一个event/epoch。")
            position_map[pos] = int( idx_pos[0] )
        sequence_position_indices[ seq_id ] = position_map
    data = epochs.get_data()
    n_epochs, n_chans, n_times = data.shape
    logger.info(
        f"Epoch数据: {n_epochs} epochs × {n_chans} channels × {n_times} timepoints")
    if n_epochs != len(meta):
        raise ValueError( f"epochs数量({n_epochs}) 与metadata行数({len(meta)})不一致" )
    rdms = {}
    for t_idx in time_indices:
        patterns = data[ :, :, t_idx]
        sequence_rdms = []
        for seq_id in sequence_ids:
            position_map = (sequence_position_indices[ seq_id ] )
            try:
                sorted_positions = sorted( position_map.keys(), key=lambda x: float(x) )
            except Exception:
                sorted_positions = sorted( position_map.keys())
            if len(sorted_positions) != 8:
                raise RuntimeError(f"sequence={seq_id} position排序后不是8个" )
            seq_epoch_indices = [position_map[pos] for pos in sorted_positions ]
            seq_patterns = patterns[ seq_epoch_indices, :]
            metric = ( CONFIG.get('rsa', {}).get( 'dist_metric','correlation') )
            dist_vec = pdist(seq_patterns,metric=metric )
            seq_rdm = squareform(dist_vec)
            seq_rdm = (seq_rdm +seq_rdm.T) / 2.0
            np.fill_diagonal(seq_rdm,0.0)
            sequence_rdms.append(seq_rdm)
        sequence_rdms = np.asarray(sequence_rdms,dtype=dtype )
        if sequence_rdms.shape != ( n_sequences, 8, 8 ):
            raise RuntimeError(
                f"时间点 {t_idx} sequence RDM shape异常: {sequence_rdms.shape}, 期望=({n_sequences},8,8)" )
        rdms[t_idx] = sequence_rdms
    save_dict = {f't_{t_idx:03d}': rdm for t_idx, rdm in rdms.items()}
    times_s_selected = (times[time_indices] )
    times_ms_selected = (times_s_selected * 1000.0)
    save_dict["times_ms"] = ( times_ms_selected.astype(np.float64))
    save_dict["times_s"] = (times_s_selected.astype(np.float64))
    save_dict["time_indices"] = ( np.asarray(time_indices,dtype=np.int32))
    save_dict["sfreq"] = (np.asarray(epochs.info['sfreq'],dtype=np.float64 ) )
    save_dict["rsa_tmin"] = (np.asarray(tmin, dtype=np.float64 ))
    save_dict["rsa_tmax"] = (np.asarray(tmax,dtype=np.float64 ))
    save_dict["n_sequences"] = ( np.asarray( n_sequences,dtype=np.int32 ) )
    save_dict["sequence_ids"] = np.asarray(sequence_ids, dtype=object)
    np.savez_compressed(cache_path,**save_dict)
    file_size = ( cache_path.stat().st_size/ (1024 * 1024))

    logger.info(
        f"CV序列级RDM缓存完成: "
        f"{cache_path} "
        f"(大小: {file_size:.1f} MB)")

    logger.info(
        f"时间轴: "
        f"{times_ms_selected[0]:.1f} ~ "
        f"{times_ms_selected[-1]:.1f} ms, "
        f"共 {len(times_ms_selected)} 点"
    )
    return (rdms,times_ms_selected,
        np.asarray(time_indices,dtype=np.int32) )