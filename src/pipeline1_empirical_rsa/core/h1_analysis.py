
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_1samp_test
from scipy import stats

from config import CONFIG
from src.shared_utils.models.resolve_model_instances import (
    filter_instances_by_hypothesis,
)
from .sensor_rsa_engine import compute_spearman_correlation
from ..visualization.h1_visualization import h1_plot_delta_curve, h1_all_models
from ...shared_utils.permutation_utils import fisher_z

logger = logging.getLogger(__name__)

def run_subject_level(
        subject_id,
        model_rdms,
        neural_rdms,
        result_root,
        times_ms,
        time_indices,
        suffix: str = '',
        overwrite: bool = False,):

    cache_dir = Path(result_root) / f'sub-{subject_id}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f'sub-{subject_id}_h1_rhos{suffix}.npz'
    if out_path.exists() and not overwrite:
        logger.info(f"缓存已存在，加载: {out_path}")
        data = np.load(out_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    times_ms = np.asarray(times_ms, dtype=float)
    if len(times_ms) == 0:
        raise ValueError(f"被试 {subject_id}: times_ms 为空")
    clean_time_indices = []
    for idx in time_indices:
        idx = int(idx)
        if 0 <= idx < len(times_ms):
            clean_time_indices.append(idx)
        else:
            logger.warning(f"sub-{subject_id}:忽略无效 time index={idx}")
    if not clean_time_indices:
        raise ValueError(f"被试 {subject_id}:没有有效 time_indices")
    clean_time_indices = sorted(set(clean_time_indices))
    model_rdms = filter_instances_by_hypothesis(model_rdms, 'h1')
    if not model_rdms:
        raise ValueError(f"被试 {subject_id}:H1 没有可用理论模型")
    results = {name: [] for name in model_rdms.keys()}
    used_times = []
    used_indices = []
    for t_idx in clean_time_indices:
        if t_idx not in neural_rdms:
            logger.warning(f"sub-{subject_id}: neural_rdm 缺少 time index={t_idx}")
            continue
        neural_rdm = neural_rdms[t_idx]
        used_times.append(float(times_ms[t_idx]))
        used_indices.append(int(t_idx))
        for name, model_rdm in model_rdms.items():
            rho = compute_spearman_correlation(neural_rdm, model_rdm)
            rho = float(rho)
            if not np.isfinite(rho):
                logger.warning( f"sub-{subject_id}, time={times_ms[t_idx]:.1f} ms,model={name}: rho={rho}")
                rho = np.nan
            results[name].append(rho)
    used_times = np.asarray(used_times, dtype=float)
    used_indices = np.asarray(used_indices, dtype=np.int32)
    if len(used_times) == 0:
        raise ValueError(f"被试 {subject_id}: 没有成功计算任何时间点")
    for name in results:
        results[name] = np.asarray(results[name], dtype=float)
        if len(results[name]) != len(used_times):
            raise RuntimeError(
                f"sub-{subject_id}, model={name}: rho长度={len(results[name])}, time长度={len(used_times)}" )
    if not np.all(np.diff(used_times) >= 0):
        raise RuntimeError(f"sub-{subject_id}: 时间轴不是单调递增")
    save_dict = { **results,'times_ms': used_times,'time_indices': used_indices,}
    np.savez_compressed(out_path, **save_dict)
    logger.info(f"H1 结果已保存: {out_path} | time={used_times[0]:.1f}~{used_times[-1]:.1f} ms | n_time={len(used_times)}" )
    return save_dict


def run_group_level(subject_ids: list, result_root: str = 'results',
        suffix: str = '',save_fig_dir: str = 'results/figures/group',):
    rsa_cfg = CONFIG.get('rsa', {})
    n_perm = rsa_cfg.get('n_perm', 5000)
    min_split_half = rsa_cfg.get('min_split_half', 0.05)
    stat_tmin = rsa_cfg.get('stat_tmin',  0.0)
    stat_tmax = rsa_cfg.get('stat_tmax', 800.0)
    min_cluster_points = rsa_cfg.get('min_cluster_points', 3)
    correct_pairwise_fdr = rsa_cfg.get('correct_pairwise_fdr', False)
    # split_half_qc_df = _load_split_half_qc(CONFIG.get('paths', {}))
    model_pairs = [
        ('target_priority_model', 'visual_model'),
        ('target_priority_model', 'binary_color_model'),
        ('gradient_model', 'visual_model'),
        ('binary_adjacency_model', 'visual_model'),
        ('reciprocal_adjacency_model', 'visual_model'),
        # ('serial_position_model', 'visual_model'),
        ('gradient_model', 'timestamp_model'),
    ]
    save_fig_dir = Path(save_fig_dir)
    save_fig_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    pair_min_p = {}
    for model_a, model_b in model_pairs:
        pair_name = f"{model_a}_vs_{model_b}"
        do_stat_test = not (model_a == "gradient_model" and model_b == "timestamp_model")
        X, subject_times, valid_sids = _collect_group_delta(
            subject_ids=subject_ids,model_a=model_a, model_b=model_b,
            result_root=result_root, suffix=suffix, split_half_qc_df=None,
            min_split_half=min_split_half, )
        if X.size == 0:
            logger.warning(f"{pair_name}: 没有有效被试")
            all_results[pair_name] = { 'X': np.empty((0, 0)), 'valid_sids': [],'significant_clusters': [],
                'times_ms': None, 'stat_times_ms': None, }
            continue
        n_sub, n_time = X.shape
        logger.info(f"对比 {pair_name}: 有效被试数 = {n_sub}")
        subject_times = np.asarray(subject_times, dtype=float)
        if len(subject_times) != n_time:
            raise RuntimeError( f"{pair_name}: 时间轴长度={len(subject_times)}, X时间点={n_time}")
        stat_mask = (subject_times >= stat_tmin) & (subject_times <= stat_tmax)
        if not np.any(stat_mask):
            raise ValueError(f"{pair_name}: 不存在 {stat_tmin}~{stat_tmax} ms 的时间点")
        X_stat = X[:, stat_mask]
        stat_times = subject_times[stat_mask]
        mean_delta_z = np.nanmean(X, axis=0)
        sem_delta_z = np.nanstd(X, axis=0, ddof=1) / np.sqrt(n_sub)
        mean_delta_stat = np.nanmean(X_stat, axis=0)
        sem_delta_stat = np.nanstd(X_stat, axis=0, ddof=1) / np.sqrt(n_sub)
        res_dict = {
            'X': X,
            'X_stat': X_stat,
            'valid_sids': valid_sids,
            'mean_delta_z': mean_delta_z,
            'sem_delta_z': sem_delta_z,
            'mean_delta_z_stat': mean_delta_stat,
            'sem_delta_z_stat': sem_delta_stat,
            'significant_clusters': [],
            'times_ms': subject_times,
            'stat_times_ms': stat_times,
            't_obs': None,
            'clusters': None,
            'p_values': None,
            'stat_tmin': stat_tmin,
            'stat_tmax': stat_tmax,
            'min_cluster_points': min_cluster_points,
        }
        if do_stat_test and n_sub >= 3:
            finite_time_mask = np.all(np.isfinite(X_stat), axis=0)
            if not np.all(finite_time_mask):
                logger.warning(  f"{pair_name}: 删除 {np.sum(~finite_time_mask)} 个含非有限值的时间点"  )
                X_test = X_stat[:, finite_time_mask]
                test_times = stat_times[finite_time_mask]
            else:
                X_test = X_stat
                test_times = stat_times
            if X_test.shape[1] == 0:
                logger.warning(f"{pair_name}: 没有可用于统计的时间点")
                all_results[pair_name] = res_dict
                continue
            df = n_sub - 1
            threshold = stats.t.ppf(1 - 0.05 / 2, df)
            logger.info( f"{pair_name}: cluster-forming threshold={threshold:.3f}, df={df}, n_perm={n_perm}")
            t_obs, clusters, p_values, H0 = permutation_cluster_1samp_test(X_test,n_permutations=n_perm,
                tail=0,threshold=threshold, out_type='indices',verbose=False,  seed=42, )
            res_dict['t_obs'] = t_obs
            res_dict['clusters'] = clusters
            res_dict['p_values'] = p_values
            significant_clusters = []
            for i, p in enumerate(p_values):
                if not np.isfinite(p) or p >= 0.05:
                    continue
                cluster = clusters[i]
                if isinstance(cluster, tuple):
                    cluster_idx = np.asarray(cluster[0], dtype=int)
                else:
                    cluster_idx = np.asarray(cluster, dtype=int).ravel()
                if len(cluster_idx) == 0:
                    continue
                cluster_idx = np.sort(cluster_idx)
                t_start_ms = float(test_times[cluster_idx[0]])
                t_end_ms = float(test_times[cluster_idx[-1]])
                cluster_t = t_obs[cluster_idx]
                mean_cluster = np.nanmean(X_test[:, cluster_idx])
                if mean_cluster > 0:
                    direction = "A > B"
                    interpretation = f"{model_a} > {model_b}"
                else:
                    direction = "B > A"
                    interpretation = f"{model_b} > {model_a}"
                is_interpretable = len(cluster_idx) >= min_cluster_points
                cluster_info = {
                    'p': float(p),
                    'time_indices': cluster_idx.tolist(),
                    't_win_ms': [t_start_ms, t_end_ms],
                    'n_points': int(len(cluster_idx)),
                    't_sum': float(np.sum(cluster_t)),
                    't_values': cluster_t.tolist(),
                    'mean_delta_z': float(mean_cluster),
                    'direction': direction,
                    'interpretation': interpretation,
                    'interpretable_length': is_interpretable,
                }
                significant_clusters.append(cluster_info)
                logger.info( f"{pair_name}: 显著簇 p={p:.4f}, {t_start_ms:.1f}-{t_end_ms:.1f} ms,n={len(cluster_idx)}, "
                    f"direction={direction}, interpretable={is_interpretable}")
            res_dict['significant_clusters'] = significant_clusters
            if significant_clusters:
                pair_min_p[pair_name] = min(c['p'] for c in significant_clusters)
        else:
            if not do_stat_test:
                logger.info(f"{pair_name}: Gradient/Timestamp 只做描述性观察，不进行正式置换统计" )
            elif n_sub < 3:
                logger.warning( f"{pair_name}: 被试数={n_sub}，不足以进行群体统计" )
        all_results[pair_name] = res_dict
        h1_plot_delta_curve(result_dict=res_dict, model_a=model_a, model_b=model_b,
            save_path=save_fig_dir / f"H1_delta_{model_a}_vs_{model_b}{suffix}.png", )
    if correct_pairwise_fdr and len(pair_min_p) > 0:
        from statsmodels.stats.multitest import multipletests
        pair_names = list(pair_min_p.keys())
        raw_p = np.asarray([pair_min_p[name] for name in pair_names])
        reject_fdr, p_fdr, _, _ = multipletests( raw_p, alpha=0.05, method='fdr_bh')
        for name, p_adj, reject in zip(pair_names, p_fdr, reject_fdr):
            all_results[name]['pairwise_fdr_p'] = float(p_adj)
            all_results[name]['pairwise_fdr_reject'] = bool(reject)
        logger.info("H1 pairwise FDR:")
        for name, p_adj, reject in zip(pair_names, p_fdr, reject_fdr):
            logger.info(f"  {name}: FDR-p={p_adj:.4f}, reject={reject}")
    _save_group_report(all_results, result_root, suffix)
    all_valid = set()
    for res in all_results.values():
        all_valid.update(res.get('valid_sids', []))
    if all_valid:
        h1_all_models(
            subject_ids=sorted(all_valid),
            result_root=result_root,
            suffix=suffix,
            output_dir=str(save_fig_dir),
        )
    return all_results

def _collect_group_delta( subject_ids, model_a, model_b,result_root,
        suffix, split_half_qc_df=None, min_split_half=0.05,):
    X = []
    valid_subjects = []
    common_times = None
    for sid in subject_ids:
        if split_half_qc_df is not None:
            subject_column = split_half_qc_df["subject"].astype(str)
            rows = split_half_qc_df[subject_column == str(sid)]
            if len(rows) == 0:
                logger.warning(f"sub-{sid}: split-half QC 没有对应记录")
            else:
                reliability = float(rows.iloc[0]["reliability_300_500_mean"])
                if np.isfinite(reliability) and reliability < min_split_half:
                    logger.info(f"skip sub-{sid}: split-half={reliability:.4f}")
                    continue
        cache_dir = Path(result_root) / f'sub-{sid}'
        fpath = cache_dir / f'sub-{sid}_h1_rhos{suffix}.npz'
        if not fpath.exists():
            logger.warning(f"被试 {sid} 的结果缺失，跳过: {fpath}")
            continue
        data = np.load(fpath, allow_pickle=True)
        if model_a not in data or model_b not in data:
            logger.warning(f"被试 {sid} 缺少 {model_a} / {model_b}，跳过")
            continue
        if "times_ms" not in data:
            raise RuntimeError(f"{fpath} 缺少 times_ms，请重新生成 H1 缓存。")
        subject_times = np.asarray(data["times_ms"], dtype=float)
        rho_a = np.asarray(data[model_a], dtype=float)
        rho_b = np.asarray(data[model_b], dtype=float)
        if len(subject_times) != len(rho_a) or len(subject_times) != len(rho_b):
            raise RuntimeError(f"sub-{sid}: time/rho 长度不一致")
        if common_times is None:
            common_times = subject_times.copy()
        else:
            if len(subject_times) != len(common_times):
                raise RuntimeError(f"sub-{sid}: 时间点数量不一致")
            if not np.allclose(subject_times, common_times, atol=1e-6):
                raise RuntimeError(f"sub-{sid}: 时间轴与其他被试不一致")
        z_a = _safe_fisher_z(rho_a)
        z_b = _safe_fisher_z(rho_b)
        delta_z = z_a - z_b
        if not np.all(np.isfinite(delta_z)):
            bad_n = np.sum(~np.isfinite(delta_z))
            logger.warning(f"sub-{sid}: {bad_n} 个时间点 delta_z 非有限")
        X.append(delta_z)
        valid_subjects.append(str(sid))
    if len(X) == 0:
        return np.empty((0, 0)), None, []
    X = np.asarray(X, dtype=float)
    return X, common_times, valid_subjects

def _safe_fisher_z(rho):
    rho = np.asarray(rho, dtype=float)
    rho = np.clip(rho, -0.999999, 0.999999)
    return fisher_z(rho)

def _load_split_half_qc(cfg):
    qc_path = cfg.get('split_half_qc_path', None)
    if not qc_path:
        return None
    qc_path = Path(qc_path)
    if not qc_path.exists():
        logger.warning(f"split_half_qc_path 不存在，跳过信度过滤: {qc_path}")
    return pd.read_csv(qc_path)

def _save_group_report( all_results, result_root, suffix,):
    report_path = Path(result_root) / f'group_h1_clusters{suffix}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("========== H1 群体水平结果 ==========\n")
        for pair_name, res in all_results.items():
            f.write(f"\n对比: {pair_name}\n")
            valid_sids = res.get('valid_sids', [])
            f.write(f"有效被试数: {len(valid_sids)}\n")
            stat_tmin = res.get('stat_tmin')
            stat_tmax = res.get('stat_tmax')
            if stat_tmin is not None and stat_tmax is not None:
                f.write( f"正式统计时间窗: {stat_tmin:.1f}-{stat_tmax:.1f} ms\n" )
            f.write(f"模型: {pair_name}\n")
            clusters = res.get('significant_clusters', [])
            if not clusters:
                f.write("显著簇: 无\n")
            else:
                f.write(f"显著簇数量: {len(clusters)}\n")
                for i, cl in enumerate(clusters, start=1):
                    t0, t1 = cl['t_win_ms']
                    f.write(
                        f"簇 {i}: p={cl['p']:.4f}, "
                        f"time={t0:.1f}-{t1:.1f} ms, "
                        f"n={cl['n_points']}, "
                        f"direction={cl['direction']}, "
                        f"mean_delta_z={cl['mean_delta_z']:.4f}, "
                        f"interpretable={cl['interpretable_length']}\n" )
                    if 'pairwise_fdr_p' in res:
                        f.write(
                            f"    pairwise FDR p={res['pairwise_fdr_p']:.4f}, "
                            f"reject={res['pairwise_fdr_reject']}\n"
                        )
            f.write("\n")
        f.write("======================================\n")
    logger.info(f"H1 群体报告已保存: {report_path}")
