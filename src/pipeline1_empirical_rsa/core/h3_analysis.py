import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel, spearmanr

from config import CONFIG
from src.shared_utils.models.registry import list_models_by_hypothesis
from src.shared_utils.models.resolve_model_instances import filter_instances_by_hypothesis
from .sensor_rsa_engine import compute_generalization_matrix
from src.pipeline1_empirical_rsa.visualization.h3_visualization import h3_group_generalization_heatmap, \
    h3_generalization_diff, h3_behavior_scatter

logger = logging.getLogger(__name__)


def run_subject_level(
        suffix,
        subject_id,
        model_rdms,
        neural_rdms,
        result_root,
        times_ms: np.ndarray = None,
        structure_model_name='target_priority_model',
        overwrite: bool = False,
):
    cache_dir = Path(result_root) / f'sub-{subject_id}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f'sub-{subject_id}_h3_rhos{suffix}.npz'

    if out_path.exists() and not overwrite:
        logger.info(f"  缓存已存在，加载: {out_path}")
        return _load_h3_result(out_path)

    all_h3_models = list_models_by_hypothesis('h3')
    control_model_names = [name for name in all_h3_models if name != structure_model_name]

    available_times = sorted(neural_rdms.keys())
    valid_idx_set = set(available_times)
    train_time_range = CONFIG.get('rsa', {}).get('train_indices', [200, 500])
    test_time_range = CONFIG.get('rsa', {}).get('test_indices', [600, 800])


    train_idx = time_window_to_indices(
        times_ms, train_time_range[0], train_time_range[1], valid_idx_set )
    test_idx = time_window_to_indices(
        times_ms, test_time_range[0], test_time_range[1], valid_idx_set)
    if not train_idx or not test_idx:
        raise ValueError(f"被试 {subject_id} 的编码期或维持期无有效时间点")
    gen_mat = compute_generalization_matrix(neural_rdms, train_idx, test_idx)
    first_rdm = neural_rdms[train_idx[0]]
    triu_idx = np.triu_indices_from(first_rdm, k=1)

    model_rdms = filter_instances_by_hypothesis(model_rdms, 'h3')
    struct_vec = model_rdms.get(structure_model_name)[triu_idx]
    struct_rhos = []
    for t in test_idx:
        neural_vec = neural_rdms[t][triu_idx]
        rho, _ = spearmanr(neural_vec, struct_vec)
        struct_rhos.append(rho)
    struct_mean = float(np.mean(struct_rhos))
    control_rhos = {}
    control_by_time = {}
    for cname in control_model_names:
        if cname not in model_rdms:
            logger.warning(f"控制模型 {cname} 不在 model_rdms 中，跳过")
            continue
        c_vec = model_rdms[cname][triu_idx]
        c_rhos = []
        for t in test_idx:
            neural_vec = neural_rdms[t][triu_idx]
            rho, _ = spearmanr(neural_vec, c_vec)
            c_rhos.append(rho)
        control_rhos[cname] = float(np.mean(c_rhos))
        control_by_time[cname] = np.array(c_rhos, dtype=float)
    if control_rhos:
        control_mean = float(np.mean(list(control_rhos.values())))
        ssi = struct_mean - control_mean
    else:
        ssi = struct_mean
        logger.warning("没有控制模型，SSI 直接使用结构均值")
    save_dict = {
        'gen_mat': gen_mat,
        'ssi': ssi,
        'struct_rho': struct_mean,
        'struct_rhos': np.array(struct_rhos),
        'struct_rhos_by_time': np.array(struct_rhos),
        'train_indices': np.array(train_idx, dtype=int),
        'test_indices': np.array(test_idx, dtype=int),
    }
    if control_rhos:
        cnames = list(control_rhos.keys())
        cvals = np.array([control_rhos[name] for name in cnames])
        ctime_vals = np.array([control_by_time[name] for name in cnames])
        save_dict['control_names'] = np.array(cnames, dtype=object)
        save_dict['control_values'] = cvals
        save_dict['control_by_time'] = ctime_vals
    else:
        save_dict['control_names'] = np.array([], dtype=object)
        save_dict['control_values'] = np.array([])
        save_dict['control_by_time'] = np.array([])
    if times_ms is not None:
        save_dict['train_times_ms'] = np.asarray(times_ms, dtype=float)[train_idx]
        save_dict['test_times_ms'] = np.asarray(times_ms, dtype=float)[test_idx]
    np.savez_compressed(out_path, **save_dict)
    logger.info(f"  H3 结果已保存: {out_path} | SSI={ssi:.4f}")
    return _load_h3_result(out_path)


def _load_h3_result(file_path: Path) -> dict:
    data = np.load(file_path, allow_pickle=True)
    if 'control_names' in data and 'control_values' in data:
        control_rhos = dict(zip(data['control_names'], data['control_values']))
    else:
        control_rhos = {}
    control_by_time = {}
    if 'control_by_time' in data and 'control_names' in data and data['control_by_time'].size > 0:
        names = data['control_names']
        mat = np.asarray(data['control_by_time'], dtype=float)
        for i, nm in enumerate(names):
            control_by_time[str(nm)] = mat[i]
    train_times = np.asarray(data['train_times_ms'], dtype=float) if 'train_times_ms' in data else np.array([])
    test_times = np.asarray(data['test_times_ms'], dtype=float) if 'test_times_ms' in data else np.array([])
    struct_by_time = (data['struct_rhos_by_time'] if 'struct_rhos_by_time' in data
                      else data['struct_rhos'] if 'struct_rhos' in data else np.array([]))
    return {
        'gen_mat': data['gen_mat'],
        'ssi': float(data['ssi']),
        'struct_rho': float(data['struct_rho']),
        'struct_rhos': np.asarray(struct_by_time, dtype=float),
        'struct_rhos_by_time': np.asarray(struct_by_time, dtype=float),
        'control_rhos': control_rhos,
        'control_by_time': control_by_time,
        'train_indices': data['train_indices'].tolist() if 'train_indices' in data else [],
        'test_indices': data['test_indices'].tolist() if 'test_indices' in data else [],
        'train_times_ms': train_times,
        'test_times_ms': test_times,
    }


def load_h3_results(subject_id: str, result_root: str, suffix: str = '') -> dict:
    path = Path(result_root) / f'sub-{subject_id}' / f'sub-{subject_id}_h3_rhos{suffix}.npz'
    if not path.exists():
        raise FileNotFoundError(f"未找到 H3 结果文件: {path}")
    return _load_h3_result(path)


def run_group_level(
    subject_ids: List[str],
    result_root: str = 'results',
    suffix: str = '',
    ssi_threshold: float = 0.0,
    save_fig_dir: str = 'results/figures/group',
) -> Dict:
    ssi_list = []
    struct_list = []
    control_dict = {}
    valid_subjects = []

    for sid in subject_ids:
        fpath = Path(result_root) / f'sub-{sid}' / f'sub-{sid}_h3_rhos{suffix}.npz'
        if not fpath.exists():
            logger.warning(f"被试 {sid} 的 H3 结果不存在，跳过")
            continue

        data = _load_h3_result(fpath)
        ssi_list.append(data['ssi'])
        struct_list.append(data['struct_rho'])
        valid_subjects.append(sid)

        for name, val in data['control_rhos'].items():
            if name not in control_dict:
                control_dict[name] = []
            control_dict[name].append(val)

    if len(ssi_list) < 3:
        raise ValueError(f"有效被试不足（仅 {len(ssi_list)} 个），无法进行统计检验")

    ssi_arr = np.asarray(ssi_list, dtype=float)
    struct_arr = np.asarray(struct_list, dtype=float)
    n_sub = len(ssi_arr)
    ssi_valid = ssi_arr[np.isfinite(ssi_arr)]
    n_ssi_valid = len(ssi_valid)
    if n_ssi_valid >= 3:
        t_stat, p_two = ttest_1samp(ssi_valid, popmean=ssi_threshold)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        mean_ssi = float(np.mean(ssi_valid))
        std_ssi = float(np.std(ssi_valid, ddof=1))
        sem_ssi = float(std_ssi / np.sqrt(n_ssi_valid))
    else:
        t_stat, p_two, p_one = np.nan, np.nan, np.nan
        mean_ssi, std_ssi, sem_ssi = np.nan, np.nan, np.nan

    struct_valid = struct_arr[np.isfinite(struct_arr)]
    n_struct_valid = len(struct_valid)
    if n_struct_valid >= 3:
        t_struct, p_struct_two = ttest_1samp(struct_valid, popmean=0)
        p_struct_one = p_struct_two / 2 if t_struct > 0 else 1 - p_struct_two / 2
        struct_mean = float(np.mean(struct_valid))
        struct_sem = float(np.std(struct_valid, ddof=1) / np.sqrt(n_struct_valid))
    else:
        t_struct, p_struct_two, p_struct_one = np.nan, np.nan, np.nan
        struct_mean, struct_sem = np.nan, np.nan

    stats = {
        'n_subjects': n_sub,
        'n_ssi_valid': n_ssi_valid,
        'valid_subjects': valid_subjects,
        'mean_ssi': mean_ssi,
        'std_ssi': std_ssi,
        'sem_ssi': sem_ssi,
        't_stat': float(t_stat) if np.isfinite(t_stat) else np.nan,
        'p_two': float(p_two) if np.isfinite(p_two) else np.nan,
        'p_one': float(p_one) if np.isfinite(p_one) else np.nan,
        't_ssi': float(t_stat) if np.isfinite(t_stat) else np.nan,
        'p_ssi': float(p_one) if np.isfinite(p_one) else np.nan,
        't_struct': float(t_struct) if np.isfinite(t_struct) else np.nan,
        'p_struct': float(p_struct_one) if np.isfinite(p_struct_one) else np.nan,
        'ssi_list': ssi_arr,
        'struct_mean': struct_mean,
        'struct_sem': struct_sem,
        'control_models': {},

    }
    logger.info("=" * 60)
    logger.info(f"H3 群体统计 (N={n_sub}, SSI有效={n_ssi_valid})")
    logger.info(
        f"SSI: mean={mean_ssi:.4f} ± SEM={sem_ssi:.4f}, "
        f"t={t_stat:.3f}, p_one={p_one:.4f}"
    )
    logger.info(
        f"结构模型平均拟合: {struct_mean:.4f} ± SEM={struct_sem:.4f}, "
        f"t={t_struct:.3f}, p_one={p_struct_one:.4f}"
    )

    for name, vals in control_dict.items():
        vals_arr = np.asarray(vals, dtype=float)
        pair_mask = np.isfinite(struct_arr) & np.isfinite(vals_arr)
        struct_pair = struct_arr[pair_mask]
        vals_pair = vals_arr[pair_mask]

        if len(struct_pair) >= 3:
            t_rel, p_rel_two = ttest_rel(struct_pair, vals_pair)
            p_rel_one = p_rel_two / 2 if t_rel > 0 else 1 - p_rel_two / 2
            mean_c = float(np.mean(vals_pair))
            sem_c = float(np.std(vals_pair, ddof=1) / np.sqrt(len(vals_pair)))
        else:
            t_rel, p_rel_two, p_rel_one = np.nan, np.nan, np.nan
            mean_c, sem_c = np.nan, np.nan

        stats['control_models'][name] = {
            'n': int(len(vals_pair)),
            'mean': mean_c,
            'sem': sem_c,
            't_rel': float(t_rel) if np.isfinite(t_rel) else np.nan,
            'p_rel_one': float(p_rel_one) if np.isfinite(p_rel_one) else np.nan,
            'p_rel_two': float(p_rel_two) if np.isfinite(p_rel_two) else np.nan,
        }
        logger.info(
            f"  vs {name}: mean={mean_c:.4f} ± SEM={sem_c:.4f}, "
            f"t={t_rel:.3f}, df={len(struct_pair) - 1}, p_one={p_rel_one:.4f}"
        )

    logger.info("=" * 60)
    behavior_df = get_behavior(subject_ids)
    run_h3_figures(
        subject_ids=subject_ids,
        result_root=result_root,
        suffix=suffix,
        behavior_df=behavior_df,
        rt_col='residual_rt',
        acc_col='accuracy',
        output_dir=save_fig_dir,
    )
    _save_group_report(stats, result_root, suffix)

    return stats


def get_behavior(subject_ids) -> Optional[pd.DataFrame]:

    metadata_root = str(CONFIG.get('paths', {}).get('deriv_root', './data/derivatives'))
    root = Path(metadata_root)
    root.mkdir(parents=True, exist_ok=True)

    cache_file = root / 'behavior_subject.csv'
    if cache_file.exists():
        logger.info(f"行为表缓存已存在，加载: {cache_file}")
        return pd.read_csv(cache_file)

    RESP_ROLES = {'remembered_correct', 'remembered_incorrect',
                  'ignored_correct', 'ignored_incorrect'}
    CORRECT_ROLES = {'remembered_correct', 'ignored_correct'}
    bids_root = Path(CONFIG.get('paths', {}).get('bids_root', 'data/raw'))
    trial_rows = []
    for sid in subject_ids:
        ev_files = sorted(bids_root.glob(f'sub-{sid}/**/*_events.tsv'))
        if not ev_files:
            logger.warning(f"被试 {sid} 无 events.tsv，跳过")
            continue
        frames = []
        for f in ev_files:
            run_id = int(f.stem.split('_run-')[1].split('_')[0]) if '_run-' in f.stem else 0
            try:
                df = pd.read_csv(f, sep='\t')
            except Exception as e:
                logger.warning(f"读取 {f} 失败: {e}")
                continue
            if df.empty:
                continue
            df['run'] = run_id
            frames.append(df)
        if not frames:
            continue
        raw = pd.concat(frames, ignore_index=True)
        logger.info(f"被试 {sid}: {len(frames)} 个 run, {len(raw)} 行事件")
        for (run_id, tid), g in raw.groupby(['run', 'trial']):
            if (g['task_role'].astype(str) == 'bad_trial').any():
                continue
            probe = g[g['task_role'].astype(str).str.startswith('probe')]
            resp = g[g['task_role'].astype(str).isin(RESP_ROLES)]  # 排除 indicate_ready
            if len(probe) == 0 or len(resp) == 0:
                continue
            role = str(resp['task_role'].iloc[0])
            rt_s = float(resp['onset'].iloc[0]) - float(probe['onset'].iloc[0])
            load = pd.to_numeric(g['memory_cond'], errors='coerce').dropna()
            trial_rows.append({
                'subject': str(sid),
                'rt_ms': rt_s * 1000,
                'correct': 1.0 if role in CORRECT_ROLES else 0.0,
                'load': float(load.iloc[0]) if len(load) else np.nan,
            })

    tr = pd.DataFrame(trial_rows)
    if len(tr) < 10:
        logger.warning("有效 trial 过少，返回 None")
        return None
    logger.info(f"有效 trial: {len(tr)}（{tr['subject'].nunique()} 个被试）")

    rows = []
    for sid, g in tr.groupby('subject'):
        x, y = g['load'].values, g['rt_ms'].values
        valid = np.isfinite(x) & np.isfinite(y)
        residual_rt = float(np.mean(y))
        if valid.sum() >= 3 and np.std(x[valid]) > 0:
            beta1 = np.cov(x[valid], y[valid], ddof=1)[0, 1] / np.var(x[valid], ddof=1)
            residual_rt = float(np.mean(y[valid]) - beta1 * np.mean(x[valid]))
        rows.append({'subject': str(sid),
                     'rt': float(np.mean(y)),
                     'accuracy': float(np.mean(g['correct'])),
                     'residual_rt': residual_rt})
    behavior_df = pd.DataFrame(rows)
    behavior_df['subject'] = behavior_df['subject'].map(_norm_subject)
    behavior_df.to_csv(cache_file, index=False)
    logger.info(f"行为表已保存: {cache_file} (N={len(behavior_df)})")
    return behavior_df

def _norm_subject(x) -> str:
    return str(x).replace('sub-', '').zfill(3)

def _save_group_report(stats: dict, result_root: str, suffix: str) -> str:
    report_path = Path(result_root) / f'group_h3_results{suffix}.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    n_sub = stats.get('n_subjects', 0)
    n_ssi_valid = stats.get('n_ssi_valid', n_sub)
    n_struct_valid = stats.get('n_struct_valid', n_sub)

    lines = []
    lines.append("=" * 60)
    lines.append("========== H3 群体水平结果 ==========")
    lines.append(f"有效被试数: {n_sub} (SSI有效: {n_ssi_valid})")
    lines.append("")
    lines.append("--- SSI 单样本单侧t检验 (检验 > 0) ---")
    lines.append(
        f"SSI: mean={stats.get('mean_ssi', np.nan):.4f} ± SEM={stats.get('sem_ssi', np.nan):.4f}, "
        f"std={stats.get('std_ssi', np.nan):.4f}"
    )
    lines.append(
        f"  t={stats.get('t_ssi', np.nan):.3f}, df={n_ssi_valid - 1}, "
        f"p_one={stats.get('p_ssi', np.nan):.4f}"
    )
    lines.append("")
    lines.append("--- 结构模型平均拟合 单样本t检验 (检验 > 0) ---")
    lines.append(
        f"Structure rho: mean={stats.get('struct_mean', np.nan):.4f} ± SEM={stats.get('struct_sem', np.nan):.4f}"
    )
    lines.append(
        f"  t={stats.get('t_struct', np.nan):.3f}, df={n_struct_valid - 1}, "
        f"p_one={stats.get('p_struct', np.nan):.4f}"
    )
    lines.append("")
    lines.append("--- 结构模型 vs 各控制模型 (配对t检验, 单侧: 结构>控制) ---")
    for name, info in stats.get('control_models', {}).items():
        n_pair = info.get('n', n_sub)
        lines.append(f"vs {name} (N={n_pair}):")
        lines.append(
            f"  控制模型 mean={info.get('mean', np.nan):.4f} ± SEM={info.get('sem', np.nan):.4f}"
        )
        lines.append(
            f"  t={info.get('t_rel', np.nan):.3f}, df={n_pair - 1}, "
            f"p_one={info.get('p_rel_one', np.nan):.4f}"
        )
    lines.append("")
    lines.append("有效被试列表: " + ", ".join(map(str, stats.get('valid_subjects', []))))
    lines.append("=" * 60)
    lines.append("")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    logger.info(f"H3 群体报告已保存: {report_path}")
    return str(report_path)



def run_h3b_behavior_correlation(
        subject_ids: List[str],
        behavior_df: pd.DataFrame,
        result_root: str = 'results',
        suffix: str = '',
        rt_col: str = 'residual_rt',
        acc_col: str = 'accuracy',
) -> Dict:
    ssi_vals = []
    rt_vals = []
    acc_vals = []
    valid_sids = []

    for sid in subject_ids:
        try:
            res = load_h3_results(sid, result_root, suffix=suffix)
        except FileNotFoundError:
            continue
        beh_row = behavior_df[behavior_df['subject'].astype(str) == str(sid)]
        if len(beh_row) == 0:
            continue
        ssi_vals.append(res['ssi'])
        rt_vals.append(float(beh_row.iloc[0][rt_col]))
        acc_vals.append(float(beh_row.iloc[0][acc_col]))
        valid_sids.append(sid)

    if len(ssi_vals) < 3:
        raise ValueError("有效样本不足，无法进行行为关联分析")
    ssi_arr = np.array(ssi_vals, dtype=float)
    rt_arr = np.array(rt_vals, dtype=float)
    acc_arr = np.array(acc_vals, dtype=float)
    r_rt, p_rt_two = spearmanr(ssi_arr, rt_arr)
    p_rt_one = p_rt_two / 2 if r_rt < 0 else 1
    r_acc, p_acc_two = spearmanr(ssi_arr, acc_arr)
    p_acc_one = p_acc_two / 2 if r_acc > 0 else 1 - p_acc_two / 2

    results = {
        'n': len(valid_sids),
        'valid_subjects': valid_sids,
        'rt': {
            'r': float(r_rt),
            'p_one': float(p_rt_one),
            'p_two': float(p_rt_two), },
        'acc': {
            'r': float(r_acc),
            'p_one': float(p_acc_one),
            'p_two': float(p_acc_two), }, }
    logger.info(f"H3b 行为关联 (N={len(valid_sids)})")
    logger.info(f"SSI vs {rt_col}: r={r_rt:.3f}, p_one={p_rt_one:.4f}")
    logger.info(f"SSI vs {acc_col}: r={r_acc:.3f}, p_one={p_acc_one:.4f}")
    return results


def time_window_to_indices(
        times_ms: np.ndarray,
        t_start: float,
        t_end: float,
        valid_indices: set = None
) -> list:
    times_ms = np.asarray(times_ms, dtype=float)
    mask = (times_ms >= t_start) & (times_ms <= t_end)
    indices = np.where(mask)[0].tolist()
    if valid_indices is not None:
        indices = [i for i in indices if i in valid_indices]
    return indices

def run_h3_figures(
    subject_ids: List[str],
    result_root: str = 'results',
    suffix: str = '',
    behavior_df=None,
    rt_col: str = 'residual_rt',
    acc_col: str = 'accuracy',
    output_dir: str = 'figures/paper',
):
    gen_mats, struct_by_time, ssi_list, valid_sids = [], [], [], []
    control_by_time: Dict[str, List[np.ndarray]] = {}
    train_times, test_times = None, None

    for sid in subject_ids:
        try:
            res = load_h3_results(sid, result_root, suffix=suffix)
        except FileNotFoundError:
            logger.warning(f"被试 {sid} H3结果缺失，跳过")
            continue

        gen_mats.append(res['gen_mat'])
        struct_by_time.append(res['struct_rhos_by_time'])
        ssi_list.append(res['ssi'])
        valid_sids.append(sid)

        for m, v in res['control_by_time'].items():
            control_by_time.setdefault(m, []).append(v)

        if train_times is None and len(res['train_times_ms']):
            train_times = res['train_times_ms']
        if test_times is None and len(res['test_times_ms']):
            test_times = res['test_times_ms']

    if len(gen_mats) < 3:
        logger.error("有效H3结果不足3个，无法生成群体图")
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    if train_times is not None and test_times is not None:
        h3_group_generalization_heatmap(
            suffix, gen_mats, train_times, test_times, output_dir=output_dir)
    else:
        logger.warning("缺少 train/test_times_ms，跳过")

    if struct_by_time and control_by_time and test_times is not None:
        h3_generalization_diff(
            suffix, struct_by_time, control_by_time, test_times, output_dir=output_dir)
    else:
        logger.warning("缺少 control_by_time，跳过 ")

    if behavior_df is not None:
        ssi_arr = np.asarray(ssi_list, dtype=float)
        rt_vals = []
        for s in valid_sids:
            sub = behavior_df[behavior_df['subject'].map(_norm_subject) == _norm_subject(s)]
            if len(sub) == 0:
                rt_vals.append(np.nan)
            else:
                v = sub.iloc[0][rt_col]
                rt_vals.append(np.nan if pd.isna(v) else float(v))
        rt_arr = np.asarray(rt_vals, dtype=float)
        h3_behavior_scatter(suffix, ssi_arr, rt_arr, output_dir=output_dir)
    else:
        logger.info("未提供 behavior_df，跳过")