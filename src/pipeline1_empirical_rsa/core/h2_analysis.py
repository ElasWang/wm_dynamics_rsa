import gc
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from config import CONFIG

from src.shared_utils.models.resolve_model_instances import (
    resolve_condition_model_instances,
)

from .sensor_rsa_engine import (
    extract_upper_triangular,
    load_all_neural_rdms,
)
from ..visualization.h2_visualization import h2_beta_structure_bar, h2_delta_r2_bar

from ...shared_utils.plotting import set_publication_style

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parents[3]
os.chdir(project_root)


def run_subject_level(
    suffix,
    subject_id: str,
    neural_rdms_dict: dict,
    model_rdms_dict: dict,
    time_indices: list,
    result_root: str = 'results',
    n_folds: int = 5,
    structure_model_name: str = 'target_priority_model',
    overwrite: bool = False,
) -> dict:
    cache_dir = Path(result_root) / f'sub-{subject_id}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f'sub-{subject_id}_h2_beta{suffix}.npz'
    if out_path.exists() and not overwrite:
        logger.info(f"  H2 缓存已存在: {out_path}")
        data = np.load(out_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    rdms = [neural_rdms_dict[t] for t in time_indices if t in neural_rdms_dict]
    if not rdms:
        raise ValueError(f"被试 {subject_id} 在指定时间窗口没有可用 RDM")
    model_specs = {
        'M1': ['content_model_content_lambda_param_1'],
        'M2': ['content_model_content_lambda_param_1', 'visual_model'],
        'M3': [
            'content_model_content_lambda_param_1',
            'visual_model',
            'binary_color_model',
        ],
        'M4': [
            'content_model_content_lambda_param_1',
            'visual_model',
            'binary_color_model',
            structure_model_name,
        ],
    }
    base_model_names = [
        'content_model_content_lambda_param_1',
        'visual_model',
        'binary_color_model',
        structure_model_name,
    ]
    neural_rdm = np.mean(rdms, axis=0)
    y_vec = extract_upper_triangular(neural_rdm)
    model_vecs = {}
    for name in base_model_names:
        m_rdm = _get_h2_model_rdm(model_rdms_dict, name)
        model_vecs[name] = extract_upper_triangular(m_rdm)

    X_list = [model_vecs[name] for name in base_model_names]
    X = np.column_stack(X_list)
    y = y_vec
    groups = None

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_r2 = {'M1': [], 'M2': [], 'M3': [], 'M4': []}
    fold_beta_m4_struct = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        X_train_all = X[train_idx]
        X_test_all = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        y_scaler = StandardScaler()
        y_train_z = y_scaler.fit_transform(y_train.reshape(-1,1)).ravel()
        y_test_z = y_scaler.transform(y_test.reshape(-1,1)).ravel()

        for m_key in ['M1','M2','M3','M4']:
            pred_names = model_specs[m_key]
            pred_idx = [base_model_names.index(n) for n in pred_names]
            X_train_raw = X_train_all[:, pred_idx]
            X_test_raw = X_test_all[:, pred_idx]

            x_scaler = StandardScaler()
            X_train_z = x_scaler.fit_transform(X_train_raw)
            X_test_z = x_scaler.transform(X_test_raw)

            if groups is not None:
                reg = _fit_ridge_group_cv(X_train_z, y_train_z, groups[train_idx])
            else:
                gs = GridSearchCV(
                    estimator=Ridge(),
                    param_grid={"alpha":[0.01,0.1,1.0,10.0,100.0]},
                    scoring="r2",
                    cv=KFold(n_splits=4, shuffle=True, random_state=42),
                    n_jobs=-1, refit=True
                )
                gs.fit(X_train_z, y_train_z)
                reg = gs.best_estimator_

            y_pred = reg.predict(X_test_z)
            r2_val = r2_score(y_test_z, y_pred)
            fold_r2[m_key].append(r2_val)
            if m_key == "M4" and structure_model_name in pred_names:
                loc_idx = pred_names.index(structure_model_name)
                fold_beta_m4_struct.append(reg.coef_[loc_idx])

        delta12 = fold_r2['M2'][-1] - fold_r2['M1'][-1]
        delta23 = fold_r2['M3'][-1] - fold_r2['M2'][-1]
        delta34 = fold_r2['M4'][-1] - fold_r2['M3'][-1]
        logger.info(
            f"    Fold {fold}: M1={fold_r2['M1'][-1]:.4f}, M2={fold_r2['M2'][-1]:.4f}, "
            f"M3={fold_r2['M3'][-1]:.4f}, M4={fold_r2['M4'][-1]:.4f} | "
            f"ΔVisual={delta12:.4f}, ΔTask={delta23:.4f}, ΔStructure={delta34:.4f}"
        )
    r2_m1 = float(np.mean(fold_r2['M1']))
    r2_m2 = float(np.mean(fold_r2['M2']))
    r2_m3 = float(np.mean(fold_r2['M3']))
    r2_m4 = float(np.mean(fold_r2['M4']))

    delta_r2_visual = r2_m2 - r2_m1
    delta_r2_task = r2_m3 - r2_m2
    delta_r2_structure = r2_m4 - r2_m3

    valid_beta = np.array(fold_beta_m4_struct, dtype=float)
    valid_beta = valid_beta[np.isfinite(valid_beta)]
    beta_struct = float(np.median(valid_beta)) if len(valid_beta)>0 else np.nan
    save_dict = {
        'analysis_type': 'condition_level',
        'n_folds': n_folds,
        'r2_m1': r2_m1,
        'r2_m2': r2_m2,
        'r2_m3': r2_m3,
        'r2_m4': r2_m4,
        'delta_r2_visual': delta_r2_visual,
        'delta_r2_task': delta_r2_task,
        'delta_r2_structure': delta_r2_structure,
        'delta_r2': delta_r2_structure,
        'r2_full': r2_m4,
        'r2_reduced': r2_m3,
        'beta_struct': beta_struct,
        'betas_all': np.asarray([beta_struct], dtype=float),
        'model_names': np.array([structure_model_name], dtype=object),
        'fold_r2_m1': np.array(fold_r2['M1']),
        'fold_r2_m2': np.array(fold_r2['M2']),
        'fold_r2_m3': np.array(fold_r2['M3']),
        'fold_r2_m4': np.array(fold_r2['M4']),
        'fold_delta_r2_visual': np.array(fold_r2['M2']) - np.array(fold_r2['M1']),
        'fold_delta_r2_task': np.array(fold_r2['M3']) - np.array(fold_r2['M2']),
        'fold_delta_r2_structure': np.array(fold_r2['M4']) - np.array(fold_r2['M3']),
        'fold_beta_struct': np.array(fold_beta_m4_struct),
    }
    np.savez_compressed(out_path, **save_dict)
    logger.info(f"  H2 个体结果已保存: {out_path}")
    logger.info(f"    M1={r2_m1:.6f}, M2={r2_m2:.6f}, M3={r2_m3:.6f}, M4={r2_m4:.6f}")
    logger.info(f"    ΔVisual={delta_r2_visual:.6f}, ΔTask={delta_r2_task:.6f}, ΔStructure={delta_r2_structure:.6f}, beta={beta_struct:.6f}")
    return save_dict


def run_group_level(
    subject_ids: list,
    result_root: str = 'results',
    suffix: str = '_cv',
    structure_model_name: str = 'target_priority_model',
    save_fig_dir: str = 'results/figures/group',
) -> dict:
    all_m1 = []
    all_m2 = []
    all_m3 = []
    all_m4 = []
    all_delta_visual = []
    all_delta_task = []
    all_delta_structure = []
    all_betas = []
    valid_subjects = []

    for sid in subject_ids:
        fpath = (Path(result_root) / f'sub-{sid}' / f'sub-{sid}_h2_beta{suffix}.npz')
        if not fpath.exists():
            logger.warning(f"被试 {sid} 的H2结果不存在，跳过")
            continue
        data = np.load(fpath, allow_pickle=True)

        m1 = float(np.asarray(data['r2_m1']).squeeze())
        m2 = float(np.asarray(data['r2_m2']).squeeze())
        m3 = float(np.asarray(data['r2_m3']).squeeze())
        m4 = float(np.asarray(data['r2_m4']).squeeze())
        delta_visual = float(np.asarray(data['delta_r2_visual']).squeeze())
        delta_task = float(np.asarray(data['delta_r2_task']).squeeze())
        delta_structure = float(np.asarray(data['delta_r2_structure']).squeeze())
        beta = float(np.asarray(data['beta_struct']).squeeze())

        values = [m1, m2, m3, m4, delta_visual, delta_task, delta_structure, beta]
        if not all(np.isfinite(v) for v in values):
            logger.warning(f"被试 {sid} H2含NaN/Inf，跳过")
            continue

        all_m1.append(m1)
        all_m2.append(m2)
        all_m3.append(m3)
        all_m4.append(m4)
        all_delta_visual.append(delta_visual)
        all_delta_task.append(delta_task)
        all_delta_structure.append(delta_structure)
        all_betas.append(beta)
        valid_subjects.append(sid)

    n_sub = len(valid_subjects)
    if n_sub < 3:
        logger.error(f"有效被试数不足3: {n_sub}")
        return {}

    m1 = np.asarray(all_m1, dtype=float)
    m2 = np.asarray(all_m2, dtype=float)
    m3 = np.asarray(all_m3, dtype=float)
    m4 = np.asarray(all_m4, dtype=float)
    delta_visual = np.asarray(all_delta_visual, dtype=float)
    delta_task = np.asarray(all_delta_task, dtype=float)
    delta_structure = np.asarray(all_delta_structure, dtype=float)
    betas = np.asarray(all_betas, dtype=float)

    t_visual, p_visual = ttest_1samp(delta_visual, 0, alternative='greater')
    t_task, p_task = ttest_1samp(delta_task, 0, alternative='greater')
    t_structure, p_structure = ttest_1samp(delta_structure, 0, alternative='greater')
    t_beta, p_beta = ttest_1samp(betas, 0, alternative='greater')

    def _mean_sem(x):
        return (float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x))),)

    mean_m1, sem_m1 = _mean_sem(m1)
    mean_m2, sem_m2 = _mean_sem(m2)
    mean_m3, sem_m3 = _mean_sem(m3)
    mean_m4, sem_m4 = _mean_sem(m4)
    mean_visual, sem_visual = _mean_sem(delta_visual)
    mean_task, sem_task = _mean_sem(delta_task)
    mean_structure, sem_structure = _mean_sem(delta_structure)
    mean_beta, sem_beta = _mean_sem(betas)

    logger.info("=" * 60)
    logger.info(f"H2 群体统计结果 (结构模型: {structure_model_name}, suffix={suffix})")
    logger.info(f"有效被试数: {n_sub}")
    logger.info(f"M1 Content: R²={mean_m1:.4f} ± {sem_m1:.4f}")
    logger.info(f"M2 Content+Visual: R²={mean_m2:.4f} ± {sem_m2:.4f}")
    logger.info(
        f"ΔR² Visual: {mean_visual:.4f} ± {sem_visual:.4f}, "
        f"t={t_visual:.3f}, p={p_visual:.4f}"
    )
    logger.info(f"M3 Content+Visual+Task: R²={mean_m3:.4f} ± {sem_m3:.4f}")
    logger.info(
        f"ΔR² Task: {mean_task:.4f} ± {sem_task:.4f}, "
        f"t={t_task:.3f}, p={p_task:.4f}"
    )
    logger.info(f"M4 Full: R²={mean_m4:.4f} ± {sem_m4:.4f}")
    logger.info(
        f"ΔR² Structure: {mean_structure:.4f} ± {sem_structure:.4f}, "
        f"t={t_structure:.3f}, p={p_structure:.4f}"
    )
    logger.info(
        f"Structure β: {mean_beta:.4f} ± {sem_beta:.4f}, "
        f"t={t_beta:.3f}, p={p_beta:.4f}"
    )
    logger.info("=" * 60)

    report_path = (
        Path(result_root)
        / f'group_h2_results_{structure_model_name}{suffix}.txt'
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("========== H2 群体水平结果 ==========\n")
        f.write(f"结构模型: {structure_model_name}\n")
        f.write(f"suffix: {suffix}\n")
        f.write(f"有效被试数: {n_sub}\n\n")
        f.write(f"M1 Content: {mean_m1:.6f} ± {sem_m1:.6f}\n")
        f.write(f"M2 Content+Visual: {mean_m2:.6f} ± {sem_m2:.6f}\n")
        f.write(
            f"ΔR² Visual: {mean_visual:.6f} ± {sem_visual:.6f}, "
            f"t={t_visual:.3f}, p={p_visual:.6f}\n\n"
        )
        f.write(f"M3 Content+Visual+Task: {mean_m3:.6f} ± {sem_m3:.6f}\n")
        f.write(
            f"ΔR² Task: {mean_task:.6f} ± {sem_task:.6f}, "
            f"t={t_task:.3f}, p={p_task:.6f}\n\n"
        )
        f.write(f"M4 Full: {mean_m4:.6f} ± {sem_m4:.6f}\n")
        f.write(
            f"ΔR² Structure: {mean_structure:.6f} ± {sem_structure:.6f}, "
            f"t={t_structure:.3f}, p={p_structure:.6f}\n\n"
        )
        f.write(
            f"Structure β: {mean_beta:.6f} ± {sem_beta:.6f}, "
            f"t={t_beta:.3f}, p={p_beta:.6f}\n"
        )
        f.write("======================================\n")
    logger.info(f"群体报告已保存: {report_path}")

    results = {
        'n_sub': n_sub,
        'valid_subjects': valid_subjects,
        'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4,
        'mean_m1': mean_m1, 'sem_m1': sem_m1,
        'mean_m2': mean_m2, 'sem_m2': sem_m2,
        'mean_m3': mean_m3, 'sem_m3': sem_m3,
        'mean_m4': mean_m4, 'sem_m4': sem_m4,
        'delta_visual': delta_visual,
        'delta_task': delta_task,
        'delta_structure': delta_structure,
        'mean_delta_visual': mean_visual,
        'sem_delta_visual': sem_visual,
        'mean_delta_task': mean_task,
        'sem_delta_task': sem_task,
        'mean_delta_structure': mean_structure,
        'sem_delta_structure': sem_structure,
        't_visual': float(t_visual), 'p_visual': float(p_visual),
        't_task': float(t_task), 'p_task': float(p_task),
        't_structure': float(t_structure), 'p_structure': float(p_structure),
        'betas': betas,
        'beta_struct': mean_beta,
        'sem_beta': sem_beta,
        't_beta': float(t_beta),
        'p_beta': float(p_beta),
        'mean_delta': mean_structure,
        'sem_delta': sem_structure,
        't_delta': float(t_structure),
        'p_delta': float(p_structure),
        'r2_full': m4,
        'r2_reduced': m3,
    }
    try:
        set_publication_style()
        plot_results = {
            'beta_struct': mean_beta,
            'sem_beta': sem_beta,
            't_beta': float(t_beta),
            'p_beta': float(p_beta),
            'mean_delta': mean_structure,
            'sem_delta': sem_structure,
            't_delta': float(t_structure),
            'p_delta': float(p_structure),
            'mean_betas': [mean_beta],
            'sem_betas': [sem_beta],
            't_stats': [float(t_beta)],
            'p_values': [float(p_beta)],
            'model_names': [structure_model_name],
            'deltas': delta_structure,
            'r2_full': m4,
            'r2_reduced': m3,
        }
        h2_beta_structure_bar(suffix, plot_results, output_dir=save_fig_dir)
        h2_delta_r2_bar(suffix, plot_results, output_dir=save_fig_dir)
        logger.info(f"Figure 2 已保存至 {save_fig_dir}")
    except Exception as e:
        logger.warning(f"Figure 2 生成失败（不影响统计结果）: {e}")
    finally:
        plt.close('all')
        gc.collect()

    return results


def run_sensitivity_analysis(
    subject_ids: list,
    alternative_model: str,
    suffix: str,
    result_root: str = 'results',
    n_folds: int = 5,
    overwrite: bool = False,
) -> dict:
    for sid in subject_ids:
        try:
            rdms, times_ms, time_indices = load_all_neural_rdms(
                sid,
                cache_root=CONFIG.get('paths', {}).get('neural_rdm_root', 'results/rdms/neural'),
            )
        except FileNotFoundError as e:
            logger.warning(f"被试 {sid} 神经RDM缺失: {e}")
            continue
        mask = (times_ms >= 300) & (times_ms <= 500)
        idx_h2 = np.where(mask)[0].tolist()
        if not idx_h2:
            logger.warning(f"被试 {sid} 300‑500ms无有效时间点")
            continue
        try:
            model_rdms = resolve_condition_model_instances(sid, hypothesis='h2', force=overwrite)
        except Exception as e:
            logger.warning(f"被试 {sid} 理论模型RDM加载失败: {e}")
            continue
        run_subject_level(
            suffix=suffix,
            subject_id=sid,
            neural_rdms_dict=rdms,
            model_rdms_dict=model_rdms,
            time_indices=idx_h2,
            result_root=result_root,
            n_folds=n_folds,
            structure_model_name=alternative_model,
            overwrite=overwrite,
        )
    return run_group_level(
        subject_ids=subject_ids,
        result_root=result_root,
        suffix=suffix,
        structure_model_name=alternative_model,
    )


def _fit_ridge_group_cv(X, y, groups, n_inner_folds=4):
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(f"训练数据中 sequence 数量不足: {len(unique_groups)}")
    n_splits = min(n_inner_folds, len(unique_groups))
    inner_cv = GroupKFold(n_splits=n_splits)
    search = GridSearchCV(
        estimator=Ridge(),
        param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="r2",
        cv=inner_cv.split(X, y, groups),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_


def _get_h2_model_rdm(model_rdms_dict, model_name):
    if model_name not in model_rdms_dict:
        raise KeyError(f"缺少 H2 模型: {model_name}")
    model_rdm = np.asarray(model_rdms_dict[model_name], dtype=float)
    return model_rdm
