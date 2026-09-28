import gc
import logging
import os
from pathlib import Path
from statistics import LinearRegression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import CONFIG

from src.shared_utils.models.resolve_model_instances import (
    resolve_condition_model_instances, resolve_model_instances,
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
    overwrite: bool = False,
) -> dict:
    structure_model_names = ['target_priority_model','gradient_model']

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

    shared_models = [
        'content_model_content_lambda_param_1',
        'visual_model',
        'binary_color_model',
    ]
    valid_structure_models = [m for m in structure_model_names if m in model_rdms_dict]
    missing = [m for m in structure_model_names if m not in model_rdms_dict]
    if missing:
        logger.warning(f"被试 {subject_id} 缺少结构模型: {missing}")
    if not valid_structure_models:
        raise ValueError(f"被试 {subject_id} 没有任何可用结构模型")

    neural_rdm = np.mean(rdms, axis=0)
    y = extract_upper_triangular(neural_rdm)

    shared_vecs = {
        name: extract_upper_triangular(_get_h2_model_rdm(model_rdms_dict, name))
        for name in shared_models
    }
    struct_vecs = {
        name: extract_upper_triangular(_get_h2_model_rdm(model_rdms_dict, name))
        for name in valid_structure_models
    }

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_r2_m1, fold_r2_m2, fold_r2_m3 = [], [], []
    fold_r2_m4_by_model = {m: [] for m in valid_structure_models}
    fold_beta_by_model = {m: [] for m in valid_structure_models}

    def _fit_ridge(X_train_z, y_train_z):
        gs = GridSearchCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            scoring="r2",
            cv=KFold(n_splits=4, shuffle=True, random_state=42),
            n_jobs=-1, refit=True,
        )
        gs.fit(X_train_z, y_train_z)
        return gs.best_estimator_

    for fold, (train_idx, test_idx) in enumerate(cv.split(y)):
        y_train, y_test = y[train_idx], y[test_idx]
        y_scaler = StandardScaler()
        y_train_z = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_z = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

        def _eval_model(feature_vecs):
            X_tr = np.column_stack([vecs[train_idx] for vecs in feature_vecs])
            X_te = np.column_stack([vecs[test_idx] for vecs in feature_vecs])
            x_scaler = StandardScaler()
            X_tr_z = x_scaler.fit_transform(X_tr)
            X_te_z = x_scaler.transform(X_te)
            reg = _fit_ridge(X_tr_z, y_train_z)
            return r2_score(y_test_z, reg.predict(X_te_z)), reg

        r2, _ = _eval_model([shared_vecs[m] for m in shared_models[:1]])
        fold_r2_m1.append(r2)
        r2, _ = _eval_model([shared_vecs[m] for m in shared_models[:2]])
        fold_r2_m2.append(r2)
        r2, _ = _eval_model([shared_vecs[m] for m in shared_models[:3]])
        fold_r2_m3.append(r2)

        for m_name in valid_structure_models:
            vecs_list = [shared_vecs[m] for m in shared_models[:3]] + [struct_vecs[m_name]]
            r2, reg = _eval_model(vecs_list)
            fold_r2_m4_by_model[m_name].append(r2)
            fold_beta_by_model[m_name].append(reg.coef_[-1])

    r2_m1 = float(np.mean(fold_r2_m1))
    r2_m2 = float(np.mean(fold_r2_m2))
    r2_m3 = float(np.mean(fold_r2_m3))

    save_dict = {
        'analysis_type': 'condition_level',
        'n_folds': n_folds,
        'r2_m1': r2_m1,
        'r2_m2': r2_m2,
        'r2_m3': r2_m3,
        'delta_r2_visual': r2_m2 - r2_m1,
        'delta_r2_task': r2_m3 - r2_m2,
        'fold_r2_m1': np.array(fold_r2_m1),
        'fold_r2_m2': np.array(fold_r2_m2),
        'fold_r2_m3': np.array(fold_r2_m3),
        'structure_model_names': np.array(valid_structure_models, dtype=object),
    }

    for m_name in valid_structure_models:
        fold_r2_m4 = fold_r2_m4_by_model[m_name]
        fold_beta = fold_beta_by_model[m_name]
        r2_m4 = float(np.mean(fold_r2_m4))
        delta_r2_structure = r2_m4 - r2_m3

        valid_beta = np.array(fold_beta, dtype=float)
        valid_beta = valid_beta[np.isfinite(valid_beta)]
        beta_struct = float(np.median(valid_beta)) if len(valid_beta) > 0 else np.nan

        save_dict[f'r2_m4__{m_name}'] = r2_m4
        save_dict[f'delta_r2_structure__{m_name}'] = delta_r2_structure
        save_dict[f'beta_struct__{m_name}'] = beta_struct
        save_dict[f'fold_r2_m4__{m_name}'] = np.array(fold_r2_m4)
        save_dict[f'fold_beta_struct__{m_name}'] = np.array(fold_beta)

    np.savez_compressed(out_path, **save_dict)
    logger.info(f"  H2 个体结果已保存: {out_path}")
    for m_name in valid_structure_models:
        logger.info(
            f"    {m_name}: M4={save_dict[f'r2_m4__{m_name}']:.4f}, "
            f"ΔStructure={save_dict[f'delta_r2_structure__{m_name}']:.4f}, "
            f"beta={save_dict[f'beta_struct__{m_name}']:.4f}"
        )
    return save_dict


def run_group_level(
    subject_ids: list,
    result_root: str = 'results',
    suffix: str = '_cv',
    save_fig_dir: str = 'results/figures/group',
    run_collinearity: bool = True,
) -> dict:

    structure_model_names = ['target_priority_model','gradient_model']

    loaded = {}
    for sid in subject_ids:
        fpath = Path(result_root) / f'sub-{sid}' / f'sub-{sid}_h2_beta{suffix}.npz'
        if not fpath.exists():
            logger.warning(f"被试 {sid} H2 结果不存在，跳过")
            continue
        loaded[sid] = dict(np.load(fpath, allow_pickle=True))

    if len(loaded) < 3:
        logger.error(f"有效被试数不足3: {len(loaded)}")
        return {}

    shared_arrays = {
        k: [] for k in ['r2_m1', 'r2_m2', 'r2_m3',
                        'delta_r2_visual', 'delta_r2_task']
    }
    valid_subjects = []
    for sid, data in loaded.items():
        try:
            for k in shared_arrays:
                shared_arrays[k].append(float(np.asarray(data[k]).squeeze()))
            valid_subjects.append(sid)
        except KeyError:
            logger.warning(f"被试 {sid} 缺少共享字段，跳过")
            continue

    n_sub = len(valid_subjects)
    if n_sub < 3:
        logger.error("有效被试不足")
        return {}
    shared = {k: np.asarray(v, dtype=float) for k, v in shared_arrays.items()}

    per_model = {}
    for m_name in structure_model_names:
        key_beta = f'beta_struct__{m_name}'
        key_delta = f'delta_r2_structure__{m_name}'

        beta_list, delta_list = [], []
        for sid in valid_subjects:
            data = loaded[sid]
            if key_beta not in data or key_delta not in data:
                continue
            b = float(np.asarray(data[key_beta]).squeeze())
            d = float(np.asarray(data[key_delta]).squeeze())
            if not (np.isfinite(b) and np.isfinite(d)):
                continue
            beta_list.append(b)
            delta_list.append(d)

        if len(beta_list) < 3:
            logger.warning(f"结构模型 {m_name} 有效被试不足，跳过")
            continue

        betas = np.asarray(beta_list, dtype=float)
        deltas = np.asarray(delta_list, dtype=float)
        t_beta, p_beta = ttest_1samp(betas, 0, alternative='greater')
        t_delta, p_delta = ttest_1samp(deltas, 0, alternative='greater')

        per_model[m_name] = {
            'model': m_name,
            'n_sub': len(betas),
            'beta_struct': float(np.mean(betas)),
            'sem_beta': float(np.std(betas, ddof=1) / np.sqrt(len(betas))),
            't_beta': float(t_beta),
            'p_beta': float(p_beta),
            'delta_structure': float(np.mean(deltas)),
            'sem_delta': float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))),
            't_delta': float(t_delta),
            'p_delta': float(p_delta),
            'betas': betas,
            'deltas': deltas,
        }
        logger.info(
            f"  {m_name}: beta={per_model[m_name]['beta_struct']:.4f} "
            f"(t={t_beta:.3f}, p={p_beta:.4f}), "
            f"ΔR²={per_model[m_name]['delta_structure']:.4f} "
            f"(t={t_delta:.3f}, p={p_delta:.4f})"
        )

    summary = summarize_multi_structure_results(
        per_model, shared, valid_subjects,
        result_root, suffix, save_fig_dir
    )

    collinearity = {}
    # if run_collinearity and per_model:
    #     logger.info("=" * 60)
    #     logger.info("运行 H2 共线性诊断")
    #     logger.info("=" * 60)
    #     collinearity = run_h2_collinearity_diagnostics(
    #         subject_ids=subject_ids,
    #         result_root=result_root,
    #         suffix=suffix,
    #         structure_model_name='target_priority_model',
    #         control_structure_model='gradient_model',
    #         output_dir=save_fig_dir,
    #     )

    result = {
        'n_sub': n_sub,
        'valid_subjects': valid_subjects,
        'shared': shared,
        'per_model': per_model,
        'summary': summary,
        'collinearity': collinearity,
    }
    main_model = structure_model_names[0]
    if main_model in per_model:
        res = per_model[main_model]
        result.update({
            'beta_struct': res['beta_struct'],
            'sem_beta': res['sem_beta'],
            't_beta': res['t_beta'],
            'p_beta': res['p_beta'],
            'delta_structure': res['delta_structure'],
            'sem_delta': res['sem_delta'],
            't_delta': res['t_delta'],
            'p_delta': res['p_delta'],
            'mean_delta_structure': res['delta_structure'],
            'sem_delta_structure': res['sem_delta'],
            't_structure': res['t_delta'],
            'p_structure': res['p_delta'],
        })

    return result

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
            if suffix == '_event':
                model_rdms = resolve_model_instances(sid, hypothesis='h2', force=False)
            else:
                model_rdms = resolve_condition_model_instances(sid, hypothesis='h2', force=False)
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
def summarize_multi_structure_results(
    per_model: dict,
    shared: dict,
    valid_subjects: list,
    result_root: str = 'results',
    suffix: str = '_cv',
    save_fig_dir: str = 'results/figures/group',
) -> dict:
    rows = []
    for m_name, res in per_model.items():
        rows.append({
            'model': m_name,
            'n_sub': res['n_sub'],
            'beta_struct': res['beta_struct'],
            'sem_beta': res['sem_beta'],
            't_beta': res['t_beta'],
            'p_beta': res['p_beta'],
            'delta_structure': res['delta_structure'],
            'sem_delta': res['sem_delta'],
            't_delta': res['t_delta'],
            'p_delta': res['p_delta'],
        })

    if not rows:
        logger.warning("无可汇总的结构模型结果")
        return {}

    df = pd.DataFrame(rows)
    all_sig_beta = bool((df['p_beta'] < 0.05).all())
    all_sig_delta = bool((df['p_delta'] < 0.05).all())

    csv_path = Path(result_root) / f'group_h2_multi_structure_summary{suffix}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"多结构模型汇总已保存: {csv_path}")

    logger.info("=" * 70)
    logger.info("多结构模型 H2 汇总")
    logger.info("\n" + df.to_string(index=False))
    logger.info(f"所有结构模型 β 显著: {all_sig_beta}")
    logger.info(f"所有结构模型 ΔR² 显著: {all_sig_delta}")

    os.makedirs(save_fig_dir, exist_ok=True)
    x = np.arange(len(df))
    labels = df['model'].tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, df['beta_struct'], yerr=df['sem_beta'],
           capsize=5, color='steelblue', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('β_structure')
    ax.set_title('β_structure across Structural Models')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row['beta_struct'] + row['sem_beta'] + 0.005,
                f"p={row['p_beta']:.3f}", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(Path(save_fig_dir) / f'h2_multi_structure_beta{suffix}.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, df['delta_structure'], yerr=df['sem_delta'],
           capsize=5, color='salmon', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('ΔR² (Structure)')
    ax.set_title('ΔR² across Structural Models')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row['delta_structure'] + row['sem_delta'] + 0.005,
                f"p={row['p_delta']:.3f}", ha='center', fontsize=9)
    plt.subplots_adjust(bottom=0.3, top=0.92, left=0.1, right=0.95)
    plt.savefig(Path(save_fig_dir) / f'h2_multi_structure_delta_r2{suffix}.png', dpi=300)
    plt.close()

    report_path = Path(result_root) / f'group_h2_multi_structure_report{suffix}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("多结构模型 H2 汇总报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"有效被试数: {len(valid_subjects)}\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write(f"所有结构模型 β_structure 显著: {all_sig_beta}\n")
        f.write(f"所有结构模型 ΔR² 显著: {all_sig_delta}\n\n")
        if all_sig_beta:
            f.write("结论: '存在结构信息'这一结论稳健，但不支持某一特定模型最优。\n")
        else:
            f.write("结论: 并非所有结构模型都显著，结论可能依赖特定模型，需谨慎。\n")
    logger.info(f"多结构模型报告已保存: {report_path}")

    return {
        'table': df,
        'all_sig_beta': all_sig_beta,
        'all_sig_delta': all_sig_delta,
    }

DISPLAY_LABELS = {
    'content_model_content_lambda_param_1': 'Content',
    'visual_model': 'Visual',
    'binary_color_model': 'Color',
    'target_priority_model': 'TPR',
    'gradient_model': 'Gradient',
    'binary_adjacency_model': 'Adjacency',
    'reciprocal_adjacency_model': 'Reciprocal',
    'serial_position_model': 'SerialPosition',
    'timestamp_model': 'Timestamp',
}

def _display(name):
    return DISPLAY_LABELS.get(name, name)

def diagnose_h2_internal_collinearity(
    subject_ids: list,
    result_root: str = 'results',
    suffix: str = '_cv',
    structure_model_name: str = 'target_priority_model',
    output_dir: str = 'results/figures/group',
) -> dict:
    base_model_names = [
        'content_model_content_lambda_param_1',
        'visual_model',
        'binary_color_model',
        structure_model_name,
    ]
    display_names = [_display(n) for n in base_model_names]

    corr_matrices, vif_values, condition_numbers, valid_subjects = [], [], [], []

    for sid in subject_ids:
        try:
            rdms, times_ms, _ = load_all_neural_rdms(
                sid,
                cache_root=CONFIG.get('paths', {}).get(
                    'neural_rdm_root', 'results/rdms/neural'),
            )
        except FileNotFoundError:
            continue
        mask = (times_ms >= 300) & (times_ms <= 500)
        idx = np.where(mask)[0].tolist()
        if not idx:
            continue
        neural_rdm = np.mean([rdms[t] for t in idx if t in rdms], axis=0)

        try:
            if suffix == '_event':
                model_rdms = resolve_model_instances(sid, hypothesis='h2', force=False)
            else:
                model_rdms = resolve_condition_model_instances(sid, hypothesis='h2', force=False)
        except Exception:
            continue
        if any(m not in model_rdms for m in base_model_names):
            continue

        triu_idx = np.triu_indices_from(neural_rdm, k=1)
        X = np.column_stack([model_rdms[n][triu_idx] for n in base_model_names])
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

        corr_matrices.append(np.corrcoef(X.T))
        vif_values.append([variance_inflation_factor(X, i) for i in range(X.shape[1])])
        condition_numbers.append(np.linalg.cond(X))
        valid_subjects.append(sid)

    if len(valid_subjects) < 3:
        logger.warning("有效被试不足，跳过 H2 内部共线性")
        return {}

    corr_mean = np.mean(corr_matrices, axis=0)
    vif_mean = np.mean(vif_values, axis=0)
    cond_mean = float(np.mean(condition_numbers))

    logger.info("=" * 60)
    logger.info("H2 设计矩阵内部共线性")
    logger.info(f"  条件数: {cond_mean:.2f}")
    for n, v in zip(display_names, vif_mean):
        logger.info(f"  {n}: VIF={v:.2f}")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_mean, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(display_names)))
    ax.set_yticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=30, ha='right')
    ax.set_yticklabels(display_names)
    for i in range(len(display_names)):
        for j in range(len(display_names)):
            ax.text(j, i, f'{corr_mean[i, j]:.2f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title('H2 Design Matrix Correlation')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'h2_internal_collinearity{suffix}.png', dpi=300)
    plt.close()

    return {
        'corr_mean': corr_mean,
        'vif_mean': vif_mean,
        'condition_number': cond_mean,
        'model_names': base_model_names,
        'display_names': display_names,
        'valid_subjects': valid_subjects,
    }

def diagnose_structure_model_collinearity(
    subject_ids: list,
    suffix: str = '_cv',
    output_dir: str = 'results/figures/group',
) -> dict:
    structure_models = [
        'target_priority_model',
        'gradient_model',
        'binary_adjacency_model',
        'reciprocal_adjacency_model',
        'serial_position_model',
    ]
    corr_matrices, valid_subjects = [], []

    for sid in subject_ids:
        try:
            if suffix == '_event':
                model_rdms = resolve_model_instances(sid, hypothesis='h2', force=False)
            else:
                model_rdms = resolve_condition_model_instances(sid, hypothesis='h2', force=False)
        except Exception:
            continue
        available = [m for m in structure_models if m in model_rdms]
        if len(available) < 2:
            continue

        first = model_rdms[available[0]]
        triu_idx = np.triu_indices_from(first, k=1)
        vecs = np.column_stack([model_rdms[m][triu_idx] for m in available])
        n = len(available)
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                corr[i, j] = spearmanr(vecs[:, i], vecs[:, j])[0]
        corr_matrices.append(corr)
        valid_subjects.append(sid)

    if len(valid_subjects) < 3:
        logger.warning("有效被试不足，跳过结构模型共线性")
        return {}

    corr_mean = np.mean(corr_matrices, axis=0)
    model_names = structure_models[:corr_mean.shape[0]]
    display_names = [_display(n) for n in model_names]

    logger.info("=" * 60)
    logger.info("候选结构模型之间的共线性")
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            logger.info(
                f"  {display_names[i]} vs {display_names[j]}: r={corr_mean[i, j]:.3f}"
            )
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_mean, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(display_names)))
    ax.set_yticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.set_yticklabels(display_names)
    for i in range(len(display_names)):
        for j in range(len(display_names)):
            ax.text(j, i, f'{corr_mean[i, j]:.2f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, label='Spearman r')
    ax.set_title('Structure Model RDM Correlation')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'h2_structure_model_collinearity{suffix}.png', dpi=300)
    plt.close()

    return {
        'corr_mean': corr_mean,
        'model_names': model_names,
        'display_names': display_names,
        'valid_subjects': valid_subjects,
    }

def orthogonalized_beta_stability(
    subject_ids: list,
    result_root: str = 'results',
    suffix: str = '_cv',
    structure_model_name: str = 'target_priority_model',
    control_structure_model: str = 'gradient_model',
    output_dir: str = 'results/figures/group',
) -> dict:
    base_model_names = [
        'content_model_content_lambda_param_1',
        'visual_model',
        'binary_color_model',
        structure_model_name,
    ]
    beta_orig_list, beta_orth_list, valid_subjects = [], [], []

    for sid in subject_ids:
        try:
            rdms, times_ms, _ = load_all_neural_rdms(
                sid,
                cache_root=CONFIG.get('paths', {}).get(
                    'neural_rdm_root', 'results/rdms/neural'),
            )
        except FileNotFoundError:
            continue
        mask = (times_ms >= 300) & (times_ms <= 500)
        idx = np.where(mask)[0].tolist()
        if not idx:
            continue
        neural_rdm = np.mean([rdms[t] for t in idx if t in rdms], axis=0)

        try:
            if suffix == '_event':
                model_rdms = resolve_model_instances(sid, hypothesis='h2', force=False)
            else:
                model_rdms = resolve_condition_model_instances(sid, hypothesis='h2', force=False)
        except Exception:
            continue
        if any(m not in model_rdms for m in base_model_names):
            continue
        if control_structure_model not in model_rdms:
            continue

        triu_idx = np.triu_indices_from(neural_rdm, k=1)
        y = neural_rdm[triu_idx]
        y_std = (y - y.mean()) / y.std(ddof=1)

        X = np.column_stack([model_rdms[n][triu_idx] for n in base_model_names])
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

        model_full = Ridge(alpha=1.0)
        model_full.fit(X, y_std)
        beta_orig = model_full.coef_[-1]

        tpr_vec = model_rdms[structure_model_name][triu_idx]
        ctrl_vec = model_rdms[control_structure_model][triu_idx]
        residualizer = LinearRegression()
        residualizer.fit(ctrl_vec.reshape(-1, 1), tpr_vec)
        tpr_orth = tpr_vec - residualizer.predict(ctrl_vec.reshape(-1, 1))

        X_orth = X.copy()
        X_orth[:, -1] = tpr_orth
        X_orth[:, -1] = (X_orth[:, -1] - X_orth[:, -1].mean()) / X_orth[:, -1].std(ddof=1)

        model_orth = Ridge(alpha=1.0)
        model_orth.fit(X_orth, y_std)
        beta_orth = model_orth.coef_[-1]

        beta_orig_list.append(beta_orig)
        beta_orth_list.append(beta_orth)
        valid_subjects.append(sid)

    if len(valid_subjects) < 3:
        logger.warning("有效被试不足，跳过正交化 β 稳定性")
        return {}

    beta_orig = np.asarray(beta_orig_list)
    beta_orth = np.asarray(beta_orth_list)
    t_o, p_o = ttest_1samp(beta_orig, 0, alternative='greater')
    t_h, p_h = ttest_1samp(beta_orth, 0, alternative='greater')

    logger.info("=" * 60)
    logger.info("正交化后 β 稳定性")
    logger.info(f"  正交化前 β ({structure_model_name}): "
                f"mean={beta_orig.mean():.4f}, t={t_o:.3f}, p={p_o:.4f}")
    logger.info(f"  正交化后 β ({structure_model_name} 对 {control_structure_model} 正交化): "
                f"mean={beta_orth.mean():.4f}, t={t_h:.3f}, p={p_h:.4f}")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    ax.bar(
        x,
        [beta_orig.mean(), beta_orth.mean()],
        yerr=[beta_orig.std(ddof=1) / np.sqrt(len(beta_orig)),
              beta_orth.std(ddof=1) / np.sqrt(len(beta_orth))],
        capsize=5, color=['steelblue', 'salmon'],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([
        f'Original\n({_display(structure_model_name)})',
        f'Orthogonalized\n({_display(structure_model_name)} ⊥ {_display(control_structure_model)})',
    ])
    ax.set_ylabel('β_structure')
    ax.set_title('β Stability after Orthogonalization')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'h2_beta_stability{suffix}.png', dpi=300)
    plt.close()

    return {
        'beta_original': beta_orig,
        'beta_orthogonal': beta_orth,
        't_original': t_o,
        'p_original': p_o,
        't_orthogonal': t_h,
        'p_orthogonal': p_h,
        'structure_model_name': structure_model_name,
        'control_structure_model': control_structure_model,
        'valid_subjects': valid_subjects,
    }

def run_h2_collinearity_diagnostics(
    subject_ids: list,
    result_root: str = 'results',
    suffix: str = '_cv',
    structure_model_name: str = 'target_priority_model',
    control_structure_model: str = 'gradient_model',
    output_dir: str = 'results/figures/group',
) -> dict:
    logger.info("=" * 60)
    logger.info("开始 H2 共线性诊断")
    logger.info("=" * 60)

    internal = diagnose_h2_internal_collinearity(
        subject_ids, result_root, suffix, structure_model_name, output_dir
    )
    structure = diagnose_structure_model_collinearity(
        subject_ids, suffix, output_dir
    )
    orth = orthogonalized_beta_stability(
        subject_ids, result_root, suffix,
        structure_model_name, control_structure_model, output_dir
    )

    report_path = Path(result_root) / f'h2_collinearity_report{suffix}.txt'
    os.makedirs(report_path.parent, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("H2 共线性诊断报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("--- 1. H2 设计矩阵内部共线性 ---\n")
        if internal:
            f.write(f"条件数: {internal['condition_number']:.2f}\n")
            for name, vif in zip(internal['display_names'], internal['vif_mean']):
                f.write(f"  {name}: VIF={vif:.2f}\n")
        f.write("\n")

        f.write("--- 2. 候选结构模型之间的共线性 ---\n")
        if structure:
            names = structure['display_names']
            corr = structure['corr_mean']
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    f.write(f"  {names[i]} vs {names[j]}: r={corr[i, j]:.3f}\n")
        f.write("\n")

        f.write("--- 3. 正交化后 β 稳定性 ---\n")
        if orth:
            f.write(f"结构模型: {orth['structure_model_name']}\n")
            f.write(f"控制模型: {orth['control_structure_model']}\n")
            f.write(f"正交化前 β: mean={orth['beta_original'].mean():.4f}, "
                    f"t={orth['t_original']:.3f}, p={orth['p_original']:.4f}\n")
            f.write(f"正交化后 β: mean={orth['beta_orthogonal'].mean():.4f}, "
                    f"t={orth['t_orthogonal']:.3f}, p={orth['p_orthogonal']:.4f}\n")
    logger.info(f"H2 共线性诊断报告已保存: {report_path}")

    return {
        'internal_collinearity': internal,
        'structure_model_collinearity': structure,
        'orthogonalized_beta': orth,
    }
