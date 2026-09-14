
import argparse
import logging
import os
import sys
from pathlib import Path

import mne

from config import CONFIG
from src.pipeline1_empirical_rsa.core.h1_analysis import run_subject_level as h1_subject
from src.pipeline1_empirical_rsa.core.h2_analysis import run_subject_level as h2_subject
from src.pipeline1_empirical_rsa.core.h3_analysis import run_subject_level as h3_subject
from src.pipeline1_empirical_rsa.core.sensor_rsa_engine import cache_sensor_rdms, cache_cv_condition_rdms, \
    cache_cv_sequence_rdms
from src.pipeline1_empirical_rsa.core.source_rsa_engine import cache_cv_source_condition_rdms
from src.shared_utils.metadata_util import filter_epochs
from src.shared_utils.models.resolve_model_instances import resolve_model_instances, resolve_condition_model_instances


def main():
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    paths_cfg = CONFIG.get('paths', {})
    deriv_root = Path(paths_cfg.get('deriv_root', 'data/derivatives'))
    result_root = Path(paths_cfg.get('rsa_result'))

    parser = argparse.ArgumentParser(description="Study 1：RSA分析")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--stage", choices=['sensor', 'source', 'all'], default='sensor',
                        help="sensor: 仅传感器RSA | source: 仅源空间RSA | all: 串行执行（不推荐）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新计算所有缓存对象")
    args = parser.parse_args()
    subject_id = args.subject
    logger.info(f"加载被试 {subject_id} 的数据...")
    epochs = mne.read_epochs(f"{deriv_root}/sub-{subject_id}_coding_epo.fif", preload=True)

    n_original = len(epochs.drop_log)
    n_kept = len(epochs)
    drop_ratio = 1.0 - (n_kept / n_original)
    if drop_ratio > 0.25:
        logger.warning(f"被试 {args.subject} 剔除比例 {drop_ratio:.1%} > 25%，排除此被试")
        with open('excluded_subjects.txt', 'a') as f:
            f.write(f"{args.subject}\n")
        sys.exit(0)
    else:
        logger.info(f"试次剔除比例: {drop_ratio:.1%} (保留 {n_kept}/{n_original})")

    epochs = filter_epochs(epochs)
    debug_dir = Path(paths_cfg.get('figures_debug')) / f'sub-{args.subject}'
    debug_dir.mkdir(parents=True, exist_ok=True)

    # ---- 获取理论模型 RDM ----
    logger.info("获取理论模型 event-level RDM...")
    model_rdms = resolve_model_instances(subject_id=subject_id)

    logger.info("获取理论模型 cv-level RDM...")
    model_rdms_cond = resolve_condition_model_instances(subject_id=subject_id)

    # plot_debug_for_model_dict(model_rdms_cond, 'cond', debug_dir)

    debug_dir = Path('figures/debug') / f'sub-{args.subject}'
    debug_dir.mkdir(parents=True, exist_ok=True)

    # ---- 获取神经 RDM ----
    logger.info("获取神经 event-level RDM...")
    neural_rdms, times_ms, time_indices = cache_sensor_rdms(
        subject_id,
        epochs,
        overwrite=args.force
    )
    logger.info("获取神经 cv-level RDM...")
    cv_rdms, cv_times_ms, cv_time_indices = cache_cv_condition_rdms(
        subject_id=subject_id,
        epochs=epochs,
        condition_col='position',
        n_repeats=10,
        overwrite=args.force
    )
    logger.info("获取神经 cv-sequence-level RDM...")
    cv_sequence_rdms, cv_sequence_times_ms, cv_sequence_time_indices = cache_cv_sequence_rdms(
        subject_id=subject_id,
        epochs=epochs,
        overwrite=args.force
    )
    logger.info("获取神经 cv_source-level RDM...")
    cv_source_rdms, cv_source_times_ms, cv_source_time_indices = cache_cv_source_condition_rdms(
        subject_id=subject_id,
        epochs=epochs,
        tmin=-0.2, tmax=0.8,
        overwrite=args.force
    )

    # ---- 运行 H1 分析 ----
    logger.info("运行 H1 event-level ...")
    h1_subject(
        subject_id=subject_id,
        model_rdms=model_rdms,
        neural_rdms=neural_rdms,
        result_root=result_root,
        times_ms=times_ms,
        time_indices=time_indices,
        suffix='_event',
        overwrite=args.force,
    )
    logger.info("运行 H1 cv-level ...")
    h1_subject(
        subject_id=subject_id,
        model_rdms=model_rdms_cond,
        neural_rdms=cv_rdms,
        result_root=result_root,
        times_ms=cv_times_ms,
        time_indices=cv_time_indices,
        suffix='_cv',
        overwrite=args.force,
    )
    logger.info("运行 H1 cv_source-level  ...")
    h1_subject(
        subject_id=subject_id,
        model_rdms=model_rdms_cond,
        neural_rdms=cv_source_rdms,
        result_root=result_root,
        times_ms=cv_source_times_ms,
        time_indices=cv_source_time_indices,
        suffix='_source_cv',
        overwrite=args.force,
    )

    logger.info("运行 H2 cv-level ...")
    h2_subject(
        suffix='_cv',
        subject_id=subject_id,
        neural_rdms_dict=cv_rdms,
        model_rdms_dict=model_rdms_cond,
        time_indices=cv_time_indices,
        result_root=result_root,
        structure_model_name='target_priority_model',
        overwrite=args.force,
    )
    logger.info("运行 H2 event-level ...")
    h2_subject(
        suffix='_event',
        subject_id=subject_id,
        neural_rdms_dict=neural_rdms,
        model_rdms_dict=model_rdms,
        time_indices=time_indices,
        result_root=result_root,
        structure_model_name='target_priority_model',
        overwrite=args.force,
    )
    logger.info("运行 H2 cv-source-level ...")
    h2_subject(
        subject_id=subject_id,
        neural_rdms_dict=cv_source_rdms,
        model_rdms_dict=model_rdms_cond,
        time_indices=cv_source_time_indices,
        result_root=result_root,
        suffix='_source_cv',
        structure_model_name='target_priority_model',
        overwrite=args.force,
    )
    # ---- 运行 H3 分析 ----
    logger.info("运行 H3 event-level ...")
    h3_subject(
        suffix='_event',
        subject_id=subject_id,
        model_rdms=model_rdms,
        neural_rdms=neural_rdms,
        result_root=result_root,
        times_ms=times_ms,
        structure_model_name='target_priority_model',
        overwrite=args.force,
    )
    logger.info("运行 H3 cv-level ...")
    h3_subject(
        suffix='_cv',
        subject_id=subject_id,
        model_rdms=model_rdms_cond,
        neural_rdms=cv_rdms,
        result_root=result_root,
        times_ms=cv_times_ms,
        structure_model_name='target_priority_model',
        overwrite=args.force,
    )
    logger.info("运行 H3 cv-source-level ...")
    h3_subject(
        suffix='_source_cv',
        subject_id=subject_id,
        model_rdms=model_rdms_cond,
        neural_rdms=cv_source_rdms,
        result_root=result_root,
        times_ms=cv_source_times_ms,
        structure_model_name='target_priority_model',
        overwrite=args.force,
    )

    logger.info(f" 被试 {subject_id} 单被试计算完成")


if __name__ == "__main__":
    main()
