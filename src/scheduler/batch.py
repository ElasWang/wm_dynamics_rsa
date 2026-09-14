

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)


def resolve_subject_list(subject_arg: str, data_dir: Path) -> List[str]:
    if subject_arg == 'all':
        raw_path = data_dir / "raw"
        if not raw_path.exists():
            return []
        subjects = [f.name.replace('sub-', '')
                    for f in raw_path.glob("sub-*") if f.is_dir()]
        subjects = [s for s in subjects if re.match(r'^\d+$', s)]
        return sorted(subjects, key=lambda x: int(x))
    elif ',' in subject_arg:
        return [s.strip() for s in subject_arg.split(',')]
    else:
        return [subject_arg]


def check_subject_status(subject_id: str, data_dir: Path) -> Dict[str, bool]:
    deriv_path = data_dir / "derivatives"
    results_path = data_dir.parent / "results" / "stats"
    return {
        'preprocessed': (deriv_path / f"sub-{subject_id}_coding_epo.fif").exists(),
        'rsa_done': (results_path / f"sub-{subject_id}_rsa_results.npy").exists(),
    }


def preview_subjects(
        subject_list: List[str],
        data_dir: Path,
        pipeline_name: str,
        force: bool = False
) -> Tuple[List[str], List[str], List[str]]:
    to_process = []
    already_done = []
    skipped_missing = []
    for sub in subject_list:
        status = check_subject_status(sub, data_dir)
        if pipeline_name == 'preprocess':
            if status['preprocessed'] and not force:
                already_done.append(sub)
            else:
                to_process.append(sub)
        elif pipeline_name == 'empirical':
            if not status['preprocessed']:
                skipped_missing.append(sub)
            elif status['rsa_done'] and not force:
                already_done.append(sub)
            else:
                to_process.append(sub)
        elif pipeline_name == 'all':
            to_process.append(sub)
        else:
            to_process.append(sub)
    return to_process, already_done, skipped_missing


def run_batch_pipelines(
    pipeline_name: str,
    subject_arg: str,
    config: Dict[str, Any],
    dry_run: bool,
    project_root: Path,
    registry: Dict[str, Dict[str, Any]],
        executor_func: Callable,
        force: bool = False,
) -> int:
    data_dir = project_root / "data"
    subject_list = resolve_subject_list(subject_arg, data_dir)
    if not subject_list:
        logger.error("未找到任何被试（请确保 data/raw/ 下有 sub-* 文件夹）")
        return 1
    logger.info(f" 发现 {len(subject_list)} 个被试: {subject_list}")
    if pipeline_name in ['all', 'preprocess', 'empirical']:
        to_process, already_done, skipped_missing = preview_subjects(
            subject_list, data_dir, pipeline_name, force)
        logger.info("\n" + "=" * 60)
        logger.info(f" 批量处理预览 (阶段: {pipeline_name})")
        logger.info(f"  总被试数: {len(subject_list)}")
        if already_done:
            logger.info(f"   已完成（跳过）: {len(already_done)} 个 -> {already_done}")
        if skipped_missing:
            logger.info(f"   跳过（缺失前置文件）: {len(skipped_missing)} 个 -> {skipped_missing}")
        if to_process:
            logger.info(f"   待处理: {len(to_process)} 个 -> {to_process}")
        else:
            logger.info(f"   所有被试已处理完毕！")
            return 0
        logger.info("=" * 60 + "\n")
        if dry_run:
            logger.info(" DRY-RUN 模式：仅预览，不实际执行")
            return 0
        subject_list = to_process
        if not subject_list:
            logger.info(" 没有需要处理的被试")
            return 0
    if pipeline_name != 'all':
        logger.info(f" 执行阶段: {pipeline_name}")
        failed = []
        for idx, sub in enumerate(subject_list, 1):
            logger.info(f"\n [{idx}/{len(subject_list)}] 处理被试 sub-{sub}")
            exit_code = executor_func(pipeline_name, sub, config, dry_run, project_root, registry)
            if exit_code != 0:
                failed.append(sub)
        if failed:
            logger.error(f" 失败被试: {failed}")
            return 1
        logger.info(" 单阶段批量处理全部成功！")
        return 0
    stages = ['preprocess', 'empirical', 'group_stats']
    failed_subjects = {stage: [] for stage in stages}
    stage_success_counts = {stage: 0 for stage in stages}
    if not subject_list:
        subject_list = resolve_subject_list(subject_arg, data_dir)
    for stage in stages:
        logger.info("\n" + "=" * 70)
        logger.info(f"阶段 [{stage.upper()}] 开始")
        logger.info(f"   待处理被试: {len(subject_list)} 个")
        logger.info("=" * 70)
        if stage == 'group_stats':
            exit_code = executor_func('group_stats', '', config, dry_run, project_root, registry)
            if exit_code != 0:
                logger.error(f" 组水平统计失败，终止流水线")
                return exit_code
            continue
        stage_success = []
        stage_failed = []
        for idx, sub in enumerate(subject_list, 1):
            logger.info(f"\n [{stage.upper()}] 进度: {idx}/{len(subject_list)} -> sub-{sub}")
            if stage == 'empirical':
                deriv_file = project_root / "data" / "derivatives" / f"sub-{sub}_coding_epo.fif"
                if not deriv_file.exists():
                    logger.warning(f"  跳过 sub-{sub}（预处理文件不存在）")
                    stage_failed.append(sub)
                    continue
            exit_code = executor_func(stage, sub, config, dry_run, project_root, registry)
            if exit_code == 0:
                stage_success.append(sub)
                logger.info(f" [{stage.upper()}] sub-{sub} 成功")
            else:
                stage_failed.append(sub)
                logger.error(f" [{stage.upper()}] sub-{sub} 失败")
        failed_subjects[stage] = stage_failed
        stage_success_counts[stage] = len(stage_success)
        logger.info("\n" + "=" * 70)
        logger.info(f"阶段 [{stage.upper()}] 完成")
        logger.info(f"  成功: {len(stage_success)} 个 -> {stage_success}")
        if stage_failed:
            logger.warning(f"   失败: {len(stage_failed)} 个 -> {stage_failed}")
        logger.info("=" * 70)
    logger.info("\n" + "=" * 60)
    logger.info(" 全流程批量处理完成!")
    total_failed = set()
    for stage, lst in failed_subjects.items():
        if lst:
            total_failed.update(lst)
    if total_failed:
        logger.error(f" 存在失败被试: {total_failed}")
        return 1
    logger.info("所有阶段所有被试全部成功！")
    return 0