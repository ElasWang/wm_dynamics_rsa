# src/scheduler/batch.py
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def resolve_subject_list(subject_arg: str, data_dir: Path) -> List[str]:
    """
    解析 --subject 参数，返回被试编号列表。
    支持: '001', 'all', '001,005,010'
    """
    if subject_arg == 'all':
        subjects = []
        for f in (data_dir / "derivatives").glob("sub-*_coding_epo.fif"):
            match = re.search(r'sub-(\d+)_coding_epo\.fif', f.name)
            if match:
                subjects.append(match.group(1))
        return sorted(subjects, key=lambda x: int(x))
    elif ',' in subject_arg:
        return [s.strip() for s in subject_arg.split(',')]
    else:
        return [subject_arg]


def run_batch_pipelines(
    pipeline_name: str,
    subject_arg: str,
    config: Dict[str, Any],
    dry_run: bool,
    project_root: Path,
    registry: Dict[str, Dict[str, Any]],
    executor_func,  # 注入 run_pipeline_by_name
) -> int:
    """
    批量执行流水线。
    通过注入 executor_func 避免循环依赖。
    """
    subject_list = resolve_subject_list(subject_arg, project_root / "data")
    if not subject_list:
        logger.error("未找到任何被试")
        return 1

    logger.info(f" 批量处理 {len(subject_list)} 个被试: {subject_list}")
    failed = []
    for idx, sub in enumerate(subject_list, 1):
        logger.info(f"\n [{idx}/{len(subject_list)}] 处理被试 sub-{sub}")
        exit_code = executor_func(
            pipeline_name, sub, config, dry_run, project_root, registry
        )
        if exit_code != 0:
            failed.append(sub)
    if failed:
        logger.error(f" 失败被试: {failed}")
        return 1
    logger.info(" 批量处理全部成功！")
    return 0