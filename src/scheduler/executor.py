
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def run_pipeline_by_name(
    pipeline_name: str,
    subject: str,
    config: Dict,
    dry_run: bool,
    project_root: Path,
    pipeline_registry: Dict[str, Dict[str, Any]]
) -> int:
    info = pipeline_registry.get(pipeline_name)
    if not info:
        logger.error(f"未知流水线: {pipeline_name}")
        return 1
    done_file = info.get("done_file")
    if done_file is not None:
        try:
            if callable(done_file):
                done_path = done_file(subject)
            else:
                done_path = done_file
            if done_path is not None and done_path.exists():
                logger.info(f" 跳过 {info['name']}（结果已存在: {done_path}）")
                return 0
        except Exception as e:
            logger.warning(f"检查 done_file 失败: {e}，继续执行")
    script_path = info["script"]
    if not script_path.exists():
        logger.error(f"脚本不存在: {script_path}")
        return 1
    logger.info(f"执行: {info['name']} (被试: {subject})")
    if dry_run:
        cmd_display = f"python {script_path}"
        if info["requires_subject"]:
            cmd_display += f" --subject {subject}"
        logger.info(f"[DRY-RUN] 将执行: {cmd_display}")
        return 0
    cmd = [sys.executable, str(script_path)]
    if info["requires_subject"] and subject:
        cmd.extend(["--subject", subject])
    env = os.environ.copy()
    env["BIDS_ROOT"] = str(config.get('paths', {}).get('bids_root', './data/raw'))
    env["DERIV_ROOT"] = str(config.get('paths', {}).get('deriv_root', './data/derivatives'))
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        subprocess.run(cmd, env=env, cwd=project_root, check=True)
        logger.info(f" {info['name']} 完成")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f" {info['name']} 失败: {e}")
        return e.returncode


def run_all_pipelines(
    args,
    config: Dict,
    project_root: Path,
    pipeline_registry: Dict[str, Dict[str, Any]]
) -> int:
    pipeline_order = ["preprocess", "empirical", "group_stats", "simulation", "rnn"]
    skip_cache = {}
    for name in pipeline_order:
        info = pipeline_registry[name]
        done_file = info["done_file"](args.subject)
        skip_cache[name] = done_file.exists() if done_file is not None else False
    for name in pipeline_order:
        if skip_cache.get(name, False) and not args.force:
            logger.info(f" 跳过 {pipeline_registry[name]['name']}（结果已存在，使用 --force 强制重跑）")
            continue
        deps = pipeline_registry[name].get("depends_on", [])
        for dep in deps:
            if not skip_cache.get(dep, False):
                logger.info(f"依赖 {dep} 未完成，先执行...")
                exit_code = run_pipeline_by_name(dep, args.subject,
                                                 config, args.dry_run, project_root, pipeline_registry)
                if exit_code != 0:
                    return exit_code
                skip_cache[dep] = True
        exit_code = run_pipeline_by_name(name, args.subject, config,
                                         args.dry_run, project_root, pipeline_registry)
        if exit_code != 0:
            logger.error(f" {pipeline_registry[name]['name']} 执行失败")
            return exit_code
        skip_cache[name] = True
    return 0