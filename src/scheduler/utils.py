# src/scheduler/utils.py
import os
import sys
import subprocess
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(config_dir: Path) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    config_path = config_dir / "global_config.yaml"
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        logger.error(f"加载配置文件成功: {config_path}")
        return yaml.safe_load(f)


def run_download(data_dir: Path, project_root: Path, subject: Optional[str] = None) -> int:
    """下载 ds004117 数据集"""
    logger.info(" 启动数据下载流程")
    target_dir = data_dir / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and any(target_dir.iterdir()):
        logger.warning("data/raw/ 目录非空，可能存在已有数据")
        response = input("是否继续下载（将覆盖已有文件）？[y/N]: ")
        if response.lower() != 'y':
            logger.info("下载已取消")
            return 0

    try:
        from openneuro import download
    except ImportError:
        logger.error("未安装 openneuro-py，正在尝试自动安装...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "openneuro-py"], cwd=project_root, check=True)
            from openneuro import download
        except Exception as e:
            logger.error(f"安装失败：{e}")
            logger.error(f"请手动执行： pip install openneuro-py")
            return 1
    try:
        dataset = 'ds004117'
        include = f'sub-{subject}/eeg' if subject else None
        logger.info(f"开始下载 {dataset}")
        if subject:
            logger.info(f"  仅下载被试: {subject}")
        else:
            logger.info("  下载全部被试（约5.8GB，可能需要较长时间）")

        download(dataset=dataset, target_dir=str(target_dir), include=include)

        logger.info("数据下载完成！")
        logger.info(f"数据存放位置: {target_dir}")
        return 0
    except Exception as e:
        logger.error(f"下载失败: {e}")
        logger.info("手动备选方案:")
        logger.info("  1. 浏览器打开 https://openneuro.org/datasets/ds004117/versions/1.0.1")
        logger.info("  2. 点击 'Download' 下载压缩包")
        logger.info(f"  3. 解压后放入: {target_dir}")
        return 1



def list_subjects(data_dir: Path) -> None:
    """列出 data/raw/ 下所有可用被试"""
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        logger.error(f"数据目录不存在: {raw_dir}")
        return

    subjects = [d.name for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    if not subjects:
        logger.warning("未找到任何被试数据")
        return

    logger.info(f"找到 {len(subjects)} 个被试:")
    for sub in sorted(subjects):
        subject_id = sub.replace("sub-", "")
        deriv_file = data_dir / "derivatives" / f"sub-{subject_id}_coding_epo.fif"
        status = "✅ 已预处理" if deriv_file.exists() else "⏳ 未预处理"
        logger.info(f"  - {sub} {status}")