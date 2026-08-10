#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WM Dynamics RSA 项目顶级调度器 (Production Entry Point)

用法:
    python run.py --help
    python run.py --download                              # 下载数据集
    python run.py --pipeline all --subject 001            # 运行完整流水线
    python run.py --pipeline empirical --subject 002      # 只运行工程一
    python run.py --pipeline all --force                  # 强制重跑所有步骤
    python run.py --list-subjects                         # 列出所有可用被试

维护指南:
    1. 新增子工程时，在 PIPELINE_REGISTRY 中注册
    2. 新增命令行参数时，在 _parse_args() 中添加
    3. 每次发布前更新 __version__
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import yaml

# 如果上面成功了，再执行你原来的 import
from src.shared_utils import preprocess_util

# ==================== 版本与元信息 ====================
__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"


# ==================== 路径解析 ====================
def get_project_root() -> Path:
    """
    获取项目根目录（无论从哪里执行都能准确找到）
    原理：__file__ 是当前文件的绝对路径，不断调用 .parent 向上回溯
    """
    return Path(__file__).resolve().parent


PROJECT_ROOT = get_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# ==================== 子工程注册表 ====================
# 单一事实来源：所有子工程的元信息集中管理
# 新增子工程时，只需在此添加一条记录
PIPELINE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "preprocess": {
        "name": "数据预处理",
        "script": SRC_DIR / "pipeline1_empirical_rsa" / "run_preprocess.py",
        "requires_subject": True,
        "depends_on": [],
        "done_file": lambda subj: DATA_DIR / "derivatives" / f"sub-{subj}_coding_epo.fif"
    },
    "empirical": {
        "name": "工程一：动态RSA",
        "script": SRC_DIR / "pipeline1_empirical_rsa" / "run_rsa_analysis.py",
        "requires_subject": True,
        "depends_on": ["preprocess"],
        "done_file": lambda subj: RESULTS_DIR / "stats" / f"sub-{subj}_rsa_results.npy"
    },
    "group_stats": {
        "name": "组水平统计",
        "script": SRC_DIR / "pipeline1_empirical_rsa" / "run_group_stats.py",
        "requires_subject": False,
        "depends_on": ["empirical"],
        "done_file": lambda _: RESULTS_DIR / "stats" / "group_p_values.npy"
    },
    "simulation": {
        "name": "工程二：模拟验证",
        "script": SRC_DIR / "pipeline2_simulation_validate" / "run_simulation.py",
        "requires_subject": False,
        "depends_on": [],
        "done_file": lambda _: RESULTS_DIR / "simulation_sweep.png"
    },
    "rnn": {
        "name": "工程三：RNN建模",
        "script": SRC_DIR / "pipeline3_rnn_modeling" / "run_rnn.py",
        "requires_subject": False,
        "depends_on": [],
        "done_file": lambda _: RESULTS_DIR / "models" / "ctrnn_seq_memory.pt"
    }
}


# ==================== 日志系统 ====================
def setup_logging(verbose: bool = False) -> None:
    """
    配置全局日志系统（修复 Windows 控制台 emoji 编码问题）
    """
    # --- 修复 Windows 控制台编码问题 ---
    if sys.platform == 'win32':
        # 如果 stdout 编码不是 utf-8，则重新包装以支持 emoji
        if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        # 同样处理 stderr（以防万一）
        if sys.stderr.encoding and 'utf' not in sys.stderr.encoding.lower():
            import codecs
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

    # 确保日志目录存在
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 文件处理器指定 utf-8 编码，防止写入日志文件时也乱码
    file_handler = logging.FileHandler(
        LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),  # stdout 已重新包装
            file_handler
        ]
    )

    # 抑制第三方库的过度日志
    for lib in ["mne", "matplotlib", "torch", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ==================== 配置加载 ====================
def load_config() -> Dict[str, Any]:
    """加载 config/global_config.yaml 配置文件"""
    config_path = CONFIG_DIR / "global_config.yaml"
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        logger.error("请从 config/global_config.yaml.example 复制并修改")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f"✅ 配置加载成功: {config_path}")
    return config


# ==================== 参数解析 ====================
def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="WM Dynamics RSA Pipeline - 工作记忆序列表征动态分析流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py --download                            # 下载 ds004117 数据集
  python run.py --download --subject 001              # 仅下载被试 001
  python run.py --pipeline all --subject 001          # 运行完整流水线
  python run.py --pipeline empirical --subject 002    # 只运行工程一
  python run.py --pipeline all --force                # 强制重跑所有步骤
  python run.py --list-subjects                       # 列出所有可用被试
        """
    )

    # ---------- 数据下载 ----------
    parser.add_argument(
        "--download",
        action="store_true",
        help="从 OpenNeuro 下载 ds004117 数据集（约5.8GB）"
    )

    # ---------- 流水线选择 ----------
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=['all', 'preprocess', 'empirical', 'group_stats', 'simulation', 'rnn'],
        default='all',
        help="选择要运行的流水线阶段 (默认: all)"
    )

    parser.add_argument(
        "--subject",
        type=str,
        default="001",
        help="被试编号，如 001, 002 (默认: 001)"
    )

    # ---------- 控制开关 ----------
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出详细调试日志"
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="仅打印将要执行的操作，不实际运行"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已有结果（跳过增量检测）"
    )

    # ---------- 信息查询 ----------
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        help="列出 data/raw/ 下所有可用被试"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"WM Dynamics RSA v{__version__}"
    )

    return parser.parse_args()


# ==================== 核心调度器 ====================
def run_pipeline_by_name(pipeline_name: str, subject: str, config: Dict, dry_run: bool = False) -> int:
    """
    通用流水线执行器

    Args:
        pipeline_name: 注册表中的流水线名称
        subject: 被试编号
        dry_run: 是否仅预览

    Returns:
        执行结果码 (0 表示成功)
    """
    info = PIPELINE_REGISTRY.get(pipeline_name)
    if not info:
        logger.error(f"未知流水线: {pipeline_name}")
        return 1

    script_path = info["script"]
    if not script_path.exists():
        logger.error(f"脚本不存在: {script_path}")
        return 1

    logger.info(f"🚀 执行: {info['name']}")

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
    env["PYTHONPATH"] = str(PROJECT_ROOT)+ os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(cmd, env=env,cwd=PROJECT_ROOT, check=True)
        logger.info(f"✅ {info['name']} 完成")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {info['name']} 失败: {e}")
        return e.returncode


def run_all_pipelines(args, config) -> int:
    """
    智能依赖拓扑执行（带增量跳过）

    流程:
        1. 检查每个步骤是否已有有效结果
        2. 按依赖顺序执行，跳过已完成的步骤（除非 --force）
        3. 若某步骤失败，终止整个流水线
    """
    pipeline_order = [
        "preprocess",  # 工程一：数据准备
        "empirical",  # 工程一：个体表征
        "group_stats",  # 工程一：群体推断
        "simulation",  # 工程二：独立验证
        "rnn"  # 工程三：计算建模
    ]

    # 检查已完成状态
    skip_cache = {}
    for name in pipeline_order:
        info = PIPELINE_REGISTRY[name]
        done_file = info["done_file"](args.subject)
        # done_file 可能不存在（旧版本兼容）
        if done_file is not None:
            skip_cache[name] = done_file.exists()
        else:
            skip_cache[name] = False

    # 按顺序执行
    for name in pipeline_order:
        # 检查是否已完成
        if skip_cache.get(name, False) and not args.force:
            logger.info(f"⏭️ 跳过 {PIPELINE_REGISTRY[name]['name']}（结果已存在，使用 --force 强制重跑）")
            continue

        # 检查依赖是否满足
        deps = PIPELINE_REGISTRY[name].get("depends_on", [])
        for dep in deps:
            if not skip_cache.get(dep, False):
                # 依赖未完成，先递归执行
                logger.info(f"📌 依赖 {dep} 未完成，先执行...")
                exit_code = run_pipeline_by_name(dep, args.subject,config,args.dry_run)
                if exit_code != 0:
                    logger.error(f"❌ 依赖 {dep} 执行失败，终止流水线")
                    return exit_code
                skip_cache[dep] = True

        # 执行当前步骤
        exit_code = run_pipeline_by_name(name, args.subject, config,args.dry_run)
        if exit_code != 0:
            logger.error(f"❌ {PIPELINE_REGISTRY[name]['name']} 执行失败，终止流水线")
            return exit_code

        # 标记为已完成
        skip_cache[name] = True

    return 0


# ==================== 数据下载 ====================
def run_download(subject: Optional[str] = None) -> int:
    """
    从 OpenNeuro 下载 ds004117 数据集

    Args:
        subject: 指定被试编号，若为 None 则下载全部
    """
    logger.info("📦 启动数据下载流程")

    target_dir = DATA_DIR / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已存在数据
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.warning("⚠️ data/raw/ 目录非空，可能存在已有数据")
        logger.warning("如需重新下载，请先手动删除 data/raw/ 目录")
        response = input("是否继续下载（将覆盖已有文件）？[y/N]: ")
        if response.lower() != 'y':
            logger.info("下载已取消")
            return 0

    try:
        from openneuro import download
    except ImportError:
        logger.error("未安装 openneuro-py，正在尝试自动安装...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "openneuro-py"],cwd=PROJECT_ROOT,  check=True)
            from openneuro import download
        except Exception as e:
            logger.error(f"安装失败: {e}")
            logger.error("请手动执行: pip install openneuro-py")
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

        logger.info("✅ 数据下载完成！")
        logger.info(f"📁 数据存放位置: {target_dir}")
        return 0
    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        logger.info("手动备选方案:")
        logger.info("  1. 浏览器打开 https://openneuro.org/datasets/ds004117/versions/1.0.1")
        logger.info("  2. 点击 'Download' 下载压缩包")
        logger.info(f"  3. 解压后放入: {target_dir}")
        return 1


# ==================== 辅助功能 ====================
def list_subjects() -> None:
    """列出 data/raw/ 下所有可用被试"""
    raw_dir = DATA_DIR / "raw"
    if not raw_dir.exists():
        logger.error(f"数据目录不存在: {raw_dir}")
        return

    # 获取所有子目录（BIDS标准：sub-XXX）
    subjects = [d.name for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    if not subjects:
        logger.warning("未找到任何被试数据")
        return

    logger.info(f"找到 {len(subjects)} 个被试:")
    for sub in sorted(subjects):
        # 检查预处理状态
        subject_id = sub.replace("sub-", "")
        deriv_file = DATA_DIR / "derivatives" / f"sub-{subject_id}_coding_epo.fif"
        status = "✅ 已预处理" if deriv_file.exists() else "⏳ 未预处理"
        logger.info(f"  - {sub} {status}")


# ==================== 主入口 ====================
def main() -> int:
    """程序主入口"""
    # 1. 解析参数
    args = parse_args()

    # 2. 设置日志
    setup_logging(args.verbose)

    # 3. 打印启动信息
    logger.info("=" * 60)
    logger.info(f"🧠 WM Dynamics RSA Pipeline v{__version__}")
    logger.info(f"📂 项目根目录: {PROJECT_ROOT}")
    logger.info(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 4. 数据下载模式
    if args.download:
        return run_download(args.subject if args.subject != "001" else None)

    # 5. 列表查询模式
    if args.list_subjects:
        list_subjects()
        return 0

    # 6. 正常流水线模式
    config = load_config()

    if args.dry_run:
        logger.info("⚠️ DRY-RUN 模式: 仅预览操作，不实际执行")

    if args.pipeline == "all":
        exit_code = run_all_pipelines(args, config)
    else:
        exit_code = run_pipeline_by_name(args.pipeline, args.subject, config,args.dry_run)

    # 7. 输出结果
    if exit_code == 0:
        logger.info("=" * 60)
        logger.info("🎉 所有任务执行成功！")
        logger.info(f"📁 结果保存在: {RESULTS_DIR}")
        if not args.dry_run:
            logger.info("📊 查看结果:")
            logger.info(f"   - 统计图: {RESULTS_DIR / 'figures/'}")
            logger.info(f"   - 统计数据: {RESULTS_DIR / 'stats/'}")
            logger.info(f"   - 模型权重: {RESULTS_DIR / 'models/'}")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("❌ 流水线执行失败，请检查上方错误日志")
        logger.error("提示: 使用 --verbose 查看详细调试信息")
        logger.error("=" * 60)

    return exit_code


# ==================== 标准入口 ====================
if __name__ == "__main__":
    sys.exit(main())