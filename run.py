import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# ==================== 版本与元信息 ====================
__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"


# ==================== 路径解析 ====================
def get_project_root() -> Path:
    return Path(__file__).resolve().parent

PROJECT_ROOT = get_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# ==================== 导入调度器 ====================
from src.scheduler import (
    get_pipeline_registry,
    run_pipeline_by_name,
    run_all_pipelines,
    load_config,
    run_download,
    list_subjects,
    run_batch_pipelines
)

# ==================== 日志系统 ====================
def setup_logging(verbose: bool = False) -> None:
    """
    配置全局日志系统
    """
    if sys.platform == 'win32':
        if sys.stdout.encoding and 'utf' not in sys.stdout.encoding.lower():
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        if sys.stderr.encoding and 'utf' not in sys.stderr.encoding.lower():
            import codecs
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    file_handler = logging.FileHandler(
        LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler
        ]
    )
    for lib in ["mne", "matplotlib", "torch", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


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
        help="被试编号，如 '001'；或 'all' 处理全部；或 'sub-01,sub-02' 指定列表"
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

# ==================== 主入口 ====================
def main() -> int:
    # ---- 解析参数 ----
    args = parse_args()

    # ---- 设置日志 ----
    setup_logging(args.verbose)

    # ---- 打印启动信息 ----
    logger.info("=" * 60)
    logger.info(f" WM Dynamics RSA Pipeline v{__version__}")
    logger.info(f" 项目根目录: {PROJECT_ROOT}")
    logger.info(f" 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ---- 数据下载模式 ----
    if args.download:
        return run_download(DATA_DIR, PROJECT_ROOT, args.subject if args.subject != "001" else None)

    # ---- 列表查询模式 ----
    if args.list_subjects:
        list_subjects(DATA_DIR)
        return 0

    # ---- 获取配置文件 ----
    config = load_config(CONFIG_DIR)

    # ---- 获取子进程注册列表 ----
    registry = get_pipeline_registry(SRC_DIR, DATA_DIR, RESULTS_DIR)

    if args.dry_run:
        logger.info("DRY-RUN 模式: 仅预览操作，不实际执行")

    if args.subject == 'all':
        exit_code = run_batch_pipelines(
            pipeline_name=args.pipeline,
            subject_arg=args.subject,
            config=config,
            dry_run=args.dry_run,
            project_root=PROJECT_ROOT,
            registry=registry,
            executor_func=run_pipeline_by_name
        )
    else:
        if args.pipeline == "all":
            exit_code = run_all_pipelines(args, config, PROJECT_ROOT, registry)
        else:
            exit_code = run_pipeline_by_name(args.pipeline, args.subject, config, args.dry_run, PROJECT_ROOT, registry)

    # ---- 输出结果 ----
    if exit_code == 0:
        logger.info("=" * 60)
        logger.info(" 所有任务执行成功！")
        logger.info(f"结果保存在: {RESULTS_DIR}")
        if not args.dry_run:
            logger.info("查看结果:")
            logger.info(f"   - 统计图: {RESULTS_DIR / 'figures/'}")
            logger.info(f"   - 统计数据: {RESULTS_DIR / 'stats/'}")
            logger.info(f"   - 模型权重: {RESULTS_DIR / 'models/'}")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error(" 流水线执行失败，请检查上方错误日志")
        logger.error("提示: 使用 --verbose 查看详细调试信息")
        logger.error("=" * 60)

    return exit_code


# ==================== 标准入口 ====================
if __name__ == "__main__":
    sys.exit(main())
