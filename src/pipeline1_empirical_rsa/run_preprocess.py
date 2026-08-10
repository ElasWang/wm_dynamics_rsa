from pathlib import Path
import os
import logging
from config import CONFIG
from src.shared_utils.preprocess_util import process_subject
import sys
import argparse

# ==================== 路径修正 ====================
# 用于独立运行时（非通过 run.py 调度）自动找到项目根目录
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:

    # ===== 子进程日志初始化 =====
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    # ---- 1. 解析命令行参数 ----
    parser = argparse.ArgumentParser(description="EEG 数据预处理")
    parser.add_argument("--subject", required=True, help="被试编号，如 001")
    parser.add_argument("--force", action="store_true", help="强制重跑（目前未实现）")
    args = parser.parse_args()

    # ---- 2. 从配置文件读取预处理参数 ----
    preproc_cfg = CONFIG.get('preprocessing', {})
    paths_cfg = CONFIG.get('paths', {})

    resample_sfreq = preproc_cfg.get('resample_sfreq', 250)
    filter_low = preproc_cfg.get('filter_low', 0.5)
    filter_high = preproc_cfg.get('filter_high', 40.0)
    ica_n_components = preproc_cfg.get('ica_n_components', 30)
    eog_threshold = preproc_cfg.get('eog_threshold', 2.5)

    # ---- 3. 构造路径（优先使用环境变量，否则基于项目根目录） ----
    # 当通过 run.py 调度时，BIDS_ROOT 和 DERIV_ROOT 会通过环境变量传入
    bids_root = Path(os.getenv("BIDS_ROOT", PROJECT_ROOT / paths_cfg.get('bids_root', 'data/raw')))
    deriv_root = Path(os.getenv("DERIV_ROOT", PROJECT_ROOT / paths_cfg.get('deriv_root', 'data/derivatives')))

    logger.info("=" * 60)
    logger.info(f"预处理被试: sub-{args.subject}")
    logger.info(f"  项目根目录: {PROJECT_ROOT}")
    logger.info(f"  数据源: {bids_root}")
    logger.info(f"  输出: {deriv_root}")
    logger.info(f"  重采样: {resample_sfreq} Hz")
    logger.info(f"  滤波: {filter_low} - {filter_high} Hz")
    logger.info("=" * 60)

    # ---- 4. 执行预处理 ----
    try:
        epochs, metadata = process_subject(
            subject_id=args.subject,
            bids_root=bids_root,
            deriv_root=deriv_root,
            task="WorkingMemory",
            session="01",
            resample_sfreq=resample_sfreq,
            filter_low=filter_low,
            filter_high=filter_high,
            ica_n_components=ica_n_components,
            eog_threshold=eog_threshold,
        )
        logger.info(f"✅ 预处理完成！共 {len(epochs)} 个 Epochs")
        return 0
    except FileNotFoundError as e:
        logger.error(f"数据文件不存在: {e}")
        logger.error("请检查 BIDS_ROOT 路径是否正确，或运行 python run.py --download 下载数据")
        return 1
    except Exception as e:
        logger.error(f"预处理失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())