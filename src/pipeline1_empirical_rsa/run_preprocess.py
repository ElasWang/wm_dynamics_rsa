import argparse
import logging
import os
import sys
from pathlib import Path

from config import CONFIG
from src.pipeline1_empirical_rsa.core.preprocess_engine import process_subject
from src.pipeline1_empirical_rsa.visualization.plot_erp_and_topomap import (
    plot_erp_overlay, plot_topomap_at_times
)
from src.shared_utils.metadata_util import filter_epochs

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="EEG 数据预处理")
    parser.add_argument("--subject", required=True, help="被试编号，如 001")
    parser.add_argument("--force", action="store_true", help="强制重跑（目前未实现）")
    args = parser.parse_args()

    try:
        epochs, metadata = process_subject(subject_id=args.subject)
        logger.info(f" 预处理完成！共 {len(epochs)} 个 Epochs")
        paths_cfg = CONFIG.get('paths', {})
        epochs = filter_epochs(epochs)
        qc_dir = (Path(paths_cfg.get("figures_result", "/results/figures"))
                  / "QC" / f"sub-{args.subject}")
        qc_dir.mkdir(parents=True, exist_ok=True)
        plot_erp_overlay(epochs,subject_id=args.subject,
                         save_path=qc_dir / 'erp_overlay.png',show=False,)
        plot_topomap_at_times(epochs,subject_id=args.subject,times_ms=[100, 300, 500],
            save_path=qc_dir / 'topomap.png',show=False,)
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