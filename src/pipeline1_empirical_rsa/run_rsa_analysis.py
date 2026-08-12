"""
工程一：RSA分析总入口
"""
import argparse
import gc
import logging
import os
import sys

import mne
import numpy as np
import pandas as pd

from config import CONFIG
from src.shared_utils.metadata_util import compute_positions
from src.pipeline1_empirical_rsa.core.generate_all_models_engine import get_all_cached_objects


# from src.pipeline1_empirical_rsa.core.sensor_rsa_engine import run_sensor_rsa
# from src.pipeline1_empirical_rsa.core.source_rsa_engine import run_source_rsa
# from src.pipeline1_empirical_rsa.core.source_localization import compute_source_estimates


def main():
    # ---- 子进程日志初始化 ----
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    # ---- 获取配置文件中的数据存放路径----
    paths_cfg = CONFIG.get('paths', {})
    deriv_root = paths_cfg.get('deriv_root', 'data/derivatives')
    visual_similarity_root = paths_cfg.get('visual_similarity_root', 'data/visual_similarity')
    cache_dir = paths_cfg.get('model_rdm_root', './result/rdms/model')
    null_dir = paths_cfg.get('model_rdm_root', './result/rdms/model')

    # ---- 获取配置文件中的理论模型参数 ----
    structure_weights = CONFIG.get('structure_content_mixture', {}).get('structure_weights')
    distractor_collapse_levels = CONFIG.get('structure_content_mixture', {}).get('distractor_collapse_levels')
    green_retention_weights = CONFIG.get('parametric_color_position', {}).get('green_retention_weights')
    chunk_pattern = CONFIG.get('chunk_pattern', {}).get('pattern')
    n_permutations = CONFIG.get('statistics', {}).get('n_permutations')
    # ---- 解析命令行 ----
    parser = argparse.ArgumentParser(description="工程一：RSA分析")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--stage", choices=['sensor', 'source', 'all'], default='sensor',
                        help="sensor: 仅传感器RSA | source: 仅源空间RSA | all: 串行执行（不推荐）")
    args = parser.parse_args()

    # ----加载原始数据 ----
    logger.info(f"加载被试 {args.subject} 的数据...")
    epochs = mne.read_epochs(f"{deriv_root}/sub-{args.subject}_coding_epo.fif", preload=True)
    metadata = pd.read_csv(f"{deriv_root}/sub-{args.subject}_metadata.csv")

    # ---- 构建理论模型 ----
    logger.info("构建理论模型 RDM...")
    metadata['position'] = compute_positions(metadata)
    model_vectors, mask, null_vectors = get_all_cached_objects(
        subject_id=args.subject,
        metadata=metadata,
        visual_similarity_root=visual_similarity_root,
        cache_dir=cache_dir,
        null_dir=null_dir,
        structure_weights=structure_weights,
        distractor_collapse_levels=distractor_collapse_levels,
        green_retention_weights=green_retention_weights,
        chunk_pattern=chunk_pattern,
        n_permutations=n_permutations,
        force=args.force,
    )

    # ---- 执行传感器RSA ----
    if args.stage in ['sensor', 'all']:
        logger.info(">>> 执行传感器水平 RSA <<<")
        # results_sensor = run_sensor_rsa(epochs, model_vectors, CONFIG)
        # np.save(f"results/sensor_rsa/sub-{args.subject}_sensor.npy", results_sensor)
        logger.info("传感器RSA完成，结果已保存。")
        if args.stage == 'all':
            # del results_sensor
            gc.collect()

    # ---- 执行源空间RSA ----
    if args.stage in ['source', 'all']:
        logger.info(">>> 执行源空间 RSA <<<")
        # stcs = compute_source_estimates(epochs, subject_id=args.subject)
        # results_source = run_source_rsa(stcs, model_vectors, CONFIG)
        # np.save(f"results/source_rsa/sub-{args.subject}_source.npy", results_source)
        logger.info("源空间RSA完成，结果已保存。")

    logger.info(f"✅ 所有指定分析阶段完成: {args.stage}")


if __name__ == "__main__":
    main()
