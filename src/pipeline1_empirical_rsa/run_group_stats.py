import logging
import os
import sys
from pathlib import Path

from config import CONFIG
from src.pipeline1_empirical_rsa.core import h2_analysis, h3_analysis, h1_analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parents[2]
os.chdir(project_root)

def main():
    result_root = Path(CONFIG.get('paths', {}).get('rsa_result', 'results'))
    figures_group = Path(CONFIG.get('paths', {}).get('figures_group', 'figures_group'))
    excluded_subjects = CONFIG.get('rsa', {}).get('excluded_subjects', [])
    suffixes = ['_event', '_cv', '_source_cv']
    subject_ids = get_valid_subject_ids(result_root)
    subject_ids = [s for s in subject_ids if s not in excluded_subjects]
    if len(subject_ids) < 3:
        logger.error(f"有效被试不足 (n={len(subject_ids)})，无法进行群体统计")
        sys.exit(1)
    logger.info(f"纳入群体统计的被试 (n={len(subject_ids)}): {subject_ids}")
    h1_results = {}
    for suffix in suffixes:
        logger.info(f"\n===== H1 群体统计 (suffix={suffix}) =====")
        try:
            clusters = h1_analysis.run_group_level(subject_ids, result_root=result_root,
                suffix=suffix, save_fig_dir=figures_group,)
            h1_results[suffix] = clusters
            logger.info(f"  -> 发现 {len(clusters)} 个显著簇")
        except Exception as e:
            logger.error(f"H1 ({suffix}) 失败: {e}")
            h1_results[suffix] = []

    logger.info("\n===== H2 群体统计 =====")
    for suffix in suffixes:
        try:
            h2_results = h2_analysis.run_group_level(subject_ids=subject_ids, result_root=result_root,
                suffix=suffix,structure_model_name='target_priority_model',
                save_fig_dir=figures_group,)
            if h2_results and h2_results['p_beta'] < 0.05:
                logger.info(" H2 成立：结构模型具有显著独特贡献")
            else:
                logger.info(" H2 未成立：未检测到显著独特贡献")
        except Exception as e:
            logger.error(f"H2 群体统计失败: {e}")

    logger.info("\n===== H3 群体统计 =====")
    # suffixes.append('_sequence_cv')
    for suffix in suffixes:
        try:
            h3_results = h3_analysis.run_group_level(subject_ids=subject_ids,
                result_root=str(result_root),suffix = suffix, save_fig_dir=figures_group,)
            logger.info(f"  SSI: t={h3_results['t_ssi']:.3f}, p={h3_results['p_ssi']:.4f}")
            logger.info(f"  Structure rho: t={h3_results['t_struct']:.3f}, p={h3_results['p_struct']:.4f}")
        except Exception as e:
            logger.error(f"H3 群体统计失败: {e}")

def get_valid_subject_ids(result_root='results'):
    result_root = Path(result_root)
    subject_ids = []
    for sub_dir in result_root.glob('sub-*'):
        h1_event = sub_dir / f"{sub_dir.name}_h1_rhos_event.npz"
        if h1_event.exists():
            subject_ids.append(sub_dir.name[4:])
        else:
            logger.warning(f"跳过 {sub_dir.name}：缺少 H1 event-level 结果")
    return sorted(subject_ids)


if __name__ == "__main__":
    main()
