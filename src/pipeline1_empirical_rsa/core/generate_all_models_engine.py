import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
import pandas as pd

from src.shared_utils.model_rdm_builder import generate_all_models
from src.shared_utils.model_rdm_builder import build_boundary_model
from src.shared_utils.permutation_utils import generate_shuffled_null_distribution

logger = logging.getLogger(__name__)

# ==================== 理论模型 RDM 缓存管理 ====================
def get_or_build_model_vectors(
    subject_id: str,
    metadata: pd.DataFrame,
    visual_similarity_root: Path,
    cache_dir: Path,
    structure_weights: list,
    distractor_collapse_levels: list,
    green_retention_weights: list,
    chunk_pattern: list,
    force: bool = False,

) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
        - 检查 results/rdms/model/ 下是否存在该被试的模型 RDM 缓存
        - 若存在且非强制重跑，直接加载
        - 若不存在或强制重跑，调用 shared_utils 构建，并保存为 .npz
        - 提取上三角向量（model_vectors）供传感器和源空间引擎复用
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"sub-{subject_id}_model_rdms.npz"

    if cache_path.exists() and not force:
        logger.info(f" 从缓存加载模型 RDM: {cache_path}")
        model_rdms = dict(np.load(cache_path, allow_pickle=True))
    else:
        logger.info(" 构建理论模型 RDM（缓存未命中或 --force）...")
        model_rdms = generate_all_models(
            metadata=metadata,
            visual_similarity_root=visual_similarity_root,
            structure_weights=structure_weights,
            distractor_collapse_levels=distractor_collapse_levels,
            green_retention_weights=green_retention_weights,
            chunk_pattern=chunk_pattern,
        )
        np.savez_compressed(cache_path, **model_rdms)
        logger.info(f" 模型 RDM 已缓存至: {cache_path}")

    n_trials = len(metadata)
    mask = np.triu_indices(n_trials, k=1)
    model_vectors = {}
    for name, rdm in model_rdms.items():
        if rdm.shape != (n_trials, n_trials):
            logger.warning(f"模型 {name} 维度 {rdm.shape} 与试次数 {n_trials} 不匹配，跳过")
            continue
        model_vectors[name] = rdm[mask].copy()

    logger.info(f" 生成 {len(model_vectors)} 个理论模型向量，每个长度 {len(mask[0])}")
    return model_vectors, mask


# ==================== 打乱基线零分布缓存管理 ====================
def get_or_generate_null_vectors(
    subject_id: str,
    metadata: pd.DataFrame,
    null_dir: Path,
    model_func: Optional[Callable] = None,
    n_permutations: int = 1000,
    force: bool = False,
) -> np.ndarray:
    """
   -获取或生成指定被试的零分布向量（缓存优先策略）
    """
    if model_func is None:
        model_func = build_boundary_model(metadata)

    null_dir.mkdir(parents=True, exist_ok=True)
    null_path = null_dir / f"sub-{subject_id}_boundary_null_vectors.npy"

    if null_path.exists() and not force:
        logger.info(f" 加载被试 {subject_id} 的零分布缓存: {null_path}")
        return np.load(null_path)


    logger.info(f" 生成被试 {subject_id} 的零分布（{n_permutations}次置换）...")

    shuffled_rdms = generate_shuffled_null_distribution(
        metadata=metadata,
        model_func=model_func,
        n_permutations=n_permutations,
    )

    mask = np.triu_indices(len(metadata), k=1)
    shuffled_vectors = np.array([rdm[mask] for rdm in shuffled_rdms])

    np.save(null_path, shuffled_vectors)
    logger.info(f" 被试 {subject_id} 零分布已保存: {null_path} (形状: {shuffled_vectors.shape})")

    return shuffled_vectors


# ==================== 便捷函数（可选）：一次性获取模型向量 + 零分布 ====================
def get_all_cached_objects(
        subject_id: str,
        metadata: pd.DataFrame,
        visual_similarity_root: Path,
        cache_dir: Path,
        null_dir: Path,
        structure_weights: list,
        distractor_collapse_levels: list,
        green_retention_weights: list,
        chunk_pattern: list,
        n_permutations: int = 1000,
        force: bool = False,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    一次性获取模型向量和零分布向量（内部顺序调用）

    返回
    -------
    model_vectors : dict
    mask : np.ndarray
    null_vectors : np.ndarray
    """
    model_vectors, mask = get_or_build_model_vectors(
        subject_id=subject_id,
        metadata=metadata,
        visual_similarity_root=visual_similarity_root,
        cache_dir=cache_dir,
        structure_weights=structure_weights,
        distractor_collapse_levels=distractor_collapse_levels,
        green_retention_weights=green_retention_weights,
        chunk_pattern=chunk_pattern,
        force=force,
    )

    null_vectors = get_or_generate_null_vectors(
        subject_id=subject_id,
        metadata=metadata,
        null_dir=null_dir,
        n_permutations=n_permutations,
        force=force,
    )

    return model_vectors, mask, null_vectors