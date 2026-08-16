#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型RDM构建器
根据设计的10个认知模型，构建对应的表征不相似性矩阵（RDM）
"""
import logging
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from src.shared_utils.simpson_rdm_util import load_simpson_rdm


logger = logging.getLogger(__name__)


# ==================== 构建表征相异度矩阵 ====================

def _features_to_rdm(features: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    将特征矩阵转换为 RDM，并进行 [0,1] 归一化
    """

    if features.shape[1] == 1 and metric == 'correlation':
        metric = 'euclidean'
        logger.debug("特征维度为1，自动切换至欧氏距离")
    if metric == 'correlation':
        rdm = 1 - np.corrcoef(features)
        rdm = np.nan_to_num(rdm, nan=0.0)
    else:
        rdm = squareform(pdist(features, metric=metric))
    if rdm.max() > 0:
        rdm = rdm / rdm.max()
    np.fill_diagonal(rdm, 0)
    return rdm


# ==================== 构建特征矩阵 ====================

def build_null_model(metadata: pd.DataFrame) -> np.ndarray:
    """
    M1: 空模型
    -所有特征恒为0，检验数据是否干净。
    """
    n = len(metadata)
    features = np.zeros((n, 1))
    return _features_to_rdm(features, metric='euclidean')


def build_visual_model(visual_similarity_root: Path, metadata: pd.DataFrame) -> np.ndarray:
    """
    M2: 视觉相似性模型 (基于 Simpson 混淆矩阵)
    -根据 metadata 中的字母，构建 NxN RDM。
    """
    full_rdm = load_simpson_rdm(visual_similarity_root)
    letters = metadata['letter'].str.upper().values
    idx = np.array([ord(l) - ord('A') for l in letters], dtype=int)
    n = len(metadata)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = full_rdm[idx[i], idx[j]]

    return rdm


def build_boundary_model(metadata: pd.DataFrame) -> np.ndarray:
    """
    M3: 边界强化模型
    -每个试次的首尾位置标记为1，其余为0。
    """
    df = metadata.sort_values(['run', 'trial', 'onset']).copy()
    n = len(df)
    feat = np.zeros(n, dtype=float)
    for _, group in df.groupby(['run', 'trial']):
        pos = group['position'].values
        idx = group.index
        if len(pos) == 0:
            continue
        min_pos = pos.min()
        max_pos = pos.max()
        mask = (pos == min_pos) | (pos == max_pos)
        feat[idx[mask]] = 1.0
    return _features_to_rdm(feat.reshape(-1, 1), metric='euclidean')

def build_gradient_model(metadata: pd.DataFrame) -> np.ndarray:
    """
    M4: 位置梯度模型
    -在每个试次内部独立归一化，避免跨试次长度差异带来的尺度偏差。
    - 位置 1 → 0（开头）
    - 位置 N → 1（结尾，N 为该试次总字母数）
    - 中间位置线性插值
    """
    df = metadata.sort_values(['run', 'trial', 'onset']).copy()
    n = len(df)
    feat = np.zeros(n, dtype=float)
    for _, group in df.groupby(['run', 'trial']):
        idx = group.index
        if len(idx) == 0:
            continue
        pos = group['position'].values
        max_pos = pos.max()
        if max_pos > 1:
            feat[idx] = (pos - 1) / (max_pos - 1)
        else:
            feat[idx] = 0.0

    return _features_to_rdm(feat.reshape(-1, 1), metric='euclidean')


def build_parametric_color_position_rdm(metadata: pd.DataFrame,
    alpha: float) -> np.ndarray:
    """
    M5:参数化颜色-位置权衡模型
    对每个试次构造 2 维特征向量:
        - 黑色字母: [1.0, position_norm]
        - 绿色字母: [alpha, 0.0]
    """
    positions = metadata['position'].copy()
    max_pos = positions.max() if positions.max() > 0 else 1
    pos_norm = (positions - 1) / (max_pos - 1) if max_pos > 1 else positions.astype(float)
    colors = metadata['color'].map({'black': 1.0, 'green': 0.0}).fillna(0.0).values
    n = len(metadata)
    features = np.zeros((n, 2))
    for i in range(n):
        if colors[i] == 1.0:
            features[i, :] = [1.0, pos_norm[i]]
        else:
            features[i, :] = [alpha, 0.0]
    rdm = _features_to_rdm(features, metric='euclidean')
    return rdm


def build_parametric_color_position_rdms(
        metadata: pd.DataFrame,
        green_retention_weights: Optional[List[float]] = None
) -> Dict[str, np.ndarray]:
    """
    M5:参数化颜色-位置权衡模型
    -生成多个 alpha 下的参数化颜色-位置模型 RDM
    """
    if green_retention_weights is None:
        green_retention_weights = np.linspace(0, 1, 11).tolist()
    rdms = {}
    for alpha in green_retention_weights:
        key = f'M5_parametric_color_pos_alpha_{alpha:.2f}'
        rdms[key] = build_parametric_color_position_rdm(metadata, alpha)
    return rdms


def build_chunking_model(metadata: pd.DataFrame, chunk_pattern: Optional[List[int]] =  [2, 2, 3, 3]) -> np.ndarray:
    """
    M6: 自发组块化模型
    -黑色按配置文件中的chunking组块规则分组赋值（块内相同）
    -绿色=-1（独立废弃标签）。
    -归一化
    """
    df = metadata.sort_values(['run', 'trial',  'position']).copy()
    n = len(df)
    feat = np.zeros(n, dtype=float)


    for _, group in df.groupby(['run', 'trial']):
        idx = group.index
        if len(idx) == 0:
            continue
        colors = group['color'].values
        black_indices = [i for i, c in enumerate(colors) if c == 'black']
        for i, c in enumerate(colors):
            if c != 'black':
                feat[idx[i]] = -1.0
        n_black = len(black_indices)
        if n_black == 0:
            continue
        group_id = 1
        count_in_group = 0
        pattern_iter = iter(chunk_pattern)
        current_capacity = next(pattern_iter, None)

        for j, original_idx in enumerate(black_indices):
            if current_capacity is None:
                feat[idx[original_idx]] = group_id
                continue
            if count_in_group >= current_capacity:
                try:
                    current_capacity = next(pattern_iter)
                    group_id += 1
                    count_in_group = 0
                except StopIteration:
                    current_capacity = None
                    group_id += 1
            feat[idx[original_idx]] = group_id
            count_in_group += 1

    return _features_to_rdm(feat.reshape(-1, 1), metric='euclidean')



def build_structure_model(metadata: pd.DataFrame) -> np.ndarray:
    """
    M7: 结构分离模型（结构RDM）
    -仅使用位置梯度（同M4），作为结构系统的锚点。
    """
    return build_gradient_model(metadata)


def build_content_model(
        metadata: pd.DataFrame,
        lambda_param: float = 1.0
) -> np.ndarray:
    """
    M7: 内容分离模型（内容RDM）
    -26维字母独热 + 第27维干扰项
    """
    n = len(metadata)
    features = np.zeros((n, 27))
    for i, row in metadata.iterrows():
        if row['color'] == 'black':
            idx = ord(row['letter'].upper()) - ord('A')
            if 0 <= idx < 26:
                features[i, idx] = 1.0
        else:
            idx = ord(row['letter'].upper()) - ord('A')
            if 0 <= idx < 26:
                features[i, idx] = 1.0 - lambda_param
            features[i, 26] = lambda_param
    return _features_to_rdm(features, metric='correlation')


def build_dynamic_mixture_model(
        metadata: pd.DataFrame,
        alpha: float,
        lambda_param: float
) -> np.ndarray:
    """
    M8: 动态加权混合模型
    -混合RDM = α * 结构RDM (M7: 结构分离模型) + (1-α) * 内容RDM (M7: 内容分离模型)
    """
    structure_rdm = build_structure_model(metadata)
    content_rdm = build_content_model(metadata, lambda_param)
    mixed_rdm = alpha * structure_rdm + (1 - alpha) * content_rdm
    np.fill_diagonal(mixed_rdm, 0)
    if mixed_rdm.max() > 0:
        mixed_rdm = mixed_rdm / mixed_rdm.max()
    return mixed_rdm


# ==================== 批量生成模型RDM====================

def generate_all_models(
        metadata: pd.DataFrame,
        visual_similarity_root: Path,
        structure_weights: List[float] = None,
        distractor_collapse_levels: List[float] = None,
        green_retention_weights: List[float] = None ,
        chunk_pattern: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:

    if structure_weights is None:
        structure_weights =  [0.0, 0.3, 0.5, 0.7, 1.0]
    if distractor_collapse_levels is None:
        distractor_collapse_levels = [0.0, 0.5, 1.0]
    if green_retention_weights is None:
        green_retention_weights = np.linspace(0, 1, 11).tolist()

    rdms = {}
    # M1
    rdms['M1_null'] = build_null_model(metadata)
    # M2
    rdms['M2_visual'] = build_visual_model(visual_similarity_root, metadata)
    # M3
    rdms['M3_boundary'] = build_boundary_model(metadata)
    # M4
    rdms['M4_gradient'] = build_gradient_model(metadata)
    # M5 key为parametric_color_pos_alpha_{alpha:.2f}
    param_rdms = build_parametric_color_position_rdms(metadata, green_retention_weights)
    rdms.update(param_rdms)
    # M6
    rdms['M6_chunking'] = build_chunking_model(metadata, chunk_pattern)
    # M7
    rdms['M7_structure'] = build_structure_model(metadata)
    rdms['M7_content'] = build_content_model(metadata, 1.0)
    # M8
    for alpha in structure_weights:
        for lam in distractor_collapse_levels:
            rdms[f'M8_mixture_alpha_{alpha:.1f}_lambda_{lam:.1f}'] = build_dynamic_mixture_model(
                metadata, alpha, lam
            )
    return rdms

