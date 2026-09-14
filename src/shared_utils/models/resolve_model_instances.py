

import itertools
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import get_path_params, CONFIG
from .rdm.aggregate_rdm_to_condition_level import aggregate_rdm_to_condition_level
from .rdm.compute_rdm import features_to_rdm
from .registry import MODEL_REGISTRY, list_models_by_hypothesis

logger = logging.getLogger(__name__)

paths_cfg = get_path_params()


def resolve_model_instances(
        subject_id: str,
        hypothesis: Optional[str] = None,
        force: bool = False,
) -> Dict[str, np.ndarray]:
    cache_dir = Path(paths_cfg.get('model_rdm_root', 'results/rdms/model'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"sub-{subject_id}_model_rdms.npz"
    if cache_path.exists() and not force:
        logger.info(f" 从缓存加载模型 RDM: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        all_instances = {key: data[key] for key in data.files}
        instances = filter_instances_by_hypothesis(all_instances, hypothesis)
        return instances
    from src.shared_utils.metadata_util import compute_positions
    deriv_root = Path(paths_cfg.get('deriv_root', 'data/derivatives'))
    logger.info(" 构建理论模型 RDM（缓存未命中或 --force）...")
    metadata = pd.read_csv(f"{deriv_root}/sub-{subject_id}_metadata.csv")
    metadata = metadata[metadata['task_role'].isin(['to_remember', 'to_ignore'])]
    metadata = compute_positions(metadata)
    metadata = metadata.reset_index(drop=True)
    if hypothesis:
        model_names = list_models_by_hypothesis(hypothesis)
    else:
        model_names = list(MODEL_REGISTRY.keys())
    instances = {}
    for model_name in model_names:
        model_cfg = MODEL_REGISTRY[model_name]
        is_parametric = model_cfg.get("is_parametric", False)
        if model_name == "visual_model":
            extractor = model_cfg["rdm_builder"]
            visual_similarity_root = Path(paths_cfg.get('visual_similarity_root', 'data/visual_similarity'))
            rdm = extractor(metadata, visual_similarity_root=visual_similarity_root)
            instances[model_name] = rdm
        elif not is_parametric:
            extractor = model_cfg["feature_extractor"]
            metric = model_cfg["default_metric"]
            features = extractor(metadata)
            extra_kwargs = {}
            if metric == 'target_priority':
                extra_kwargs['trial_loads'] = metadata['load'].values
            rdm = features_to_rdm(features, metric=metric, normalize=True, **extra_kwargs)
            instances[model_name] = rdm
            logger.debug(f"  生成模型: {model_name}")
        else:
            if "rdm_builder" in model_cfg:
                builder = model_cfg["rdm_builder"]
                keys, values = _get_params(model_cfg)
                for combo in itertools.product(*values):
                    kwargs = dict(zip(keys, combo))
                    key = model_name + "_" + "_".join([f"{k}_{v:.2f}" for k, v in kwargs.items()])
                    rdm = builder(metadata, **kwargs)
                    instances[key] = rdm
                    logger.debug(f"  生成模型实例: {key}")
            elif "feature_extractor" in model_cfg:
                extractor = model_cfg["feature_extractor"]
                metric = model_cfg["default_metric"]
                keys, values = _get_params(model_cfg)
                for combo in itertools.product(*values):
                    kwargs = dict(zip(keys, combo))
                    features = extractor(metadata, **kwargs)
                    rdm = features_to_rdm(features, metric=metric, normalize=True)
                    key = model_name + "_" + "_".join([f"{k}_{v}" for k, v in kwargs.items()])
                    instances[key] = rdm
                    logger.debug(f"  生成模型实例: {key}")
    np.savez_compressed(cache_path, **instances)
    logger.info(f" 模型 RDM 已缓存至: {cache_path} (共 {len(instances)} 个实例)")
    return instances


def resolve_condition_model_instances(
        subject_id: str,
        hypothesis: Optional[str] = None,
        force: bool = False,
        condition_col: str = 'position'
) -> Dict[str, np.ndarray]:
    paths_cfg = get_path_params()
    cache_dir = Path(paths_cfg.get('model_rdm_root', 'results/rdms/model'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"sub-{subject_id}_model_rdms_cond.npz"
    if cache_path.exists() and not force:
        logger.info(f" 从缓存加载条件级模型 RDM: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        all_instances = {key: data[key] for key in data.files}
        return filter_instances_by_hypothesis(all_instances, hypothesis)
    event_instances = resolve_model_instances(subject_id, hypothesis=hypothesis, force=force)
    deriv_root = Path(paths_cfg.get('deriv_root', 'data/derivatives'))
    metadata = pd.read_csv(deriv_root / f'sub-{subject_id}_metadata.csv')
    metadata = metadata[metadata['task_role'].isin(['to_remember', 'to_ignore'])]
    if 'position' not in metadata.columns or (metadata['position'] == 0).all():
        from src.shared_utils.metadata_util import compute_positions
        metadata = compute_positions(metadata)
    metadata = metadata.reset_index(drop=True)
    cond_instances = {}
    for name, rdm_event in event_instances.items():
        cond_rdm = aggregate_rdm_to_condition_level(
            rdm_event, metadata, condition_col=condition_col
        )
        cond_instances[name] = cond_rdm
    np.savez_compressed(cache_path, **cond_instances)
    logger.info(f" 条件级模型 RDM 已缓存至: {cache_path} (共 {len(cond_instances)} 个模型)")
    return cond_instances


def _get_params(model_cfg):
    param_names = model_cfg.get("param_ranges", [])
    param_cfg = CONFIG.get(model_cfg.get("param_config_key", {}))
    param_values = {}
    for p in param_names:
        val = param_cfg.get(f"{p}_values")
        if val is None:
            val = param_cfg.get(p)
        if val is None:
            raise KeyError(f"在配置 {model_cfg.get('param_config_key')} 中找不到参数 {p} 或 {p}_values")
        if not isinstance(val, list):
            val = [val]
        param_values[p] = val
    keys = param_names
    values = [param_values[k] for k in keys]
    return keys, values


def filter_instances_by_hypothesis(
        all_instances: Dict[str, np.ndarray],
        hypothesis: Optional[str] = None
) -> Dict[str, np.ndarray]:
    if hypothesis is None:
        logger.info(f"  加载了 {len(all_instances)} 个模型实例")
        return all_instances
    target_names = list_models_by_hypothesis(hypothesis)
    filtered = {}
    for key, rdm in all_instances.items():
        matched = any(key.startswith(name + '_') or key == name for name in target_names)
        if matched:
            filtered[key] = rdm
    logger.info(f"  按 hypothesis='{hypothesis}' 过滤，保留 {len(filtered)} 个模型实例")
    return filtered
