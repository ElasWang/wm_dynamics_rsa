
from typing import Dict, Callable, Any

from .features import (
    target_priority_feature,
    gradient_feature,
    onehot_feature,
    timestamp_feature,
    content_feature,
    chunking_feature,
    binary_color_feature,
    position_scalar_feature,
)
from .rdm.visual import visual_rdm

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "target_priority_model": {
        "name": "目标优先重索引",
        "feature_extractor": target_priority_feature,
        "default_metric": "target_priority",
        "tags": ["h1","h2","h3"],
        "predicted_window": (0.2, 0.4),
        "description": "忽略干扰字母，按照黑色字母出现的时间顺序梯度赋值",
        "is_parametric": False,
        "permutable": True,
    },
    "gradient_model": {
        "name": "位置梯度模型",
        "feature_extractor": gradient_feature,
        "default_metric": "euclidean",
        "tags": ["h1"],
        "predicted_window": (0.2, 0.4),
        "description": "包括干扰字母，按照字母出现的时间顺序梯度赋值",
        "is_parametric": False,
        "permutable": True,
    },
    "binary_adjacency_model": {
        "name": "二值邻近模型",
        "feature_extractor": position_scalar_feature,
        "default_metric": "adjacency_rdm_binary",
        "tags": ["h1"],
        "predicted_window": (0.2, 0.4),
        "description": "检验大脑是否存在“邻近聚类”效应",
        "is_parametric": False,
        "permutable": True,
    },
    "reciprocal_adjacency_model": {
        "name": "倒数邻近性模型",
        "feature_extractor": position_scalar_feature,
        "default_metric": "adjacency_rdm_reciprocal",
        "tags": ["h1"],
        "predicted_window": (0.2, 0.4),
        "description": "检验大脑是否存在“连续衰减的局部性”梯度（近因效应强烈）",
        "is_parametric": False,
        "permutable": True,
    },
    "timestamp_model": {
        "name": "时间戳模型",
        "feature_extractor": timestamp_feature,
        "default_metric": "manhattan",
        "tags": ["h1", "h3"],
        "predicted_window": (0.2, 0.4),
        "description": "计算每个trail内的SOA",
        "is_parametric": False,
    },
    "serial_position_model": {
        "name": "绝对位置向量",
        "feature_extractor": onehot_feature,
        "default_metric": "euclidean",
        "tags": ["test"],
        "predicted_window": (0.2, 0.4),
        "description": "H1 的基线（阳性对照）：证明大脑至少区分了不同位置",
        "is_parametric": False,
        "permutable": True,
    },
    "visual_model": {
        "name": "视觉相似性模型",
        "rdm_builder": visual_rdm,
        "predicted_window": (0.1, 0.2),
        "tags": ["h1", "h2", "h3"],
        "description": "基于Simpson混淆矩阵的字母形状相似性,直接返回RDM",
        "is_parametric": False,
    },
    "structure_model": {
        "name": "结构-内容分离结构模型",
        "feature_extractor": onehot_feature,
        "default_metric": "euclidean",
        "tags": ["test"],
        "predicted_window": (0.2, 0.4),
        "description": "用于与内容模型（27维独热）竞争",
        "is_parametric": False,
        "permutable": True,
    },
    "content_model": {
        "name": "结构-内容分离内容模型",
        "feature_extractor": content_feature,
        "default_metric": "euclidean",
        "tags": ["h2", "h3"],
        "predicted_window": (0.2, 0.4),
        "description": "字母之间是完全离散的、非序数的类别，只表示字母身份",
        "is_parametric": True,
        "param_ranges": ['content_lambda_param'],
        "param_config_key": "model_rdm",
    },
    "binary_color_model": {
        "name": "颜色控制模型",
        "feature_extractor": binary_color_feature,
        "default_metric": "euclidean",
        "tags": ["h1", "h2", "h3"],
        "predicted_window": (0.2, 0.4),
        "description": "颜色控制模型",
        "is_parametric": False,
    },
    "chunking_model": {
        "name": "自发组块化模型",
        "feature_extractor": chunking_feature,
        "default_metric": "euclidean",
        "tags": ["h2"],
        "predicted_window": (0.2, 0.4),
        "description": "自发组块化模型",
        "is_parametric": True,
        "param_ranges": ['chunking_pattern'],
        "param_config_key": "model_rdm",
        "permutable": True,
    },
}


def get_model_builder(model_name: str) -> Callable:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"未知模型: {model_name}")
    return MODEL_REGISTRY[model_name]["builder"]


def list_models_by_hypothesis(tag: str) -> list:
    return [
        name for name, info in MODEL_REGISTRY.items()
        if tag in info.get("tags", [])
    ]
