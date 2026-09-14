# config/__init__.py

import os
from pathlib import Path
import yaml
from typing import Dict, Any

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "global_config.yaml"

_CONFIG_CACHE: Dict[str, Any] = {}


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if not _CONFIG_CACHE:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE


CONFIG = get_config()


def get_preproc_params() -> Dict[str, Any]:
    return CONFIG.get('preprocessing', {})


def get_rsa_params() -> Dict[str, Any]:
    return CONFIG.get('rsa', {})

def get_path_params() -> Dict[str, Any]:
    return CONFIG.get('paths', {})