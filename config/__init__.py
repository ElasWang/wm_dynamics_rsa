# config/__init__.py
"""
配置中心：加载 global.yaml，提供全局唯一配置对象
用法： from config import CONFIG
"""
import os
from pathlib import Path
import yaml
from typing import Dict, Any

# 获取 config 文件夹的绝对路径
CONFIG_DIR = Path(__file__).resolve().parent
# 默认配置文件路径
DEFAULT_CONFIG_PATH = CONFIG_DIR / "global_config.yaml"

# 缓存变量（防止多次读取硬盘）
_CONFIG_CACHE: Dict[str, Any] = {}


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    加载 YAML 配置文件并返回字典
    如果文件不存在，抛出异常（避免静默使用错误参数）
    """
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_config() -> Dict[str, Any]:
    """获取全局配置（单例模式，只加载一次）"""
    global _CONFIG_CACHE
    if not _CONFIG_CACHE:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE


# 导出一个默认的配置对象供各模块导入
CONFIG = get_config()


# ================ 便捷访问函数（可选，提升代码可读性） ================
def get_preproc_params() -> Dict[str, Any]:
    """返回预处理参数字典"""
    return CONFIG.get('preprocessing', {})


def get_rsa_params() -> Dict[str, Any]:
    """返回RSA参数字典"""
    return CONFIG.get('rsa', {})