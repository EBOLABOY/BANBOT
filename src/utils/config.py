"""
配置文件加载与处理
"""

import os
import yaml
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """
    加载配置文件
    
    参数:
        config_path: 配置文件路径
        
    返回:
        配置字典
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 替换环境变量
        config_str = yaml.dump(config)
        for key, value in os.environ.items():
            placeholder = f"${{{key}}}"
            config_str = config_str.replace(placeholder, value)
        
        config = yaml.safe_load(config_str)
        logger.info(f"配置文件 {config_path} 已加载")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

def get_config_value(config, path, default=None):
    """
    从嵌套配置字典中获取值
    
    参数:
        config: 配置字典
        path: 以点分隔的键路径，如 "data.storage.host"
        default: 默认值，如果路径不存在
        
    返回:
        配置值或默认值
    """
    keys = path.split(".")
    result = config
    
    try:
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError):
        logger.warning(f"配置路径 {path} 不存在，使用默认值 {default}")
        return default 