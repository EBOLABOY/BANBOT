"""
日志配置模块
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from .config import load_config, get_config_value

def setup_logging(config_path="config.yaml", default_level=logging.INFO):
    """
    设置日志配置
    
    参数:
        config_path: 配置文件路径
        default_level: 默认日志级别
    """
    try:
        config = load_config(config_path)
        log_config = get_config_value(config, "system.logging", {})
        
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "logs/crypto_prediction.log")
        log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 配置根日志记录器
        handlers = [
            # 控制台处理器
            logging.StreamHandler(sys.stdout),
            # 文件处理器
            RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding="utf-8"
            )
        ]
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers
        )
        
        # 设置第三方库的日志级别
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("tensorflow").setLevel(logging.WARNING)
        
        logging.info("日志系统已初始化")
    except Exception as e:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        logging.error(f"初始化日志系统时出错: {e}")

def get_logger(name):
    """
    获取命名的日志记录器
    
    参数:
        name: 日志记录器名称
        
    返回:
        Logger对象
    """
    return logging.getLogger(name) 