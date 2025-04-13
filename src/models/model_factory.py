"""
模型工厂模块 - 提供统一的模型创建和加载接口
"""

import os
import json
from typing import Dict, List, Union, Optional, Any

# 导入各类模型
from models.base_model import BaseModel
from models.machine_learning_models import create_ml_model
from models.time_series_models import create_time_series_model
from models.deep_learning_models import create_deep_learning_model
from models.ensemble_models import create_ensemble_model, EnsembleModel
from utils.logger import get_logger

logger = get_logger(__name__)


def create_model(model_config: Dict) -> BaseModel:
    """
    根据配置创建模型
    
    参数:
        model_config (Dict): 模型配置字典，必须包含：
            - type: 模型类型，如"ml", "time_series", "deep_learning", "ensemble"
            - model_type: 具体模型类型，如"random_forest", "lstm", "arima"等
            - target_type: 目标类型，如"price_change_pct"或"direction"
            - model_params: 模型参数
            - name: 模型名称（可选）
            
    返回:
        BaseModel: 创建的模型实例
    """
    # 验证配置
    if "type" not in model_config:
        raise ValueError("模型配置必须包含'type'字段")
    if "model_type" not in model_config:
        raise ValueError("模型配置必须包含'model_type'字段")
    if "target_type" not in model_config:
        raise ValueError("模型配置必须包含'target_type'字段")
    
    # 获取配置参数
    model_type = model_config["type"].lower()
    specific_model_type = model_config["model_type"].lower()
    target_type = model_config["target_type"]
    model_params = model_config.get("model_params", {})
    name = model_config.get("name")
    
    # 根据模型类型创建模型
    if model_type == "ml":
        return create_ml_model(
            model_type=specific_model_type,
            target_type=target_type,
            model_params=model_params,
            name=name
        )
    elif model_type == "time_series":
        return create_time_series_model(
            model_type=specific_model_type,
            target_type=target_type,
            model_params=model_params,
            name=name
        )
    elif model_type == "deep_learning":
        return create_deep_learning_model(
            model_type=specific_model_type,
            target_type=target_type,
            model_params=model_params,
            name=name
        )
    elif model_type == "ensemble":
        # 对于集成模型，需要先创建基础模型
        if "models" not in model_config:
            raise ValueError("集成模型配置必须包含'models'字段")
        
        models = []
        for model_conf in model_config["models"]:
            models.append(create_model(model_conf))
        
        # 如果是堆叠集成，还需要创建元学习模型
        meta_model = None
        if specific_model_type == "stacking" and "meta_model" in model_config:
            meta_model = create_model(model_config["meta_model"])
        
        return create_ensemble_model(
            ensemble_type=specific_model_type,
            models=models,
            target_type=target_type,
            meta_model=meta_model,
            model_params=model_params,
            name=name
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def save_model(model: BaseModel, filepath: str) -> None:
    """
    保存模型到文件
    
    参数:
        model (BaseModel): 要保存的模型
        filepath (str): 文件路径
    """
    logger.info(f"保存模型到{filepath}...")
    model.save(filepath)
    logger.info(f"模型已保存")


def load_model(filepath: str) -> BaseModel:
    """
    从文件加载模型
    
    参数:
        filepath (str): 文件路径
        
    返回:
        BaseModel: 加载的模型实例
    """
    logger.info(f"从{filepath}加载模型...")
    
    # 加载模型元数据以确定模型类型
    with open(filepath, 'r') as f:
        model_data = json.load(f)
    
    # 获取模型类型信息
    if "model_type" in model_data:
        model_type = model_data["model_type"]
    elif "model_paths" in model_data:
        # 这是一个集成模型
        model_type = "ensemble"
    else:
        # 尝试根据文件名推断模型类型
        filename = os.path.basename(filepath).lower()
        if "ensemble" in filename:
            model_type = "ensemble"
        elif any(ts_model in filename for ts_model in ["arima", "prophet"]):
            model_type = "time_series"
        elif any(dl_model in filename for dl_model in ["lstm", "rnn", "cnn"]):
            model_type = "deep_learning"
        else:
            model_type = "ml"
    
    # 根据模型类型创建相应的模型并加载
    if model_type == "ensemble":
        # 先加载基础模型
        base_models = []
        for model_path in model_data["model_paths"]:
            base_models.append(load_model(model_path))
        
        # 加载元学习模型（如果有）
        meta_model = None
        if "meta_model_path" in model_data and model_data["meta_model_path"]:
            meta_model = load_model(model_data["meta_model_path"])
        
        # 创建集成模型
        model = EnsembleModel(
            models=base_models,
            target_type=model_data["target_type"],
            model_params=model_data["model_params"],
            name=model_data["name"]
        )
        
        # 设置元学习模型
        model.meta_model = meta_model
        
        # 设置权重
        model.weights = model_data["weights"]
        
        # 设置其他属性
        model.model_performances = model_data["model_performances"]
        model.dynamic_weights = model_data["dynamic_weights"]
        model.window_size = model_data["window_size"]
        model.recent_errors = model_data["recent_errors"]
    else:
        # 创建具体模型
        model = create_model({
            "type": model_type,
            "model_type": model_data.get("model_type", ""),
            "target_type": model_data["target_type"],
            "model_params": model_data["model_params"],
            "name": model_data["name"]
        })
        
        # 加载模型
        model.load(filepath)
    
    logger.info(f"模型已加载: {model.name}")
    return model


def get_available_model_types() -> Dict:
    """
    获取可用的模型类型
    
    返回:
        Dict: 可用的模型类型列表
    """
    return {
        "ml": [
            "random_forest",
            "xgboost",
            "linear",
            "svm",
            "knn",
            "gradient_boosting",
            "adaboost",
            "extra_trees"
        ],
        "time_series": [
            "arima",
            "prophet"
        ],
        "deep_learning": [
            "lstm",
            "rnn",
            "cnn_lstm"
        ],
        "ensemble": [
            "simple",
            "voting",
            "stacking"
        ]
    }


def get_default_model_params(model_type: str, specific_model_type: str) -> Dict:
    """
    获取模型的默认参数
    
    参数:
        model_type (str): 模型类型，如"ml", "time_series", "deep_learning", "ensemble"
        specific_model_type (str): 具体模型类型，如"random_forest", "lstm", "arima"等
        
    返回:
        Dict: 默认参数字典
    """
    model_type = model_type.lower()
    specific_model_type = specific_model_type.lower()
    
    # 创建临时模型并返回其默认参数
    if model_type == "ml":
        temp_model = create_ml_model(specific_model_type)
        return temp_model.model_params
    elif model_type == "time_series":
        temp_model = create_time_series_model(specific_model_type)
        return temp_model.model_params
    elif model_type == "deep_learning":
        temp_model = create_deep_learning_model(specific_model_type)
        return temp_model.model_params
    elif model_type == "ensemble":
        # 集成模型需要基础模型，返回通用参数
        if specific_model_type == "simple":
            return {
                "weights": None,
                "dynamic_weights": False,
                "window_size": 20
            }
        elif specific_model_type == "voting":
            return {
                "voting": "hard"
            }
        elif specific_model_type == "stacking":
            return {
                "cv": 5,
                "use_features": False
            }
    
    return {}


# 单例模式的模型工厂
class ModelFactory:
    """
    单例模式的模型工厂类
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelFactory, cls).__new__(cls)
            cls._instance._models = {}  # 存储已创建的模型
        return cls._instance
    
    def create_model(self, model_config: Dict, replace: bool = False) -> BaseModel:
        """
        创建模型并缓存
        
        参数:
            model_config (Dict): 模型配置
            replace (bool): 是否替换现有模型
            
        返回:
            BaseModel: 创建的模型
        """
        name = model_config.get("name")
        if name and name in self._models and not replace:
            logger.info(f"使用缓存中的模型: {name}")
            return self._models[name]
        
        model = create_model(model_config)
        
        if name:
            self._models[name] = model
        
        return model
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """
        获取已创建的模型
        
        参数:
            name (str): 模型名称
            
        返回:
            Optional[BaseModel]: 模型实例，如果不存在则返回None
        """
        return self._models.get(name)
    
    def list_models(self) -> List[str]:
        """
        列出所有已创建的模型
        
        返回:
            List[str]: 模型名称列表
        """
        return list(self._models.keys())
    
    def load_model(self, filepath: str, name: Optional[str] = None) -> BaseModel:
        """
        加载模型并缓存
        
        参数:
            filepath (str): 文件路径
            name (Optional[str]): 模型名称，如果不提供则使用加载的模型名称
            
        返回:
            BaseModel: 加载的模型
        """
        model = load_model(filepath)
        
        # 使用提供的名称或模型自身的名称
        model_name = name or model.name
        
        self._models[model_name] = model
        return model
    
    def remove_model(self, name: str) -> bool:
        """
        从缓存中移除模型
        
        参数:
            name (str): 模型名称
            
        返回:
            bool: 是否成功移除
        """
        if name in self._models:
            del self._models[name]
            return True
        return False


# 导出单例实例
model_factory = ModelFactory() 