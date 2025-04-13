"""
模型基类模块 - 定义所有模型的公共接口
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple, Optional, Any

from utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """模型基类，定义模型的公共接口"""
    
    def __init__(self, 
                target_type: str = "price_change_pct",
                model_params: Dict = None,
                name: str = "BaseModel"):
        """
        初始化模型基类
        
        参数:
            target_type (str): 目标类型，如"price_change_pct"或"direction"
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        self.target_type = target_type
        self.model_params = model_params or {}
        self.name = name
    
    @abstractmethod
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        训练模型
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 特征数据
            y (Union[pd.Series, np.ndarray]): 目标数据
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 特征数据
            
        返回:
            np.ndarray: 预测结果
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        保存模型到文件
        
        参数:
            filepath (str): 文件路径
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        从文件加载模型
        
        参数:
            filepath (str): 文件路径
        """
        pass
    
    def update(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        使用新数据更新模型（默认实现是重新训练）
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 新的特征数据
            y (Union[pd.Series, np.ndarray]): 新的目标数据
        """
        self.train(X, y)
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict:
        """
        评估模型性能
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 特征数据
            y (Union[pd.Series, np.ndarray]): 目标数据
            
        返回:
            Dict: 评估指标
        """
        # 基本实现，子类可以重写
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = self.predict(X)
        
        # 根据目标类型计算指标
        metrics = {}
        if self.target_type == "direction":
            # 分类指标
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['precision'] = precision_score(y, y_pred, average='binary')
            metrics['recall'] = recall_score(y, y_pred, average='binary')
            metrics['f1'] = f1_score(y, y_pred, average='binary')
        else:
            # 回归指标
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y, y_pred)
            metrics['r2'] = r2_score(y, y_pred)
        
        return metrics
    
    def get_params(self) -> Dict:
        """
        获取模型参数
        
        返回:
            Dict: 模型参数
        """
        return self.model_params.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """
        设置模型参数
        
        参数:
            **params: 参数键值对
            
        返回:
            BaseModel: 模型实例
        """
        for key, value in params.items():
            self.model_params[key] = value
        return self
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预测概率（分类模型使用）
        
        参数:
            X: 预测特征数据
            
        返回:
            np.ndarray: 预测概率 
        """
        raise NotImplementedError("此模型不支持概率预测")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性（如果模型支持）
        
        返回:
            Optional[Dict[str, float]]: 特征重要性字典，如果模型不支持则返回None
        """
        if not self.trained:
            logger.warning("模型尚未训练，无法获取特征重要性")
            return None
            
        # 默认实现 - 子类可以覆盖此方法
        importance = None
        try:
            # 尝试访问不同的特征重要性属性
            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                importance = np.abs(self.model.coef_)
                if len(importance.shape) > 1:
                    importance = np.mean(importance, axis=0)
        except (AttributeError, Exception) as e:
            logger.warning(f"无法获取特征重要性: {str(e)}")
            return None
            
        if importance is not None and self.feature_names is not None:
            # 返回特征名称和重要性的字典
            importance_dict = dict(zip(self.feature_names, importance))
            # 按重要性降序排序
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        return None
    
    def save(self, filepath: str = None, include_model: bool = True) -> str:
        """
        保存模型到指定路径
        
        参数:
            filepath (str): 保存路径。如果为None，则自动生成路径
            include_model (bool): 是否包含模型对象。如果为False，只保存元数据
            
        返回:
            str: 保存的文件路径
        """
        if not self.trained and include_model:
            logger.warning("模型尚未训练，无法保存模型对象")
            include_model = False
            
        # 如果未指定文件路径，则自动生成
        if filepath is None:
            # 创建目录（如果不存在）
            os.makedirs(self.model_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            horizon = self.prediction_horizon
            target = self.target_type.split("_")[0]  # 简化目标类型名称
            filename = f"{self.name}_{target}_h{horizon}_{timestamp}.pkl"
            
            filepath = os.path.join(self.model_dir, filename)
            
        # 更新元数据中的保存时间
        self.metadata["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存对象
        if include_model:
            with open(filepath, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "metadata": self.metadata,
                    "model_params": self.model_params,
                    "trained": self.trained,
                    "class_name": self.__class__.__name__
                }, f)
            logger.info(f"模型已保存至 {filepath}")
        else:
            # 仅保存元数据
            metadata_path = filepath.replace(".pkl", "_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"模型元数据已保存至 {metadata_path}")
            return metadata_path
            
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        从文件加载模型
        
        参数:
            filepath (str): 模型文件路径
            
        返回:
            BaseModel: 加载的模型实例
        """
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                
            # 获取模型类
            class_name = data.get("class_name", cls.__name__)
            
            # 导入模型模块
            import sys
            import importlib
            
            # 查找模型类
            if class_name != cls.__name__:
                # 尝试导入特定的模型类
                try:
                    module = importlib.import_module("models")
                    model_class = getattr(module, class_name)
                except (ImportError, AttributeError):
                    try:
                        # 尝试从子模块导入
                        module = importlib.import_module("models.traditional_models")
                        model_class = getattr(module, class_name)
                    except (ImportError, AttributeError):
                        logger.warning(f"找不到类 {class_name}，使用基类加载")
                        model_class = cls
            else:
                model_class = cls
                
            # 创建模型实例
            metadata = data.get("metadata", {})
            model_instance = model_class(
                name=filepath.split("/")[-1].split("_")[0],
                model_params=data.get("model_params", {}),
                prediction_horizon=metadata.get("prediction_horizon", 1),
                target_type=metadata.get("target_type", "price_change_pct")
            )
            
            # 恢复模型状态
            model_instance.model = data.get("model")
            model_instance.feature_names = data.get("feature_names")
            model_instance.metadata = metadata
            model_instance.trained = data.get("trained", False)
            
            logger.info(f"已从 {filepath} 加载模型")
            return model_instance
            
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise
            
    def __str__(self) -> str:
        """模型的字符串表示"""
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"horizon={self.prediction_horizon}, "
                f"target='{self.target_type}', "
                f"trained={self.trained})")
    
    def summary(self) -> str:
        """返回模型摘要信息"""
        summary_lines = [
            f"模型类型: {self.__class__.__name__}",
            f"模型名称: {self.name}",
            f"预测周期: {self.prediction_horizon}",
            f"目标类型: {self.target_type}",
            f"已训练: {self.trained}"
        ]
        
        # 添加参数信息
        if self.model_params:
            summary_lines.append("模型参数:")
            for key, value in self.model_params.items():
                summary_lines.append(f"  - {key}: {value}")
        
        # 添加评估指标
        metrics = self.metadata.get("metrics", {})
        if metrics:
            summary_lines.append("评估指标:")
            for metric, value in metrics.items():
                summary_lines.append(f"  - {metric}: {value:.4f}")
        
        # 添加特征信息
        if self.feature_names:
            feature_count = len(self.feature_names)
            summary_lines.append(f"特征数量: {feature_count}")
            
            # 添加特征重要性（如果可用）
            importance = self.get_feature_importance()
            if importance:
                summary_lines.append("前5个重要特征:")
                for i, (feature, imp) in enumerate(list(importance.items())[:5]):
                    summary_lines.append(f"  {i+1}. {feature}: {imp:.4f}")
        
        return "\n".join(summary_lines) 