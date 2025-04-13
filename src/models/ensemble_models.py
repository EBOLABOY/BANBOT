"""
模型集成策略模块 - 提供各种模型组合方法
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
from tqdm import tqdm

# 导入模型评估相关库
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleModel(BaseModel):
    """模型集成基类"""
    
    def __init__(self, 
                models: List[BaseModel],
                target_type: str = "price_change_pct",
                model_params: Dict = None,
                name: str = "Ensemble"):
        """
        初始化模型集成
        
        参数:
            models (List[BaseModel]): 模型列表
            target_type (str): 目标类型，如"price_change_pct"或"direction"
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        # 默认参数
        default_params = {
            "weights": None,  # 默认为None，表示等权重
            "meta_model": None,  # 元学习模型，用于元学习集成
            "dynamic_weights": False,  # 是否使用动态权重
        }
        
        # 合并默认参数和用户参数
        model_params = model_params or {}
        for key, value in default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        super().__init__(target_type=target_type, model_params=model_params, name=name)
        
        # 确保所有模型的目标类型一致
        for model in models:
            if model.target_type != target_type:
                raise ValueError(f"模型 {model.name} 的目标类型 ({model.target_type}) 与集成目标类型 ({target_type}) 不一致")
        
        self.models = models
        self.weights = model_params.get("weights")
        
        # 如果提供了权重，确保权重数量与模型数量一致
        if self.weights is not None and len(self.weights) != len(self.models):
            raise ValueError(f"权重数量 ({len(self.weights)}) 与模型数量 ({len(self.models)}) 不一致")
        
        # 如果没有提供权重，则使用等权重
        if self.weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        # 记录各个模型的性能
        self.model_performances = {}
        
        # 元学习模型
        self.meta_model = model_params.get("meta_model")
        
        # 动态权重
        self.dynamic_weights = model_params.get("dynamic_weights", False)
        self.window_size = model_params.get("window_size", 20)
        self.recent_errors = [[] for _ in range(len(self.models))]
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        训练集成模型（注意：此方法假设各个基础模型已经分别训练过）
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 特征数据
            y (Union[pd.Series, np.ndarray]): 目标数据
        """
        logger.info(f"开始训练{self.name}集成模型...")
        
        # 如果使用元学习模型，需要对其进行训练
        if self.meta_model is not None:
            logger.info("使用元学习模型进行集成训练...")
            
            # 收集各个基础模型的预测结果
            base_predictions = []
            for i, model in enumerate(self.models):
                logger.info(f"获取基础模型 {model.name} 的预测结果...")
                pred = model.predict(X)
                base_predictions.append(pred)
            
            # 将各个模型的预测结果组合为元学习的特征
            meta_features = np.column_stack(base_predictions)
            
            # 训练元学习模型
            logger.info("训练元学习模型...")
            self.meta_model.train(meta_features, y)
        
        # 计算各个模型的性能（可选，用于调整权重）
        if self.dynamic_weights:
            logger.info("计算各个模型的初始性能...")
            for i, model in enumerate(self.models):
                pred = model.predict(X)
                error = self._calculate_error(pred, y)
                self.model_performances[model.name] = error
            
            # 根据性能调整权重
            self._update_weights()
        
        logger.info(f"{self.name}集成模型训练完成")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用集成模型进行预测
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 特征数据
            
        返回:
            np.ndarray: 预测结果
        """
        logger.info(f"使用{self.name}集成模型进行预测...")
        
        # 收集各个基础模型的预测结果
        predictions = []
        for i, model in enumerate(self.models):
            logger.info(f"获取基础模型 {model.name} 的预测结果...")
            pred = model.predict(X)
            predictions.append(pred)
        
        # 如果使用元学习模型
        if self.meta_model is not None:
            # 将各个模型的预测结果组合为元学习的特征
            meta_features = np.column_stack(predictions)
            
            # 使用元学习模型进行最终预测
            logger.info("使用元学习模型进行最终预测...")
            final_prediction = self.meta_model.predict(meta_features)
        else:
            # 否则使用加权平均
            logger.info(f"使用加权平均进行集成，权重为: {self.weights}")
            weighted_sum = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                weighted_sum += self.weights[i] * pred
            
            final_prediction = weighted_sum
            
            # 如果是分类问题，转换为类别标签
            if self.target_type == "direction":
                final_prediction = (final_prediction > 0.5).astype(int)
        
        return final_prediction
    
    def update(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        使用新数据更新集成模型
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 新的特征数据
            y (Union[pd.Series, np.ndarray]): 新的目标数据
        """
        logger.info(f"更新{self.name}集成模型...")
        
        # 更新各个基础模型（如果支持更新）
        for model in self.models:
            if hasattr(model, 'update') and callable(getattr(model, 'update')):
                logger.info(f"更新基础模型 {model.name}...")
                model.update(X, y)
        
        # 如果使用元学习模型，也更新元学习模型
        if self.meta_model is not None and hasattr(self.meta_model, 'update') and callable(getattr(self.meta_model, 'update')):
            # 收集各个基础模型的预测结果
            base_predictions = []
            for model in self.models:
                pred = model.predict(X)
                base_predictions.append(pred)
            
            # 将各个模型的预测结果组合为元学习的特征
            meta_features = np.column_stack(base_predictions)
            
            # 更新元学习模型
            logger.info("更新元学习模型...")
            self.meta_model.update(meta_features, y)
        
        # 如果使用动态权重，更新权重
        if self.dynamic_weights:
            logger.info("更新模型权重...")
            
            # 收集新数据上各个模型的预测结果
            for i, model in enumerate(self.models):
                pred = model.predict(X)
                error = self._calculate_error(pred, y)
                
                # 更新最近误差
                self.recent_errors[i].append(error)
                if len(self.recent_errors[i]) > self.window_size:
                    self.recent_errors[i].pop(0)
                
                # 更新模型性能
                self.model_performances[model.name] = np.mean(self.recent_errors[i])
            
            # 根据性能调整权重
            self._update_weights()
        
        logger.info(f"{self.name}集成模型更新完成")
    
    def _calculate_error(self, pred: np.ndarray, y: Union[pd.Series, np.ndarray]) -> float:
        """
        计算预测误差
        
        参数:
            pred (np.ndarray): 预测值
            y (Union[pd.Series, np.ndarray]): 真实值
            
        返回:
            float: 误差值
        """
        # 确保y是numpy数组
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.values
        
        # 计算误差
        if self.target_type == "direction":
            # 对于分类问题，使用错误率
            error = 1.0 - accuracy_score(y, pred)
        else:
            # 对于回归问题，使用RMSE
            error = np.sqrt(mean_squared_error(y, pred))
        
        return error
    
    def _update_weights(self) -> None:
        """
        根据模型性能更新权重
        """
        # 获取所有模型的误差
        errors = np.array([self.model_performances[model.name] for model in self.models])
        
        # 避免除以零
        eps = 1e-10
        errors = np.maximum(errors, eps)
        
        # 计算权重：误差越小，权重越大
        inv_errors = 1.0 / errors
        self.weights = inv_errors / np.sum(inv_errors)
        
        logger.info(f"更新后的权重: {self.weights}")
    
    def save(self, filepath: str) -> None:
        """
        保存集成模型到文件
        
        参数:
            filepath (str): 文件路径
        """
        logger.info(f"保存{self.name}集成模型到{filepath}...")
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型参数和权重
        model_dir = os.path.dirname(filepath)
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        
        # 保存基础模型（如果需要）
        model_paths = []
        for i, model in enumerate(self.models):
            model_path = os.path.join(model_dir, f"{model_name}_base_{i}_{model.name}.json")
            model.save(model_path)
            model_paths.append(model_path)
        
        # 保存元学习模型（如果有）
        meta_model_path = None
        if self.meta_model is not None:
            meta_model_path = os.path.join(model_dir, f"{model_name}_meta_model.json")
            self.meta_model.save(meta_model_path)
        
        # 保存集成模型元数据
        model_data = {
            "model_params": self.model_params,
            "target_type": self.target_type,
            "name": self.name,
            "model_paths": model_paths,
            "meta_model_path": meta_model_path,
            "weights": self.weights.tolist(),
            "model_performances": self.model_performances,
            "dynamic_weights": self.dynamic_weights,
            "window_size": self.window_size,
            "recent_errors": self.recent_errors
        }
        
        # 保存元数据
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=4)
        
        logger.info(f"{self.name}集成模型已保存")
    
    def load(self, filepath: str) -> None:
        """
        从文件加载集成模型
        
        参数:
            filepath (str): 文件路径
        """
        logger.info(f"从{filepath}加载{self.name}集成模型...")
        
        # 加载模型元数据
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # 加载模型参数
        self.model_params = model_data["model_params"]
        self.target_type = model_data["target_type"]
        self.name = model_data["name"]
        
        # 加载权重
        self.weights = np.array(model_data["weights"])
        
        # 加载模型性能
        self.model_performances = model_data["model_performances"]
        
        # 加载动态权重参数
        self.dynamic_weights = model_data["dynamic_weights"]
        self.window_size = model_data["window_size"]
        self.recent_errors = model_data["recent_errors"]
        
        # 加载基础模型
        from src.models.model_factory import load_model
        self.models = []
        for model_path in model_data["model_paths"]:
            model = load_model(model_path)
            self.models.append(model)
        
        # 加载元学习模型（如果有）
        if model_data["meta_model_path"] is not None:
            self.meta_model = load_model(model_data["meta_model_path"])
        else:
            self.meta_model = None
        
        logger.info(f"{self.name}集成模型已加载")


class VotingEnsemble(EnsembleModel):
    """投票集成模型"""
    
    def __init__(self, 
                models: List[BaseModel],
                target_type: str = "direction",
                model_params: Dict = None,
                name: str = "VotingEnsemble"):
        """
        初始化投票集成模型（适用于分类问题）
        
        参数:
            models (List[BaseModel]): 模型列表
            target_type (str): 目标类型，应为"direction"
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        # 确保目标类型是方向预测
        if target_type != "direction":
            raise ValueError("投票集成模型仅适用于方向预测 (target_type='direction')")
        
        # 默认参数
        voting_default_params = {
            "voting": "hard",  # "hard"或"soft"
        }
        
        # 合并默认参数和用户参数
        model_params = model_params or {}
        for key, value in voting_default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        super().__init__(models=models, target_type=target_type, model_params=model_params, name=name)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用投票集成模型进行预测
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 特征数据
            
        返回:
            np.ndarray: 预测结果
        """
        logger.info(f"使用{self.name}投票集成模型进行预测...")
        
        # 收集各个基础模型的预测结果
        predictions = []
        for i, model in enumerate(self.models):
            logger.info(f"获取基础模型 {model.name} 的预测结果...")
            pred = model.predict(X)
            predictions.append(pred)
        
        # 转换为numpy数组
        predictions = [pred.flatten() if isinstance(pred, np.ndarray) else pred for pred in predictions]
        predictions = np.array(predictions)
        
        # 根据投票类型进行集成
        voting_type = self.model_params.get("voting", "hard")
        
        if voting_type == "hard":
            # 硬投票：少数服从多数
            final_prediction = np.zeros(predictions.shape[1])
            
            for i in range(predictions.shape[1]):
                votes = predictions[:, i]
                # 计算多数类别
                unique_values, counts = np.unique(votes, return_counts=True)
                most_common_idx = np.argmax(counts)
                final_prediction[i] = unique_values[most_common_idx]
        else:
            # 软投票：加权平均概率
            weighted_sum = np.zeros(predictions.shape[1])
            for i, pred in enumerate(predictions):
                weighted_sum += self.weights[i] * pred
            
            # 转换为类别标签
            final_prediction = (weighted_sum > 0.5).astype(int)
        
        return final_prediction


class StackingEnsemble(EnsembleModel):
    """堆叠集成模型"""
    
    def __init__(self, 
                models: List[BaseModel],
                meta_model: BaseModel,
                target_type: str = "price_change_pct",
                model_params: Dict = None,
                name: str = "StackingEnsemble"):
        """
        初始化堆叠集成模型
        
        参数:
            models (List[BaseModel]): 基础模型列表
            meta_model (BaseModel): 元学习模型
            target_type (str): 目标类型
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        # 默认参数
        stacking_default_params = {
            "cv": 5,  # 交叉验证折数
            "use_features": False,  # 是否将原始特征与模型预测结合
        }
        
        # 合并默认参数和用户参数
        model_params = model_params or {}
        for key, value in stacking_default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        # 添加元学习模型
        model_params["meta_model"] = meta_model
        
        super().__init__(models=models, target_type=target_type, model_params=model_params, name=name)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        训练堆叠集成模型
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 特征数据
            y (Union[pd.Series, np.ndarray]): 目标数据
        """
        logger.info(f"开始训练{self.name}堆叠集成模型...")
        
        # 获取交叉验证折数
        cv = self.model_params.get("cv", 5)
        
        # 将数据划分为cv折
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # 收集各个基础模型的预测结果
        base_predictions = []
        
        # 对每个模型进行交叉验证预测
        for i, model in enumerate(self.models):
            logger.info(f"为基础模型 {model.name} 进行交叉验证预测...")
            
            # 初始化预测结果数组
            if isinstance(X, pd.DataFrame):
                pred = np.zeros(X.shape[0])
            else:
                pred = np.zeros(len(X))
            
            # 对每一折进行训练和预测
            for train_idx, val_idx in kf.split(X):
                # 获取训练集和验证集
                if isinstance(X, pd.DataFrame):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train = y.iloc[train_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else y[train_idx]
                else:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y[train_idx] if isinstance(y, (pd.Series, pd.DataFrame)) else y[train_idx]
                
                # 训练模型
                model.train(X_train, y_train)
                
                # 预测验证集
                val_pred = model.predict(X_val)
                
                # 将预测结果存入对应位置
                pred[val_idx] = val_pred.flatten()
            
            # 保存模型的交叉验证预测结果
            base_predictions.append(pred)
            
            # 使用全部数据重新训练模型
            logger.info(f"使用全部数据重新训练基础模型 {model.name}...")
            model.train(X, y)
        
        # 将各个模型的预测结果组合为元学习的特征
        meta_features = np.column_stack(base_predictions)
        
        # 如果需要使用原始特征
        if self.model_params.get("use_features", False):
            logger.info("结合原始特征与模型预测结果...")
            if isinstance(X, pd.DataFrame):
                meta_features = np.column_stack([meta_features, X.values])
            else:
                meta_features = np.column_stack([meta_features, X])
        
        # 训练元学习模型
        logger.info("训练元学习模型...")
        self.meta_model.train(meta_features, y)
        
        logger.info(f"{self.name}堆叠集成模型训练完成")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用堆叠集成模型进行预测
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 特征数据
            
        返回:
            np.ndarray: 预测结果
        """
        logger.info(f"使用{self.name}堆叠集成模型进行预测...")
        
        # 收集各个基础模型的预测结果
        base_predictions = []
        for i, model in enumerate(self.models):
            logger.info(f"获取基础模型 {model.name} 的预测结果...")
            pred = model.predict(X)
            base_predictions.append(pred.flatten())
        
        # 将各个模型的预测结果组合为元学习的特征
        meta_features = np.column_stack(base_predictions)
        
        # 如果需要使用原始特征
        if self.model_params.get("use_features", False):
            logger.info("结合原始特征与模型预测结果...")
            if isinstance(X, pd.DataFrame):
                meta_features = np.column_stack([meta_features, X.values])
            else:
                meta_features = np.column_stack([meta_features, X])
        
        # 使用元学习模型进行预测
        logger.info("使用元学习模型进行最终预测...")
        final_prediction = self.meta_model.predict(meta_features)
        
        return final_prediction


# 工厂函数
def create_ensemble_model(ensemble_type: str, models: List[BaseModel], target_type: str = "price_change_pct", meta_model: Optional[BaseModel] = None, model_params: Dict = None, name: str = None) -> BaseModel:
    """
    创建集成模型
    
    参数:
        ensemble_type (str): 集成类型，如"simple", "voting", "stacking"
        models (List[BaseModel]): 基础模型列表
        target_type (str): 目标类型
        meta_model (Optional[BaseModel]): 元学习模型，用于堆叠集成
        model_params (Dict): 模型参数
        name (str): 模型名称
        
    返回:
        BaseModel: 集成模型实例
    """
    ensemble_type = ensemble_type.lower()
    
    if ensemble_type == "simple":
        name = name or "SimpleEnsemble"
        return EnsembleModel(models=models, target_type=target_type, model_params=model_params, name=name)
    elif ensemble_type == "voting":
        name = name or "VotingEnsemble"
        return VotingEnsemble(models=models, target_type=target_type, model_params=model_params, name=name)
    elif ensemble_type == "stacking":
        if meta_model is None:
            raise ValueError("堆叠集成需要提供元学习模型 (meta_model)")
        name = name or "StackingEnsemble"
        return StackingEnsemble(models=models, meta_model=meta_model, target_type=target_type, model_params=model_params, name=name)
    else:
        raise ValueError(f"不支持的集成类型: {ensemble_type}") 