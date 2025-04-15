"""
模型集成和预测平滑脚本，用于组合多个基础模型预测和平滑预测结果
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict

from src.utils.logger import get_logger
from src.models.base_model import BaseModel

logger = get_logger(__name__)

class EnsembleModel(BaseModel):
    """
    集成模型，组合多个基础模型的预测结果
    """
    def __init__(
        self,
        name: str = "ensemble",
        base_models: List[BaseModel] = None,
        weights: List[float] = None,
        ensemble_method: str = "weighted",
        target_type: str = "price_change_pct",
        prediction_horizon: int = 60,
        smoothing_window: int = 0
    ):
        """
        初始化集成模型
        
        参数:
            name: 模型名称
            base_models: 基础模型列表
            weights: 各模型权重
            ensemble_method: 集成方法 (weighted, voting, stacking)
            target_type: 目标类型
            prediction_horizon: 预测时间范围
            smoothing_window: 平滑窗口大小，0表示不平滑
        """
        super().__init__(name, target_type, prediction_horizon)
        
        self.base_models = base_models if base_models else []
        self.weights = weights if weights else self._get_equal_weights()
        self.ensemble_method = ensemble_method
        self.smoothing_window = smoothing_window
        
        # 用于stacking方法的元模型
        self.meta_model = None
        
        # 验证权重
        self._validate_weights()
        
        # 模型元数据
        self.metadata = {
            "model_type": "ensemble",
            "name": name,
            "ensemble_method": ensemble_method,
            "base_model_count": len(self.base_models) if self.base_models else 0,
            "target_type": target_type,
            "prediction_horizon": prediction_horizon,
            "smoothing_window": smoothing_window,
            "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _get_equal_weights(self) -> List[float]:
        """
        生成等权重列表
        """
        if not self.base_models:
            return []
        
        n_models = len(self.base_models)
        return [1.0 / n_models] * n_models
    
    def _validate_weights(self):
        """
        验证权重是否有效
        """
        if not self.base_models or not self.weights:
            return
        
        if len(self.weights) != len(self.base_models):
            raise ValueError(f"权重数量 ({len(self.weights)}) 必须等于模型数量 ({len(self.base_models)})")
        
        if abs(sum(self.weights) - 1.0) > 1e-6:
            logger.warning(f"权重和 ({sum(self.weights)}) 不为1，将进行归一化")
            self.weights = [w / sum(self.weights) for w in self.weights]
    
    def add_model(self, model: BaseModel, weight: float = None):
        """
        添加基础模型
        
        参数:
            model: 要添加的模型
            weight: 模型权重
        """
        self.base_models.append(model)
        
        # 更新权重
        if weight is not None:
            self.weights.append(weight)
            # 重新归一化权重
            self.weights = [w / sum(self.weights) for w in self.weights]
        else:
            # 使用等权重
            self.weights = self._get_equal_weights()
        
        self.metadata["base_model_count"] = len(self.base_models)
    
    def _smooth_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        对预测结果进行平滑处理
        
        参数:
            predictions: 预测结果数组
            
        返回:
            平滑后的预测结果
        """
        if self.smoothing_window <= 1:
            return predictions
        
        # 确保预测是一维数组
        if len(predictions.shape) > 1:
            if predictions.shape[1] == 1:
                predictions = predictions.ravel()
            else:
                # 对于多维预测，单独平滑每一列
                smoothed = np.zeros_like(predictions)
                for i in range(predictions.shape[1]):
                    smoothed[:, i] = self._smooth_predictions(predictions[:, i])
                return smoothed
        
        # 应用滑动窗口平滑
        window = min(self.smoothing_window, len(predictions))
        if window <= 1:
            return predictions
        
        # 使用pandas的滑动窗口平均
        df = pd.Series(predictions)
        smoothed = df.rolling(window=window, min_periods=1, center=False).mean().values
        
        return smoothed
    
    def _weighted_ensemble(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """
        加权集成
        
        参数:
            predictions_list: 各模型的预测结果列表
            
        返回:
            加权集成后的预测结果
        """
        # 确保所有预测形状一致
        shapes = [pred.shape for pred in predictions_list]
        if len(set(shapes)) > 1:
            raise ValueError(f"预测形状不一致: {shapes}")
        
        # 加权平均
        ensemble_pred = np.zeros_like(predictions_list[0])
        for i, pred in enumerate(predictions_list):
            ensemble_pred += pred * self.weights[i]
        
        return ensemble_pred
    
    def _voting_ensemble(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """
        投票集成（适用于分类任务）
        
        参数:
            predictions_list: 各模型的预测结果列表
            
        返回:
            投票集成后的预测结果
        """
        # 目前仅支持二元分类
        ensemble_pred = np.zeros_like(predictions_list[0])
        
        # 将预测转换为二元值 (0,1)
        binary_preds = [pred > 0.5 for pred in predictions_list]
        
        # 加权投票
        for i, pred in enumerate(binary_preds):
            ensemble_pred += pred * self.weights[i]
        
        # 转换为最终预测（多数投票）
        ensemble_pred = (ensemble_pred > 0.5).astype(float)
        
        return ensemble_pred
    
    def _stacking_ensemble(self, X: pd.DataFrame, predictions_list: List[np.ndarray]) -> np.ndarray:
        """
        堆叠集成
        
        参数:
            X: 特征数据
            predictions_list: 各模型的预测结果列表
            
        返回:
            堆叠集成后的预测结果
        """
        # 如果元模型未训练，使用加权集成
        if self.meta_model is None:
            logger.warning("元模型未训练，将使用加权集成")
            return self._weighted_ensemble(predictions_list)
        
        # 组合所有基模型的预测结果作为元模型的输入
        stacking_features = np.column_stack(predictions_list)
        
        # 使用元模型进行预测
        try:
            ensemble_pred = self.meta_model.predict(stacking_features)
            return ensemble_pred
        except Exception as e:
            logger.error(f"元模型预测失败: {str(e)}")
            return self._weighted_ensemble(predictions_list)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        训练集成模型
        
        参数:
            X: 特征数据
            y: 目标变量
            
        返回:
            训练结果信息
        """
        if not self.base_models:
            raise ValueError("没有基础模型可训练")
        
        logger.info(f"开始训练集成模型 ({self.ensemble_method})")
        
        # 训练所有基础模型
        base_model_results = []
        for i, model in enumerate(self.base_models):
            try:
                logger.info(f"训练基础模型 {i+1}/{len(self.base_models)}: {model.name}")
                result = model.train(X, y)
                base_model_results.append(result)
                logger.info(f"基础模型 {model.name} 训练完成, 训练 R²: {result.get('train_r2', 'N/A'):.4f}, "
                           f"验证 R²: {result.get('val_r2', 'N/A'):.4f}")
            except Exception as e:
                logger.error(f"基础模型 {model.name} 训练失败: {str(e)}")
                # 移除失败的模型
                self.base_models.pop(i)
                self.weights.pop(i)
                self._validate_weights()
        
        # 检查是否有足够的模型
        if not self.base_models:
            raise ValueError("所有基础模型训练失败")
        
        # 对于stacking方法，训练元模型
        if self.ensemble_method == "stacking":
            logger.info("训练stacking元模型")
            self._train_meta_model(X, y)
        
        # 生成集成预测结果
        train_predictions = self.predict(X)
        
        # 计算评估指标
        train_mse = mean_squared_error(y, train_predictions)
        train_r2 = r2_score(y, train_predictions)
        
        logger.info(f"集成模型训练完成，训练 MSE: {train_mse:.6f}, R²: {train_r2:.6f}")
        
        # 更新metadata
        self.metadata.update({
            "train_mse": train_mse,
            "train_r2": train_r2,
            "base_model_results": base_model_results,
            "training_samples": len(X),
            "feature_count": X.shape[1],
            "updated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return {
            "train_mse": train_mse,
            "train_r2": train_r2,
            "base_model_results": base_model_results
        }
    
    def _train_meta_model(self, X: pd.DataFrame, y: pd.Series):
        """
        训练stacking的元模型
        
        参数:
            X: 特征数据
            y: 目标变量
        """
        from sklearn.linear_model import Ridge
        
        # 从各基础模型获取预测
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            base_predictions.append(pred)
        
        # 组合预测作为元模型特征
        meta_features = np.column_stack(base_predictions)
        
        # 使用岭回归作为元模型
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_features, y)
        
        # 获取元模型的系数作为权重
        coeffs = self.meta_model.coef_
        
        # 归一化系数作为权重
        if np.sum(np.abs(coeffs)) > 0:
            self.weights = (np.abs(coeffs) / np.sum(np.abs(coeffs))).tolist()
        
        logger.info(f"元模型训练完成，权重更新为: {[round(w, 4) for w in self.weights]}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用集成模型进行预测
        
        参数:
            X: 特征数据
            
        返回:
            预测结果
        """
        if not self.base_models:
            raise ValueError("没有基础模型可预测")
        
        # 收集各基础模型的预测
        predictions_list = []
        for model in self.base_models:
            try:
                pred = model.predict(X)
                predictions_list.append(pred)
            except Exception as e:
                logger.error(f"模型 {model.name} 预测失败: {str(e)}")
        
        # 确保至少有一个模型预测成功
        if not predictions_list:
            raise ValueError("所有模型预测都失败了")
        
        # 根据集成方法获取集成预测
        if self.ensemble_method == "weighted":
            ensemble_pred = self._weighted_ensemble(predictions_list)
        elif self.ensemble_method == "voting":
            ensemble_pred = self._voting_ensemble(predictions_list)
        elif self.ensemble_method == "stacking":
            ensemble_pred = self._stacking_ensemble(X, predictions_list)
        else:
            # 默认使用加权集成
            ensemble_pred = self._weighted_ensemble(predictions_list)
        
        # 应用平滑
        if self.smoothing_window > 1:
            ensemble_pred = self._smooth_predictions(ensemble_pred)
        
        return ensemble_pred
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        评估模型性能
        
        参数:
            X: 特征数据
            y: 目标变量
            
        返回:
            评估指标
        """
        # 获取预测
        y_pred = self.predict(X)
        
        # 计算评估指标
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # 计算各基础模型的性能
        base_models_metrics = []
        for model in self.base_models:
            try:
                base_pred = model.predict(X)
                base_mse = mean_squared_error(y, base_pred)
                base_r2 = r2_score(y, base_pred)
                base_models_metrics.append({
                    "name": model.name,
                    "mse": base_mse,
                    "r2": base_r2
                })
            except Exception as e:
                logger.error(f"基础模型 {model.name} 评估失败: {str(e)}")
        
        # 更新模型元数据
        evaluation = {
            "mse": mse,
            "r2": r2,
            "base_models": base_models_metrics
        }
        
        # 添加到元数据
        self.metadata["evaluation"] = evaluation
        
        return evaluation
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        返回:
            Dict[str, float]: 特征名称到重要性的映射
        """
        # 合并所有基础模型的特征重要性
        feature_importance = defaultdict(float)
        
        # 对每个基础模型
        for i, model in enumerate(self.base_models):
            try:
                # 获取该模型的特征重要性
                model_importance = model.get_feature_importance()
                
                # 按权重加入总体特征重要性
                for feature, importance in model_importance.items():
                    feature_importance[feature] += importance * self.weights[i]
            except Exception as e:
                logger.warning(f"无法获取模型 {model.name} 的特征重要性: {str(e)}")
        
        # 如果没有获取到任何特征重要性，返回空字典
        if not feature_importance:
            return {}
        
        # 归一化特征重要性
        total = sum(feature_importance.values())
        if total > 0:
            normalized_importance = {f: imp/total for f, imp in feature_importance.items()}
            return normalized_importance
        
        return dict(feature_importance)
    
    def save(self, filepath: str):
        """
        保存模型
        
        参数:
            filepath: 保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        # 保存元数据
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"模型已保存至: {filepath}")
        logger.info(f"元数据已保存至: {metadata_path}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EnsembleModel':
        """
        加载模型
        
        参数:
            filepath: 模型文件路径
            
        返回:
            加载的模型
        """
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"已从 {filepath} 加载模型")
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        返回:
            Dict[str, Any]: 模型信息
        """
        return {
            "name": self.name,
            "model_type": "ensemble",
            "ensemble_method": self.ensemble_method,
            "base_models": [model.name for model in self.base_models],
            "weights": self.weights,
            "prediction_horizon": self.prediction_horizon,
            "target_type": self.target_type,
            "smoothing_window": self.smoothing_window,
            **self.metadata
        }

class ModelSmoother:
    """
    模型预测平滑器，用于平滑单个模型的预测结果
    """
    def __init__(
        self, 
        base_model: BaseModel,
        window_size: int = 5,
        smoothing_method: str = "moving_avg"
    ):
        """
        初始化平滑器
        
        参数:
            base_model: 基础模型
            window_size: 滑动窗口大小
            smoothing_method: 平滑方法
        """
        self.base_model = base_model
        self.window_size = window_size
        self.smoothing_method = smoothing_method
        
        # 验证参数
        if window_size < 2:
            logger.warning("窗口大小小于2，不会进行平滑处理")
            self.window_size = 1
    
    def smooth(self, predictions: np.ndarray) -> np.ndarray:
        """
        平滑预测结果
        
        参数:
            predictions: 原始预测结果
            
        返回:
            平滑后的预测结果
        """
        if self.window_size <= 1:
            return predictions
        
        # 确保预测是一维数组
        if len(predictions.shape) > 1:
            if predictions.shape[1] == 1:
                predictions = predictions.ravel()
            else:
                # 对于多维预测，单独平滑每一列
                smoothed = np.zeros_like(predictions)
                for i in range(predictions.shape[1]):
                    smoothed[:, i] = self.smooth(predictions[:, i])
                return smoothed
        
        # 选择平滑方法
        if self.smoothing_method == "moving_avg":
            return self._moving_average(predictions)
        elif self.smoothing_method == "ewm":
            return self._exponential_weighted_moving_average(predictions)
        else:
            logger.warning(f"未知的平滑方法: {self.smoothing_method}，使用默认的移动平均")
            return self._moving_average(predictions)
    
    def _moving_average(self, predictions: np.ndarray) -> np.ndarray:
        """
        移动平均平滑
        
        参数:
            predictions: 原始预测
            
        返回:
            平滑后的预测
        """
        window = min(self.window_size, len(predictions))
        if window <= 1:
            return predictions
        
        df = pd.Series(predictions)
        smoothed = df.rolling(window=window, min_periods=1, center=False).mean().values
        
        return smoothed
    
    def _exponential_weighted_moving_average(self, predictions: np.ndarray) -> np.ndarray:
        """
        指数加权移动平均平滑
        
        参数:
            predictions: 原始预测
            
        返回:
            平滑后的预测
        """
        alpha = 2.0 / (self.window_size + 1)  # 常用EWM参数计算方式
        
        df = pd.Series(predictions)
        smoothed = df.ewm(alpha=alpha, adjust=False).mean().values
        
        return smoothed
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测并平滑结果
        
        参数:
            X: 特征数据
            
        返回:
            平滑后的预测结果
        """
        # 获取基础模型预测
        predictions = self.base_model.predict(X)
        
        # 平滑预测
        smoothed_predictions = self.smooth(predictions)
        
        return smoothed_predictions 