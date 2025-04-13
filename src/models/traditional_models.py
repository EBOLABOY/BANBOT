"""
传统机器学习模型 - 包括线性模型和树模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Any
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LinearModel(BaseModel):
    """
    线性模型类，支持Ridge和Lasso回归
    """
    
    def __init__(self, 
                 name: str = "linear",
                 model_params: Dict = None,
                 prediction_horizon: int = 1,
                 target_type: str = "price_change_pct",
                 model_type: str = "ridge",
                 model_dir: str = "models/saved_models"):
        """
        初始化线性模型
        
        参数:
            name (str): 模型名称
            model_params (Dict): 模型参数字典
            prediction_horizon (int): 预测周期（步数）
            target_type (str): 目标变量类型
            model_type (str): 线性模型类型，可选 'ridge' 或 'lasso'
            model_dir (str): 模型保存目录
        """
        super().__init__(name, model_params, prediction_horizon, target_type, model_dir)
        
        self.model_type = model_type
        
        # 设置默认参数
        default_params = {
            "alpha": 1.0, 
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 1e-3,
            "random_state": 42
        }
        
        # 更新参数
        if model_params:
            default_params.update(model_params)
        
        self.model_params = default_params
        self.metadata["model_subtype"] = model_type
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], 
              validation_data: Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]] = None) -> Dict:
        """
        训练线性模型
        
        参数:
            X: 训练特征数据
            y: 训练目标数据
            validation_data: 可选的验证数据 (X_val, y_val)
            
        返回:
            Dict: 包含训练指标的字典
        """
        # 保存特征名称（如果可用）
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # 创建模型
        if self.model_type.lower() == "ridge":
            self.model = Ridge(**self.model_params)
        elif self.model_type.lower() == "lasso":
            self.model = Lasso(**self.model_params)
        else:
            raise ValueError(f"不支持的线性模型类型: {self.model_type}")
        
        # 训练模型
        logger.info(f"开始训练 {self.model_type} 线性模型...")
        self.model.fit(X, y)
        self.trained = True
        
        # 计算训练指标
        train_pred = self.model.predict(X)
        train_metrics = {
            "train_r2": self.model.score(X, y)
        }
        
        # 如果提供了验证数据，则计算验证指标
        if validation_data is not None:
            X_val, y_val = validation_data
            val_pred = self.model.predict(X_val)
            val_metrics = self.evaluate(X_val, y_val)
            train_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        
        # 添加正则化参数到指标
        train_metrics["alpha"] = self.model_params["alpha"]
        
        # 更新元数据
        self.metadata["metrics"].update(train_metrics)
        
        logger.info(f"{self.model_type} 模型训练完成，训练 R²: {train_metrics['train_r2']:.4f}")
        
        return train_metrics
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X: 预测特征数据
            
        返回:
            np.ndarray: 预测结果
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        return self.model.predict(X)


class TreeModel(BaseModel):
    """
    树模型类，支持随机森林和梯度提升树
    """
    
    def __init__(self, 
                 name: str = "tree",
                 model_params: Dict = None,
                 prediction_horizon: int = 1,
                 target_type: str = "price_change_pct",
                 model_type: str = "random_forest",
                 model_dir: str = "models/saved_models"):
        """
        初始化树模型
        
        参数:
            name (str): 模型名称
            model_params (Dict): 模型参数字典
            prediction_horizon (int): 预测周期（步数）
            target_type (str): 目标变量类型
            model_type (str): 树模型类型，可选 'random_forest' 或 'gradient_boosting'
            model_dir (str): 模型保存目录
        """
        super().__init__(name, model_params, prediction_horizon, target_type, model_dir)
        
        self.model_type = model_type
        
        # 设置默认参数
        if model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1
            }
        else:  # gradient_boosting
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "subsample": 1.0,
                "random_state": 42
            }
        
        # 更新参数
        if model_params:
            default_params.update(model_params)
        
        self.model_params = default_params
        self.metadata["model_subtype"] = model_type
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], 
              validation_data: Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]] = None) -> Dict:
        """
        训练树模型
        
        参数:
            X: 训练特征数据
            y: 训练目标数据
            validation_data: 可选的验证数据 (X_val, y_val)
            
        返回:
            Dict: 包含训练指标的字典
        """
        # 保存特征名称（如果可用）
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # 确定任务类型（回归或分类）
        is_classification = self.target_type == "direction"
        
        # 创建模型
        if self.model_type == "random_forest":
            if is_classification:
                self.model = RandomForestClassifier(**self.model_params)
            else:
                self.model = RandomForestRegressor(**self.model_params)
        elif self.model_type == "gradient_boosting":
            if is_classification:
                self.model = GradientBoostingClassifier(**self.model_params)
            else:
                self.model = GradientBoostingRegressor(**self.model_params)
        else:
            raise ValueError(f"不支持的树模型类型: {self.model_type}")
        
        # 训练模型
        logger.info(f"开始训练 {self.model_type} {'分类' if is_classification else '回归'}模型...")
        self.model.fit(X, y)
        self.trained = True
        
        # 计算训练指标
        train_metrics = {
            "train_score": self.model.score(X, y)
        }
        
        # 如果提供了验证数据，则计算验证指标
        if validation_data is not None:
            X_val, y_val = validation_data
            val_metrics = self.evaluate(X_val, y_val)
            train_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        
        # 更新元数据
        self.metadata["metrics"].update(train_metrics)
        
        # 日志输出
        metric_name = "准确率" if is_classification else "R²"
        logger.info(f"{self.model_type} 模型训练完成，训练 {metric_name}: {train_metrics['train_score']:.4f}")
        
        return train_metrics
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X: 预测特征数据
            
        返回:
            np.ndarray: 预测结果
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预测概率（仅分类模型）
        
        参数:
            X: 预测特征数据
            
        返回:
            np.ndarray: 预测概率
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        if self.target_type != "direction":
            raise NotImplementedError("概率预测仅适用于分类模型")
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("此模型不支持概率预测")


class XGBoostModel(BaseModel):
    """
    XGBoost模型类，支持回归和分类
    """
    
    def __init__(self, 
                 name: str = "xgboost",
                 model_params: Dict = None,
                 prediction_horizon: int = 1,
                 target_type: str = "price_change_pct",
                 model_dir: str = "models/saved_models"):
        """
        初始化XGBoost模型
        
        参数:
            name (str): 模型名称
            model_params (Dict): 模型参数字典
            prediction_horizon (int): 预测周期（步数）
            target_type (str): 目标变量类型
            model_dir (str): 模型保存目录
        """
        super().__init__(name, model_params, prediction_horizon, target_type, model_dir)
        
        # 设置默认参数
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": 42,
            "n_jobs": -1
        }
        
        # 更新参数
        if model_params:
            default_params.update(model_params)
        
        self.model_params = default_params
        self.early_stopping_rounds = self.model_params.pop("early_stopping_rounds", 10)
        self.eval_metric = self.model_params.pop("eval_metric", None)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], 
              validation_data: Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]] = None) -> Dict:
        """
        训练XGBoost模型
        
        参数:
            X: 训练特征数据
            y: 训练目标数据
            validation_data: 可选的验证数据 (X_val, y_val)
            
        返回:
            Dict: 包含训练指标的字典
        """
        # 保存特征名称（如果可用）
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # 确定任务类型（回归或分类）
        is_classification = self.target_type == "direction"
        objective = "binary:logistic" if is_classification else "reg:squarederror"
        
        # 设置评估指标
        if self.eval_metric is None:
            self.eval_metric = "auc" if is_classification else "rmse"
        
        # 更新模型参数
        self.model_params["objective"] = objective
        
        # 创建模型
        if is_classification:
            self.model = xgb.XGBClassifier(**self.model_params)
        else:
            self.model = xgb.XGBRegressor(**self.model_params)
        
        # 准备训练参数
        fit_params = {}
        
        # 如果提供了验证数据，添加到训练参数
        if validation_data is not None:
            X_val, y_val = validation_data
            fit_params["eval_set"] = [(X, y), (X_val, y_val)]
            fit_params["early_stopping_rounds"] = self.early_stopping_rounds
            fit_params["eval_metric"] = self.eval_metric
        
        # 训练模型
        logger.info(f"开始训练 XGBoost {'分类' if is_classification else '回归'}模型...")
        self.model.fit(X, y, **fit_params)
        self.trained = True
        
        # 计算训练指标
        train_metrics = {
            "train_score": self.model.score(X, y)
        }
        
        # 如果提供了验证数据，则计算验证指标
        if validation_data is not None:
            X_val, y_val = validation_data
            val_metrics = self.evaluate(X_val, y_val)
            train_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            
            # 添加最佳迭代轮数
            if hasattr(self.model, "best_iteration"):
                train_metrics["best_iteration"] = self.model.best_iteration
                logger.info(f"最佳迭代轮数: {self.model.best_iteration}")
        
        # 更新元数据
        self.metadata["metrics"].update(train_metrics)
        
        # 日志输出
        metric_name = "准确率" if is_classification else "R²"
        logger.info(f"XGBoost模型训练完成，训练 {metric_name}: {train_metrics['train_score']:.4f}")
        
        return train_metrics
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X: 预测特征数据
            
        返回:
            np.ndarray: 预测结果
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预测概率（仅分类模型）
        
        参数:
            X: 预测特征数据
            
        返回:
            np.ndarray: 预测概率
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        if self.target_type != "direction":
            raise NotImplementedError("概率预测仅适用于分类模型")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取XGBoost模型的特征重要性
        
        返回:
            Dict[str, float]: 特征重要性字典
        """
        if not self.trained:
            logger.warning("模型尚未训练，无法获取特征重要性")
            return None
        
        # 获取特征重要性
        importance_type = "gain"
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        # 如果缺少特征，则填充为0
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in importance:
                    importance[feature] = 0
        
        # 按重要性降序排序
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)) 