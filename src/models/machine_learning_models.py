"""
机器学习模型实现模块 - 包括随机森林、XGBoost等模型
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
import joblib

# 导入机器学习模型
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import xgboost as xgb

# 导入评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.base_model import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)


class MLModel(BaseModel):
    """机器学习模型基类"""
    
    def __init__(self, 
                target_type: str = "price_change_pct",
                model_params: Dict = None,
                name: str = "ML"):
        """
        初始化机器学习模型
        
        参数:
            target_type (str): 目标类型，如"price_change_pct"或"direction"
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        super().__init__(target_type=target_type, model_params=model_params, name=name)
        self.model = None
        self.feature_importances_ = None
    
    def _get_model_instance(self) -> Any:
        """
        获取模型实例（子类将重写此方法）
        
        返回:
            Any: 模型实例
        """
        raise NotImplementedError("子类必须实现_get_model_instance方法")
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        训练机器学习模型
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 特征数据
            y (Union[pd.Series, np.ndarray]): 目标数据
        """
        logger.info(f"开始训练{self.name}模型...")
        
        # 获取模型实例
        if self.model is None:
            self.model = self._get_model_instance()
        
        # 训练模型
        self.model.fit(X, y)
        
        # 记录特征重要性（如果模型支持）
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
            
            # 如果X是DataFrame，可以将特征名与重要性关联
            if isinstance(X, pd.DataFrame):
                self.feature_importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # 记录前10个重要特征
                top_features = self.feature_importance_df.head(10)
                logger.info(f"前10个重要特征:\n{top_features}")
        
        logger.info(f"{self.name}模型训练完成")
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 特征数据
            
        返回:
            np.ndarray: 预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        logger.info(f"使用{self.name}模型进行预测...")
        
        # 进行预测
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用训练好的模型进行概率预测（仅适用于分类问题）
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 特征数据
            
        返回:
            np.ndarray: 预测概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        if self.target_type != "direction":
            raise ValueError("predict_proba方法仅适用于方向预测 (target_type='direction')")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.name}模型不支持概率预测")
        
        logger.info(f"使用{self.name}模型进行概率预测...")
        
        # 进行概率预测
        proba = self.model.predict_proba(X)
        
        return proba
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict:
        """
        评估模型性能
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 特征数据
            y (Union[pd.Series, np.ndarray]): 目标数据
            
        返回:
            Dict: 评估指标
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        logger.info(f"评估{self.name}模型性能...")
        
        # 进行预测
        y_pred = self.predict(X)
        
        # 根据目标类型计算指标
        metrics = {}
        if self.target_type == "direction":
            # 分类指标
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['precision'] = precision_score(y, y_pred, average='binary')
            metrics['recall'] = recall_score(y, y_pred, average='binary')
            metrics['f1'] = f1_score(y, y_pred, average='binary')
            
            # 如果模型支持概率预测，还可以计算ROC-AUC
            if hasattr(self.model, 'predict_proba'):
                from sklearn.metrics import roc_auc_score
                y_proba = self.predict_proba(X)
                # 确保二分类概率取正类概率
                if y_proba.shape[1] == 2:
                    y_proba = y_proba[:, 1]
                metrics['roc_auc'] = roc_auc_score(y, y_proba)
        else:
            # 回归指标
            metrics['mse'] = mean_squared_error(y, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y, y_pred)
            metrics['r2'] = r2_score(y, y_pred)
        
        # 打印指标
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        保存模型到文件
        
        参数:
            filepath (str): 文件路径
        """
        logger.info(f"保存{self.name}模型到{filepath}...")
        
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型
        model_dir = os.path.dirname(filepath)
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        joblib.dump(self.model, model_path)
        
        # 保存特征重要性（如果有）
        feature_importance_path = None
        if hasattr(self, 'feature_importance_df') and self.feature_importance_df is not None:
            feature_importance_path = os.path.join(model_dir, f"{model_name}_feature_importance.csv")
            self.feature_importance_df.to_csv(feature_importance_path, index=False)
        
        # 保存元数据
        metadata = {
            "model_params": self.model_params,
            "target_type": self.target_type,
            "name": self.name,
            "model_path": model_path,
            "feature_importance_path": feature_importance_path,
            "model_type": self.__class__.__name__
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"{self.name}模型已保存")
    
    def load(self, filepath: str) -> None:
        """
        从文件加载模型
        
        参数:
            filepath (str): 文件路径
        """
        logger.info(f"从{filepath}加载{self.name}模型...")
        
        # 加载元数据
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        # 更新属性
        self.model_params = metadata["model_params"]
        self.target_type = metadata["target_type"]
        self.name = metadata["name"]
        
        # 加载模型
        model_path = metadata["model_path"]
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"模型文件{model_path}不存在")
        
        # 加载特征重要性（如果有）
        if "feature_importance_path" in metadata and metadata["feature_importance_path"]:
            feature_importance_path = metadata["feature_importance_path"]
            if os.path.exists(feature_importance_path):
                self.feature_importance_df = pd.read_csv(feature_importance_path)
                self.feature_importances_ = self.feature_importance_df['importance'].values
        
        logger.info(f"{self.name}模型已加载")
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        绘制特征重要性
        
        参数:
            top_n (int): 显示前几个重要特征
            figsize (Tuple[int, int]): 图形大小
        """
        if not hasattr(self, 'feature_importance_df') or self.feature_importance_df is None:
            raise ValueError("模型没有特征重要性信息")
        
        # 获取前N个重要特征
        top_features = self.feature_importance_df.head(top_n)
        
        # 绘制条形图
        plt.figure(figsize=figsize)
        plt.barh(
            np.arange(len(top_features)),
            top_features['importance'],
            align='center'
        )
        plt.yticks(np.arange(len(top_features)), top_features['feature'])
        plt.xlabel('特征重要性')
        plt.ylabel('特征')
        plt.title(f'{self.name}模型 - 前{top_n}个重要特征')
        plt.tight_layout()
        plt.show()


class RandomForestModel(MLModel):
    """随机森林模型"""
    
    def __init__(self, 
                target_type: str = "price_change_pct",
                model_params: Dict = None,
                name: str = "RandomForest"):
        """
        初始化随机森林模型
        
        参数:
            target_type (str): 目标类型，如"price_change_pct"或"direction"
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        # 默认参数
        default_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "auto",
            "random_state": 42
        }
        
        # 合并默认参数和用户参数
        model_params = model_params or {}
        for key, value in default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        super().__init__(target_type=target_type, model_params=model_params, name=name)
    
    def _get_model_instance(self) -> Any:
        """
        获取随机森林模型实例
        
        返回:
            Any: 随机森林模型实例
        """
        if self.target_type == "direction":
            return RandomForestClassifier(**self.model_params)
        else:
            return RandomForestRegressor(**self.model_params)


class XGBoostModel(MLModel):
    """XGBoost模型"""
    
    def __init__(self, 
                target_type: str = "price_change_pct",
                model_params: Dict = None,
                name: str = "XGBoost"):
        """
        初始化XGBoost模型
        
        参数:
            target_type (str): 目标类型，如"price_change_pct"或"direction"
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        # 默认参数
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        
        # 合并默认参数和用户参数
        model_params = model_params or {}
        for key, value in default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        super().__init__(target_type=target_type, model_params=model_params, name=name)
    
    def _get_model_instance(self) -> Any:
        """
        获取XGBoost模型实例
        
        返回:
            Any: XGBoost模型实例
        """
        if self.target_type == "direction":
            return xgb.XGBClassifier(**self.model_params)
        else:
            return xgb.XGBRegressor(**self.model_params)


class LinearModel(MLModel):
    """线性模型"""
    
    def __init__(self, 
                target_type: str = "price_change_pct",
                model_params: Dict = None,
                name: str = "Linear"):
        """
        初始化线性模型
        
        参数:
            target_type (str): 目标类型，如"price_change_pct"或"direction"
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        # 默认参数
        default_params = {
            "linear_type": "linear",  # "linear", "ridge", "lasso", "elastic_net"
            "alpha": 1.0,  # 正则化强度，用于Ridge和Lasso
            "l1_ratio": 0.5,  # 用于ElasticNet
            "random_state": 42
        }
        
        # 合并默认参数和用户参数
        model_params = model_params or {}
        for key, value in default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        super().__init__(target_type=target_type, model_params=model_params, name=name)
    
    def _get_model_instance(self) -> Any:
        """
        获取线性模型实例
        
        返回:
            Any: 线性模型实例
        """
        linear_type = self.model_params.get("linear_type", "linear").lower()
        
        if self.target_type == "direction":
            # 分类问题使用逻辑回归
            return LogisticRegression(
                C=1.0/self.model_params.get("alpha", 1.0),
                random_state=self.model_params.get("random_state", 42)
            )
        else:
            # 回归问题根据linear_type选择不同的线性模型
            if linear_type == "ridge":
                return Ridge(
                    alpha=self.model_params.get("alpha", 1.0),
                    random_state=self.model_params.get("random_state", 42)
                )
            elif linear_type == "lasso":
                return Lasso(
                    alpha=self.model_params.get("alpha", 1.0),
                    random_state=self.model_params.get("random_state", 42)
                )
            elif linear_type == "elastic_net":
                return ElasticNet(
                    alpha=self.model_params.get("alpha", 1.0),
                    l1_ratio=self.model_params.get("l1_ratio", 0.5),
                    random_state=self.model_params.get("random_state", 42)
                )
            else:
                return LinearRegression()


class SVMModel(MLModel):
    """SVM模型"""
    
    def __init__(self, 
                target_type: str = "price_change_pct",
                model_params: Dict = None,
                name: str = "SVM"):
        """
        初始化SVM模型
        
        参数:
            target_type (str): 目标类型，如"price_change_pct"或"direction"
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        # 默认参数
        default_params = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "random_state": 42
        }
        
        # 合并默认参数和用户参数
        model_params = model_params or {}
        for key, value in default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        super().__init__(target_type=target_type, model_params=model_params, name=name)
    
    def _get_model_instance(self) -> Any:
        """
        获取SVM模型实例
        
        返回:
            Any: SVM模型实例
        """
        if self.target_type == "direction":
            return SVC(
                kernel=self.model_params.get("kernel", "rbf"),
                C=self.model_params.get("C", 1.0),
                gamma=self.model_params.get("gamma", "scale"),
                probability=True,
                random_state=self.model_params.get("random_state", 42)
            )
        else:
            return SVR(
                kernel=self.model_params.get("kernel", "rbf"),
                C=self.model_params.get("C", 1.0),
                gamma=self.model_params.get("gamma", "scale")
            )


class KNNModel(MLModel):
    """K近邻模型"""
    
    def __init__(self, 
                target_type: str = "price_change_pct",
                model_params: Dict = None,
                name: str = "KNN"):
        """
        初始化K近邻模型
        
        参数:
            target_type (str): 目标类型，如"price_change_pct"或"direction"
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        # 默认参数
        default_params = {
            "n_neighbors": 5,
            "weights": "uniform",
            "p": 2  # 欧几里得距离
        }
        
        # 合并默认参数和用户参数
        model_params = model_params or {}
        for key, value in default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        super().__init__(target_type=target_type, model_params=model_params, name=name)
    
    def _get_model_instance(self) -> Any:
        """
        获取K近邻模型实例
        
        返回:
            Any: K近邻模型实例
        """
        if self.target_type == "direction":
            return KNeighborsClassifier(
                n_neighbors=self.model_params.get("n_neighbors", 5),
                weights=self.model_params.get("weights", "uniform"),
                p=self.model_params.get("p", 2)
            )
        else:
            return KNeighborsRegressor(
                n_neighbors=self.model_params.get("n_neighbors", 5),
                weights=self.model_params.get("weights", "uniform"),
                p=self.model_params.get("p", 2)
            )


class GradientBoostingModel(MLModel):
    """梯度提升模型"""
    
    def __init__(self, 
                target_type: str = "price_change_pct",
                model_params: Dict = None,
                name: str = "GradientBoosting"):
        """
        初始化梯度提升模型
        
        参数:
            target_type (str): 目标类型，如"price_change_pct"或"direction"
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        # 默认参数
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "subsample": 1.0,
            "random_state": 42
        }
        
        # 合并默认参数和用户参数
        model_params = model_params or {}
        for key, value in default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        super().__init__(target_type=target_type, model_params=model_params, name=name)
    
    def _get_model_instance(self) -> Any:
        """
        获取梯度提升模型实例
        
        返回:
            Any: 梯度提升模型实例
        """
        if self.target_type == "direction":
            return GradientBoostingClassifier(**self.model_params)
        else:
            return GradientBoostingRegressor(**self.model_params)


class AdaBoostModel(MLModel):
    """AdaBoost模型"""
    
    def __init__(self, 
                target_type: str = "price_change_pct",
                model_params: Dict = None,
                name: str = "AdaBoost"):
        """
        初始化AdaBoost模型
        
        参数:
            target_type (str): 目标类型，如"price_change_pct"或"direction"
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        # 默认参数
        default_params = {
            "n_estimators": 50,
            "learning_rate": 1.0,
            "random_state": 42
        }
        
        # 合并默认参数和用户参数
        model_params = model_params or {}
        for key, value in default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        super().__init__(target_type=target_type, model_params=model_params, name=name)
    
    def _get_model_instance(self) -> Any:
        """
        获取AdaBoost模型实例
        
        返回:
            Any: AdaBoost模型实例
        """
        if self.target_type == "direction":
            return AdaBoostClassifier(**self.model_params)
        else:
            return AdaBoostRegressor(**self.model_params)


class ExtraTreesModel(MLModel):
    """极端随机树模型"""
    
    def __init__(self, 
                target_type: str = "price_change_pct",
                model_params: Dict = None,
                name: str = "ExtraTrees"):
        """
        初始化极端随机树模型
        
        参数:
            target_type (str): 目标类型，如"price_change_pct"或"direction"
            model_params (Dict): 模型参数
            name (str): 模型名称
        """
        # 默认参数
        default_params = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "auto",
            "random_state": 42
        }
        
        # 合并默认参数和用户参数
        model_params = model_params or {}
        for key, value in default_params.items():
            if key not in model_params:
                model_params[key] = value
        
        super().__init__(target_type=target_type, model_params=model_params, name=name)
    
    def _get_model_instance(self) -> Any:
        """
        获取极端随机树模型实例
        
        返回:
            Any: 极端随机树模型实例
        """
        if self.target_type == "direction":
            return ExtraTreesClassifier(**self.model_params)
        else:
            return ExtraTreesRegressor(**self.model_params)


# 工厂函数
def create_ml_model(model_type: str, target_type: str = "price_change_pct", model_params: Dict = None, name: str = None) -> BaseModel:
    """
    创建机器学习模型
    
    参数:
        model_type (str): 模型类型，如"random_forest", "xgboost", "linear", "svm", "knn", "gradient_boosting", "adaboost", "extra_trees"
        target_type (str): 目标类型，如"price_change_pct"或"direction"
        model_params (Dict): 模型参数
        name (str): 模型名称
        
    返回:
        BaseModel: 机器学习模型实例
    """
    model_type = model_type.lower()
    
    if model_type == "random_forest":
        name = name or "RandomForest"
        return RandomForestModel(target_type=target_type, model_params=model_params, name=name)
    elif model_type == "xgboost":
        name = name or "XGBoost"
        return XGBoostModel(target_type=target_type, model_params=model_params, name=name)
    elif model_type == "linear":
        name = name or "Linear"
        return LinearModel(target_type=target_type, model_params=model_params, name=name)
    elif model_type == "svm":
        name = name or "SVM"
        return SVMModel(target_type=target_type, model_params=model_params, name=name)
    elif model_type == "knn":
        name = name or "KNN"
        return KNNModel(target_type=target_type, model_params=model_params, name=name)
    elif model_type == "gradient_boosting":
        name = name or "GradientBoosting"
        return GradientBoostingModel(target_type=target_type, model_params=model_params, name=name)
    elif model_type == "adaboost":
        name = name or "AdaBoost"
        return AdaBoostModel(target_type=target_type, model_params=model_params, name=name)
    elif model_type == "extra_trees":
        name = name or "ExtraTrees"
        return ExtraTreesModel(target_type=target_type, model_params=model_params, name=name)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}") 