"""
特征选择与模型集成模块，用于筛选重要特征并对预测结果进行平滑处理。
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple
import matplotlib.pyplot as plt
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator


class FeatureSelector:
    """
    特征选择器，用于基于特征重要性或SHAP值筛选重要特征
    """
    
    def __init__(self, method='importance', threshold=0.0, top_n=None, 
                 min_features=5, output_dir='data/processed/features/selected'):
        """
        初始化特征选择器
        
        Args:
            method (str): 选择方法，支持'importance'和'shap'
            threshold (float): 特征重要性阈值，低于此值的特征将被过滤
            top_n (int, optional): 选择前N个重要特征
            min_features (int): 最少保留的特征数量
            output_dir (str): 输出目录
        """
        self.method = method
        self.threshold = threshold
        self.top_n = top_n
        self.min_features = min_features
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.selected_features = None
        self.feature_importance = None
        
    def select_features_from_model(self, model, feature_names=None):
        """
        基于模型提取重要特征
        
        Args:
            model: 训练好的模型对象
            feature_names (list, optional): 特征名称列表
            
        Returns:
            list: 选中的特征名称列表
        """
        if self.method == 'importance':
            return self._select_by_importance(model, feature_names)
        elif self.method == 'shap':
            return self._select_by_shap(model, feature_names)
        else:
            raise ValueError(f"不支持的特征选择方法: {self.method}")
            
    def _select_by_importance(self, model, feature_names=None):
        """使用特征重要性选择特征"""
        if not hasattr(model, 'feature_importances_'):
            self.logger.error("模型没有feature_importances_属性")
            return []
        
        # 获取特征重要性
        importance = model.feature_importances_
        
        # 创建特征名称
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        # 确保特征名称和特征重要性长度匹配
        if len(feature_names) != len(importance):
            self.logger.warning(f"特征名称数量({len(feature_names)})与特征重要性数量({len(importance)})不匹配")
            feature_names = [f"feature_{i}" for i in range(len(importance))]
            
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        # 使用阈值筛选
        if self.threshold > 0:
            selected_df = importance_df[importance_df['importance'] >= self.threshold]
            
            # 确保至少保留min_features个特征
            if len(selected_df) < self.min_features:
                selected_df = importance_df.head(self.min_features)
                
            self.selected_features = selected_df['feature'].tolist()
                
        # 使用top_n筛选
        elif self.top_n is not None:
            top_n = min(self.top_n, len(importance_df))
            self.selected_features = importance_df.head(top_n)['feature'].tolist()
            
        # 默认全部选择
        else:
            self.selected_features = feature_names
            
        self.logger.info(f"基于特征重要性选择了{len(self.selected_features)}个特征")
        
        return self.selected_features
    
    def _select_by_shap(self, model, feature_names=None, X_sample=None):
        """使用SHAP值选择特征"""
        try:
            import shap
        except ImportError:
            self.logger.error("请安装shap包: pip install shap")
            return []
        
        if X_sample is None:
            self.logger.error("使用SHAP方法需要提供样本数据X_sample")
            return self._select_by_importance(model, feature_names)
            
        # 创建特征名称
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_sample.shape[1])]
            
        # 计算SHAP值
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)
        
        # 计算每个特征的平均绝对SHAP值
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        # 使用阈值筛选
        if self.threshold > 0:
            selected_df = importance_df[importance_df['importance'] >= self.threshold]
            
            # 确保至少保留min_features个特征
            if len(selected_df) < self.min_features:
                selected_df = importance_df.head(self.min_features)
                
            self.selected_features = selected_df['feature'].tolist()
                
        # 使用top_n筛选
        elif self.top_n is not None:
            top_n = min(self.top_n, len(importance_df))
            self.selected_features = importance_df.head(top_n)['feature'].tolist()
            
        # 默认全部选择
        else:
            self.selected_features = feature_names
            
        self.logger.info(f"基于SHAP值选择了{len(self.selected_features)}个特征")
        
        return self.selected_features
    
    def plot_feature_importance(self, max_features=20, save_path=None):
        """
        绘制特征重要性图
        
        Args:
            max_features (int): 最多显示的特征数量
            save_path (str, optional): 保存路径
            
        Returns:
            plt.Figure: 图表对象
        """
        if self.feature_importance is None:
            self.logger.error("请先调用select_features_from_model方法")
            return None
            
        # 最多显示max_features个特征
        if len(self.feature_importance) > max_features:
            plot_df = self.feature_importance.head(max_features)
        else:
            plot_df = self.feature_importance
            
        # 绘制图表
        plt.figure(figsize=(12, 8))
        bars = plt.barh(plot_df['feature'], plot_df['importance'])
        
        # 高亮选中的特征
        if self.selected_features:
            for i, (feature, _) in enumerate(zip(plot_df['feature'], plot_df['importance'])):
                if feature in self.selected_features:
                    bars[i].set_color('orange')
                    
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"特征重要性图保存到: {save_path}")
            
        return plt.gcf()
    
    def save_selected_features(self, name=None):
        """
        保存选中的特征
        
        Args:
            name (str, optional): 特征集名称
            
        Returns:
            str: 保存文件路径
        """
        if self.selected_features is None:
            self.logger.error("请先调用select_features_from_model方法")
            return None
            
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 使用时间戳作为名称
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"selected_features_{timestamp}"
            
        # 保存特征列表
        file_path = os.path.join(self.output_dir, f"{name}.json")
        with open(file_path, 'w') as f:
            json.dump(self.selected_features, f, indent=4)
            
        # 如果有特征重要性，也保存
        if self.feature_importance is not None:
            importance_path = os.path.join(self.output_dir, f"{name}_importance.csv")
            self.feature_importance.to_csv(importance_path, index=False)
            
        self.logger.info(f"选中的特征保存到: {file_path}")
        
        return file_path


class PredictionSmoother:
    """
    预测结果平滑处理器
    """
    
    def __init__(
        self,
        window_size: int = 5,
        method: str = 'moving_avg',
        center: bool = True
    ):
        """
        初始化预测平滑处理器
        
        Args:
            window_size: 滑动窗口大小
            method: 平滑方法，'moving_avg', 'ewm'(指数加权移动平均), 'median'
            center: 是否使用中心窗口(对称)
        """
        self.window_size = window_size
        self.method = method
        self.center = center
    
    def smooth(self, predictions: pd.Series) -> pd.Series:
        """
        平滑预测结果
        
        Args:
            predictions: 原始预测结果，时间序列
        
        Returns:
            pd.Series: 平滑后的预测结果
        """
        if self.method == 'moving_avg':
            smoothed = predictions.rolling(
                window=self.window_size,
                center=self.center,
                min_periods=1
            ).mean()
        elif self.method == 'ewm':
            # alpha = 2/(window_size + 1)
            alpha = 2 / (self.window_size + 1)
            smoothed = predictions.ewm(alpha=alpha, adjust=True).mean()
        elif self.method == 'median':
            smoothed = predictions.rolling(
                window=self.window_size,
                center=self.center,
                min_periods=1
            ).median()
        else:
            raise ValueError(f"不支持的平滑方法: {self.method}")
        
        # 确保不会产生NaN
        if smoothed.isna().any():
            # 用原始值填充NaN
            smoothed = smoothed.fillna(predictions)
        
        return smoothed
    
    def plot_comparison(
        self,
        original: pd.Series,
        smoothed: pd.Series,
        targets: Optional[pd.Series] = None,
        title: str = 'Original vs Smoothed Predictions',
        save_path: Optional[str] = None
    ) -> None:
        """
        绘制原始预测、平滑预测和目标值的对比图

        Args:
            original: 原始预测序列
            smoothed: 平滑后的预测序列
            targets: 目标值序列 (可选)
            title: 图表标题
            save_path: 保存路径 (可选)
        """
        plt.figure(figsize=(12, 6))

        # 绘制原始预测
        plt.plot(original.index, original.values, label='Original Predictions', alpha=0.7)

        # 绘制平滑预测
        plt.plot(smoothed.index, smoothed.values, label='Smoothed Predictions', linewidth=2)

        # 绘制目标值
        if targets is not None:
            plt.plot(targets.index, targets.values, label='Actual Values', alpha=0.5, linestyle=':')

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")

        plt.close()


class ModelEnsemble:
    """
    模型集成类，支持多种集成方法
    """
    
    def __init__(
        self,
        model_paths: List[str],
        ensemble_method: str = 'average',
        weights: Optional[List[float]] = None,
        model_names: Optional[List[str]] = None,
        selected_features: Optional[Dict[str, List[str]]] = None
    ):
        """
        初始化模型集成
        
        Args:
            model_paths: 模型路径列表
            ensemble_method: 集成方法，'average', 'weighted', 'stacking'
            weights: 权重列表(用于weighted方法)
            model_names: 模型名称列表(可选)
            selected_features: 每个模型使用的特征列表(如果不同)
        """
        self.model_paths = model_paths
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.model_names = model_names or [f"model_{i}" for i in range(len(model_paths))]
        self.selected_features = selected_features or {}
        
        # 加载所有模型
        self.models = self._load_models()
        
        # 验证权重
        if ensemble_method == 'weighted' and weights is None:
            # 默认平均权重
            self.weights = [1.0 / len(model_paths)] * len(model_paths)
        elif ensemble_method == 'weighted' and len(weights) != len(model_paths):
            raise ValueError(f"权重数量 ({len(weights)}) 必须与模型数量 ({len(model_paths)}) 相同")
    
    def _load_models(self) -> List[BaseEstimator]:
        """
        加载所有模型
        
        Returns:
            List[BaseEstimator]: 模型列表
        """
        models = []
        for path in self.model_paths:
            try:
                model = joblib.load(path)
                models.append(model)
            except Exception as e:
                print(f"加载模型失败 ({path}): {e}")
                raise
        
        return models
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        apply_smoothing: bool = False,
        smoother: Optional[PredictionSmoother] = None
    ) -> pd.Series:
        """
        使用集成模型进行预测
        
        Args:
            X: 输入特征
            apply_smoothing: 是否应用平滑处理
            smoother: 平滑处理器实例(如果apply_smoothing为True)
            
        Returns:
            pd.Series: 预测结果
        """
        # 获取每个模型的单独预测结果
        predictions = []
        for i, model in enumerate(self.models):
            # 检查是否需要为此模型选择特定特征
            if self.model_names[i] in self.selected_features:
                features = self.selected_features[self.model_names[i]]
                X_model = X[features]
            else:
                X_model = X
            
            # 进行预测
            pred = model.predict(X_model)
            
            if isinstance(pred, np.ndarray) and pred.ndim == 1:
                pred_series = pd.Series(pred, index=X.index if hasattr(X, 'index') else None)
            else:
                pred_series = pd.Series(pred.flatten(), index=X.index if hasattr(X, 'index') else None)
            
            predictions.append(pred_series)
        
        # 根据集成方法组合预测结果
        if self.ensemble_method == 'average':
            ensemble_pred = sum(predictions) / len(predictions)
        elif self.ensemble_method == 'weighted':
            ensemble_pred = sum(w * p for w, p in zip(self.weights, predictions))
        elif self.ensemble_method == 'median':
            # 计算每个样本的预测中位数
            all_preds = pd.concat(predictions, axis=1)
            ensemble_pred = all_preds.median(axis=1)
        else:
            raise ValueError(f"不支持的集成方法: {self.ensemble_method}")
        
        # 应用平滑处理
        if apply_smoothing:
            if smoother is None:
                smoother = PredictionSmoother()
            
            ensemble_pred = smoother.smooth(ensemble_pred)
        
        return ensemble_pred
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        apply_smoothing: bool = False,
        smoother: Optional[PredictionSmoother] = None
    ) -> Dict[str, float]:
        """
        评估模型集成性能
        
        Args:
            X: 输入特征
            y: 目标变量
            apply_smoothing: 是否应用平滑处理
            smoother: 平滑处理器实例(如果apply_smoothing为True)
            
        Returns:
            Dict[str, float]: 评估指标
        """
        # 获取预测结果
        predictions = self.predict(X, apply_smoothing, smoother)
        
        # 计算评估指标
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        # 返回评估结果
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        }
    
    def plot_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        title: str = 'Ensemble Model Predictions Comparison',
        save_path: Optional[str] = None,
        apply_smoothing: bool = False,
        smoother: Optional[PredictionSmoother] = None,
        show_individual: bool = True
    ) -> None:
        """
        绘制集成模型的预测结果与实际值的对比图

        Args:
            X: 特征数据
            y: 实际目标值
            title: 图表标题
            save_path: 保存路径 (可选)
            apply_smoothing: 是否应用平滑 (可选)
            smoother: 平滑器实例 (可选)
            show_individual: 是否显示单个模型的预测 (可选)
        """
        logger.info(f"Generating prediction plot: {title}")

        # 获取集成预测
        ensemble_pred = self.predict(X, apply_smoothing=apply_smoothing, smoother=smoother)

        plt.figure(figsize=(15, 7))

        # Plot actual values
        plt.plot(y.index, y.values, label='Actual Values', alpha=0.7, linewidth=1.5)

        # Plot ensemble prediction
        label_suffix = " (Smoothed)" if apply_smoothing and smoother else ""
        plt.plot(ensemble_pred.index, ensemble_pred.values, label=f'Ensemble Prediction{label_suffix}', linewidth=2, color='blue')

        # Plot individual model predictions if requested
        if show_individual:
            for i, model in enumerate(self.models):
                # Check if specific features are needed for this model
                X_model = X
                model_key = self.model_names[i] if self.model_names else f"model_{i}"
                if self.selected_features and model_key in self.selected_features:
                    X_model = X[self.selected_features[model_key]]
                elif self.selected_features and 'default' in self.selected_features: # Fallback? Maybe not needed
                     X_model = X[self.selected_features['default']]


                pred = model.predict(X_model)

                if isinstance(pred, np.ndarray) and pred.ndim == 1:
                    pred_series = pd.Series(pred, index=X.index)
                else:
                    pred_series = pd.Series(pred.flatten(), index=X.index)

                model_display_name = self.model_names[i] if self.model_names else f"Model {i}"
                plt.plot(pred_series.index, pred_series.values, '--', alpha=0.4, label=f'{model_display_name} Prediction')

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Predicted Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction plot saved to: {save_path}")

        plt.close()
    
    def save_weights(self, save_path: str) -> None:
        """
        保存模型权重
        
        Args:
            save_path: 保存路径
        """
        # 创建权重信息
        weight_info = {
            'model_paths': self.model_paths,
            'model_names': self.model_names,
            'ensemble_method': self.ensemble_method,
            'weights': self.weights,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存权重信息
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            import json
            json.dump(weight_info, f, indent=4)
        
        print(f"模型权重已保存至: {save_path}")


def retrain_with_selected_features(model_type, X_train, y_train, X_val, y_val, X_test, y_test, 
                                   selected_features, model_params=None):
    """
    使用选中的特征重新训练模型
    
    Args:
        model_type (str): 模型类型，支持'xgboost'和'lightgbm'
        X_train (pd.DataFrame): 训练特征
        y_train (pd.Series): 训练目标
        X_val (pd.DataFrame): 验证特征
        y_val (pd.Series): 验证目标
        X_test (pd.DataFrame): 测试特征
        y_test (pd.Series): 测试目标
        selected_features (list): 选中的特征列表
        model_params (dict, optional): 模型参数
        
    Returns:
        tuple: (训练好的模型, 评估指标)
    """
    logger = logging.getLogger(__name__)
    
    # 使用选中的特征
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    logger.info(f"使用{len(selected_features)}个选中的特征重新训练模型")
    
    # 创建模型
    if model_type == 'xgboost':
        import xgboost as xgb
        if model_params is None:
            model_params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1
            }
        
        # 创建模型
        model = xgb.XGBRegressor(**model_params)
        
        # 训练模型
        model.fit(
            X_train_selected, y_train,
            eval_set=[(X_val_selected, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
    elif model_type == 'lightgbm':
        import lightgbm as lgb
        if model_params is None:
            model_params = {
                'objective': 'regression',
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 5,
                'num_leaves': 31,
                'min_child_weight': 0.001,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1
            }
        
        # 创建模型
        model = lgb.LGBMRegressor(**model_params)
        
        # 训练模型
        model.fit(
            X_train_selected, y_train,
            eval_set=[(X_val_selected, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 评估模型
    y_pred = model.predict(X_test_selected)
    
    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    logger.info(f"使用选中特征的模型性能 - RMSE: {rmse}, MAE: {mae}, R²: {r2}")
    
    return model, metrics


def create_model_ensemble(model_files, X, weights=None, method='average'):
    """
    创建模型集成
    
    Args:
        model_files (list): 模型文件路径列表
        X (pd.DataFrame): 特征数据
        weights (list, optional): 模型权重
        method (str): 集成方法
        
    Returns:
        tuple: (集成对象, 模型列表, 特征重要性DataFrame)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"创建{len(model_files)}个模型的集成")
    
    # 创建集成对象
    ensemble = ModelEnsemble(
        model_paths=model_files,
        ensemble_method=method,
        weights=weights,
        model_names=None
    )
    
    # 提取特征重要性
    feature_importances = []
    
    for i, model in enumerate(ensemble.models):
        if hasattr(model, 'feature_importances_'):
            feature_names = X.columns.tolist()
            importance = model.feature_importances_
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance,
                'model': f"model_{i}"
            })
            
            feature_importances.append(importance_df)
    
    # 合并特征重要性
    if feature_importances:
        combined_importance = pd.concat(feature_importances)
        
        # 计算每个特征的平均重要性
        avg_importance = combined_importance.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
    else:
        avg_importance = None
    
    return ensemble, ensemble.models, avg_importance


def main():
    """
    主函数，处理命令行参数并执行特征选择或模型集成
    """
    parser = argparse.ArgumentParser(description="特征选择与模型集成")
    parser.add_argument("--mode", type=str, required=True, choices=['select', 'ensemble', 'smooth'],
                        help="操作模式")
    
    # 特征选择参数
    parser.add_argument("--model_file", type=str, help="模型文件路径")
    parser.add_argument("--features_file", type=str, help="特征文件路径")
    parser.add_argument("--method", type=str, default='importance', choices=['importance', 'shap'],
                        help="特征选择方法")
    parser.add_argument("--threshold", type=float, default=0.0, help="特征重要性阈值")
    parser.add_argument("--top_n", type=int, help="选择前N个特征")
    parser.add_argument("--min_features", type=int, default=5, help="最少保留的特征数量")
    parser.add_argument("--output_dir", type=str, default="data/processed/features/selected",
                        help="输出目录")
    parser.add_argument("--feature_set_name", type=str, help="特征集名称")
    
    # 模型集成参数
    parser.add_argument("--model_files", type=str, nargs='+', help="模型文件路径列表")
    parser.add_argument("--weights", type=float, nargs='+', help="模型权重列表")
    parser.add_argument("--ensemble_method", type=str, default='average',
                        choices=['average', 'weighted', 'median'], help="集成方法")
    parser.add_argument("--predictions_file", type=str, help="预测结果文件路径")
    parser.add_argument("--output_file", type=str, help="输出文件路径")
    
    # 预测平滑参数
    parser.add_argument("--window_size", type=int, default=5, help="滑动窗口大小")
    
    # 重新训练参数
    parser.add_argument("--retrain", action="store_true", help="是否使用选中的特征重新训练模型")
    parser.add_argument("--model_type", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm"], help="模型类型")
    parser.add_argument("--target_file", type=str, help="目标文件路径")
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--val_size", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 特征选择模式
    if args.mode == 'select':
        if not args.model_file:
            logger.error("请提供模型文件路径")
            return
            
        # 加载模型
        logger.info(f"加载模型: {args.model_file}")
        model = joblib.load(args.model_file)
        
        # 创建特征选择器
        selector = FeatureSelector(
            method=args.method,
            threshold=args.threshold,
            top_n=args.top_n,
            min_features=args.min_features,
            output_dir=args.output_dir
        )
        
        # 如果提供了特征文件，加载特征名称
        feature_names = None
        if args.features_file:
            logger.info(f"从文件加载特征: {args.features_file}")
            features_df = pd.read_csv(args.features_file, index_col=0)
            feature_names = features_df.columns.tolist()
        
        # 选择特征
        selected_features = selector.select_features_from_model(model, feature_names)
        
        # 保存选中的特征
        selector.save_selected_features(args.feature_set_name)
        
        # 绘制特征重要性
        plot_path = os.path.join(args.output_dir, 
                                f"{args.feature_set_name or 'feature_importance'}.png")
        selector.plot_feature_importance(save_path=plot_path)
        
        # 重新训练模型
        if args.retrain:
            if not args.features_file or not args.target_file:
                logger.error("重新训练需要提供特征文件和目标文件")
                return
                
            # 加载数据
            X = pd.read_csv(args.features_file, index_col=0)
            targets = pd.read_csv(args.target_file, index_col=0)
            y = targets.iloc[:, 0]
            
            # 删除包含NaN的行
            valid_indices = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_indices]
            y = y[valid_indices]
            
            # 数据分割
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.random_state
            )
            
            # 从剩余数据中分割验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=args.val_size / (1 - args.test_size),
                random_state=args.random_state
            )
            
            # 重新训练
            new_model, metrics = retrain_with_selected_features(
                args.model_type,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                selected_features
            )
            
            # 保存模型
            model_dir = os.path.join(args.output_dir, args.feature_set_name or "retrained_model")
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, "model.pkl")
            joblib.dump(new_model, model_path)
            
            # 保存指标
            metrics_path = os.path.join(model_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
                
            logger.info(f"重新训练的模型保存到: {model_path}")
        
    # 模型集成模式
    elif args.mode == 'ensemble':
        if not args.model_files:
            logger.error("请提供模型文件路径列表")
            return
            
        if not args.features_file:
            logger.error("请提供特征文件路径")
            return
            
        # 加载特征
        logger.info(f"加载特征: {args.features_file}")
        X = pd.read_csv(args.features_file, index_col=0)
        
        # 创建模型集成
        ensemble, models, importance = create_model_ensemble(
            args.model_files,
            X,
            weights=args.weights,
            method=args.ensemble_method
        )
        
        # 生成预测
        predictions = ensemble.predict(X)
        
        # 保存预测结果
        if args.output_file:
            # 创建包含预测的DataFrame
            pred_df = pd.DataFrame(index=X.index)
            pred_df['prediction'] = predictions
            
            # 保存到文件
            pred_df.to_csv(args.output_file)
            logger.info(f"集成预测结果保存到: {args.output_file}")
            
        # 保存特征重要性
        if importance is not None and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            importance_file = os.path.join(args.output_dir, "ensemble_importance.csv")
            importance.to_csv(importance_file, index=False)
            
            # 绘制图表
            plt.figure(figsize=(12, 8))
            top_features = importance.head(20)
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Ensemble Model Feature Importance')
            plt.tight_layout()
            
            plot_file = os.path.join(args.output_dir, "ensemble_importance.png")
            plt.savefig(plot_file)
            logger.info(f"集成特征重要性保存到: {importance_file}")
        
    # 预测平滑模式
    elif args.mode == 'smooth':
        if not args.predictions_file:
            logger.error("请提供预测文件路径")
            return
            
        # 加载预测
        logger.info(f"加载预测: {args.predictions_file}")
        pred_df = pd.read_csv(args.predictions_file, index_col=0)
        
        # 获取预测列
        if len(pred_df.columns) > 1:
            logger.warning(f"发现多个列，使用第一列作为预测: {pred_df.columns[0]}")
            
        predictions = pred_df.iloc[:, 0].values
        
        # 创建平滑处理器
        smoother = PredictionSmoother(
            window_size=args.window_size,
            method=args.ensemble_method
        )
        
        # 平滑预测
        smoothed = smoother.smooth(pd.Series(predictions))
        
        # 创建包含平滑预测的DataFrame
        smooth_df = pd.DataFrame(index=pred_df.index)
        smooth_df['smoothed_prediction'] = smoothed
        
        # 添加原始预测进行对比
        smooth_df['original_prediction'] = predictions
        
        # 保存到文件
        if args.output_file:
            smooth_df.to_csv(args.output_file)
            logger.info(f"平滑预测结果保存到: {args.output_file}")
            
        # 绘制对比图
        smoother.plot_comparison(
            original=pd.Series(predictions),
            smoothed=smoothed,
            targets=None,
            title=f'Original vs Smoothed Predictions (Window Size: {args.window_size}, Method: {args.ensemble_method})',
            save_path=os.path.join(args.output_dir, "prediction_smoothing.png")
        )
    
    logger.info(f"{args.mode}模式操作完成")


if __name__ == "__main__":
    main() 