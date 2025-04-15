"""
特征选择和模型集成模块，用于基于特征重要性筛选重要特征，并对预测结果进行集成或平滑处理
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Optional, Tuple
import argparse
from datetime import datetime


class FeatureSelector:
    """
    特征选择器，基于模型重要性或SHAP值选择重要特征
    """
    
    def __init__(
        self,
        selection_method: str = 'importance',
        n_features: int = 20,
        output_dir: str = 'data/processed/features/selected',
        importance_threshold: float = 0.01,
        verbose: bool = True
    ):
        """
        初始化特征选择器
        
        Args:
            selection_method: 特征选择方法，支持'importance'和'shap'
            n_features: 保留的特征数量，当selection_method为'importance'时使用
            output_dir: 输出目录
            importance_threshold: 重要性阈值，低于此值的特征将被过滤
            verbose: 是否输出详细信息
        """
        self.selection_method = selection_method
        self.n_features = n_features
        self.output_dir = output_dir
        self.importance_threshold = importance_threshold
        self.verbose = verbose
        
        self.logger = logging.getLogger(__name__)
        self.selected_features = None
        self.feature_importance = None
        
    def select_features_by_importance(
        self,
        model,
        feature_names: List[str]
    ) -> List[str]:
        """
        基于模型特征重要性选择特征
        
        Args:
            model: 训练好的模型，需要有feature_importances_属性
            feature_names: 特征名称列表
            
        Returns:
            List[str]: 选定的特征名称
        """
        if not hasattr(model, 'feature_importances_'):
            self.logger.error("模型没有feature_importances_属性")
            return []
        
        # 获取特征重要性
        importances = model.feature_importances_
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 过滤低重要性特征
        filtered_df = importance_df[importance_df['importance'] > self.importance_threshold]
        
        # 选择前n个特征
        if len(filtered_df) > self.n_features:
            selected_df = filtered_df.head(self.n_features)
        else:
            selected_df = filtered_df
            
        self.selected_features = selected_df['feature'].tolist()
        self.feature_importance = dict(zip(selected_df['feature'], selected_df['importance']))
        
        if self.verbose:
            self.logger.info(f"基于重要性选择了{len(self.selected_features)}个特征")
            
        return self.selected_features
    
    def select_features_by_shap(
        self,
        model,
        X: pd.DataFrame,
        n_background_samples: int = 100,
        random_state: int = 42
    ) -> List[str]:
        """
        基于SHAP值选择特征
        
        Args:
            model: 训练好的模型
            X: 特征数据
            n_background_samples: 用于SHAP计算的背景样本数量
            random_state: 随机种子
            
        Returns:
            List[str]: 选定的特征名称
        """
        try:
            import shap
        except ImportError:
            self.logger.error("使用SHAP需要安装shap库：pip install shap")
            return []
        
        self.logger.info("开始计算SHAP值...")
        
        # 准备背景数据
        if len(X) > n_background_samples:
            X_sample = X.sample(n_background_samples, random_state=random_state)
        else:
            X_sample = X
            
        # 判断模型类型
        if hasattr(model, 'predict_proba'):
            explainer = shap.Explainer(model, X_sample)
        else:
            explainer = shap.Explainer(model)

        # 计算SHAP值
        shap_values = explainer(X_sample)
        
        # 计算SHAP重要性
        feature_names = X.columns.tolist()
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values.values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        # 过滤低重要性特征
        filtered_df = shap_importance[shap_importance['importance'] > self.importance_threshold]
        
        # 选择前n个特征
        if len(filtered_df) > self.n_features:
            selected_df = filtered_df.head(self.n_features)
        else:
            selected_df = filtered_df
            
        self.selected_features = selected_df['feature'].tolist()
        self.feature_importance = dict(zip(selected_df['feature'], selected_df['importance']))
        
        if self.verbose:
            self.logger.info(f"基于SHAP值选择了{len(self.selected_features)}个特征")
            
        return self.selected_features
    
    def plot_feature_importance(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制特征重要性图
        
        Args:
            save_path: 图表保存路径
            
        Returns:
            plt.Figure: 图表对象
        """
        if self.feature_importance is None or not self.feature_importance:
            self.logger.warning("特征重要性为空，无法绘制图表")
            return None
        
        # 准备数据
        features = list(self.feature_importance.keys())
        importances = list(self.feature_importance.values())
        
        # 排序
        indices = np.argsort(importances)
        features = [features[i] for i in indices[::-1]]
        importances = [importances[i] for i in indices[::-1]]
        
        # 绘制图表
        plt.figure(figsize=(12, 8))
        plt.barh(features, importances)
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.title('特征重要性')
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"特征重要性图保存到: {save_path}")
            
        return plt.gcf()
    
    def save_selected_features(self, output_file: Optional[str] = None) -> str:
        """
        保存选定的特征
        
        Args:
            output_file: 输出文件路径，默认使用时间戳
            
        Returns:
            str: 输出文件路径
        """
        if self.selected_features is None or not self.selected_features:
            self.logger.warning("选定的特征为空，无法保存")
            return None
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 使用时间戳作为文件名
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"selected_features_{timestamp}.json")
        
        # 保存特征列表
        with open(output_file, 'w') as f:
            json.dump(self.selected_features, f, indent=4)
            
        self.logger.info(f"选定的特征保存到: {output_file}")
        
        # 如果有特征重要性，也保存它
        if self.feature_importance:
            importance_file = output_file.replace('.json', '_importance.json')
            with open(importance_file, 'w') as f:
                json.dump(self.feature_importance, f, indent=4)
                
            self.logger.info(f"特征重要性保存到: {importance_file}")
        
        return output_file


class PredictionEnsemble:
    """
    预测集成器，用于集成多个模型的预测结果或平滑单个模型的预测结果
    """
    
    def __init__(
        self,
        ensemble_method: str = 'average',
        weights: Optional[List[float]] = None,
        window_size: int = 5,
        verbose: bool = True
    ):
        """
        初始化预测集成器
        
        Args:
            ensemble_method: 集成方法，支持'average'、'weighted_average'、'median'
            weights: 权重列表，当ensemble_method为'weighted_average'时使用
            window_size: 滑动窗口大小，用于平滑预测结果
            verbose: 是否输出详细信息
        """
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.window_size = window_size
        self.verbose = verbose
        
        self.logger = logging.getLogger(__name__)
        
    def ensemble_predictions(
        self,
        predictions: List[np.ndarray]
    ) -> np.ndarray:
        """
        集成多个模型的预测结果
        
        Args:
            predictions: 预测结果列表，每个元素是一个模型的预测
            
        Returns:
            np.ndarray: 集成后的预测结果
        """
        if not predictions:
            self.logger.error("预测结果列表为空")
            return None
        
        # 确保所有预测具有相同的形状
        shapes = [p.shape for p in predictions]
        if len(set(shapes)) > 1:
            self.logger.error(f"预测结果形状不一致: {shapes}")
            return None
        
        # 根据集成方法生成结果
        if self.ensemble_method == 'average':
            result = np.mean(predictions, axis=0)
            
        elif self.ensemble_method == 'weighted_average':
            if self.weights is None or len(self.weights) != len(predictions):
                self.logger.warning(f"权重列表为空或长度不匹配，使用均匀权重")
                self.weights = [1.0 / len(predictions)] * len(predictions)
                
            # 归一化权重
            weights = np.array(self.weights) / sum(self.weights)
            
            # 计算加权平均
            result = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                result += pred * weights[i]
                
        elif self.ensemble_method == 'median':
            result = np.median(predictions, axis=0)
            
        else:
            self.logger.error(f"不支持的集成方法: {self.ensemble_method}")
            return None
            
        if self.verbose:
            self.logger.info(f"使用{self.ensemble_method}方法集成了{len(predictions)}个预测结果")
            
        return result
    
    def smooth_prediction(
        self,
        prediction: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> np.ndarray:
        """
        平滑单个模型的预测结果
        
        Args:
            prediction: 预测结果
            dates: 日期索引，用于按时间排序
            
        Returns:
            np.ndarray: 平滑后的预测结果
        """
        if prediction.size == 0:
            self.logger.error("预测结果为空")
            return np.array([])
        
        # 如果提供了日期，按日期排序
        if dates is not None:
            sorted_indices = np.argsort(dates)
            prediction = prediction[sorted_indices]
        
        # 应用滑动窗口平均
        smoothed = np.zeros_like(prediction)
        
        for i in range(len(prediction)):
            start = max(0, i - self.window_size // 2)
            end = min(len(prediction), i + self.window_size // 2 + 1)
            smoothed[i] = np.mean(prediction[start:end])
        
        # 如果之前有排序，恢复原始顺序
        if dates is not None:
            inverse_indices = np.argsort(sorted_indices)
            smoothed = smoothed[inverse_indices]
            
        if self.verbose:
            self.logger.info(f"使用大小为{self.window_size}的滑动窗口平滑了预测结果")
            
        return smoothed


def retrain_with_selected_features(
    model_path: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: List[str],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    output_path: Optional[str] = None,
    early_stopping_rounds: int = 10,
    verbose: bool = True
) -> str:
    """
    使用选定的特征重新训练模型
    
    Args:
        model_path: 原始模型路径
        X_train: 训练特征
        y_train: 训练目标
        selected_features: 选定的特征列表
        X_val: 验证特征
        y_val: 验证目标
        output_path: 输出模型路径
        early_stopping_rounds: 早停轮数
        verbose: 是否输出详细信息
        
    Returns:
        str: 输出模型路径
    """
    logger = logging.getLogger(__name__)
    
    # 加载原始模型
    logger.info(f"加载原始模型: {model_path}")
    original_model = joblib.load(model_path)
    
    # 获取模型参数
    if hasattr(original_model, 'get_params'):
        params = original_model.get_params()
    else:
        logger.error("模型不支持get_params方法")
        return None
        
    # 筛选特征
    X_train_selected = X_train[selected_features]
    if X_val is not None:
        X_val_selected = X_val[selected_features]
    else:
        X_val_selected = None
        
    # 创建新模型
    if 'xgb' in str(type(original_model)).lower():
        import xgboost as xgb
        
        # 移除无法传递给XGBoost的参数
        if 'callbacks' in params:
            del params['callbacks']
        
        # 创建新模型
        model = xgb.XGBRegressor(**params)
        
        # 训练模型
        if X_val_selected is not None and y_val is not None:
            # 使用回调而不是直接参数
            callbacks = [
                xgb.callback.EarlyStopping(
                    rounds=early_stopping_rounds,
                    save_best=True
                )
            ]
            model.fit(
                X_train_selected, y_train,
                eval_set=[(X_val_selected, y_val)],
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            model.fit(X_train_selected, y_train)
            
    elif 'lightgbm' in str(type(original_model)).lower():
        import lightgbm as lgb
        
        # 创建新模型
        model = lgb.LGBMRegressor(**params)
        
        # 训练模型
        if X_val_selected is not None and y_val is not None:
            model.fit(
                X_train_selected, y_train,
                eval_set=[(X_val_selected, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose
            )
        else:
            model.fit(X_train_selected, y_train)
            
    else:
        logger.error(f"不支持的模型类型: {type(original_model)}")
        return None
        
    # 保存模型
    if output_path is None:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        output_path = os.path.join(model_dir, f"selected_{model_name}")
        
    logger.info(f"保存模型到: {output_path}")
    joblib.dump(model, output_path)
    
    # 保存选定的特征
    features_path = output_path + ".features.json"
    with open(features_path, 'w') as f:
        json.dump(selected_features, f, indent=4)
        
    logger.info(f"保存选定的特征到: {features_path}")
    
    return output_path


def create_model_ensemble(
    model_paths: List[str],
    X: pd.DataFrame,
    selected_features_paths: Optional[List[str]] = None,
    ensemble_method: str = 'average',
    weights: Optional[List[float]] = None,
    use_smoothing: bool = False,
    window_size: int = 5,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, str]:
    """
    创建模型集成并生成预测
    
    Args:
        model_paths: 模型路径列表
        X: 预测数据
        selected_features_paths: 选定特征路径列表，与model_paths对应
        ensemble_method: 集成方法
        weights: 权重列表
        use_smoothing: 是否使用平滑
        window_size: 滑动窗口大小
        output_path: 输出路径
        verbose: 是否输出详细信息
        
    Returns:
        Tuple[np.ndarray, str]: (集成预测结果, 输出路径)
    """
    logger = logging.getLogger(__name__)
    
    if not model_paths:
        logger.error("模型路径列表为空")
        return None, None
        
    # 加载选定的特征
    if selected_features_paths:
        if len(selected_features_paths) != len(model_paths):
            logger.warning(f"特征路径数量({len(selected_features_paths)})与模型数量({len(model_paths)})不匹配")
            selected_features_list = [None] * len(model_paths)
        else:
            selected_features_list = []
            for path in selected_features_paths:
                if path and os.path.exists(path):
                    with open(path, 'r') as f:
                        selected_features_list.append(json.load(f))
                else:
                    selected_features_list.append(None)
    else:
        selected_features_list = [None] * len(model_paths)
    
    # 加载模型并生成预测
    predictions = []
    model_info = []
    
    for i, model_path in enumerate(model_paths):
        # 加载模型
        logger.info(f"加载模型: {model_path}")
        model = joblib.load(model_path)
        
        # 获取特征列表
        selected_features = selected_features_list[i]
        if selected_features:
            X_model = X[selected_features]
            logger.info(f"使用{len(selected_features)}个选定特征进行预测")
        else:
            # 尝试使用模型的特征名称
            if hasattr(model, 'feature_names'):
                feature_names = model.feature_names
                X_model = X[feature_names]
                logger.info(f"使用模型的{len(feature_names)}个特征进行预测")
            elif hasattr(model, 'feature_name_'):
                feature_names = model.feature_name_
                X_model = X[feature_names]
                logger.info(f"使用模型的{len(feature_names)}个特征进行预测")
            else:
                # 使用所有特征
                X_model = X
                logger.info(f"使用所有{X.shape[1]}个特征进行预测")
        
        # 生成预测
        pred = model.predict(X_model)
        predictions.append(pred)
        
        # 记录模型信息
        model_type = type(model).__name__
        model_info.append({
            'model_path': model_path,
            'model_type': model_type,
            'features_used': selected_features if selected_features else 'all'
        })
    
    # 创建集成器
    ensembler = PredictionEnsemble(
        ensemble_method=ensemble_method,
        weights=weights,
        window_size=window_size,
        verbose=verbose
    )
    
    # 集成预测结果
    ensemble_pred = ensembler.ensemble_predictions(predictions)
    
    # 应用平滑
    if use_smoothing:
        ensemble_pred = ensembler.smooth_prediction(ensemble_pred)
    
    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, ensemble_pred)
        
        # 保存模型信息
        info_path = output_path.replace('.npy', '_info.json')
        with open(info_path, 'w') as f:
            json.dump({
                'models': model_info,
                'ensemble_method': ensemble_method,
                'weights': weights,
                'smoothing': {
                    'used': use_smoothing,
                    'window_size': window_size
                }
            }, f, indent=4)
            
        logger.info(f"集成预测保存到: {output_path}")
        logger.info(f"集成信息保存到: {info_path}")
    
    return ensemble_pred, output_path


def main():
    """
    主函数，处理命令行参数并执行相应操作
    """
    parser = argparse.ArgumentParser(description="特征选择和模型集成")
    
    # 基本参数
    parser.add_argument("--mode", type=str, required=True,
                        choices=["select", "ensemble", "smooth"],
                        help="操作模式：select(特征选择), ensemble(模型集成), smooth(预测平滑)")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="输出目录")
    parser.add_argument("--verbose", action="store_true", help="是否输出详细信息")
    
    # 特征选择参数
    feature_selection_group = parser.add_argument_group("特征选择参数")
    feature_selection_group.add_argument("--model_file", type=str,
                        help="模型文件路径")
    feature_selection_group.add_argument("--features_file", type=str,
                        help="特征文件路径")
    feature_selection_group.add_argument("--selection_method", type=str,
                        choices=["importance", "shap"],
                        default="importance", help="特征选择方法")
    feature_selection_group.add_argument("--n_features", type=int,
                        default=20, help="选择的特征数量")
    feature_selection_group.add_argument("--importance_threshold", type=float,
                        default=0.01, help="重要性阈值")
    feature_selection_group.add_argument("--plot_importance", action="store_true",
                        help="是否绘制特征重要性图")
    
    # 特征重选模型参数
    feature_reselection_group = parser.add_argument_group("特征重选模型参数")
    feature_reselection_group.add_argument("--retrain_model", action="store_true",
                        help="是否使用选定的特征重新训练模型")
    feature_reselection_group.add_argument("--train_features_file", type=str,
                        help="训练特征文件路径")
    feature_reselection_group.add_argument("--train_target_file", type=str,
                        help="训练目标文件路径")
    feature_reselection_group.add_argument("--val_features_file", type=str,
                        help="验证特征文件路径")
    feature_reselection_group.add_argument("--val_target_file", type=str,
                        help="验证目标文件路径")
    feature_reselection_group.add_argument("--early_stopping_rounds", type=int,
                        default=10, help="早停轮数")
    
    # 模型集成参数
    ensemble_group = parser.add_argument_group("模型集成参数")
    ensemble_group.add_argument("--model_files", type=str, nargs='+',
                        help="模型文件路径列表")
    ensemble_group.add_argument("--selected_features_files", type=str, nargs='+',
                        help="已选择特征文件路径列表")
    ensemble_group.add_argument("--predict_features_file", type=str,
                        help="预测特征文件路径")
    ensemble_group.add_argument("--ensemble_method", type=str,
                        choices=["average", "weighted_average", "median"],
                        default="average", help="集成方法")
    ensemble_group.add_argument("--weights", type=float, nargs='+',
                        help="模型权重列表")
    
    # 平滑参数
    smoothing_group = parser.add_argument_group("平滑参数")
    smoothing_group.add_argument("--window_size", type=int,
                        default=5, help="滑动窗口大小")
    smoothing_group.add_argument("--prediction_file", type=str,
                        help="预测文件路径")
    smoothing_group.add_argument("--dates_file", type=str,
                        help="日期文件路径")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据模式执行相应操作
    if args.mode == "select":
        # 检查必要参数
        if not args.model_file:
            logger.error("请提供模型文件路径")
            return
            
        if not args.features_file:
            logger.error("请提供特征文件路径")
            return
            
        # 加载模型
        logger.info(f"加载模型: {args.model_file}")
        model = joblib.load(args.model_file)
        
        # 加载特征
        logger.info(f"加载特征: {args.features_file}")
        X = pd.read_csv(args.features_file, index_col=0)
        
        # 创建特征选择器
        selector = FeatureSelector(
            selection_method=args.selection_method,
            n_features=args.n_features,
            output_dir=os.path.join(args.output_dir, "features/selected"),
            importance_threshold=args.importance_threshold,
            verbose=args.verbose
        )
        
        # 选择特征
        if args.selection_method == "importance":
            selected_features = selector.select_features_by_importance(
                model, X.columns.tolist()
            )
        elif args.selection_method == "shap":
            selected_features = selector.select_features_by_shap(
                model, X
            )
        
        # 绘制特征重要性
        if args.plot_importance:
            plot_path = os.path.join(args.output_dir, "features/plots",
                                     f"importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            selector.plot_feature_importance(save_path=plot_path)
        
        # 保存选定的特征
        features_file = selector.save_selected_features()
        
        # 重新训练模型
        if args.retrain_model:
            if not args.train_features_file or not args.train_target_file:
                logger.error("要重新训练模型，请提供训练特征和目标文件")
                return
                
            # 加载训练数据
            logger.info(f"加载训练特征: {args.train_features_file}")
            X_train = pd.read_csv(args.train_features_file, index_col=0)
            
            logger.info(f"加载训练目标: {args.train_target_file}")
            y_train = pd.read_csv(args.train_target_file, index_col=0).iloc[:, 0]
            
            # 加载验证数据
            X_val = None
            y_val = None
            if args.val_features_file and args.val_target_file:
                logger.info(f"加载验证特征: {args.val_features_file}")
                X_val = pd.read_csv(args.val_features_file, index_col=0)
                
                logger.info(f"加载验证目标: {args.val_target_file}")
                y_val = pd.read_csv(args.val_target_file, index_col=0).iloc[:, 0]
                
            # 重新训练模型
            output_model_dir = os.path.join(args.output_dir, "models/selected")
            os.makedirs(output_model_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_model_path = os.path.join(
                output_model_dir,
                f"selected_model_{timestamp}.pkl"
            )
            
            retrain_with_selected_features(
                model_path=args.model_file,
                X_train=X_train,
                y_train=y_train,
                selected_features=selected_features,
                X_val=X_val,
                y_val=y_val,
                output_path=output_model_path,
                early_stopping_rounds=args.early_stopping_rounds,
                verbose=args.verbose
            )
            
    elif args.mode == "ensemble":
        # 检查必要参数
        if not args.model_files:
            logger.error("请提供模型文件路径列表")
            return
            
        if not args.predict_features_file:
            logger.error("请提供预测特征文件路径")
            return
            
        # 加载预测特征
        logger.info(f"加载预测特征: {args.predict_features_file}")
        X = pd.read_csv(args.predict_features_file, index_col=0)
        
        # 创建输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            args.output_dir,
            "predictions",
            f"ensemble_pred_{timestamp}.npy"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建模型集成
        _, _ = create_model_ensemble(
            model_paths=args.model_files,
            X=X,
            selected_features_paths=args.selected_features_files,
            ensemble_method=args.ensemble_method,
            weights=args.weights,
            use_smoothing=False,  # 在smooth模式中处理平滑
            window_size=args.window_size,
            output_path=output_path,
            verbose=args.verbose
        )
        
    elif args.mode == "smooth":
        # 检查必要参数
        if not args.prediction_file:
            logger.error("请提供预测文件路径")
            return
            
        # 加载预测
        logger.info(f"加载预测: {args.prediction_file}")
        if args.prediction_file.endswith('.npy'):
            prediction = np.load(args.prediction_file)
        else:
            prediction = pd.read_csv(args.prediction_file, index_col=0).values.flatten()
            
        # 加载日期
        dates = None
        if args.dates_file:
            logger.info(f"加载日期: {args.dates_file}")
            dates = pd.read_csv(args.dates_file, parse_dates=True, index_col=0).index
            
        # 创建平滑器
        smoother = PredictionEnsemble(
            window_size=args.window_size,
            verbose=args.verbose
        )
        
        # 平滑预测
        smoothed = smoother.smooth_prediction(prediction, dates)
        
        # 创建输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            args.output_dir,
            "predictions",
            f"smoothed_pred_{timestamp}.npy"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存结果
        np.save(output_path, smoothed)
        logger.info(f"平滑预测保存到: {output_path}")
        
        # 如果原始预测是CSV，也以CSV格式保存平滑结果
        if not args.prediction_file.endswith('.npy'):
            csv_path = output_path.replace('.npy', '.csv')
            
            # 尝试使用与原始预测相同的格式
            try:
                original_df = pd.read_csv(args.prediction_file, index_col=0)
                smoothed_df = pd.DataFrame(
                    smoothed.reshape(original_df.shape),
                    index=original_df.index,
                    columns=original_df.columns
                )
                smoothed_df.to_csv(csv_path)
                logger.info(f"平滑预测CSV保存到: {csv_path}")
            except Exception as e:
                logger.warning(f"无法以CSV格式保存平滑结果: {e}")
    
    logger.info("处理完成")


if __name__ == "__main__":
    main() 