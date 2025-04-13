"""
模型优化模块

用于调整XGBoost的正则化参数，减轻过拟合问题并提高模型泛化能力
"""

import os
import sys
import logging
import argparse
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# 导入模块
try:
    from src.model_regularization import RegularizationOptimizer
    from src.feature_selection_ensemble import FeatureSelector, PredictionEnsemble, create_model_ensemble
except ImportError:
    # 处理相对导入
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from src.model_regularization import RegularizationOptimizer
    from src.feature_selection_ensemble import FeatureSelector, PredictionEnsemble, create_model_ensemble


class ModelOptimizer:
    """
    模型优化器类，用于调整模型参数以减轻过拟合
    """
    
    def __init__(
        self,
        search_method: str = 'random',
        n_iter: int = 20,
        cv: int = 5,
        scoring: str = 'neg_root_mean_squared_error',
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
        output_dir: str = 'data/results/optimization'
    ):
        """
        初始化模型优化器
        
        Args:
            search_method: 参数搜索方法，'grid' 或 'random'
            n_iter: 随机搜索迭代次数
            cv: 交叉验证折数
            scoring: 评分标准
            n_jobs: 并行任务数，-1表示使用所有可用核心
            random_state: 随机种子
            verbose: 详细程度
            output_dir: 输出目录
        """
        self.search_method = search_method
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ModelOptimizer')
        
        # 存储搜索结果
        self.search_results = None
        self.best_params = None
        self.best_score = None
    
    def get_xgboost_param_grid(
        self,
        is_random: bool = True
    ) -> Dict[str, Union[List, Any]]:
        """
        获取XGBoost参数网格
        
        Args:
            is_random: 是否为随机搜索提供连续参数范围
            
        Returns:
            Dict: 参数网格
        """
        if is_random:
            # 随机搜索的参数范围，可以是连续的
            param_grid = {
                'max_depth': [3, 4, 5, 6, 7, 8],
                'min_child_weight': [1, 3, 5, 7],
                'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.001, 0.01, 0.1, 1, 10, 100],
                'reg_lambda': [0.01, 0.1, 1, 10, 100],
                'learning_rate': [0.01, 0.05, 0.1, 0.2]
            }
        else:
            # 网格搜索的参数范围，必须是离散的
            param_grid = {
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.2, 0.4],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0.1, 1, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        return param_grid
    
    def optimize_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        val_size: float = 0.2,
        param_grid: Optional[Dict[str, Union[List, Any]]] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 50,
        n_estimators: int = 1000
    ) -> Tuple[Dict[str, Any], XGBRegressor]:
        """
        优化XGBoost模型参数
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征(可选)
            y_val: 验证目标(可选)
            val_size: 验证集比例(如果X_val和y_val为None)
            param_grid: 参数网格(可选)
            fixed_params: 固定参数(可选)
            early_stopping_rounds: 早停轮数
            n_estimators: 最大迭代次数
            
        Returns:
            Tuple[Dict[str, Any], XGBRegressor]: 最佳参数和优化后的模型
        """
        # 创建验证集(如果未提供)
        if X_val is None or y_val is None:
            self.logger.info(f"使用 {val_size} 的比例分割训练集创建验证集")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=self.random_state
            )
        
        # 获取参数网格(如果未提供)
        if param_grid is None:
            param_grid = self.get_xgboost_param_grid(is_random=(self.search_method == 'random'))
        
        # 设置基本参数
        base_params = {
            'objective': 'reg:squarederror',
            'n_estimators': n_estimators,
            'random_state': self.random_state,
            'tree_method': 'hist'  # 使用直方图方法加速训练
        }
        
        # 合并固定参数
        if fixed_params:
            base_params.update(fixed_params)
        
        # 创建基础模型
        model = XGBRegressor(**base_params)
        
        # 创建搜索对象
        if self.search_method == 'grid':
            self.logger.info("使用网格搜索优化参数")
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        else:
            self.logger.info("使用随机搜索优化参数")
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose
            )
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行搜索
        search.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 保存搜索结果
        self.search_results = search
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        
        # 打印最佳参数和分数
        self.logger.info(f"参数优化完成，耗时 {elapsed_time:.2f} 秒")
        self.logger.info(f"最佳参数: {self.best_params}")
        self.logger.info(f"最佳分数 ({self.scoring}): {self.best_score:.6f}")
        
        # 使用最佳参数重新训练模型
        best_model = XGBRegressor(**{**base_params, **self.best_params})
        
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        # 返回最佳参数和模型
        return self.best_params, best_model
    
    def save_optimization_results(
        self,
        model: Any,
        best_params: Dict[str, Any],
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        model_name: str = 'xgboost',
        feature_names: Optional[List[str]] = None
    ) -> str:
        """
        保存优化结果
        
        Args:
            model: 优化后的模型
            best_params: 最佳参数
            X_test: 测试特征(可选)
            y_test: 测试目标(可选)
            model_name: 模型名称
            feature_names: 特征名称列表(可选)
            
        Returns:
            str: 保存的模型路径
        """
        # 创建时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存模型
        model_path = os.path.join(self.output_dir, f"optimized_{model_name}_{timestamp}.joblib")
        joblib.dump(model, model_path)
        self.logger.info(f"优化后的模型已保存至: {model_path}")
        
        # 保存参数
        params_path = os.path.join(self.output_dir, f"params_{model_name}_{timestamp}.json")
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        self.logger.info(f"最佳参数已保存至: {params_path}")
        
        # 如果提供了测试集，评估模型
        if X_test is not None and y_test is not None:
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 保存指标
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
            
            metrics_path = os.path.join(self.output_dir, f"metrics_{model_name}_{timestamp}.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"评估指标已保存至: {metrics_path}")
            
            # 绘制特征重要性
            if hasattr(model, 'feature_importances_') and feature_names is not None:
                self.plot_feature_importance(
                    model, feature_names,
                    save_path=os.path.join(self.output_dir, f"feature_importance_{model_name}_{timestamp}.png")
                )
        
        return model_path
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> None:
        """
        绘制特征重要性
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            top_n: 显示前N个重要特征
            save_path: 保存路径(可选)
        """
        if not hasattr(model, 'feature_importances_'):
            self.logger.warning("模型没有feature_importances_属性，无法绘制特征重要性")
            return
        
        # 获取特征重要性
        importances = model.feature_importances_
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # 取前N个特征
        if top_n is not None and top_n < len(importance_df):
            importance_df = importance_df.head(top_n)
        
        # 绘制特征重要性
        plt.figure(figsize=(10, 8))
        plt.barh(
            importance_df['feature'][::-1],
            importance_df['importance'][::-1]
        )
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.title(f'前 {len(importance_df)} 个重要特征')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"特征重要性图已保存至: {save_path}")
        
        plt.close()


def optimize_model(
    model_type: str,
    X_train_file: str,
    y_train_file: str,
    X_val_file: Optional[str] = None,
    y_val_file: Optional[str] = None,
    param_ranges: Optional[Dict] = None,
    n_iter: int = 20,
    cv: int = 5,
    eval_metric: str = 'neg_mean_squared_error',
    early_stopping_rounds: int = 10,
    output_dir: str = 'data/models/optimized',
    verbose: bool = True
) -> str:
    """
    对模型进行正则化参数优化
    
    Args:
        model_type: 模型类型，支持'xgboost'和'lightgbm'
        X_train_file: 训练特征文件路径
        y_train_file: 训练目标文件路径
        X_val_file: 验证特征文件路径
        y_val_file: 验证目标文件路径
        param_ranges: 自定义参数范围
        n_iter: 随机搜索迭代次数
        cv: 交叉验证折数
        eval_metric: 评估指标
        early_stopping_rounds: 早停轮数
        output_dir: 输出目录
        verbose: 是否输出详细信息
        
    Returns:
        str: 优化后模型路径
    """
    logger = logging.getLogger(__name__)
    
    # 加载数据
    logger.info(f"加载训练特征: {X_train_file}")
    X_train = pd.read_csv(X_train_file, index_col=0)
    
    logger.info(f"加载训练目标: {y_train_file}")
    y_train = pd.read_csv(y_train_file, index_col=0).iloc[:, 0]
    
    # 加载验证数据
    X_val = None
    y_val = None
    if X_val_file and y_val_file:
        logger.info(f"加载验证特征: {X_val_file}")
        X_val = pd.read_csv(X_val_file, index_col=0)
        
        logger.info(f"加载验证目标: {y_val_file}")
        y_val = pd.read_csv(y_val_file, index_col=0).iloc[:, 0]
    
    # 创建优化器
    optimizer = RegularizationOptimizer(
        model_type=model_type,
        optimization_method='random',
        n_iter=n_iter,
        cv=cv,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        output_dir=output_dir,
        verbose=verbose
    )
    
    # 如果提供了自定义参数范围，设置参数网格
    if param_ranges:
        optimizer.param_grid = param_ranges
    
    # 优化参数
    optimizer.optimize_parameters(X_train, y_train)
    
    # 使用最佳参数训练模型
    model_path = optimizer.train_with_best_params(
        X_train, y_train, X_val, y_val
    )
    
    # 评估模型
    if X_val is not None and y_val is not None:
        optimizer.evaluate_model(X_val, y_val)
    
    # 保存结果
    optimizer.save_results()
    
    # 绘制特征重要性
    plot_path = os.path.join(output_dir, 'plots', f'importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    optimizer.plot_feature_importance(plot_path)
    
    return model_path


def select_features_and_retrain(
    model_path: str,
    X_train_file: str,
    y_train_file: str,
    selection_method: str = 'importance',
    n_features: int = 20,
    X_val_file: Optional[str] = None,
    y_val_file: Optional[str] = None,
    early_stopping_rounds: int = 10,
    output_dir: str = 'data/models/feature_selected',
    verbose: bool = True
) -> Dict:
    """
    基于特征重要性选择特征并重新训练模型
    
    Args:
        model_path: 模型路径
        X_train_file: 训练特征文件路径
        y_train_file: 训练目标文件路径
        selection_method: 特征选择方法
        n_features: 选择的特征数量
        X_val_file: 验证特征文件路径
        y_val_file: 验证目标文件路径
        early_stopping_rounds: 早停轮数
        output_dir: 输出目录
        verbose: 是否输出详细信息
        
    Returns:
        Dict: 包含模型路径和选定特征的字典
    """
    logger = logging.getLogger(__name__)
    
    # 从src.feature_selection_ensemble导入必要函数
    from src.feature_selection_ensemble import (
        FeatureSelector, retrain_with_selected_features
    )
    
    # 加载数据
    logger.info(f"加载训练特征: {X_train_file}")
    X_train = pd.read_csv(X_train_file, index_col=0)
    
    logger.info(f"加载训练目标: {y_train_file}")
    y_train = pd.read_csv(y_train_file, index_col=0).iloc[:, 0]
    
    # 加载验证数据
    X_val = None
    y_val = None
    if X_val_file and y_val_file:
        logger.info(f"加载验证特征: {X_val_file}")
        X_val = pd.read_csv(X_val_file, index_col=0)
        
        logger.info(f"加载验证目标: {y_val_file}")
        y_val = pd.read_csv(y_val_file, index_col=0).iloc[:, 0]
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model = joblib.load(model_path)
    
    # 创建特征选择器
    selector = FeatureSelector(
        selection_method=selection_method,
        n_features=n_features,
        output_dir=os.path.join(output_dir, 'features'),
        verbose=verbose
    )
    
    # 选择特征
    if selection_method == 'importance':
        selected_features = selector.select_features_by_importance(
            model, X_train.columns.tolist()
        )
    else:  # shap
        selected_features = selector.select_features_by_shap(
            model, X_train
        )
    
    # 绘制特征重要性
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    selector.plot_feature_importance(save_path=plot_path)
    
    # 保存选定的特征
    features_file = selector.save_selected_features()
    
    # 使用选定的特征重新训练模型
    new_model_path = retrain_with_selected_features(
        model_path=model_path,
        X_train=X_train,
        y_train=y_train,
        selected_features=selected_features,
        X_val=X_val,
        y_val=y_val,
        output_path=None,  # 自动生成
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose
    )
    
    return {
        'model_path': new_model_path,
        'features_file': features_file,
        'selected_features': selected_features
    }


def create_prediction_ensemble(
    model_paths: List[str],
    X_predict_file: str,
    selected_features_files: Optional[List[str]] = None,
    ensemble_method: str = 'average',
    weights: Optional[List[float]] = None,
    use_smoothing: bool = True,
    window_size: int = 5,
    output_dir: str = 'data/predictions/ensemble',
    verbose: bool = True
) -> str:
    """
    创建预测集成
    
    Args:
        model_paths: 模型路径列表
        X_predict_file: 预测特征文件路径
        selected_features_files: 选定特征文件路径列表
        ensemble_method: 集成方法
        weights: 权重列表
        use_smoothing: 是否使用平滑
        window_size: 滑动窗口大小
        output_dir: 输出目录
        verbose: 是否输出详细信息
        
    Returns:
        str: 输出文件路径
    """
    logger = logging.getLogger(__name__)
    
    # 加载数据
    logger.info(f"加载预测特征: {X_predict_file}")
    X = pd.read_csv(X_predict_file, index_col=0)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建输出路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"ensemble_pred_{timestamp}.npy")
    
    # 创建模型集成
    _, output_file = create_model_ensemble(
        model_paths=model_paths,
        X=X,
        selected_features_paths=selected_features_files,
        ensemble_method=ensemble_method,
        weights=weights,
        use_smoothing=use_smoothing,
        window_size=window_size,
        output_path=output_path,
        verbose=verbose
    )
    
    return output_file


def main():
    """
    主函数，处理命令行参数并执行相应操作
    """
    parser = argparse.ArgumentParser(description="模型优化脚本")
    
    # 基本参数
    parser.add_argument("--mode", type=str, required=True,
                        choices=["regularize", "select_features", "ensemble", "full"],
                        help="操作模式：regularize(正则化参数优化), select_features(特征选择), ensemble(预测集成), full(完整流程)")
    parser.add_argument("--model_type", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm"],
                        help="模型类型")
    parser.add_argument("--output_dir", type=str, default="data/models/optimized",
                        help="输出目录")
    parser.add_argument("--verbose", action="store_true",
                        help="是否输出详细信息")
    
    # 数据参数
    data_group = parser.add_argument_group("数据参数")
    data_group.add_argument("--X_train_file", type=str,
                        help="训练特征文件路径")
    data_group.add_argument("--y_train_file", type=str,
                        help="训练目标文件路径")
    data_group.add_argument("--X_val_file", type=str,
                        help="验证特征文件路径")
    data_group.add_argument("--y_val_file", type=str,
                        help="验证目标文件路径")
    data_group.add_argument("--X_predict_file", type=str,
                        help="预测特征文件路径")
    
    # 正则化参数
    regularization_group = parser.add_argument_group("正则化参数")
    regularization_group.add_argument("--n_iter", type=int, default=20,
                        help="随机搜索迭代次数")
    regularization_group.add_argument("--cv", type=int, default=5,
                        help="交叉验证折数")
    regularization_group.add_argument("--eval_metric", type=str,
                        default="neg_mean_squared_error",
                        help="评估指标")
    regularization_group.add_argument("--early_stopping_rounds", type=int,
                        default=10, help="早停轮数")
    regularization_group.add_argument("--param_ranges_file", type=str,
                        help="参数范围配置文件路径")
    
    # 特征选择参数
    feature_selection_group = parser.add_argument_group("特征选择参数")
    feature_selection_group.add_argument("--model_file", type=str,
                        help="模型文件路径")
    feature_selection_group.add_argument("--selection_method", type=str,
                        choices=["importance", "shap"],
                        default="importance", help="特征选择方法")
    feature_selection_group.add_argument("--n_features", type=int,
                        default=20, help="选择的特征数量")
    
    # 集成参数
    ensemble_group = parser.add_argument_group("集成参数")
    ensemble_group.add_argument("--model_files", type=str, nargs='+',
                        help="模型文件路径列表")
    ensemble_group.add_argument("--selected_features_files", type=str, nargs='+',
                        help="已选择特征文件路径列表")
    ensemble_group.add_argument("--ensemble_method", type=str,
                        choices=["average", "weighted_average", "median"],
                        default="average", help="集成方法")
    ensemble_group.add_argument("--weights", type=float, nargs='+',
                        help="模型权重列表")
    ensemble_group.add_argument("--use_smoothing", action="store_true",
                        help="是否使用平滑")
    ensemble_group.add_argument("--window_size", type=int,
                        default=5, help="滑动窗口大小")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载自定义参数范围（如果有）
    param_ranges = None
    if args.param_ranges_file:
        try:
            with open(args.param_ranges_file, 'r') as f:
                param_ranges = json.load(f)
                logger.info(f"已加载自定义参数范围: {param_ranges}")
        except Exception as e:
            logger.warning(f"无法加载参数范围文件: {e}")
    
    # 根据模式执行相应操作
    if args.mode == "regularize" or args.mode == "full":
        # 检查必要参数
        if not args.X_train_file or not args.y_train_file:
            logger.error("正则化参数优化需要训练特征和目标文件")
            return
            
        # 创建正则化参数优化输出目录
        regularization_output_dir = os.path.join(args.output_dir, "regularized")
        os.makedirs(regularization_output_dir, exist_ok=True)
        
        # 执行正则化参数优化
        logger.info("开始正则化参数优化")
        model_path = optimize_model(
            model_type=args.model_type,
            X_train_file=args.X_train_file,
            y_train_file=args.y_train_file,
            X_val_file=args.X_val_file,
            y_val_file=args.y_val_file,
            param_ranges=param_ranges,
            n_iter=args.n_iter,
            cv=args.cv,
            eval_metric=args.eval_metric,
            early_stopping_rounds=args.early_stopping_rounds,
            output_dir=regularization_output_dir,
            verbose=args.verbose
        )
        
        # 如果是完整流程，使用优化后的模型继续特征选择
        if args.mode == "full":
            args.model_file = model_path
    
    if args.mode == "select_features" or args.mode == "full":
        # 检查必要参数
        if not args.model_file:
            logger.error("特征选择需要模型文件")
            return
            
        if not args.X_train_file or not args.y_train_file:
            logger.error("特征选择需要训练特征和目标文件")
            return
            
        # 创建特征选择输出目录
        feature_selection_output_dir = os.path.join(args.output_dir, "feature_selected")
        os.makedirs(feature_selection_output_dir, exist_ok=True)
        
        # 执行特征选择
        logger.info("开始特征选择")
        selection_result = select_features_and_retrain(
            model_path=args.model_file,
            X_train_file=args.X_train_file,
            y_train_file=args.y_train_file,
            selection_method=args.selection_method,
            n_features=args.n_features,
            X_val_file=args.X_val_file,
            y_val_file=args.y_val_file,
            early_stopping_rounds=args.early_stopping_rounds,
            output_dir=feature_selection_output_dir,
            verbose=args.verbose
        )
        
        # 如果是完整流程，使用选定特征的模型继续集成
        if args.mode == "full":
            args.model_files = [selection_result['model_path']]
            args.selected_features_files = [selection_result['features_file']]
    
    if args.mode == "ensemble" or args.mode == "full":
        # 检查必要参数
        if not args.model_files:
            logger.error("模型集成需要模型文件")
            return
            
        if not args.X_predict_file:
            logger.error("模型集成需要预测特征文件")
            return
            
        # 创建集成输出目录
        ensemble_output_dir = os.path.join(args.output_dir, "ensemble")
        os.makedirs(ensemble_output_dir, exist_ok=True)
        
        # 执行模型集成
        logger.info("开始模型集成")
        output_file = create_prediction_ensemble(
            model_paths=args.model_files,
            X_predict_file=args.X_predict_file,
            selected_features_files=args.selected_features_files,
            ensemble_method=args.ensemble_method,
            weights=args.weights,
            use_smoothing=args.use_smoothing,
            window_size=args.window_size,
            output_dir=ensemble_output_dir,
            verbose=args.verbose
        )
        
        logger.info(f"模型集成完成，输出文件: {output_file}")
    
    logger.info("处理完成")


if __name__ == "__main__":
    main() 