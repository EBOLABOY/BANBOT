"""
模型验证模块 - 实现交叉验证和时间序列特殊处理
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, KFold
from tqdm import tqdm

from src.models.base_model import BaseModel
from src.models.model_evaluation import ModelEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelValidator:
    """
    模型验证器类，用于进行交叉验证和时间序列特殊处理
    """
    
    def __init__(self, 
                 cv_strategy: str = "time_series",
                 n_splits: int = 5,
                 gap: int = 0,
                 test_size: int = None,
                 output_dir: str = "models/validation",
                 save_results: bool = True):
        """
        初始化模型验证器
        
        参数:
            cv_strategy (str): 交叉验证策略，可选 'time_series'、'kfold'
            n_splits (int): 交叉验证折数
            gap (int): 训练集和测试集之间的间隔（仅用于时间序列交叉验证）
            test_size (int): 测试集大小（仅用于时间序列交叉验证）
            output_dir (str): 输出目录
            save_results (bool): 是否保存结果
        """
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size
        self.output_dir = output_dir
        self.save_results = save_results
        
        # 创建输出目录
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        # 创建模型评估器
        self.evaluator = ModelEvaluator(output_dir=output_dir, save_results=save_results)
    
    def get_cv_splitter(self) -> Union[TimeSeriesSplit, KFold]:
        """
        获取交叉验证分割器
        
        返回:
            Union[TimeSeriesSplit, KFold]: 交叉验证分割器
        """
        if self.cv_strategy == "time_series":
            return TimeSeriesSplit(
                n_splits=self.n_splits,
                gap=self.gap,
                test_size=self.test_size
            )
        elif self.cv_strategy == "kfold":
            return KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=42
            )
        else:
            raise ValueError(f"不支持的交叉验证策略: {self.cv_strategy}")
    
    def cross_validate(self, 
                     model_factory: Callable[[], BaseModel],
                     X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray],
                     model_name: str = None) -> Dict[str, List[float]]:
        """
        进行交叉验证
        
        参数:
            model_factory: 模型工厂函数，返回一个新的模型实例
            X: 特征数据
            y: 目标数据
            model_name: 模型名称（可选）
            
        返回:
            Dict[str, List[float]]: 包含各折指标的字典
        """
        # 获取交叉验证分割器
        cv = self.get_cv_splitter()
        
        # 准备结果字典
        results = {}
        
        # 获取一个示例模型以确定评估指标
        sample_model = model_factory()
        model_name = model_name or sample_model.name
        is_classification = sample_model.target_type == "direction"
        
        # 确定要跟踪的指标
        if is_classification:
            metrics = ["accuracy", "precision", "recall", "f1"]
            if hasattr(sample_model, "predict_proba"):
                metrics.append("auc")
        else:
            metrics = ["rmse", "mae", "r2"]
            if sample_model.target_type == "price_change_pct":
                metrics.append("direction_accuracy")
        
        # 初始化结果字典
        for metric in metrics:
            results[metric] = []
        
        # 进行交叉验证
        logger.info(f"开始{self.cv_strategy}交叉验证 (n_splits={self.n_splits})...")
        
        for fold, (train_idx, test_idx) in enumerate(tqdm(cv.split(X), total=self.n_splits, desc="交叉验证")):
            # 分割数据
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]
                
            if isinstance(y, pd.Series):
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                y_train, y_test = y[train_idx], y[test_idx]
            
            # 创建新的模型实例
            model = model_factory()
            
            # 训练模型
            logger.info(f"正在训练第 {fold+1}/{self.n_splits} 折模型...")
            model.train(X_train, y_train)
            
            # 评估模型
            test_metrics = self.evaluator.evaluate_model(model, X_test, y_test, f"{model_name}_fold{fold+1}")
            
            # 保存指标
            for metric in metrics:
                if metric in test_metrics:
                    results[metric].append(test_metrics[metric])
            
            # 保存模型（如果需要）
            if self.save_results:
                # 创建模型目录
                model_fold_dir = os.path.join(self.output_dir, f"{model_name}_fold{fold+1}")
                os.makedirs(model_fold_dir, exist_ok=True)
                
                # 保存模型
                model_path = os.path.join(model_fold_dir, f"{model_name}_fold{fold+1}.pkl")
                model.save(model_path)
        
        # 计算和记录平均指标
        logger.info("交叉验证完成，平均指标:")
        for metric in metrics:
            if metric in results and results[metric]:
                avg_metric = np.mean(results[metric])
                std_metric = np.std(results[metric])
                logger.info(f"  - {metric}: {avg_metric:.4f} (±{std_metric:.4f})")
        
        # 如果需要保存结果
        if self.save_results:
            self._save_cv_results(results, model_name)
            self._plot_cv_results(results, model_name)
        
        return results
    
    def walk_forward_validation(self, 
                              model_factory: Callable[[], BaseModel],
                              X: Union[pd.DataFrame, np.ndarray],
                              y: Union[pd.Series, np.ndarray],
                              initial_train_size: float = 0.5,
                              step_size: int = 1,
                              n_steps: int = None,
                              refit_on_each_step: bool = True,
                              model_name: str = None) -> Dict[str, List[float]]:
        """
        进行前向验证（Walk-Forward Validation）
        
        参数:
            model_factory: 模型工厂函数，返回一个新的模型实例
            X: 特征数据
            y: 目标数据
            initial_train_size: 初始训练集比例
            step_size: 每步大小
            n_steps: 总步数（如果为None，则自动计算）
            refit_on_each_step: 是否在每步重新训练模型
            model_name: 模型名称（可选）
            
        返回:
            Dict[str, List[float]]: 包含各步指标的字典
        """
        # 获取数据大小
        data_size = len(X)
        
        # 计算初始训练集大小
        initial_train_samples = int(data_size * initial_train_size)
        
        # 计算总步数（如果未指定）
        if n_steps is None:
            n_steps = (data_size - initial_train_samples) // step_size
        
        # 准备结果字典
        results = {}
        
        # 获取一个示例模型以确定评估指标
        sample_model = model_factory()
        model_name = model_name or sample_model.name
        is_classification = sample_model.target_type == "direction"
        
        # 确定要跟踪的指标
        if is_classification:
            metrics = ["accuracy", "precision", "recall", "f1"]
            if hasattr(sample_model, "predict_proba"):
                metrics.append("auc")
        else:
            metrics = ["rmse", "mae", "r2"]
            if sample_model.target_type == "price_change_pct":
                metrics.append("direction_accuracy")
        
        # 初始化结果字典
        for metric in metrics:
            results[metric] = []
        
        # 添加时间索引列表（用于可视化）
        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            test_times = []
        else:
            test_times = None
        
        # 进行前向验证
        logger.info(f"开始前向验证 (initial_train_size={initial_train_size:.2f}, step_size={step_size}, n_steps={n_steps})...")
        
        # 初始训练集和测试集
        train_end = initial_train_samples
        model = model_factory()
        
        for step in tqdm(range(n_steps), desc="前向验证"):
            # 确定当前步的训练集和测试集索引
            train_start = 0
            if not refit_on_each_step and step > 0:
                # 如果不重新训练，则使用累积训练集
                train_end = initial_train_samples + step * step_size
            else:
                # 否则，使用固定大小的初始训练集
                train_end = initial_train_samples
            
            test_start = train_end
            test_end = test_start + step_size
            
            # 分割数据
            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train_start:train_end]
                X_test = X.iloc[test_start:test_end]
                
                # 保存测试时间（如果是时间索引）
                if test_times is not None:
                    test_times.extend(X_test.index.tolist())
            else:
                X_train = X[train_start:train_end]
                X_test = X[test_start:test_end]
            
            if isinstance(y, pd.Series):
                y_train = y.iloc[train_start:train_end]
                y_test = y.iloc[test_start:test_end]
            else:
                y_train = y[train_start:train_end]
                y_test = y[test_start:test_end]
            
            # 如果需要重新训练或是第一步
            if refit_on_each_step or step == 0:
                # 训练模型
                logger.info(f"正在训练第 {step+1}/{n_steps} 步模型 (train: {train_start}:{train_end}, test: {test_start}:{test_end})...")
                model = model_factory()
                model.train(X_train, y_train)
            
            # 评估模型
            test_metrics = self.evaluator.evaluate_model(model, X_test, y_test, f"{model_name}_step{step+1}")
            
            # 保存指标
            for metric in metrics:
                if metric in test_metrics:
                    results[metric].append(test_metrics[metric])
            
            # 保存模型（如果需要且是最后一步）
            if self.save_results and step == n_steps - 1:
                # 创建模型目录
                model_dir = os.path.join(self.output_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)
                
                # 保存模型
                model_path = os.path.join(model_dir, f"{model_name}_final.pkl")
                model.save(model_path)
        
        # 计算和记录平均指标
        logger.info("前向验证完成，平均指标:")
        for metric in metrics:
            if metric in results and results[metric]:
                avg_metric = np.mean(results[metric])
                std_metric = np.std(results[metric])
                logger.info(f"  - {metric}: {avg_metric:.4f} (±{std_metric:.4f})")
        
        # 如果需要保存结果
        if self.save_results:
            self._save_wf_results(results, model_name, test_times)
            self._plot_wf_results(results, model_name, test_times)
        
        return results
    
    def _save_cv_results(self, results: Dict[str, List[float]], model_name: str) -> None:
        """
        保存交叉验证结果
        
        参数:
            results: 交叉验证结果
            model_name: 模型名称
        """
        # 创建输出目录
        result_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(result_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 转换为DataFrame
        result_df = pd.DataFrame()
        
        for metric, values in results.items():
            for fold, value in enumerate(values):
                result_df.at[fold, metric] = value
        
        # 添加均值和标准差
        result_df.loc["mean"] = result_df.mean()
        result_df.loc["std"] = result_df.std()
        
        # 保存为CSV
        csv_path = os.path.join(result_dir, f"cv_results_{timestamp}.csv")
        result_df.to_csv(csv_path)
        
        logger.info(f"交叉验证结果已保存至 {csv_path}")
    
    def _plot_cv_results(self, results: Dict[str, List[float]], model_name: str) -> None:
        """
        绘制交叉验证结果
        
        参数:
            results: 交叉验证结果
            model_name: 模型名称
        """
        # 创建输出目录
        result_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(result_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 计算指标数量
        n_metrics = len(results)
        
        # 计算图表布局
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        # 创建图表
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        
        # 如果只有一个指标，确保axes是可迭代的
        if n_metrics == 1:
            axes = np.array([axes])
        
        # 扁平化axes数组以便迭代
        if n_rows > 1 and n_cols > 1:
            axes = axes.flatten()
        
        # 绘制每个指标的箱线图
        for i, (metric, values) in enumerate(results.items()):
            if i < len(axes):
                ax = axes[i]
                ax.boxplot(values)
                ax.set_title(f"{metric.upper()} Cross-Validation")
                ax.set_xlabel("Metric")
                ax.set_ylabel("Value")
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 添加均值和标准差标签
                mean = np.mean(values)
                std = np.std(values)
                ax.text(0.5, 0.02, f"Mean: {mean:.4f}\nStd: {std:.4f}",
                        transform=ax.transAxes, ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(result_dir, f"cv_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"交叉验证图表已保存至 {plot_path}")
    
    def _save_wf_results(self, results: Dict[str, List[float]], model_name: str, test_times: List = None) -> None:
        """
        保存前向验证结果
        
        参数:
            results: 前向验证结果
            model_name: 模型名称
            test_times: 测试时间点列表（可选）
        """
        # 创建输出目录
        result_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(result_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 转换为DataFrame
        result_df = pd.DataFrame(results)
        
        # 添加时间索引（如果可用）
        if test_times is not None:
            result_df.index = test_times
        
        # 添加均值和标准差
        result_df.loc["mean"] = result_df.mean()
        result_df.loc["std"] = result_df.std()
        
        # 保存为CSV
        csv_path = os.path.join(result_dir, f"wf_results_{timestamp}.csv")
        result_df.to_csv(csv_path)
        
        logger.info(f"前向验证结果已保存至 {csv_path}")
    
    def _plot_wf_results(self, results: Dict[str, List[float]], model_name: str, test_times: List = None) -> None:
        """
        绘制前向验证结果
        
        参数:
            results: 前向验证结果
            model_name: 模型名称
            test_times: 测试时间点列表（可选）
        """
        # 创建输出目录
        result_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(result_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 如果有时间索引，使用时间索引作为x轴
        if test_times is not None:
            x = test_times
            for metric, values in results.items():
                if len(values) > 0:
                    plt.plot(x[:len(values)], values, label=metric.upper())
        else:
            # 否则，使用步数作为x轴
            for metric, values in results.items():
                if len(values) > 0:
                    plt.plot(range(len(values)), values, label=metric.upper())
        
        plt.title(f"{model_name} Walk-Forward Validation")
        plt.xlabel("Time" if test_times is not None else "Step")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图表
        plot_path = os.path.join(result_dir, f"wf_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"前向验证图表已保存至 {plot_path}")


def create_model_factory(model_class, **kwargs) -> Callable[[], BaseModel]:
    """
    创建模型工厂函数
    
    参数:
        model_class: 模型类
        **kwargs: 传递给模型构造函数的参数
        
    返回:
        Callable[[], BaseModel]: 模型工厂函数
    """
    def factory():
        return model_class(**kwargs)
    
    return factory 