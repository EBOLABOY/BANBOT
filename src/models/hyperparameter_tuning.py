"""
超参数调优模块 - 支持网格搜索、随机搜索和贝叶斯优化等方法
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
from datetime import datetime
import matplotlib.pyplot as plt
import json
from joblib import dump, load
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from models.base_model import BaseModel
from models.model_validation import ModelValidator, create_model_factory
from utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """
    超参数调优类，支持多种优化方法
    """
    
    def __init__(self, 
                 method: str = "random",
                 n_iter: int = 10,
                 cv: int = 5,
                 scoring: str = None,
                 n_jobs: int = -1,
                 verbose: int = 1,
                 output_dir: str = "models/tuning",
                 save_results: bool = True):
        """
        初始化超参数调优器
        
        参数:
            method (str): 优化方法，可选 'grid'、'random'、'bayes'
            n_iter (int): 随机搜索或贝叶斯优化的迭代次数
            cv (int): 交叉验证折数
            scoring (str): 评分方法，如果为None则根据目标类型自动选择
            n_jobs (int): 并行作业数量，-1表示使用所有可用处理器
            verbose (int): 详细程度
            output_dir (str): 输出目录
            save_results (bool): 是否保存结果
        """
        self.method = method
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.output_dir = output_dir
        self.save_results = save_results
        
        # 创建输出目录
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
    
    def _get_default_scoring(self, model: BaseModel) -> str:
        """
        根据模型类型获取默认评分方法
        
        参数:
            model: 模型实例
            
        返回:
            str: 评分方法
        """
        if model.target_type == "direction":
            return "f1"
        elif model.target_type == "price_change_pct":
            return "neg_mean_squared_error"
        else:
            return "neg_mean_squared_error"
    
    def _create_search_cv(self, model_factory: Callable[[], BaseModel], param_grid: Dict) -> Union[GridSearchCV, RandomizedSearchCV, BayesSearchCV]:
        """
        创建搜索交叉验证对象
        
        参数:
            model_factory: 模型工厂函数
            param_grid: 参数网格
            
        返回:
            Union[GridSearchCV, RandomizedSearchCV, BayesSearchCV]: 搜索交叉验证对象
        """
        # 创建模型实例
        model = model_factory()
        
        # 获取评分方法
        scoring = self.scoring or self._get_default_scoring(model)
        
        # 创建包装类
        class ModelWrapper:
            def __init__(self, model_factory):
                self.model_factory = model_factory
                
            def get_params(self, deep=True):
                # 获取包装的模型参数
                model = self.model_factory()
                # 扁平化参数字典
                flat_params = {}
                for key, value in model.model_params.items():
                    flat_params[key] = value
                return flat_params
            
            def set_params(self, **params):
                # 保存参数，将在fit时应用
                self.params = params
                return self
            
            def fit(self, X, y):
                # 创建一个新的模型实例
                self.model = self.model_factory()
                # 更新模型参数
                for key, value in self.params.items():
                    self.model.model_params[key] = value
                # 训练模型
                self.model.train(X, y)
                return self
            
            def predict(self, X):
                # 预测
                return self.model.predict(X)
            
            def predict_proba(self, X):
                # 预测概率
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)
                raise NotImplementedError("该模型不支持概率预测")
        
        # 创建包装实例
        model_wrapper = ModelWrapper(model_factory)
        
        # 创建搜索交叉验证对象
        if self.method == "grid":
            search_cv = GridSearchCV(
                estimator=model_wrapper,
                param_grid=param_grid,
                scoring=scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True
            )
        elif self.method == "random":
            search_cv = RandomizedSearchCV(
                estimator=model_wrapper,
                param_distributions=param_grid,
                n_iter=self.n_iter,
                scoring=scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True,
                random_state=42
            )
        elif self.method == "bayes":
            search_cv = BayesSearchCV(
                estimator=model_wrapper,
                search_spaces=param_grid,
                n_iter=self.n_iter,
                scoring=scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True,
                random_state=42
            )
        else:
            raise ValueError(f"不支持的优化方法: {self.method}")
        
        return search_cv
    
    def tune(self, 
           model_factory: Callable[[], BaseModel],
           X: Union[pd.DataFrame, np.ndarray],
           y: Union[pd.Series, np.ndarray],
           param_grid: Dict,
           model_name: str = None) -> Dict:
        """
        调优超参数
        
        参数:
            model_factory: 模型工厂函数
            X: 特征数据
            y: 目标数据
            param_grid: 参数网格（或参数分布、搜索空间）
            model_name: 模型名称（可选）
            
        返回:
            Dict: 调优结果
        """
        # 获取模型名称
        if model_name is None:
            model = model_factory()
            model_name = model.name
        
        logger.info(f"开始使用 {self.method} 方法调优 {model_name} 模型...")
        
        # 创建搜索交叉验证对象
        search_cv = self._create_search_cv(model_factory, param_grid)
        
        # 执行搜索
        logger.info("执行参数搜索...")
        search_cv.fit(X, y)
        
        # 获取最佳参数和得分
        best_params = search_cv.best_params_
        best_score = search_cv.best_score_
        
        logger.info(f"最佳参数: {best_params}")
        logger.info(f"最佳得分: {best_score:.6f}")
        
        # 构建结果字典
        result = {
            "best_params": best_params,
            "best_score": best_score,
            "cv_results": search_cv.cv_results_,
            "model_name": model_name,
            "method": self.method,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # 保存结果
        if self.save_results:
            self._save_results(result)
            self._plot_results(result)
            
            # 使用最佳参数创建和训练最终模型
            best_model = self._create_best_model(model_factory, best_params, X, y, model_name)
            self._save_best_model(best_model, model_name)
        
        return result
    
    def _create_best_model(self, 
                         model_factory: Callable[[], BaseModel],
                         best_params: Dict,
                         X: Union[pd.DataFrame, np.ndarray],
                         y: Union[pd.Series, np.ndarray],
                         model_name: str) -> BaseModel:
        """
        使用最佳参数创建和训练模型
        
        参数:
            model_factory: 模型工厂函数
            best_params: 最佳参数
            X: 特征数据
            y: 目标数据
            model_name: 模型名称
            
        返回:
            BaseModel: 训练好的模型
        """
        # 创建模型
        model = model_factory()
        
        # 更新模型参数
        for key, value in best_params.items():
            model.model_params[key] = value
        
        # 训练模型
        logger.info("使用最佳参数训练最终模型...")
        model.train(X, y)
        
        return model
    
    def _save_best_model(self, model: BaseModel, model_name: str) -> None:
        """
        保存最佳模型
        
        参数:
            model: 模型实例
            model_name: 模型名称
        """
        # 创建模型目录
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"{model_name}_best_{timestamp}.pkl")
        model.save(model_path)
        
        logger.info(f"最佳模型已保存至 {model_path}")
    
    def _save_results(self, result: Dict) -> None:
        """
        保存调优结果
        
        参数:
            result: 调优结果
        """
        # 创建结果目录
        model_name = result["model_name"]
        timestamp = result["timestamp"]
        result_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(result_dir, exist_ok=True)
        
        # 提取交叉验证结果
        cv_results = result["cv_results"]
        cv_df = pd.DataFrame(cv_results)
        
        # 保存为CSV
        csv_path = os.path.join(result_dir, f"tuning_results_{timestamp}.csv")
        cv_df.to_csv(csv_path, index=False)
        
        # 保存摘要信息
        summary = {
            "model_name": model_name,
            "method": result["method"],
            "best_params": result["best_params"],
            "best_score": result["best_score"],
            "timestamp": timestamp
        }
        
        summary_path = os.path.join(result_dir, f"tuning_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"调优结果已保存至 {result_dir}")
    
    def _plot_results(self, result: Dict) -> None:
        """
        绘制调优结果
        
        参数:
            result: 调优结果
        """
        # 创建结果目录
        model_name = result["model_name"]
        timestamp = result["timestamp"]
        result_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(result_dir, exist_ok=True)
        
        # 提取交叉验证结果
        cv_results = result["cv_results"]
        
        # 提取测试分数和参数
        test_scores = cv_results["mean_test_score"]
        
        # 找出最好的n个参数组合
        n_top = min(5, len(test_scores))
        indices = np.argsort(test_scores)[-n_top:]
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        
        # 绘制测试分数分布
        plt.subplot(1, 2, 1)
        plt.hist(test_scores, bins=20, alpha=0.7)
        plt.axvline(x=result["best_score"], color='r', linestyle='--', label=f'最佳得分: {result["best_score"]:.4f}')
        plt.title(f"{model_name} - 测试分数分布")
        plt.xlabel("分数")
        plt.ylabel("频率")
        plt.legend()
        
        # 绘制训练vs测试分数
        plt.subplot(1, 2, 2)
        plt.scatter(cv_results["mean_train_score"], cv_results["mean_test_score"], alpha=0.7)
        plt.title(f"{model_name} - 训练 vs 测试分数")
        plt.xlabel("训练分数")
        plt.ylabel("测试分数")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(result_dir, f"tuning_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 如果只有一个或两个参数，绘制参数影响图
        param_names = [key for key in result["best_params"].keys()]
        if len(param_names) <= 2:
            self._plot_param_effects(result, param_names, result_dir, timestamp)
        
        logger.info(f"调优图表已保存至 {plot_path}")
    
    def _plot_param_effects(self, result: Dict, param_names: List[str], result_dir: str, timestamp: str) -> None:
        """
        绘制参数影响图
        
        参数:
            result: 调优结果
            param_names: 参数名称列表
            result_dir: 结果目录
            timestamp: 时间戳
        """
        cv_results = result["cv_results"]
        
        if len(param_names) == 1:
            # 只有一个参数，绘制该参数与分数的关系
            param_name = param_names[0]
            param_values = cv_results[f"param_{param_name}"]
            
            # 将参数值转换为数值（如果可能）
            try:
                param_values = [float(val) for val in param_values]
                is_numeric = True
            except (ValueError, TypeError):
                is_numeric = False
            
            plt.figure(figsize=(10, 6))
            
            if is_numeric:
                # 对于数值参数，绘制散点图
                plt.scatter(param_values, cv_results["mean_test_score"], alpha=0.7)
                plt.scatter(result["best_params"][param_name], result["best_score"], 
                            color='r', marker='*', s=200, label='最佳参数')
                plt.title(f"参数 {param_name} 对测试分数的影响")
                plt.xlabel(param_name)
                plt.ylabel("测试分数")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
            else:
                # 对于类别参数，绘制箱线图
                param_groups = {}
                for i, val in enumerate(param_values):
                    if val not in param_groups:
                        param_groups[val] = []
                    param_groups[val].append(cv_results["mean_test_score"][i])
                
                plt.boxplot([param_groups[val] for val in param_groups.keys()])
                plt.xticks(range(1, len(param_groups) + 1), param_groups.keys())
                plt.scatter([list(param_groups.keys()).index(result["best_params"][param_name]) + 1], 
                            [result["best_score"]], color='r', marker='*', s=200, label='最佳参数')
                plt.title(f"参数 {param_name} 对测试分数的影响")
                plt.xlabel(param_name)
                plt.ylabel("测试分数")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
        
        elif len(param_names) == 2:
            # 两个参数，绘制热图
            param1_name, param2_name = param_names
            param1_values = cv_results[f"param_{param1_name}"]
            param2_values = cv_results[f"param_{param2_name}"]
            
            # 尝试转换为数值
            try:
                param1_values = [float(val) for val in param1_values]
                param1_is_numeric = True
            except (ValueError, TypeError):
                param1_is_numeric = False
                
            try:
                param2_values = [float(val) for val in param2_values]
                param2_is_numeric = True
            except (ValueError, TypeError):
                param2_is_numeric = False
            
            if param1_is_numeric and param2_is_numeric:
                # 两个参数都是数值，绘制等高线图
                from scipy.interpolate import griddata
                
                plt.figure(figsize=(10, 8))
                
                # 创建网格
                x_min, x_max = min(param1_values), max(param1_values)
                y_min, y_max = min(param2_values), max(param2_values)
                
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                x_min -= 0.1 * x_range
                x_max += 0.1 * x_range
                y_min -= 0.1 * y_range
                y_max += 0.1 * y_range
                
                grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                
                # 插值
                points = np.column_stack((param1_values, param2_values))
                values = cv_results["mean_test_score"]
                
                grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
                
                # 绘制等高线图
                contour = plt.contourf(grid_x, grid_y, grid_z, 50, cmap='viridis')
                plt.colorbar(contour, label='测试分数')
                
                # 标记最佳参数
                best_param1 = result["best_params"][param1_name]
                best_param2 = result["best_params"][param2_name]
                plt.scatter([best_param1], [best_param2], 
                            color='r', marker='*', s=200, label='最佳参数')
                
                plt.title(f"参数 {param1_name} 和 {param2_name} 对测试分数的影响")
                plt.xlabel(param1_name)
                plt.ylabel(param2_name)
                plt.legend()
            else:
                # 至少一个参数是类别，绘制热图
                unique_param1 = np.unique(param1_values)
                unique_param2 = np.unique(param2_values)
                
                # 创建得分矩阵
                score_matrix = np.zeros((len(unique_param1), len(unique_param2)))
                count_matrix = np.zeros((len(unique_param1), len(unique_param2)))
                
                for i, (p1, p2, score) in enumerate(zip(param1_values, param2_values, cv_results["mean_test_score"])):
                    i1 = np.where(unique_param1 == p1)[0][0]
                    i2 = np.where(unique_param2 == p2)[0][0]
                    score_matrix[i1, i2] += score
                    count_matrix[i1, i2] += 1
                
                # 计算平均值
                with np.errstate(divide='ignore', invalid='ignore'):
                    score_matrix = np.divide(score_matrix, count_matrix)
                    score_matrix[np.isnan(score_matrix)] = 0
                
                plt.figure(figsize=(10, 8))
                plt.imshow(score_matrix, cmap='viridis', aspect='auto')
                plt.colorbar(label='测试分数')
                
                # 设置刻度
                plt.xticks(np.arange(len(unique_param2)), unique_param2)
                plt.yticks(np.arange(len(unique_param1)), unique_param1)
                
                # 标记最佳参数
                best_param1 = result["best_params"][param1_name]
                best_param2 = result["best_params"][param2_name]
                best_i1 = np.where(unique_param1 == best_param1)[0][0]
                best_i2 = np.where(unique_param2 == best_param2)[0][0]
                plt.scatter([best_i2], [best_i1], 
                            color='r', marker='*', s=200, label='最佳参数')
                
                plt.title(f"参数 {param1_name} 和 {param2_name} 对测试分数的影响")
                plt.xlabel(param2_name)
                plt.ylabel(param1_name)
                plt.legend()
        
        # 保存图表
        param_plot_path = os.path.join(result_dir, f"param_effects_{timestamp}.png")
        plt.savefig(param_plot_path, dpi=300, bbox_inches='tight')
        plt.close()


# 预定义参数空间
def create_param_grid(model_type: str) -> Dict:
    """
    为不同类型的模型创建参数网格
    
    参数:
        model_type: 模型类型
        
    返回:
        Dict: 参数网格
    """
    if model_type == "linear":
        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.0, 0.2, 0.5, 0.8, 1.0]
        }
    elif model_type == "tree":
        return {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }
    elif model_type == "xgboost":
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0]
        }
    elif model_type == "arima":
        return {
            "p": [1, 2, 3],
            "d": [0, 1],
            "q": [0, 1, 2]
        }
    elif model_type == "prophet":
        return {
            "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
            "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
            "seasonality_mode": ["additive", "multiplicative"]
        }
    elif model_type == "lstm":
        return {
            "units": [32, 64, 128],
            "dropout": [0.0, 0.2, 0.4],
            "recurrent_dropout": [0.0, 0.2, 0.4],
            "activation": ["tanh", "relu"],
            "optimizer": ["adam", "rmsprop"]
        }
    else:
        logger.warning(f"未知的模型类型: {model_type}，返回空参数网格")
        return {}


# 创建贝叶斯优化搜索空间
def create_bayes_search_space(model_type: str) -> Dict:
    """
    为不同类型的模型创建贝叶斯优化搜索空间
    
    参数:
        model_type: 模型类型
        
    返回:
        Dict: 贝叶斯优化搜索空间
    """
    if model_type == "linear":
        return {
            "alpha": Real(0.0001, 100.0, prior="log-uniform"),
            "l1_ratio": Real(0.0, 1.0)
        }
    elif model_type == "tree":
        return {
            "max_depth": Integer(3, 15),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 10),
            "max_features": Categorical(["sqrt", "log2", None])
        }
    elif model_type == "xgboost":
        return {
            "n_estimators": Integer(50, 500),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "max_depth": Integer(3, 10),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "gamma": Real(0, 5)
        }
    elif model_type == "arima":
        return {
            "p": Integer(0, 5),
            "d": Integer(0, 2),
            "q": Integer(0, 5)
        }
    elif model_type == "prophet":
        return {
            "changepoint_prior_scale": Real(0.001, 0.5, prior="log-uniform"),
            "seasonality_prior_scale": Real(0.01, 10.0, prior="log-uniform"),
            "seasonality_mode": Categorical(["additive", "multiplicative"])
        }
    elif model_type == "lstm":
        return {
            "units": Integer(16, 256),
            "dropout": Real(0.0, 0.5),
            "recurrent_dropout": Real(0.0, 0.5),
            "activation": Categorical(["tanh", "relu"]),
            "optimizer": Categorical(["adam", "rmsprop"]),
            "batch_size": Categorical([16, 32, 64, 128])
        }
    else:
        logger.warning(f"未知的模型类型: {model_type}，返回空搜索空间")
        return {} 