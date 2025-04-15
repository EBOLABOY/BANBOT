"""
正则化优化脚本，用于减轻模型过拟合现象
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from src.utils.logger import get_logger
from src.models.xgboost_model import XGBoostModel
from src.data.data_loader import DataLoader

logger = get_logger(__name__)

class RegularizationOptimizer:
    """
    正则化参数优化器，用于减轻模型过拟合
    """
    def __init__(
        self, 
        model_type: str = "xgboost",
        search_method: str = "grid", 
        cv_folds: int = 5,
        n_iter: int = 20,
        random_state: int = 42,
        output_dir: str = "outputs/optimization",
        plot_results: bool = True
    ):
        """
        初始化正则化优化器
        
        参数:
            model_type: 模型类型，目前支持 "xgboost"
            search_method: 搜索方法 (grid, random)
            cv_folds: 交叉验证折数
            n_iter: 随机搜索迭代次数
            random_state: 随机种子
            output_dir: 输出目录
            plot_results: 是否绘制结果图
        """
        self.model_type = model_type
        self.search_method = search_method
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.random_state = random_state
        self.output_dir = output_dir
        self.plot_results = plot_results
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 存储优化结果
        self.optimization_results = None
        self.best_params = None
        
        # 初始化参数网格
        self._init_param_grid()
    
    def _init_param_grid(self):
        """
        初始化参数搜索网格
        """
        if self.model_type == "xgboost":
            # XGBoost正则化参数网格
            self.param_grid = {
                # 深度相关 (较小的max_depth和较大的min_child_weight减轻过拟合)
                'max_depth': [3, 4, 5, 6, 7],
                'min_child_weight': [1, 3, 5, 7],
                
                # 采样相关 (较小的subsample和colsample减轻过拟合)
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                
                # L1/L2正则化 (较大的alpha和lambda减轻过拟合)
                'reg_alpha': [0, 0.1, 0.5, 1.0, 5.0],
                'reg_lambda': [0.1, 1.0, 5.0, 10.0],
                
                # 复杂度惩罚项 (较大的gamma减轻过拟合)
                'gamma': [0, 0.1, 0.2, 0.5, 1.0],
                
                # 学习率 (较小的学习率配合较多的迭代次数减轻过拟合)
                'learning_rate': [0.01, 0.05, 0.1, 0.2]
            }
            
            # 固定参数
            self.fixed_params = {
                'n_estimators': 500,  # 使用更多的树
                'objective': 'reg:squarederror',
                'random_state': self.random_state,
                'n_jobs': -1,
                'early_stopping_rounds': 50
            }
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _get_param_subsets(self) -> List[Dict[str, Any]]:
        """
        获取参数子集，用于分阶段优化
        
        返回:
            参数子集列表
        """
        if self.model_type == "xgboost":
            return [
                # 阶段1: 优化深度相关参数
                {
                    'max_depth': self.param_grid['max_depth'],
                    'min_child_weight': self.param_grid['min_child_weight']
                },
                # 阶段2: 优化采样相关参数
                {
                    'subsample': self.param_grid['subsample'],
                    'colsample_bytree': self.param_grid['colsample_bytree']
                },
                # 阶段3: 优化正则化参数
                {
                    'reg_alpha': self.param_grid['reg_alpha'],
                    'reg_lambda': self.param_grid['reg_lambda'],
                    'gamma': self.param_grid['gamma']
                },
                # 阶段4: 优化学习率
                {
                    'learning_rate': self.param_grid['learning_rate']
                }
            ]
        else:
            return [self.param_grid]
    
    def _create_model(self, params: Dict[str, Any] = None) -> Any:
        """
        创建模型实例
        
        参数:
            params: 模型参数
            
        返回:
            模型实例
        """
        if self.model_type == "xgboost":
            model_params = {**self.fixed_params}
            if params:
                model_params.update(params)
            
            return XGBoostModel(name="regularized_xgboost", params=model_params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _setup_search(self, params: Dict[str, Any], model, X_train, y_train) -> Union[GridSearchCV, RandomizedSearchCV]:
        """
        设置参数搜索
        
        参数:
            params: 要搜索的参数网格
            model: 模型实例
            X_train: 训练特征
            y_train: 训练目标
            
        返回:
            参数搜索实例
        """
        if self.search_method == "grid":
            return GridSearchCV(
                estimator=model.model,
                param_grid=params,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                cv=self.cv_folds,
                verbose=1
            )
        elif self.search_method == "random":
            return RandomizedSearchCV(
                estimator=model.model,
                param_distributions=params,
                n_iter=self.n_iter,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                cv=self.cv_folds,
                verbose=1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"不支持的搜索方法: {self.search_method}")
    
    def optimize(self, X_train: pd.DataFrame, y_train: pd.Series, 
                X_val: pd.DataFrame = None, y_val: pd.Series = None,
                staged: bool = True) -> Dict[str, Any]:
        """
        优化正则化参数
        
        参数:
            X_train: 训练集特征
            y_train: 训练集目标
            X_val: 验证集特征 (可选)
            y_val: 验证集目标 (可选)
            staged: 是否分阶段优化
            
        返回:
            优化结果
        """
        logger.info(f"开始{self.model_type}模型正则化参数优化")
        
        # 存储训练过程信息
        optimization_history = []
        current_best_params = {}
        
        if staged:
            # 分阶段优化
            param_subsets = self._get_param_subsets()
            
            for i, param_subset in enumerate(param_subsets):
                logger.info(f"阶段 {i+1}/{len(param_subsets)}: 优化参数 {list(param_subset.keys())}")
                
                # 使用当前最佳参数创建模型
                model = self._create_model(current_best_params)
                
                # 设置搜索
                search = self._setup_search(param_subset, model, X_train, y_train)
                
                # 执行搜索
                search.fit(X_train, y_train)
                
                # 更新当前最佳参数
                current_best_params.update(search.best_params_)
                
                # 保存阶段结果
                stage_results = {
                    "stage": i+1,
                    "params_optimized": list(param_subset.keys()),
                    "best_params": search.best_params_,
                    "best_score": -search.best_score_,  # 转换回MSE
                    "all_results": self._extract_cv_results(search.cv_results_)
                }
                optimization_history.append(stage_results)
                
                logger.info(f"阶段 {i+1} 最佳参数: {search.best_params_}")
                logger.info(f"阶段 {i+1} 最佳MSE分数: {-search.best_score_:.6f}")
        else:
            # 一次性优化所有参数
            model = self._create_model()
            search = self._setup_search(self.param_grid, model, X_train, y_train)
            search.fit(X_train, y_train)
            
            # 保存结果
            current_best_params = search.best_params_
            stage_results = {
                "stage": 1,
                "params_optimized": list(self.param_grid.keys()),
                "best_params": search.best_params_,
                "best_score": -search.best_score_,
                "all_results": self._extract_cv_results(search.cv_results_)
            }
            optimization_history.append(stage_results)
            
            logger.info(f"最佳参数: {search.best_params_}")
            logger.info(f"最佳MSE分数: {-search.best_score_:.6f}")
        
        # 使用最佳参数训练最终模型
        final_model = self._create_model(current_best_params)
        
        # 训练和评估
        train_results = self._train_and_evaluate(final_model, X_train, y_train, X_val, y_val)
        
        # 存储最终结果
        self.optimization_results = {
            "model_type": self.model_type,
            "search_method": self.search_method,
            "cv_folds": self.cv_folds,
            "optimization_history": optimization_history,
            "best_params": current_best_params,
            "training_results": train_results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.best_params = current_best_params
        
        # 保存结果
        self._save_results()
        
        # 绘制结果
        if self.plot_results:
            self._plot_optimization_results()
        
        return self.optimization_results
    
    def _extract_cv_results(self, cv_results: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        提取交叉验证结果
        
        参数:
            cv_results: GridSearchCV或RandomizedSearchCV的cv_results_
            
        返回:
            交叉验证结果列表
        """
        results = []
        for i in range(len(cv_results['params'])):
            result = {
                'params': cv_results['params'][i],
                'mean_test_score': -cv_results['mean_test_score'][i],  # 转换回MSE
                'std_test_score': cv_results['std_test_score'][i],
                'rank_test_score': cv_results['rank_test_score'][i]
            }
            results.append(result)
        
        return sorted(results, key=lambda x: x['rank_test_score'])
    
    def _train_and_evaluate(self, model, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        """
        使用最佳参数训练和评估模型
        
        参数:
            model: 模型实例
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        返回:
            训练和评估结果
        """
        logger.info("使用最佳参数训练最终模型")
        
        # 训练模型
        model.train(X_train, y_train)
        
        # 训练集评估
        train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        results = {
            "train_mse": train_mse,
            "train_r2": train_r2
        }
        
        logger.info(f"训练集MSE: {train_mse:.6f}, R²: {train_r2:.6f}")
        
        # 验证集评估
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            results.update({
                "val_mse": val_mse,
                "val_r2": val_r2
            })
            
            logger.info(f"验证集MSE: {val_mse:.6f}, R²: {val_r2:.6f}")
        
        # 保存模型
        model_path = os.path.join(self.output_dir, f"optimized_{self.model_type}_model.pkl")
        model.save(model_path)
        
        return results
    
    def _save_results(self):
        """
        保存优化结果
        """
        if not self.optimization_results:
            logger.warning("没有优化结果可保存")
            return
        
        # 保存结果为JSON
        results_path = os.path.join(self.output_dir, f"{self.model_type}_optimization_results.json")
        
        with open(results_path, 'w') as f:
            json.dump(self.optimization_results, f, indent=2)
        
        logger.info(f"优化结果已保存至: {results_path}")
        
        # 保存最佳参数
        params_path = os.path.join(self.output_dir, f"{self.model_type}_best_params.json")
        
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        logger.info(f"最佳参数已保存至: {params_path}")
    
    def _plot_optimization_results(self):
        """
        绘制优化结果
        """
        if not self.optimization_results:
            logger.warning("没有优化结果可绘制")
            return
        
        # 获取优化历史
        history = self.optimization_results["optimization_history"]
        
        for stage_results in history:
            stage = stage_results["stage"]
            params_optimized = stage_results["params_optimized"]
            
            # 仅对一次优化2个参数的情况绘制热图
            if len(params_optimized) == 2:
                param1 = params_optimized[0]
                param2 = params_optimized[1]
                
                # 提取结果
                all_results = stage_results["all_results"]
                
                # 准备热图数据
                unique_param1_values = sorted(set(res['params'][param1] for res in all_results))
                unique_param2_values = sorted(set(res['params'][param2] for res in all_results))
                
                # 创建热图矩阵
                heatmap_data = np.zeros((len(unique_param1_values), len(unique_param2_values)))
                
                for res in all_results:
                    param1_idx = unique_param1_values.index(res['params'][param1])
                    param2_idx = unique_param2_values.index(res['params'][param2])
                    heatmap_data[param1_idx, param2_idx] = res['mean_test_score']
                
                # 绘制热图
                plt.figure(figsize=(10, 8))
                plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
                plt.colorbar(label='MSE')
                
                # 设置标签
                plt.xticks(range(len(unique_param2_values)), unique_param2_values, rotation=45)
                plt.yticks(range(len(unique_param1_values)), unique_param1_values)
                plt.xlabel(param2)
                plt.ylabel(param1)
                
                # 标出最佳参数
                best_params = stage_results["best_params"]
                best_param1_idx = unique_param1_values.index(best_params[param1])
                best_param2_idx = unique_param2_values.index(best_params[param2])
                plt.plot(best_param2_idx, best_param1_idx, 'r*', markersize=15)
                
                plt.title(f'阶段 {stage}: {param1} vs {param2} 参数优化热图')
                plt.tight_layout()
                
                # 保存图表
                fig_path = os.path.join(self.output_dir, f"stage{stage}_{param1}_vs_{param2}_heatmap.png")
                plt.savefig(fig_path)
                plt.close()
        
        # 绘制训练和验证性能对比
        if "training_results" in self.optimization_results:
            train_results = self.optimization_results["training_results"]
            
            metrics = ["mse", "r2"]
            plt.figure(figsize=(12, 5))
            
            for i, metric in enumerate(metrics):
                plt.subplot(1, 2, i+1)
                
                train_metric = train_results.get(f"train_{metric}")
                val_metric = train_results.get(f"val_{metric}")
                
                if train_metric is not None and val_metric is not None:
                    plt.bar(["训练集", "验证集"], [train_metric, val_metric])
                    plt.title(f'最终模型 {metric.upper()} 对比')
                    plt.ylabel(metric.upper())
            
            plt.tight_layout()
            
            # 保存图表
            fig_path = os.path.join(self.output_dir, "final_model_performance.png")
            plt.savefig(fig_path)
            plt.close()
        
        logger.info(f"优化结果图表已保存至: {self.output_dir}")

def parse_args():
    """
    解析命令行参数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="模型正则化参数优化")
    
    parser.add_argument("--model_type", type=str, default="xgboost", 
                       help="模型类型 (目前仅支持 xgboost)")
    parser.add_argument("--search_method", type=str, default="grid", 
                       choices=["grid", "random"], help="参数搜索方法")
    parser.add_argument("--cv", type=int, default=5, 
                       help="交叉验证折数")
    parser.add_argument("--n_iter", type=int, default=20, 
                       help="随机搜索迭代次数")
    parser.add_argument("--random_state", type=int, default=42, 
                       help="随机种子")
    parser.add_argument("--staged", action="store_true", 
                       help="是否分阶段优化")
    parser.add_argument("--output_dir", type=str, default="outputs/optimization", 
                       help="输出目录")
    parser.add_argument("--data_dir", type=str, default="data/processed", 
                       help="数据目录")
    parser.add_argument("--symbols", type=str, default="BTCUSDT", 
                       help="交易对符号，多个符号用逗号分隔")
    parser.add_argument("--timeframe", type=str, default="1h", 
                       help="时间框架")
    parser.add_argument("--features", type=str, nargs="+", 
                       help="要使用的特征列表")
    parser.add_argument("--target_type", type=str, default="price_change_pct", 
                       help="目标类型")
    parser.add_argument("--horizon", type=int, default=60, 
                       help="预测时间范围")
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 设置日志
    logger.info("开始正则化参数优化")
    
    # 加载数据
    symbols = args.symbols.split(",")
    features = args.features if args.features else None
    
    data_loader = DataLoader(
        data_dir=args.data_dir,
        symbols=symbols,
        timeframes=[args.timeframe],
        target_type=args.target_type,
        target_horizon=args.horizon
    )
    
    # 获取特征和目标数据
    X, y = data_loader.load_train_data(features=features)
    
    if X is None or y is None:
        logger.error("加载数据失败")
        return
    
    logger.info(f"加载数据完成，特征形状: {X.shape}, 目标形状: {y.shape}")
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )
    
    logger.info(f"训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")
    
    # 创建优化器
    optimizer = RegularizationOptimizer(
        model_type=args.model_type,
        search_method=args.search_method,
        cv_folds=args.cv,
        n_iter=args.n_iter,
        random_state=args.random_state,
        output_dir=args.output_dir,
        plot_results=True
    )
    
    # 执行优化
    results = optimizer.optimize(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        staged=args.staged
    )
    
    logger.info("正则化参数优化完成")
    logger.info(f"最佳参数: {results['best_params']}")

if __name__ == "__main__":
    main() 