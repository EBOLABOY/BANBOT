"""
模型正则化参数优化模块，通过调整正则化参数减轻过拟合
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional, Union, Tuple, Any


class RegularizationOptimizer:
    """
    模型正则化参数优化器，用于优化减轻过拟合的超参数
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        optimization_method: str = 'random',
        n_iter: int = 20,
        cv: int = 5,
        eval_metric: str = 'neg_mean_squared_error',
        early_stopping_rounds: int = 10,
        output_dir: str = 'data/models/optimized',
        verbose: bool = True
    ):
        """
        初始化模型正则化参数优化器
        
        Args:
            model_type: 模型类型，支持'xgboost'和'lightgbm'
            optimization_method: 优化方法，支持'random'和'grid'
            n_iter: 随机搜索迭代次数
            cv: 交叉验证折数
            eval_metric: 评估指标
            early_stopping_rounds: 早停轮数
            output_dir: 输出目录
            verbose: 是否显示详细信息
        """
        self.model_type = model_type.lower()
        self.optimization_method = optimization_method
        self.n_iter = n_iter
        self.cv = cv
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.output_dir = output_dir
        self.verbose = verbose
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化模型
        self._init_model()
        
        # 初始化参数网格
        self._init_param_grid()
        
        # 初始化结果存储
        self.best_params = None
        self.best_score = None
        self.search_results = None
        self.feature_importance = None
        self.trained_model = None
        
        # 记录结果
        self.training_results = {
            'model_type': self.model_type,
            'optimization_method': self.optimization_method,
            'n_iter': self.n_iter,
            'cv': self.cv,
            'eval_metric': self.eval_metric,
            'best_params': None,
            'best_score': None,
            'validation_scores': {},
            'feature_importance': None
        }
    
    def _init_model(self):
        """初始化模型"""
        if self.model_type == 'xgboost':
            import xgboost as xgb
            self.model_cls = xgb.XGBRegressor
            self.model_kwargs = {
                'objective': 'reg:squarederror',
                'n_estimators': 1000,
                'verbosity': 1 if self.verbose else 0,
                'seed': 42
            }
        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            self.model_cls = lgb.LGBMRegressor
            self.model_kwargs = {
                'objective': 'regression',
                'n_estimators': 1000,
                'verbosity': 1 if self.verbose else 0,
                'seed': 42
            }
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _init_param_grid(self):
        """初始化参数网格"""
        if self.model_type == 'xgboost':
            self.param_grid = {
                'max_depth': [3, 4, 5, 6, 7, 8],
                'min_child_weight': [1, 2, 3, 4, 5],
                'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.001, 0.01, 0.1, 1.0],
                'reg_lambda': [0, 0.001, 0.01, 0.1, 1.0, 10.0],
                'learning_rate': [0.01, 0.05, 0.1, 0.2]
            }
        elif self.model_type == 'lightgbm':
            self.param_grid = {
                'max_depth': [3, 4, 5, 6, 7, 8],
                'num_leaves': [7, 15, 31, 63, 127],
                'min_child_samples': [5, 10, 20, 30, 50],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.001, 0.01, 0.1, 1.0],
                'reg_lambda': [0, 0.001, 0.01, 0.1, 1.0, 10.0],
                'learning_rate': [0.01, 0.05, 0.1, 0.2]
            }
    
    def optimize_parameters(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Dict:
        """
        优化模型参数
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            
        Returns:
            Dict: 最佳参数
        """
        # 创建基础模型
        base_model = self.model_cls(**self.model_kwargs)
        
        # 创建参数搜索对象
        if self.optimization_method == 'random':
            search = RandomizedSearchCV(
                base_model,
                self.param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                verbose=int(self.verbose),
                scoring=self.eval_metric,
                n_jobs=-1,
                random_state=42,
                return_train_score=True
            )
        else:  # grid
            search = GridSearchCV(
                base_model,
                self.param_grid,
                cv=self.cv,
                verbose=int(self.verbose),
                scoring=self.eval_metric,
                n_jobs=-1,
                return_train_score=True
            )
        
        # 执行参数搜索
        search.fit(X_train, y_train)
        
        # 保存结果
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.search_results = search.cv_results_
        
        print(f"最佳参数: {self.best_params}")
        print(f"最佳分数: {self.best_score}")
        
        # 更新训练结果
        self.training_results['best_params'] = self.best_params
        self.training_results['best_score'] = self.best_score
        
        return self.best_params
    
    def train_with_best_params(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> str:
        """
        使用最佳参数训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            str: 模型保存路径
        """
        if not self.best_params:
            raise ValueError("请先调用optimize_parameters优化参数")
        
        # 创建带最佳参数的模型
        model = self.model_cls(**{**self.model_kwargs, **self.best_params})
        
        # 如果有验证集，设置早停
        fit_params = {}
        if X_val is not None and y_val is not None:
            if self.model_type == 'xgboost':
                fit_params = {
                    'eval_set': [(X_val, y_val)],
                    'early_stopping_rounds': self.early_stopping_rounds,
                    'verbose': bool(self.verbose)
                }
            elif self.model_type == 'lightgbm':
                fit_params = {
                    'eval_set': [(X_val, y_val)],
                    'callbacks': [
                        {'type': 'early_stopping', 'stopping_rounds': self.early_stopping_rounds},
                    ] if self.early_stopping_rounds > 0 else None,
                    'verbose': bool(self.verbose)
                }
        
        # 训练模型
        model.fit(X_train, y_train, **fit_params)
        
        # 保存模型
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = os.path.join(self.output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.model_type}_optimized_{timestamp}.joblib")
        joblib.dump(model, model_path)
        
        # 保存特征重要性
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = {
                'feature': X_train.columns.tolist(),
                'importance': model.feature_importances_.tolist()
            }
            self.training_results['feature_importance'] = self.feature_importance
        
        # 保存模型
        self.trained_model = model
        
        print(f"模型已保存至: {model_path}")
        
        return model_path
    
    def evaluate_model(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict:
        """
        评估模型性能
        
        Args:
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            Dict: 评估结果
        """
        if not self.trained_model:
            raise ValueError("请先调用train_with_best_params训练模型")
        
        # 预测
        y_pred = self.trained_model.predict(X_val)
        
        # 计算评估指标
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # 保存结果
        scores = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        self.training_results['validation_scores'] = scores
        
        print("验证集评估结果:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        
        return scores
    
    def save_results(self) -> str:
        """
        保存优化结果
        
        Returns:
            str: 结果保存路径
        """
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"optimization_results_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=4)
        
        print(f"优化结果已保存至: {results_path}")
        
        return results_path
    
    def plot_feature_importance(self, save_path: Optional[str] = None) -> None:
        """
        绘制特征重要性
        
        Args:
            save_path: 保存路径
        """
        if not self.feature_importance:
            raise ValueError("模型没有特征重要性信息")
        
        # 提取特征和重要性
        features = self.feature_importance['feature']
        importances = self.feature_importance['importance']
        
        # 创建特征重要性的DataFrame
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # 绘制前20个特征（如果有那么多）
        plt.figure(figsize=(10, 8))
        n_features = min(20, len(importance_df))
        plt.barh(importance_df['feature'][:n_features], importance_df['importance'][:n_features])
        plt.xlabel('特征重要性')
        plt.ylabel('特征')
        plt.title(f'{self.model_type.upper()} 特征重要性 (Top {n_features})')
        plt.gca().invert_yaxis()  # 重要性从高到低显示
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性图已保存至: {save_path}")
        
        plt.close()


def main():
    """
    示例使用
    """
    parser = argparse.ArgumentParser(description="模型正则化参数优化")
    
    # 基本参数
    parser.add_argument("--model_type", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm"],
                        help="模型类型")
    parser.add_argument("--X_train_file", type=str, required=True,
                        help="训练特征文件路径")
    parser.add_argument("--y_train_file", type=str, required=True,
                        help="训练目标文件路径")
    parser.add_argument("--X_val_file", type=str,
                        help="验证特征文件路径")
    parser.add_argument("--y_val_file", type=str,
                        help="验证目标文件路径")
    parser.add_argument("--output_dir", type=str, default="data/models/optimized",
                        help="输出目录")
    
    # 优化参数
    parser.add_argument("--optimization_method", type=str, default="random",
                        choices=["random", "grid"],
                        help="优化方法")
    parser.add_argument("--n_iter", type=int, default=20,
                        help="随机搜索迭代次数")
    parser.add_argument("--cv", type=int, default=5,
                        help="交叉验证折数")
    parser.add_argument("--eval_metric", type=str, default="neg_mean_squared_error",
                        help="评估指标")
    parser.add_argument("--early_stopping_rounds", type=int, default=10,
                        help="早停轮数")
    parser.add_argument("--verbose", action="store_true",
                        help="是否输出详细信息")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 检查必要参数
    if not args.X_train_file or not args.y_train_file:
        logger.error("请提供训练特征文件和目标文件")
        return
    
    # 加载数据
    logger.info(f"加载训练特征: {args.X_train_file}")
    X_train = pd.read_csv(args.X_train_file, index_col=0)
    
    logger.info(f"加载训练目标: {args.y_train_file}")
    y_train = pd.read_csv(args.y_train_file, index_col=0).iloc[:, 0]
    
    # 加载验证数据
    X_val = None
    y_val = None
    if args.X_val_file and args.y_val_file:
        logger.info(f"加载验证特征: {args.X_val_file}")
        X_val = pd.read_csv(args.X_val_file, index_col=0)
        logger.info(f"加载验证目标: {args.y_val_file}")
        y_val = pd.read_csv(args.y_val_file, index_col=0).iloc[:, 0]
    
    # 创建正则化参数优化器
    optimizer = RegularizationOptimizer(
        model_type=args.model_type,
        optimization_method=args.optimization_method,
        n_iter=args.n_iter,
        cv=args.cv,
        eval_metric=args.eval_metric,
        early_stopping_rounds=args.early_stopping_rounds,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
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
    plot_path = os.path.join(args.output_dir, 'plots', f'importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    optimizer.plot_feature_importance(plot_path)


if __name__ == "__main__":
    main() 