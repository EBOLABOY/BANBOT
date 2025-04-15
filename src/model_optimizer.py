"""
模型优化器
用于调整模型的正则化参数以减轻过拟合
支持网格搜索和随机搜索两种优化方法
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)

class RegularizationOptimizer:
    """
    正则化参数优化器
    用于调整模型的正则化参数以减轻过拟合
    支持XGBoost和LightGBM模型，以及网格搜索和随机搜索
    """
    
    def __init__(
        self,
        model_type: str,
        optimization_method: str = 'grid',
        n_iter: int = 50,
        cv: int = 5,
        eval_metric: str = 'neg_root_mean_squared_error',
        scoring: Optional[Union[str, List[str]]] = None,
        param_grid: Optional[Dict] = None,
        stages: List[str] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1,
        refit: bool = True,
        early_stopping_rounds: int = 50,
        return_train_score: bool = True,
        output_dir: str = 'outputs/optimization',
        base_model_params: Optional[Dict] = None,
        feature_importance_method: str = 'gain'
    ):
        """
        初始化参数优化器
        
        参数:
            model_type: 模型类型，'xgboost' 或 'lightgbm'
            optimization_method: 优化方法，'grid' 或 'random'
            n_iter: 随机搜索的迭代次数
            cv: 交叉验证折数
            eval_metric: 评估指标
            scoring: 额外的评分指标
            param_grid: 参数网格（可选，如果不提供则使用默认网格）
            stages: 分阶段优化的参数组（例如 ['depth', 'regularization', 'sampling']）
            random_state: 随机种子
            n_jobs: 并行任务数
            verbose: 详细程度
            refit: 是否使用最佳参数重新拟合模型
            early_stopping_rounds: 早停轮数
            return_train_score: 是否返回训练集评分
            output_dir: 输出目录
            base_model_params: 基础模型参数（不进行优化的参数）
            feature_importance_method: 特征重要性计算方法
        """
        self.model_type = model_type
        self.optimization_method = optimization_method
        self.n_iter = n_iter
        self.cv = cv
        self.eval_metric = eval_metric
        self.scoring = scoring
        self.custom_param_grid = param_grid
        self.stages = stages
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit = refit
        self.early_stopping_rounds = early_stopping_rounds
        self.return_train_score = return_train_score
        self.output_dir = output_dir
        self.base_model_params = base_model_params or {}
        self.feature_importance_method = feature_importance_method
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化参数网格
        self.param_grid = self._set_param_grid()
        
        # 初始化最佳模型和参数
        self.best_model = None
        self.best_params = None
        self.cv_results = None
        self.feature_importance = None
        self.optimization_history = {}
        
    def _set_param_grid(self) -> Dict:
        """
        设置参数网格
        
        返回:
            参数网格字典
        """
        # 如果提供了自定义参数网格，则使用它
        if self.custom_param_grid is not None:
            return self.custom_param_grid
            
        # 否则，根据模型类型设置默认参数网格
        if self.model_type == 'xgboost':
            return self._set_xgboost_param_grid()
        elif self.model_type == 'lightgbm':
            return self._set_lightgbm_param_grid()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _set_xgboost_param_grid(self) -> Dict:
        """
        设置XGBoost参数网格
        
        返回:
            XGBoost参数网格
        """
        # 默认的XGBoost参数网格，按不同阶段分组
        param_grid = {
            # 树结构参数
            'depth': {
                'max_depth': [3, 4, 5, 6, 7, 8],
                'min_child_weight': [1, 3, 5, 7]
            },
            # 正则化参数
            'regularization': {
                'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 5.0],
                'reg_lambda': [0, 0.01, 0.1, 1.0, 5.0, 10.0],
                'gamma': [0, 0.1, 0.2, 0.3, 0.4]
            },
            # 采样参数
            'sampling': {
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
            },
            # 学习参数
            'learning': {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 200, 300, 500]
            }
        }
        
        # 如果没有指定分阶段优化，则合并所有参数
        if self.stages is None:
            merged_grid = {}
            for group in param_grid.values():
                merged_grid.update(group)
            return merged_grid
        
        # 否则，只返回指定阶段的参数
        stage_grid = {}
        for stage in self.stages:
            if stage in param_grid:
                stage_grid.update(param_grid[stage])
        
        return stage_grid
    
    def _set_lightgbm_param_grid(self) -> Dict:
        """
        设置LightGBM参数网格
        
        返回:
            LightGBM参数网格
        """
        # 默认的LightGBM参数网格，按不同阶段分组
        param_grid = {
            # 树结构参数
            'depth': {
                'max_depth': [3, 5, 7, 9, -1],  # -1表示不限制深度
                'min_child_samples': [5, 10, 20, 30, 50]
            },
            # 正则化参数
            'regularization': {
                'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 5.0],
                'reg_lambda': [0, 0.01, 0.1, 1.0, 5.0, 10.0],
                'min_split_gain': [0, 0.1, 0.2, 0.3, 0.4]
            },
            # 采样参数
            'sampling': {
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'bagging_freq': [0, 1, 5, 10]
            },
            # 学习参数
            'learning': {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 200, 300, 500]
            }
        }
        
        # 如果没有指定分阶段优化，则合并所有参数
        if self.stages is None:
            merged_grid = {}
            for group in param_grid.values():
                merged_grid.update(group)
            return merged_grid
        
        # 否则，只返回指定阶段的参数
        stage_grid = {}
        for stage in self.stages:
            if stage in param_grid:
                stage_grid.update(param_grid[stage])
        
        return stage_grid
    
    def optimize(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        优化模型参数
        
        参数:
            X: 训练特征
            y: 训练目标
            X_val: 验证特征（可选）
            y_val: 验证目标（可选）
            
        返回:
            self: 优化后的对象
        """
        logger.info(f"开始优化 {self.model_type} 模型参数，使用 {self.optimization_method} 方法")
        
        # 创建模型
        model = self._create_model()
        
        # 设置交叉验证
        if self.cv is None or isinstance(self.cv, int):
            cv = TimeSeriesSplit(n_splits=self.cv if self.cv is not None else 5)
        else:
            cv = self.cv
            
        # 设置早停回调
        if self.model_type == 'xgboost':
            import xgboost as xgb
            callbacks = [xgb.callback.EarlyStopping(
                rounds=self.early_stopping_rounds, 
                save_best=True,
                maximize=False
            )]
            fit_params = {'callbacks': callbacks}
            if X_val is not None and y_val is not None:
                fit_params['eval_set'] = [(X_val, y_val)]
        else:  # lightgbm
            fit_params = {'early_stopping_rounds': self.early_stopping_rounds}
            if X_val is not None and y_val is not None:
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['eval_names'] = ['validation']
        
        # 创建搜索器
        if self.optimization_method == 'grid':
            search = GridSearchCV(
                estimator=model,
                param_grid=self.param_grid,
                scoring=self.scoring or self.eval_metric,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=self.refit,
                return_train_score=self.return_train_score
            )
        else:  # random
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring or self.eval_metric,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=self.refit,
                random_state=self.random_state,
                return_train_score=self.return_train_score
            )
        
        # 执行搜索
        logger.info(f"开始参数搜索，搜索参数: {list(self.param_grid.keys())}")
        search.fit(X, y, **fit_params)
        
        # 保存结果
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_results = search.cv_results_
        
        # 计算特征重要性
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # 记录本次优化阶段结果
        stage_name = '_'.join(self.stages) if self.stages else 'all'
        self.optimization_history[stage_name] = {
            'best_params': self.best_params,
            'best_score': search.best_score_,
            'cv_results': {k: list(v) if isinstance(v, np.ndarray) else v 
                           for k, v in search.cv_results_.items()}
        }
        
        logger.info(f"参数优化完成，最佳参数: {self.best_params}")
        logger.info(f"最佳得分: {search.best_score_}")
        
        return self
    
    def _create_model(self):
        """
        创建模型实例
        
        返回:
            模型实例
        """
        if self.model_type == 'xgboost':
            import xgboost as xgb
            # 合并基础参数
            params = {
                'objective': 'reg:squarederror',
                'random_state': self.random_state,
                **self.base_model_params
            }
            return xgb.XGBRegressor(**params)
        elif self.model_type == 'lightgbm':
            import lightgbm as lgb
            # 合并基础参数
            params = {
                'objective': 'regression',
                'random_state': self.random_state,
                **self.base_model_params
            }
            return lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train_with_best_params(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        使用最佳参数训练模型
        
        参数:
            X: 训练特征
            y: 训练目标
            X_val: 验证特征（可选）
            y_val: 验证目标（可选）
            
        返回:
            训练好的模型
        """
        if self.best_params is None:
            logger.warning("尚未找到最佳参数，请先运行optimize方法")
            return None
        
        # 创建模型
        model = self._create_model()
        
        # 设置最佳参数
        model.set_params(**self.best_params)
        
        # 设置早停和验证集
        if self.model_type == 'xgboost':
            import xgboost as xgb
            callbacks = [xgb.callback.EarlyStopping(
                rounds=self.early_stopping_rounds, 
                save_best=True,
                maximize=False
            )]
            fit_params = {'callbacks': callbacks}
            if X_val is not None and y_val is not None:
                fit_params['eval_set'] = [(X_val, y_val)]
        else:  # lightgbm
            fit_params = {'early_stopping_rounds': self.early_stopping_rounds}
            if X_val is not None and y_val is not None:
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['eval_names'] = ['validation']
        
        # 训练模型
        logger.info("使用最佳参数训练模型")
        model.fit(X, y, **fit_params)
        
        # 保存模型
        self.best_model = model
        
        # 计算特征重要性
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return model
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        评估模型性能
        
        参数:
            X: 特征数据
            y: 目标变量
            
        返回:
            评估指标
        """
        if self.best_model is None:
            logger.warning("模型尚未训练，无法评估")
            return {}
            
        y_pred = self.best_model.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"评估指标: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
        
        return metrics
    
    def save_results(self, filename_prefix: str = 'optimization'):
        """
        保存优化结果
        
        参数:
            filename_prefix: 文件名前缀
        """
        if self.best_params is None:
            logger.warning("尚未找到最佳参数，无法保存结果")
            return
            
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存最佳参数
        params_path = os.path.join(self.output_dir, f"{filename_prefix}_best_params.json")
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
            
        logger.info(f"最佳参数已保存: {params_path}")
        
        # 保存特征重要性
        if self.feature_importance is not None:
            importance_path = os.path.join(self.output_dir, f"{filename_prefix}_feature_importance.csv")
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"特征重要性已保存: {importance_path}")
            
        # 保存优化历史
        history_path = os.path.join(self.output_dir, f"{filename_prefix}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.optimization_history, f, indent=2, default=lambda x: list(x) if isinstance(x, np.ndarray) else x)
            
        logger.info(f"优化历史已保存: {history_path}")
            
        # 保存模型
        if self.best_model is not None:
            model_path = os.path.join(self.output_dir, f"{filename_prefix}_best_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
                
            logger.info(f"最佳模型已保存: {model_path}")
            
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        绘制特征重要性
        
        参数:
            top_n: 显示的特征数量
            save_path: 保存图片的路径
        """
        if self.feature_importance is None:
            logger.warning("特征重要性尚未计算，无法绘图")
            return
            
        # 获取前N个重要特征
        top_features = self.feature_importance.head(top_n)
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path)
            logger.info(f"特征重要性图已保存: {save_path}")
        else:
            plt.show()
            
    def plot_optimization_results(self, param_name: str, save_path: Optional[str] = None):
        """
        绘制参数优化结果
        
        参数:
            param_name: 参数名称
            save_path: 保存图片的路径
        """
        if self.cv_results is None:
            logger.warning("交叉验证结果尚未计算，无法绘图")
            return
            
        # 获取参数名称
        param_key = f'param_{param_name}'
        if param_key not in self.cv_results:
            logger.warning(f"参数 {param_name} 不在结果中")
            return
            
        # 提取数据
        params = self.cv_results[param_key]
        mean_test_scores = self.cv_results['mean_test_score']
        std_test_scores = self.cv_results['std_test_score']
        
        # 转换为数值类型
        param_values = [float(p) for p in params]
        
        # 创建数据框
        data = pd.DataFrame({
            'param_value': param_values,
            'mean_score': mean_test_scores,
            'std_score': std_test_scores
        })
        
        # 排序
        data = data.sort_values('param_value')
        
        # 创建图形
        plt.figure(figsize=(10, 6))
        plt.errorbar(data['param_value'], data['mean_score'], yerr=data['std_score'], marker='o')
        plt.title(f'Parameter Optimization: {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        
        # 添加最佳参数点
        best_param = self.best_params.get(param_name)
        if best_param is not None:
            best_idx = data[data['param_value'] == float(best_param)].index[0]
            best_score = data.loc[best_idx, 'mean_score']
            plt.scatter([float(best_param)], [best_score], color='red', s=100, label=f'Best: {best_param}')
            plt.legend()
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path)
            logger.info(f"参数优化结果图已保存: {save_path}")
        else:
            plt.show()
    
    @classmethod
    def load(cls, model_path: str, params_path: str) -> 'RegularizationOptimizer':
        """
        加载已保存的模型和参数
        
        参数:
            model_path: 模型文件路径
            params_path: 参数文件路径
            
        返回:
            初始化的优化器实例
        """
        # 加载模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # 加载参数
        with open(params_path, 'r') as f:
            params = json.load(f)
            
        # 确定模型类型
        if 'xgboost' in str(type(model)).lower():
            model_type = 'xgboost'
        elif 'lightgbm' in str(type(model)).lower():
            model_type = 'lightgbm'
        else:
            raise ValueError("无法确定模型类型")
            
        # 创建优化器实例
        optimizer = cls(model_type=model_type)
        optimizer.best_model = model
        optimizer.best_params = params
        
        return optimizer

def parse_args():
    """
    解析命令行参数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="优化模型正则化参数")
    
    parser.add_argument("--model_type", type=str, choices=['xgboost', 'lightgbm'], required=True,
                      help="模型类型")
    parser.add_argument("--optimization_method", type=str, choices=['grid', 'random'], default='random',
                      help="优化方法")
    parser.add_argument("--n_iter", type=int, default=50,
                      help="随机搜索的迭代次数")
    parser.add_argument("--cv", type=int, default=5,
                      help="交叉验证折数")
    parser.add_argument("--eval_metric", type=str, default="neg_root_mean_squared_error",
                      help="评估指标")
    parser.add_argument("--stages", type=str, nargs='+', choices=['depth', 'regularization', 'sampling', 'learning'],
                      help="分阶段优化的参数组")
    parser.add_argument("--random_state", type=int, default=42,
                      help="随机种子")
    parser.add_argument("--n_jobs", type=int, default=-1,
                      help="并行任务数")
    parser.add_argument("--verbose", type=int, default=1,
                      help="详细程度")
    parser.add_argument("--early_stopping_rounds", type=int, default=50,
                      help="早停轮数")
    parser.add_argument("--output_dir", type=str, default="outputs/optimization",
                      help="输出目录")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                      help="数据目录")
    parser.add_argument("--symbols", type=str, default="BTCUSDT",
                      help="交易对符号，多个符号用逗号分隔")
    parser.add_argument("--timeframe", type=str, default="1h",
                      help="时间框架")
    parser.add_argument("--target_type", type=str, default="price_change_pct",
                      help="目标类型")
    parser.add_argument("--horizon", type=int, default=60,
                      help="预测时间范围")
    parser.add_argument("--features", type=str, nargs='+',
                      help="使用的特征列表，如果不指定则使用所有可用特征")
    parser.add_argument("--test_size", type=float, default=0.2,
                      help="测试集比例")
    parser.add_argument("--feature_importance_method", type=str, default="gain",
                      help="特征重要性计算方法")
    parser.add_argument("--plot", action="store_true",
                      help="绘制优化结果图")
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 设置日志
    logger.info("开始模型参数优化")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    from src.data.data_loader import DataLoader
    
    symbols = args.symbols.split(",")
    
    data_loader = DataLoader(
        data_dir=args.data_dir,
        symbols=symbols,
        timeframes=[args.timeframe],
        target_type=args.target_type,
        target_horizon=args.horizon
    )
    
    # 获取特征和目标数据
    X_train, y_train, X_test, y_test = data_loader.load_train_test_data(test_size=args.test_size)
    
    if X_train is None or y_train is None:
        logger.error("加载数据失败")
        return
    
    # 如果指定了特定特征，则仅使用这些特征
    if args.features:
        # 检查指定的特征是否存在于数据中
        missing_features = [f for f in args.features if f not in X_train.columns]
        if missing_features:
            logger.warning(f"以下特征不存在于数据中: {missing_features}")
            args.features = [f for f in args.features if f in X_train.columns]
            
        if not args.features:
            logger.error("没有可用的特征")
            return
            
        X_train = X_train[args.features]
        if X_test is not None:
            X_test = X_test[args.features]
    
    logger.info(f"加载数据完成，训练集形状: X={X_train.shape}, y={y_train.shape}")
    if X_test is not None:
        logger.info(f"测试集形状: X={X_test.shape}, y={y_test.shape}")
        
    # 创建优化器
    optimizer = RegularizationOptimizer(
        model_type=args.model_type,
        optimization_method=args.optimization_method,
        n_iter=args.n_iter,
        cv=args.cv,
        eval_metric=args.eval_metric,
        stages=args.stages,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        early_stopping_rounds=args.early_stopping_rounds,
        output_dir=args.output_dir,
        feature_importance_method=args.feature_importance_method
    )
    
    # 执行优化
    logger.info("开始参数优化")
    optimizer.optimize(X_train, y_train)
    
    # 使用最佳参数训练模型
    logger.info("使用最佳参数训练模型")
    model = optimizer.train_with_best_params(X_train, y_train)
    
    # 评估模型
    if X_test is not None and y_test is not None:
        logger.info("评估模型性能")
        metrics = optimizer.evaluate(X_test, y_test)
        
        logger.info("评估指标:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.6f}")
    
    # 保存结果
    logger.info("保存优化结果")
    optimizer.save_results(f"{args.model_type}_{args.optimization_method}")
    
    # 绘制结果
    if args.plot:
        # 绘制特征重要性
        logger.info("绘制特征重要性")
        importance_path = os.path.join(args.output_dir, f"{args.model_type}_feature_importance.png")
        optimizer.plot_feature_importance(save_path=importance_path)
        
        # 绘制参数优化结果
        if args.stages:
            for stage in args.stages:
                for param_name in optimizer.param_grid:
                    logger.info(f"绘制参数优化结果: {param_name}")
                    param_path = os.path.join(args.output_dir, f"{args.model_type}_{param_name}_optimization.png")
                    optimizer.plot_optimization_results(param_name, save_path=param_path)
    
    logger.info("模型参数优化完成")

if __name__ == "__main__":
    main() 