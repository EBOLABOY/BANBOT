"""
XGBoost模型超参数调优脚本 - 使用贝叶斯优化寻找最佳参数
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config
from src.models.model_training import load_data, prepare_train_test_data
from src.models.traditional_models import XGBoostModel

logger = get_logger(__name__)

class XGBoostOptimizer:
    """
    XGBoost模型超参数优化器
    """
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: str = "models/hyperparameters",
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
        feature_file: str = "",
        target_file: str = "",
        target_column: str = "",
    ):
        """
        初始化XGBoost优化器
        
        参数:
            X: 特征数据
            y: 目标数据
            output_dir: 输出目录
            n_trials: 优化试验次数
            cv_folds: 交叉验证折数
            random_state: 随机种子
            feature_file: 特征文件路径
            target_file: 目标文件路径
            target_column: 目标列名
        """
        self.X = X
        self.y = y
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.feature_file = feature_file
        self.target_file = target_file
        self.target_column = target_column
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"数据加载完成: X={self.X.shape}, y={self.y.shape}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        贝叶斯优化的目标函数
        
        参数:
            trial: Optuna试验对象
            
        返回:
            float: 交叉验证的平均RMSE
        """
        # 定义参数搜索空间 - 主要关注防止过拟合的参数
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": self.random_state,
        }
        
        # 设置时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # 存储每次交叉验证的指标
        cv_scores = []
        
        # 执行交叉验证
        for train_idx, val_idx in tscv.split(self.X):
            X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
            X_val, y_val = self.X.iloc[val_idx], self.y.iloc[val_idx]
            
            # 创建并训练模型
            model = XGBoostModel(
                name="xgboost_cv",
                model_params=params,
                prediction_horizon=60,  # 默认值，不影响参数优化
                target_type="price_change_pct"
            )
            
            # 训练模型
            model.fit(X_train, y_train, X_val, y_val)
            
            # 预测
            y_pred = model.predict(X_val)
            
            # 计算指标
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2 = r2_score(y_val, y_pred)
            
            # 记录指标
            cv_scores.append((rmse, r2))
            
            # 报告当前进度
            logger.debug(f"交叉验证折 - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # 计算平均指标
        mean_rmse = np.mean([score[0] for score in cv_scores])
        mean_r2 = np.mean([score[1] for score in cv_scores])
        
        # 报告试验结果
        logger.info(f"试验 #{trial.number}: 参数={params}, 平均RMSE={mean_rmse:.4f}, 平均R2={mean_r2:.4f}")
        
        return mean_rmse
    
    def optimize(self) -> Dict[str, Any]:
        """
        执行超参数优化
        
        返回:
            Dict[str, Any]: 最佳参数
        """
        # 创建优化研究
        study = optuna.create_study(
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            direction="minimize",
        )
        
        # 运行优化
        logger.info(f"开始超参数优化，将执行 {self.n_trials} 次试验...")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # 获取最佳参数
        best_params = study.best_params
        best_rmse = study.best_value
        
        logger.info(f"最佳RMSE: {best_rmse:.4f}")
        logger.info(f"最佳参数: {best_params}")
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.output_dir, f"xgboost_best_params_{timestamp}.json")
        
        # 将参数转换为可序列化的格式
        best_params_serializable = {k: float(v) if isinstance(v, np.float32) else v 
                                  for k, v in best_params.items()}
        
        with open(result_file, "w") as f:
            json.dump({
                "best_params": best_params_serializable,
                "best_rmse": float(best_rmse),
                "n_trials": self.n_trials,
                "feature_file": self.feature_file,
                "target_file": self.target_file,
                "target_column": self.target_column,
                "timestamp": timestamp
            }, f, indent=2)
            
        logger.info(f"最佳参数已保存至: {result_file}")
        
        # 生成参数重要性图
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_param_importances
            
            # 确保输出目录存在
            os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
            
            # 保存参数重要性图
            fig = plot_param_importances(study)
            fig.write_image(os.path.join(self.output_dir, "plots", f"param_importance_{timestamp}.png"))
            logger.info(f"参数重要性图已保存")
        except Exception as e:
            logger.warning(f"生成参数重要性图失败: {str(e)}")
        
        return best_params
    
    def train_with_best_params(self, 
                              best_params: Dict[str, Any], 
                              horizon: int = 60, 
                              target_type: str = "price_change_pct") -> XGBoostModel:
        """
        使用最佳参数训练模型
        
        参数:
            best_params: 最佳超参数
            horizon: 预测周期
            target_type: 目标类型
            
        返回:
            XGBoostModel: 训练好的模型
        """
        # 准备训练集和测试集
        X_train, y_train, X_test, y_test, X_val, y_val = prepare_train_test_data(
            self.X, self.y, test_size=0.2, validation_size=0.1, 
            random_state=self.random_state, time_series_split=True
        )
        
        # 创建模型
        model = XGBoostModel(
            name=f"xgboost_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_params=best_params,
            prediction_horizon=horizon,
            target_type=target_type
        )
        
        # 训练模型
        logger.info("使用最佳参数训练模型...")
        model = model.fit(X_train, y_train, X_val, y_val)
        
        # 评估模型
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"训练集 R²: {train_score:.4f}")
        logger.info(f"测试集 R²: {test_score:.4f}")
        
        return model

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="XGBoost超参数优化工具")
    
    # 主要参数
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    
    # 数据参数
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对")
    parser.add_argument("--timeframe", type=str, default="1h", help="时间框架")
    parser.add_argument("--feature_file", type=str, help="特征文件路径，如不指定则根据symbol和timeframe自动生成")
    parser.add_argument("--target_file", type=str, help="目标文件路径，如不指定则根据symbol和timeframe自动生成")
    parser.add_argument("--target_column", type=str, default="target_pct_60", help="目标列名")
    parser.add_argument("--feature_type", type=str, default="standard", 
                      choices=["standard", "cross_timeframe"], 
                      help="特征类型：标准特征或跨周期特征")
    parser.add_argument("--features", type=str, help="要使用的特征列表(逗号分隔)或特征集名称")
    
    # 优化参数
    parser.add_argument("--n_trials", type=int, default=50, help="优化试验次数")
    parser.add_argument("--cv_folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--output_dir", type=str, default="models/hyperparameters", help="输出目录")
    parser.add_argument("--train_best", action="store_true", help="使用最佳参数训练模型")
    parser.add_argument("--horizon", type=int, default=60, help="预测周期（分钟）")
    
    # 其他参数
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(config_path=args.config, default_level=log_level)
    
    logger.info("开始XGBoost超参数优化")
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 确定特征和目标文件路径
        if args.feature_file is None:
            if args.feature_type == "standard":
                # 自动生成标准特征文件路径
                feature_dir = os.path.join("data/processed/features", args.symbol)
                args.feature_file = os.path.join(feature_dir, f"features_{args.timeframe}.csv")
            elif args.feature_type == "cross_timeframe":
                # 自动生成跨周期特征文件路径
                feature_dir = "data/processed/features/cross_timeframe"
                timeframes = ["1h", "4h", "1d"]  # 默认跨周期组合
                timeframes_str = "_".join(timeframes)
                args.feature_file = os.path.join(feature_dir, f"{args.symbol}_multi_tf_{timeframes_str}.csv")
                logger.info(f"使用跨周期特征: {timeframes_str}")
            
        if args.target_file is None:
            # 自动生成目标文件路径（目标文件总是基于基础时间框架）
            target_dir = os.path.join("data/processed/features", args.symbol)
            args.target_file = os.path.join(target_dir, f"targets_{args.timeframe}.csv")
        
        logger.info(f"特征文件: {args.feature_file}")
        logger.info(f"目标文件: {args.target_file}")
        logger.info(f"目标列: {args.target_column}")
        
        # 加载完整数据
        try:
            X_full, y_full = load_data(args.feature_file, args.target_file, args.target_column)
        except FileNotFoundError as e:
            logger.error(f"找不到数据文件: {str(e)}")
            return 1
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            return 1

        # 根据 features 参数筛选特征
        selected_features = None
        if args.features:
            feature_sets = config.get("models", {}).get("feature_sets", {})
            if args.features in feature_sets:
                selected_features = feature_sets[args.features]
                logger.info(f"使用配置文件中的特征集 '{args.features}': {selected_features}")
            else:
                selected_features = [f.strip() for f in args.features.split(',')]
                logger.info(f"使用命令行提供的特征列表: {selected_features}")
            
            # 检查特征是否存在并过滤
            missing_features = [f for f in selected_features if f not in X_full.columns]
            if missing_features:
                logger.warning(f"以下特征在数据中不存在，将被忽略: {missing_features}")
                selected_features = [f for f in selected_features if f in X_full.columns]
            
            if not selected_features:
                logger.error("没有有效的特征可用于优化。")
                return 1
                
            X = X_full[selected_features]
            y = y_full
            logger.info(f"特征筛选后数据形状: X={X.shape}")
        else:
            # 如果没有指定features，使用所有特征
            X = X_full
            y = y_full
            logger.info("使用所有特征进行优化。")

        # 创建优化器 (传递筛选后的 X, y)
        optimizer = XGBoostOptimizer(
            X=X, # 传递筛选后的特征数据
            y=y, # 传递对应的目标数据
            output_dir=args.output_dir,
            n_trials=args.n_trials,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            feature_file=args.feature_file,
            target_file=args.target_file,
            target_column=args.target_column
        )
        
        # 执行优化
        best_params = optimizer.optimize()
        
        # 如果需要，使用最佳参数训练模型
        if args.train_best:
            model = optimizer.train_with_best_params(
                best_params=best_params,
                horizon=args.horizon,
                target_type="price_change_pct"
            )
            
            # 保存模型
            model_dir = os.path.join("models/saved_models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{model.name}.pkl")
            model.save(model_path)
            logger.info(f"优化后的模型已保存至: {model_path}")
        
        logger.info("XGBoost超参数优化完成")
        return 0
    
    except Exception as e:
        logger.error(f"XGBoost超参数优化失败: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 