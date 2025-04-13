"""
模型训练脚本 - 整合特征处理、模型训练和评估的主要流程
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt

from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config
from src.models.base_model import BaseModel
from src.models.traditional_models import LinearModel, TreeModel, XGBoostModel
from src.models.model_evaluation import ModelEvaluator

logger = get_logger(__name__)


def load_data(feature_path: str, target_path: str, target_column: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    加载特征和目标数据
    
    参数:
        feature_path: 特征数据文件路径
        target_path: 目标数据文件路径
        target_column: 目标列名，如果为None则尝试自动检测
        
    返回:
        Tuple[pd.DataFrame, pd.Series]: 特征DataFrame和目标Series
    """
    # 加载特征数据
    features_df = pd.read_csv(feature_path, index_col=0, parse_dates=True)
    
    # 加载目标数据
    targets_df = pd.read_csv(target_path, index_col=0, parse_dates=True)
    
    # 如果未指定目标列名，则尝试自动检测
    if target_column is None:
        # 查找以'target_'开头的列
        target_cols = [col for col in targets_df.columns if col.startswith('target_')]
        if not target_cols:
            raise ValueError(f"在目标文件{target_path}中找不到目标列（以'target_'开头的列）")
        target_column = target_cols[0]
        logger.info(f"自动选择目标列: {target_column}")
    
    # 检查目标列是否存在
    if target_column not in targets_df.columns:
        raise ValueError(f"在目标文件{target_path}中找不到列: {target_column}")
    
    # 确保索引对齐
    common_index = features_df.index.intersection(targets_df.index)
    if len(common_index) == 0:
        raise ValueError("特征和目标数据没有共同的索引")
    
    features_df = features_df.loc[common_index]
    targets_df = targets_df.loc[common_index]
    
    # 提取目标列
    target_series = targets_df[target_column]
    
    logger.info(f"已加载数据: 特征={features_df.shape}, 目标={target_series.shape}")
    
    return features_df, target_series


def prepare_train_test_data(X: pd.DataFrame, 
                          y: pd.Series, 
                          test_size: float = 0.2, 
                          validation_size: float = 0.1,
                          random_state: int = 42,
                          time_series_split: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    准备训练集、验证集和测试集
    
    参数:
        X: 特征数据
        y: 目标数据
        test_size: 测试集比例
        validation_size: 验证集比例
        random_state: 随机种子
        time_series_split: 是否使用时间序列分割
        
    返回:
        Tuple: (X_train, y_train, X_test, y_test, X_val, y_val)
    """
    if time_series_split:
        # 按时间顺序分割
        logger.info("使用时间序列分割方法")
        
        # 计算分割点
        total_samples = len(X)
        test_samples = int(total_samples * test_size)
        val_samples = int(total_samples * validation_size)
        
        # 按时间顺序分割
        train_end = total_samples - test_samples - val_samples
        val_end = total_samples - test_samples
        
        # 训练集
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        
        # 验证集
        if validation_size > 0:
            X_val = X.iloc[train_end:val_end]
            y_val = y.iloc[train_end:val_end]
        else:
            X_val, y_val = None, None
        
        # 测试集
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]
        
    else:
        # 随机分割
        logger.info("使用随机分割方法")
        
        # 先分离测试集
        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 如果需要验证集，再分离验证集
        if validation_size > 0:
            # 计算验证集在剩余数据中的比例
            val_ratio = validation_size / (1 - test_size)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_rest, y_rest, test_size=val_ratio, random_state=random_state
            )
        else:
            X_train, y_train = X_rest, y_rest
            X_val, y_val = None, None
    
    # 记录数据集大小
    logger.info(f"数据集分割: 训练集={X_train.shape}, "
               f"测试集={X_test.shape}, "
               f"验证集={X_val.shape if X_val is not None else 'None'}")
    
    return X_train, y_train, X_test, y_test, X_val, y_val


def create_model(model_type: str, 
                model_params: Dict,
                prediction_horizon: int,
                target_type: str) -> BaseModel:
    """
    创建指定类型的模型
    
    参数:
        model_type: 模型类型
        model_params: 模型参数
        prediction_horizon: 预测周期
        target_type: 目标类型
        
    返回:
        BaseModel: 创建的模型实例
    """
    # 根据模型类型创建相应的模型
    if model_type == "linear":
        # 线性模型
        linear_type = model_params.pop("linear_type", "ridge")
        return LinearModel(
            name=f"linear_{linear_type}",
            model_params=model_params,
            prediction_horizon=prediction_horizon,
            target_type=target_type,
            model_type=linear_type
        )
    elif model_type == "tree":
        # 树模型
        tree_type = model_params.pop("tree_type", "random_forest")
        return TreeModel(
            name=f"tree_{tree_type}",
            model_params=model_params,
            prediction_horizon=prediction_horizon,
            target_type=target_type,
            model_type=tree_type
        )
    elif model_type == "xgboost":
        # XGBoost模型
        return XGBoostModel(
            name="xgboost",
            model_params=model_params,
            prediction_horizon=prediction_horizon,
            target_type=target_type
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def train_model(model: BaseModel, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_val: pd.DataFrame = None, 
              y_val: pd.Series = None,
              save_model: bool = True) -> BaseModel:
    """
    训练模型
    
    参数:
        model: 要训练的模型
        X_train: 训练特征数据
        y_train: 训练目标数据
        X_val: 验证特征数据
        y_val: 验证目标数据
        save_model: 是否保存模型
        
    返回:
        BaseModel: 训练后的模型
    """
    logger.info(f"开始训练模型: {model.name} (类型: {model.__class__.__name__})")
    
    # 准备验证数据
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    
    # 训练模型
    train_metrics = model.train(X_train, y_train, validation_data)
    
    # 输出训练指标
    logger.info("训练完成，训练指标:")
    for metric, value in train_metrics.items():
        logger.info(f"  - {metric}: {value}")
    
    # 保存模型
    if save_model:
        saved_path = model.save()
        logger.info(f"模型已保存至: {saved_path}")
    
    return model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="加密货币价格预测模型训练工具")
    
    # 数据参数
    parser.add_argument("--symbol", type=str, required=True, help="交易对，例如BTC/USDT")
    parser.add_argument("--timeframe", type=str, required=True, help="时间框架，例如1h")
    parser.add_argument("--feature_file", type=str, help="特征文件路径，如不指定则根据symbol和timeframe自动生成")
    parser.add_argument("--target_file", type=str, help="目标文件路径，如不指定则根据symbol和timeframe自动生成")
    parser.add_argument("--target_column", type=str, help="目标列名，如不指定则自动检测")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, required=True, 
                      choices=["linear", "tree", "xgboost"], 
                      help="模型类型: linear, tree, xgboost")
    parser.add_argument("--subtype", type=str, help="模型子类型，例如linear的ridge/lasso，tree的random_forest/gradient_boosting")
    parser.add_argument("--horizon", type=int, required=True, help="预测周期")
    parser.add_argument("--target_type", type=str, default="price_change_pct",
                      choices=["price_change_pct", "direction", "volatility"],
                      help="目标类型: price_change_pct, direction, volatility")
    
    # 训练参数
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--val_size", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--time_series_split", action="store_true", help="使用时间序列分割")
    parser.add_argument("--random_state", type=int, default=42, help="随机数种子")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    
    # 输出参数
    parser.add_argument("--no_save", action="store_true", help="不保存模型")
    parser.add_argument("--output_dir", type=str, help="输出目录，默认为models/saved_models")
    parser.add_argument("--eval_dir", type=str, help="评估结果目录，默认为models/evaluation")
    
    # 其他参数
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 确定特征和目标文件路径
        if args.feature_file is None:
            # 自动生成特征文件路径
            feature_dir = os.path.join("data/processed/features", args.symbol)
            args.feature_file = os.path.join(feature_dir, f"features_{args.timeframe}.csv")
            
        if args.target_file is None:
            # 自动生成目标文件路径
            target_dir = os.path.join("data/processed/features", args.symbol)
            args.target_file = os.path.join(target_dir, f"targets_{args.timeframe}.csv")
        
        logger.info(f"特征文件: {args.feature_file}")
        logger.info(f"目标文件: {args.target_file}")
        
        # 如果指定了目标列，记录
        if args.target_column:
            logger.info(f"目标列: {args.target_column}")
        
        # 加载数据
        try:
            X, y = load_data(args.feature_file, args.target_file, args.target_column)
        except FileNotFoundError as e:
            logger.error(f"找不到数据文件: {str(e)}")
            return 1
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            return 1
        
        # 准备训练、验证和测试数据
        X_train, y_train, X_test, y_test, X_val, y_val = prepare_train_test_data(
            X, y, 
            test_size=args.test_size, 
            validation_size=args.val_size,
            random_state=args.random_state,
            time_series_split=args.time_series_split
        )
        
        # 获取模型配置
        model_config = config.get("models", {}).get(args.model_type, {})
        
        # 如果命令行指定了模型子类型，更新配置
        if args.subtype:
            if args.model_type == "linear":
                model_config["linear_type"] = args.subtype
            elif args.model_type == "tree":
                model_config["tree_type"] = args.subtype
        
        # 创建模型
        model = create_model(
            model_type=args.model_type,
            model_params=model_config,
            prediction_horizon=args.horizon,
            target_type=args.target_type
        )
        
        # 训练模型
        model = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            save_model=not args.no_save
        )
        
        # 创建评估器
        eval_dir = args.eval_dir or "models/evaluation"
        evaluator = ModelEvaluator(output_dir=eval_dir)
        
        # 评估模型
        test_metrics = evaluator.evaluate_model(model, X_test, y_test)
        
        # 输出模型摘要
        logger.info("模型摘要:")
        logger.info(model.summary())
        
        # 显示特征重要性
        importance = model.get_feature_importance()
        if importance:
            logger.info("前10个重要特征:")
            for i, (feature, imp) in enumerate(list(importance.items())[:10]):
                logger.info(f"  {i+1}. {feature}: {imp:.4f}")
        
        logger.info("模型训练和评估完成")
        return 0
        
    except Exception as e:
        logger.error(f"模型训练过程中发生错误: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 