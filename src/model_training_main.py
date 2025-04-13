"""
模型训练主脚本 - 用于模型训练和评估
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config
from src.models.model_training import (
    load_data, prepare_train_test_data, create_model, 
    train_model
)
from src.models.model_evaluation import ModelEvaluator

logger = get_logger(__name__)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="加密货币交易模型训练工具")
    
    # 主要参数
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    
    # 数据参数
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对")
    parser.add_argument("--timeframe", type=str, default="1h", help="时间框架")
    parser.add_argument("--feature_file", type=str, help="特征文件路径，如不指定则根据symbol和timeframe自动生成")
    parser.add_argument("--target_file", type=str, help="目标文件路径，如不指定则根据symbol和timeframe自动生成")
    parser.add_argument("--features", type=str, help="要使用的特征列表，逗号分隔")
    parser.add_argument("--target_column", type=str, help="目标变量列名")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="xgboost",
                      choices=["linear", "tree", "xgboost"], 
                      help="模型类型")
    parser.add_argument("--subtype", type=str, help="模型子类型，例如linear的ridge/lasso，tree的random_forest/gradient_boosting")
    parser.add_argument("--horizon", type=int, default=60, 
                      help="预测周期（分钟）")
    parser.add_argument("--target_type", type=str, default="price_change_pct",
                      choices=["price_change_pct", "direction", "volatility"], 
                      help="目标类型")
    
    # 训练参数
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--val_size", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--time_series_split", action="store_true", help="使用时间序列分割")
    
    # 输出参数
    parser.add_argument("--no_save", action="store_true", help="不保存模型")
    parser.add_argument("--output_dir", type=str, default="models/saved_models", 
                      help="模型输出目录")
    parser.add_argument("--eval_dir", type=str, help="评估结果目录，默认为models/evaluation")
    
    # 其他参数
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    
    logger.info("开始模型训练流程")
    
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
        
        # 如果指定了特定特征，则只使用这些特征
        if hasattr(args, 'features') and args.features:
            selected_features = args.features.split(',')
            logger.info(f"使用选定的特征: {selected_features}")
            
            # 检查选定的特征是否存在于数据中
            missing_features = [f for f in selected_features if f not in X.columns]
            if missing_features:
                logger.warning(f"以下特征在数据中不存在: {missing_features}")
                
                # 过滤掉不存在的特征
                selected_features = [f for f in selected_features if f in X.columns]
                if not selected_features:
                    logger.error("没有有效的特征可用于训练，请检查特征名称")
                    return 1
                
                logger.info(f"使用这些有效特征进行训练: {selected_features}")
            
            # 只保留选定的特征
            X = X[selected_features]
            logger.info(f"特征选择后的数据形状: {X.shape}")
        
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
        try:
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
            if model.trained:
                test_metrics = evaluator.evaluate_model(model, X_test, y_test)
                
                # 输出模型摘要
                if hasattr(model, 'summary') and callable(model.summary):
                    logger.info("模型摘要:")
                    logger.info(model.summary())
                
                # 显示特征重要性
                importance = model.get_feature_importance()
                if importance:
                    logger.info("前10个重要特征:")
                    for i, (feature, imp) in enumerate(list(importance.items())[:10]):
                        logger.info(f"  {i+1}. {feature}: {imp:.4f}")
                
                logger.info("模型训练和评估完成")
            else:
                logger.warning("模型训练失败，跳过评估和特征重要性分析")
        except Exception as e:
            logger.error(f"模型训练或评估过程中发生错误: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return 1
            
        return 0
        
    except Exception as e:
        logger.error(f"模型训练过程中发生错误: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 