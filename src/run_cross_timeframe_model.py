"""
跨周期特征模型训练示例脚本
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config
from src.features.cross_timeframe_features import create_cross_timeframe_features, CrossTimeframeFeatureIntegrator

logger = get_logger(__name__)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="跨周期特征模型训练示例脚本")
    
    # 主要参数
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对")
    parser.add_argument("--timeframes", type=str, default="1h,4h,1d", 
                        help="要整合的时间框架，逗号分隔，如 '1h,4h,1d'")
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
    
    logger.info("开始跨周期特征模型训练流程")
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 处理交易对参数
        symbols = [args.symbol]
        
        # 处理时间框架参数
        timeframes = args.timeframes.split(',')
        
        # 第1步：确保跨周期特征已生成
        base_data_dir = config.get("data", {}).get("data_dir", "data")
        output_dir = os.path.join(base_data_dir, "processed/features/cross_timeframe")
        timeframes_str = "_".join(timeframes)
        feature_file = os.path.join(output_dir, f"{args.symbol}_multi_tf_{timeframes_str}.csv")
        
        if not os.path.exists(feature_file):
            logger.info(f"跨周期特征文件不存在，正在生成: {feature_file}")
            
            # 创建跨周期特征
            integrator = CrossTimeframeFeatureIntegrator(base_timeframe=timeframes[0])
            
            # 获取特征目录
            features_dir = os.path.join(base_data_dir, "processed/features")
            
            # 获取要选择的特征
            selected_features = config.get("feature_engineering", {}).get("cross_timeframe_features", {})
            
            # 整合特征
            integrated_df = integrator.integrate_features(
                features_dir=features_dir,
                symbol=args.symbol,
                timeframes=timeframes,
                selected_features=selected_features,
                output_dir=output_dir
            )
            
            if integrated_df is None:
                logger.error("跨周期特征生成失败")
                return 1
                
            logger.info(f"跨周期特征已生成: {feature_file}")
        else:
            logger.info(f"跨周期特征文件已存在: {feature_file}")
        
        # 第2步：准备训练命令
        # 确定目标文件路径
        target_dir = os.path.join(base_data_dir, "processed/features", args.symbol)
        target_file = os.path.join(target_dir, f"targets_{timeframes[0]}.csv")
        
        # 构建训练命令
        selected_features = []
        
        # 从配置中获取跨周期特征组
        if "feature_engineering" in config and "cross_timeframe_features" in config["feature_engineering"]:
            for tf in timeframes:
                if tf in config["feature_engineering"]["cross_timeframe_features"]:
                    for feature in config["feature_engineering"]["cross_timeframe_features"][tf]:
                        selected_features.append(f"{tf}_{feature}")
        
        if not selected_features:
            # 如果没有配置特定特征，则使用所有特征
            logger.warning("未在配置中找到特定的跨周期特征，将使用所有特征")
            try:
                # 尝试加载特征文件获取列名
                df = pd.read_csv(feature_file, index_col=0, nrows=1)
                selected_features = df.columns.tolist()
            except Exception as e:
                logger.error(f"读取特征文件出错: {str(e)}")
                return 1
        
        # 构建模型训练命令
        cmd = [
            "python -m src.model_training_main",
            f"--feature_file={feature_file}",
            f"--target_file={target_file}",
            f"--symbol={args.symbol}",
            f"--timeframe={timeframes[0]}",
            "--feature_type=cross_timeframe",
            "--model_type=xgboost",
            "--target_type=price_change_pct",
            "--horizon=60",
            "--time_series_split"
        ]
        
        # 将特征列表添加到命令中
        feature_arg = ",".join(selected_features[:20])  # 限制特征数量，避免命令行过长
        cmd.append(f"--features={feature_arg}")
        
        # 打印最终命令
        command = " ".join(cmd)
        logger.info(f"模型训练命令: {command}")
        
        # 执行命令
        logger.info("开始执行模型训练")
        return_code = os.system(command)
        
        if return_code == 0:
            logger.info("模型训练成功完成")
        else:
            logger.error(f"模型训练失败，返回代码: {return_code}")
            return 1
        
        return 0
    except Exception as e:
        logger.error(f"跨周期特征模型训练出错: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 