"""
跨周期特征整合主脚本 - 用于将不同时间周期的特征整合为一个特征集
"""

import os
import sys
import argparse
import logging
import pandas as pd
from typing import List, Dict, Optional

from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config
from src.features.cross_timeframe_features import create_cross_timeframe_features, CrossTimeframeFeatureIntegrator

logger = get_logger(__name__)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="跨周期特征整合工具")
    
    # 主要参数
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    
    # 数据参数
    parser.add_argument("--symbols", type=str, default=None, 
                        help="要处理的交易对，逗号分隔，如 'BTCUSDT,ETHUSDT'")
    parser.add_argument("--timeframes", type=str, default="1h,4h,1d", 
                        help="要整合的时间框架，逗号分隔，如 '1h,4h,1d'")
    parser.add_argument("--base_timeframe", type=str, default="1h", 
                        help="基础时间框架，所有其他时间框架将对齐到这个时间框架")
    
    # 特征参数
    parser.add_argument("--feature_dir", type=str, default=None, 
                        help="特征目录，默认为 'data/processed/features'")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="输出目录，默认为 'data/processed/features/cross_timeframe'")
    parser.add_argument("--selected_features", type=str, default=None, 
                        help="要选择的特征列表，JSON格式字符串，如 '{\"1h\":[\"vwap\",\"rsi_14\"],\"4h\":[\"sma_50\",\"ema_200\"]}'")
    
    # 其他参数
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
    
    logger.info("开始跨周期特征整合流程")
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 处理交易对参数
        symbols = None
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(',')]
        
        # 处理时间框架参数
        timeframes = ["1h", "4h", "1d"]
        if args.timeframes:
            timeframes = [t.strip() for t in args.timeframes.split(',')]
            
        # 如果基础时间框架不在时间框架列表中，将其添加进去
        if args.base_timeframe not in timeframes:
            timeframes.append(args.base_timeframe)
        
        # 处理特征选择参数
        selected_features = None
        if args.selected_features:
            import json
            try:
                selected_features = json.loads(args.selected_features)
            except json.JSONDecodeError:
                logger.error(f"无法解析JSON格式的特征列表: {args.selected_features}")
                return 1
        
        # 处理目录参数
        if args.feature_dir:
            config["data"]["data_dir"] = os.path.dirname(args.feature_dir)
        
        if args.output_dir:
            output_dir = args.output_dir
        else:
            # 获取特征目录
            base_data_dir = config.get("data", {}).get("data_dir", "data")
            output_dir = os.path.join(base_data_dir, "processed/features/cross_timeframe")
        
        # 准备特征配置
        if selected_features and "feature_engineering" not in config:
            config["feature_engineering"] = {}
        
        if selected_features:
            config["feature_engineering"]["cross_timeframe_features"] = selected_features
        
        # 创建跨周期特征整合器
        integrator = CrossTimeframeFeatureIntegrator(base_timeframe=args.base_timeframe)
        
        # 获取特征目录
        if args.feature_dir:
            features_dir = args.feature_dir
        else:
            base_data_dir = config.get("data", {}).get("data_dir", "data")
            features_dir = os.path.join(base_data_dir, "processed/features")
        
        # 获取交易对列表
        if not symbols:
            symbols = config.get("data", {}).get("target_currencies", ["BTCUSDT"])
        
        # 为每个交易对创建跨周期特征
        for symbol in symbols:
            logger.info(f"正在为交易对 {symbol} 创建跨周期特征")
            
            try:
                # 整合特征
                integrated_df = integrator.integrate_features(
                    features_dir=features_dir,
                    symbol=symbol,
                    timeframes=timeframes,
                    selected_features=selected_features,
                    output_dir=output_dir
                )
                
                if integrated_df is not None:
                    # 输出一些统计信息
                    logger.info(f"成功为 {symbol} 创建跨周期特征")
                    logger.info(f"  - 形状: {integrated_df.shape}")
                    logger.info(f"  - 特征数量: {len(integrated_df.columns)}")
                    logger.info(f"  - 时间范围: {integrated_df.index.min()} 到 {integrated_df.index.max()}")
                    
                    # 检查缺失值
                    missing_count = integrated_df.isna().sum().sum()
                    if missing_count > 0:
                        logger.warning(f"  - 存在 {missing_count} 个缺失值")
                    else:
                        logger.info("  - 无缺失值")
                    
                    # 保存特征重要性信息
                    timeframes_str = "_".join(timeframes)
                    feature_list_file = os.path.join(output_dir, f"{symbol}_multi_tf_{timeframes_str}_features.txt")
                    with open(feature_list_file, 'w') as f:
                        for col in integrated_df.columns:
                            f.write(f"{col}\n")
                    
                    logger.info(f"特征列表已保存到 {feature_list_file}")
                else:
                    logger.error(f"为 {symbol} 创建跨周期特征失败")
            except Exception as e:
                logger.error(f"为 {symbol} 创建跨周期特征时出错: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        logger.info("跨周期特征整合流程完成")
        return 0
    except Exception as e:
        logger.error(f"跨周期特征整合流程出错: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 