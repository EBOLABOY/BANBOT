"""
特征工程主脚本 - 用于特征生成和处理
"""

import os
import sys
import argparse
import logging
import traceback
from datetime import datetime
from tqdm import tqdm

from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config
from src.features.feature_engineering import FeatureEngineer

logger = get_logger(__name__)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="加密货币特征工程工具")
    
    # 主要参数
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["compute", "select", "all"], default="all",
                      help="操作模式: compute=仅计算特征, select=仅进行特征选择, all=计算并选择特征")
    
    # 数据参数
    parser.add_argument("--symbols", type=str, nargs="+", help="要处理的交易对列表，例如BTC/USDT ETH/USDT")
    parser.add_argument("--timeframes", type=str, nargs="+", help="要处理的时间框架列表, 例如1m 5m 1h")
    parser.add_argument("--start_date", type=str, help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="截止日期 (YYYY-MM-DD)")
    parser.add_argument("--data_dir", type=str, default="data/processed/merged", help="原始数据目录")
    
    # 特征计算参数
    parser.add_argument("--feature_groups", type=str, nargs="+", 
                      help="要计算的特征组，例如 price_based volume_based volatility trend momentum market_microstructure")
    
    # 特征选择参数
    parser.add_argument("--selection_method", type=str, 
                      choices=["mutual_info", "f_regression", "pca"], 
                      help="特征选择方法")
    parser.add_argument("--n_features", type=int, help="要选择的特征数量")
    parser.add_argument("--target_type", type=str, 
                      choices=["price_change_pct", "direction", "volatility"], 
                      default="price_change_pct",
                      help="用于特征选择的目标变量类型")
    parser.add_argument("--target_horizon", type=int, help="用于特征选择的预测周期")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="data/processed/features", 
                      help="输出特征目录")
    
    # GPU加速参数
    parser.add_argument("--use_gpu", action="store_true", help="使用GPU加速计算特征")
    parser.add_argument("--batch_size", type=int, default=500000, help="GPU处理的批次大小，用于避免GPU内存不足")
    
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
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(config_path=args.config, default_level=log_level)
    
    logger.info("开始特征工程流程")
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 如果启用GPU，导入GPU支持模块并配置
        if args.use_gpu:
            try:
                # 尝试导入GPU技术指标模块和GPU加速器
                from src.features.gpu_accelerator import get_accelerator
                from src.features.gpu_technical_indicators import GpuCompatibleTechnicalIndicators
                
                accelerator = get_accelerator()
                if accelerator.is_available():
                    logger.info("GPU加速已启用 - RAPIDS库已加载")
                    # 使用GPU兼容的技术指标类替换原有实现
                    from src.features.feature_engineering import FeatureEngineer
                    FeatureEngineer._orig_tech_indicators = FeatureEngineer.tech_indicators
                    FeatureEngineer.tech_indicators = GpuCompatibleTechnicalIndicators()
                    logger.info("已替换技术指标计算为GPU加速版本")
                else:
                    logger.warning("无法启用GPU加速 - 未检测到RAPIDS库或GPU设备")
            except ImportError as e:
                logger.warning(f"无法导入GPU加速库: {e}")
                logger.warning("将使用CPU进行计算，如需GPU加速，请安装RAPIDS库: 'pip install cudf-cu11 cuml-cu11 cupy-cuda11x'")
        
        # 创建特征工程器
        feature_engineer = FeatureEngineer(args.config)
        
        # 处理模式: 计算特征
        if args.mode in ["compute", "all"]:
            logger.info("开始计算特征...")
            
            # GPU处理的批量大小
            batch_size = args.batch_size if args.use_gpu else None
            
            # 处理所有交易对的数据
            processed_data = feature_engineer.process_all_data(
                symbols=args.symbols,
                timeframes=args.timeframes,
                start_date=args.start_date,
                end_date=args.end_date,
                data_dir=args.data_dir,
                batch_size=batch_size
            )
            
            logger.info(f"已为 {len(processed_data)} 个交易对计算特征")
        
        # 处理模式: 特征选择
        if args.mode in ["select", "all"] and args.selection_method:
            logger.info("开始特征选择...")
            
            # 确定目标变量类型和预测周期
            target_type = args.target_type or "price_change_pct"
            
            # 如果未指定预测周期，使用配置文件中的第一个
            if not args.target_horizon:
                horizons_config = config.get("models", {}).get("prediction_horizons", {})
                for period_list in horizons_config.values():
                    if period_list:
                        target_horizon = period_list[0]
                        break
                else:
                    target_horizon = 1
            else:
                target_horizon = args.target_horizon
            
            # 获取或创建输出目录
            output_dir = args.output_dir or "data/processed/features/selected"
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取特征目录下的所有交易对
            if not args.symbols:
                feature_base_dir = "data/processed/features"
                args.symbols = [d for d in os.listdir(feature_base_dir) 
                              if os.path.isdir(os.path.join(feature_base_dir, d))]
            
            # 获取每个交易对目录下的所有时间框架
            for symbol in tqdm(args.symbols, desc="处理交易对特征选择"):
                symbol_dir = os.path.join("data/processed/features", symbol)
                
                if not os.path.exists(symbol_dir):
                    logger.warning(f"找不到交易对 {symbol} 的特征目录")
                    continue
                
                # 获取此交易对的所有特征文件
                feature_files = [f for f in os.listdir(symbol_dir) if f.startswith("features_")]
                
                # 筛选指定的时间框架
                if args.timeframes:
                    feature_files = [f for f in feature_files if any(tf in f for tf in args.timeframes)]
                
                for feature_file in feature_files:
                    try:
                        # 从文件名中提取时间框架
                        tf_part = feature_file.replace("features_", "").split("_")[0]
                        
                        # 加载特征
                        feature_path = os.path.join(symbol_dir, feature_file)
                        features_df = feature_engineer.load_data(feature_path)
                        
                        # 获取对应的目标变量文件
                        target_file = feature_file.replace("features_", "targets_")
                        target_path = os.path.join(symbol_dir, target_file)
                        
                        if not os.path.exists(target_path):
                            logger.warning(f"找不到对应的目标变量文件: {target_path}")
                            continue
                        
                        # 加载目标变量
                        targets_df = feature_engineer.load_data(target_path)
                        
                        # 确定目标列名
                        target_col = f"target_pct_{target_horizon}"
                        
                        if target_col not in targets_df.columns:
                            logger.warning(f"找不到目标列 {target_col} 在 {target_path}")
                            continue
                        
                        # 选择特征
                        selected_features_df, selected_features = feature_engineer.select_features(
                            X=features_df,
                            y=targets_df[target_col],
                            method=args.selection_method,
                            n_features=args.n_features
                        )
                        
                        # 创建输出目录
                        symbol_output_dir = os.path.join(output_dir, symbol)
                        os.makedirs(symbol_output_dir, exist_ok=True)
                        
                        # 生成输出文件名
                        output_filename = f"selected_features_{tf_part}_{args.selection_method}_{args.n_features}.csv"
                        output_path = os.path.join(symbol_output_dir, output_filename)
                        
                        # 保存选定的特征为CSV文件
                        selected_features_df.to_csv(output_path)
                        
                        # 保存特征名称列表
                        features_list_path = output_path.replace(".csv", "_names.txt")
                        with open(features_list_path, 'w') as f:
                            for feature in selected_features:
                                f.write(f"{feature}\n")
                        
                        logger.info(f"已为 {symbol} {tf_part} 选择 {len(selected_features)} 个特征，保存至 {output_path}")
                    
                    except Exception as e:
                        logger.error(f"处理文件 {feature_file} 时出错: {str(e)}")
                        logger.debug(traceback.format_exc())
        
        logger.info("特征工程流程完成")
        
    except Exception as e:
        logger.error(f"特征工程过程中发生错误: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 