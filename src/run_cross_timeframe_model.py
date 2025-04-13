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
    parser.add_argument("--target_column", type=str, default="target_pct_60",
                        help="目标列名，如target_pct_60表示60分钟后的价格百分比变化")
    parser.add_argument("--horizon", type=int, default=60,
                        help="预测时间范围（分钟）")
    parser.add_argument("--diagnose", action="store_true", 
                        help="运行数据诊断模式，不执行实际训练")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    
    return parser.parse_args()

def diagnose_data(feature_file, target_file, target_column=None):
    """
    诊断特征和目标数据
    
    参数：
        feature_file: 特征文件路径
        target_file: 目标文件路径
        target_column: 目标列名
    """
    logger.info("===== 数据诊断开始 =====")
    
    # 检查文件是否存在
    if not os.path.exists(feature_file):
        logger.error(f"特征文件不存在: {feature_file}")
        return
    
    if not os.path.exists(target_file):
        logger.error(f"目标文件不存在: {target_file}")
        return
    
    # 加载特征数据
    try:
        features_df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
        logger.info(f"特征文件成功加载，形状: {features_df.shape}")
        logger.info(f"特征时间范围: {features_df.index.min()} 到 {features_df.index.max()}")
        
        # 检查列名
        logger.info(f"特征列数: {len(features_df.columns)}")
        logger.info(f"前10个特征列: {list(features_df.columns)[:10]}")
        
        # 检查缺失值
        nan_count = features_df.isna().sum().sum()
        logger.info(f"特征数据中的缺失值数量: {nan_count}")
        
        # 检查跨周期特征
        timeframe_prefixes = ['1h_', '4h_', '1d_', '1m_', '5m_', '15m_']
        has_prefixes = {prefix: any(col.startswith(prefix) for col in features_df.columns) for prefix in timeframe_prefixes}
        logger.info(f"跨周期特征前缀检测: {has_prefixes}")
    except Exception as e:
        logger.error(f"特征文件加载失败: {str(e)}")
    
    # 加载目标数据
    try:
        targets_df = pd.read_csv(target_file, index_col=0, parse_dates=True)
        logger.info(f"目标文件成功加载，形状: {targets_df.shape}")
        logger.info(f"目标时间范围: {targets_df.index.min()} 到 {targets_df.index.max()}")
        
        # 列出所有目标列
        logger.info(f"所有目标列: {list(targets_df.columns)}")
        
        # 如果指定了目标列，检查是否存在
        if target_column:
            if target_column in targets_df.columns:
                logger.info(f"目标列 '{target_column}' 存在")
                # 统计非缺失值的数量
                non_nan_count = targets_df[target_column].notna().sum()
                logger.info(f"目标列 '{target_column}' 中非缺失值的数量: {non_nan_count}")
                logger.info(f"目标列 '{target_column}' 中缺失值的数量: {len(targets_df) - non_nan_count}")
                
                # 如果有非缺失值，显示基本统计
                if non_nan_count > 0:
                    logger.info(f"目标列 '{target_column}' 统计: 均值={targets_df[target_column].mean():.4f}, "
                               f"标准差={targets_df[target_column].std():.4f}, "
                               f"最小值={targets_df[target_column].min():.4f}, "
                               f"最大值={targets_df[target_column].max():.4f}")
            else:
                logger.error(f"目标列 '{target_column}' 不存在于目标文件中")
                # 推荐其他可能的目标列
                target_cols = [col for col in targets_df.columns if col.startswith('target_')]
                if target_cols:
                    logger.info(f"可用的目标列: {target_cols}")
                    
                    # 检查每个目标列的非缺失值数量
                    for col in target_cols:
                        non_nan_count = targets_df[col].notna().sum()
                        logger.info(f"目标列 '{col}' 中非缺失值的数量: {non_nan_count}")
    except Exception as e:
        logger.error(f"目标文件加载失败: {str(e)}")
    
    # 检查特征和目标的时间对齐
    try:
        # 重新加载数据确保一致的处理
        features_df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
        targets_df = pd.read_csv(target_file, index_col=0, parse_dates=True)
        
        # 找出公共索引
        common_index = features_df.index.intersection(targets_df.index)
        logger.info(f"特征和目标数据的公共时间戳数量: {len(common_index)}")
        
        if len(common_index) > 0:
            logger.info(f"公共时间范围: {common_index.min()} 到 {common_index.max()}")
            
            # 如果指定了目标列并且存在
            if target_column and target_column in targets_df.columns:
                # 计算在公共索引上非缺失目标值的数量
                aligned_targets = targets_df.loc[common_index, target_column]
                non_nan_count = aligned_targets.notna().sum()
                logger.info(f"在公共时间范围内，目标列 '{target_column}' 中非缺失值的数量: {non_nan_count}")
        else:
            logger.error("特征和目标数据没有公共时间戳")
            
            # 尝试分析原因
            if features_df.index.tzinfo != targets_df.index.tzinfo:
                logger.warning("特征和目标数据的时区不一致")
            
            f_min, f_max = features_df.index.min(), features_df.index.max()
            t_min, t_max = targets_df.index.min(), targets_df.index.max()
            
            if t_max < f_min:
                logger.warning("目标数据的时间范围在特征数据之前")
            elif t_min > f_max:
                logger.warning("目标数据的时间范围在特征数据之后")
    except Exception as e:
        logger.error(f"时间对齐检查失败: {str(e)}")
    
    logger.info("===== 数据诊断结束 =====")

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
        
        # 确定目标文件路径
        target_dir = os.path.join(base_data_dir, "processed/features", args.symbol)
        target_file = os.path.join(target_dir, f"targets_{timeframes[0]}.csv")
        
        # 如果是诊断模式，运行诊断
        if args.diagnose:
            diagnose_data(feature_file, target_file, args.target_column)
            return 0
        
        # 第2步：准备训练命令
        # 验证特征文件可读
        try:
            df = pd.read_csv(feature_file, index_col=0, nrows=5)
            logger.info(f"成功读取特征文件，示例列: {list(df.columns)[:5]}")
        except Exception as e:
            logger.error(f"无法读取特征文件: {str(e)}")
            return 1
        
        # 验证目标文件可读
        try:
            target_df = pd.read_csv(target_file, index_col=0, nrows=5)
            logger.info(f"成功读取目标文件，可用列: {list(target_df.columns)}")
            
            # 确保目标列存在
            if args.target_column not in target_df.columns:
                logger.error(f"目标列 '{args.target_column}' 不在目标文件中")
                # 检查可用的目标列
                target_cols = [col for col in target_df.columns if col.startswith('target_')]
                if target_cols:
                    logger.info(f"可用的目标列有: {target_cols}")
                    # 自动选择最匹配的目标列
                    for col in target_cols:
                        if f"target_pct_{args.horizon}" in col:
                            args.target_column = col
                            logger.info(f"自动选择目标列: {args.target_column}")
                            break
                    else:
                        args.target_column = target_cols[0]
                        logger.info(f"自动选择第一个可用目标列: {args.target_column}")
                else:
                    logger.error("目标文件中没有可用的目标列")
                    return 1
        except Exception as e:
            logger.error(f"无法读取目标文件: {str(e)}")
            return 1
            
        # 构建训练命令
        selected_features = []
        
        # 先检查特征文件中的列名，验证跨周期特征的存在
        try:
            df = pd.read_csv(feature_file, index_col=0, nrows=1)
            column_prefixes = set()
            for col in df.columns:
                if '_' in col:
                    prefix = col.split('_')[0]
                    if prefix in timeframes:
                        column_prefixes.add(prefix)
            
            if not column_prefixes:
                logger.warning(f"特征文件中没有找到跨周期前缀 {timeframes}，可能不是跨周期特征文件")
                if len(df.columns) > 0:
                    logger.info(f"文件包含以下列: {list(df.columns)[:5]}...")
            else:
                logger.info(f"特征文件中发现跨周期前缀: {column_prefixes}")
                
            # 将特征列名转换为小写，用于不区分大小写的匹配
            columns_lower = {col.lower(): col for col in df.columns}
            
            # 添加调试信息 - 这会显示每个时间框架前缀的几个列示例
            for tf in timeframes:
                tf_columns = [col for col in df.columns if col.startswith(f"{tf}_")]
                if tf_columns:
                    logger.info(f"{tf} 时间框架的列示例: {tf_columns[:5]}")
                else:
                    logger.warning(f"没有找到以 {tf}_ 开头的列")
        except Exception as e:
            logger.error(f"检查特征列名时出错: {str(e)}")
        
        # 从配置中获取跨周期特征组
        if "feature_engineering" in config and "cross_timeframe_features" in config["feature_engineering"]:
            for tf in timeframes:
                if tf in config["feature_engineering"]["cross_timeframe_features"]:
                    for feature in config["feature_engineering"]["cross_timeframe_features"][tf]:
                        prefixed_feature = f"{tf}_{feature}"
                        # 尝试精确匹配
                        if prefixed_feature in df.columns:
                            selected_features.append(prefixed_feature)
                        # 尝试不区分大小写的匹配
                        elif prefixed_feature.lower() in columns_lower:
                            actual_name = columns_lower[prefixed_feature.lower()]
                            selected_features.append(actual_name)
                            logger.info(f"通过不区分大小写匹配到特征: '{prefixed_feature}' -> '{actual_name}'")
                        # 尝试部分匹配（查找包含该特征名称的列）
                        else:
                            potential_matches = [col for col in df.columns 
                                             if col.lower().startswith(f"{tf.lower()}_") and 
                                             feature.lower() in col.lower()]
                            if potential_matches:
                                best_match = potential_matches[0]  # 取第一个匹配
                                selected_features.append(best_match)
                                logger.info(f"通过部分匹配找到特征: '{prefixed_feature}' -> '{best_match}'")
                            else:
                                logger.warning(f"特征 '{prefixed_feature}' 不在特征文件中")
                                
                                # 尝试查找具有相似名称的特征
                                similar_features = []
                                for col in df.columns:
                                    if col.startswith(f"{tf}_"):
                                        similar_features.append(col)
                                
                                if similar_features:
                                    # 显示一些最相似的特征名称，以帮助诊断问题
                                    import difflib
                                    most_similar = difflib.get_close_matches(
                                        prefixed_feature, similar_features, n=3, cutoff=0.6
                                    )
                                    if most_similar:
                                        logger.info(f"'{prefixed_feature}' 的相似特征: {most_similar}")
                                        # 自动使用最相似的特征
                                        selected_features.append(most_similar[0])
                                        logger.info(f"自动使用最相似的特征: '{most_similar[0]}'")
        
        if not selected_features:
            # 如果没有配置特定特征，则使用所有带前缀的特征
            logger.warning("未找到匹配的特征，将使用所有特征")
            
            # 对于每个时间框架，选择一些最常用的特征
            for tf in timeframes:
                common_features = [
                    f"{tf}_close", f"{tf}_vwap", f"{tf}_rsi_14", f"{tf}_macd",
                    f"{tf}_ema_50", f"{tf}_sma_50"
                ]
                
                # 检查这些常用特征是否存在
                for feature in common_features:
                    exact_match = feature in df.columns
                    case_insensitive_match = feature.lower() in columns_lower
                    
                    if exact_match:
                        selected_features.append(feature)
                    elif case_insensitive_match:
                        selected_features.append(columns_lower[feature.lower()])
            
            # 如果仍然没有特征，则回退到使用前5个特征
            if not selected_features:
                selected_features = df.columns.tolist()
                # 限制特征数量以避免命令行过长
                if len(selected_features) > 20:
                    logger.warning(f"特征数量过多 ({len(selected_features)})，将只使用前20个")
                    selected_features = selected_features[:20]
            else:
                logger.info(f"已选择 {len(selected_features)} 个常用特征")
        
        logger.info(f"最终选定的特征: {selected_features}")
        
        # 构建模型训练命令
        cmd = [
            "python -m src.model_training_main",
            f"--feature_file={feature_file}",
            f"--target_file={target_file}",
            f"--target_column={args.target_column}",
            f"--symbol={args.symbol}",
            f"--timeframe={timeframes[0]}",
            "--feature_type=cross_timeframe",
            "--model_type=xgboost",
            "--target_type=price_change_pct",
            f"--horizon={args.horizon}",
            "--time_series_split"
        ]
        
        # 将特征列表添加到命令中（如有必要）
        if selected_features:
            feature_arg = ",".join(selected_features)
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