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
    加载特征和目标数据 (优先 Parquet, 兼容 CSV)
    
    参数:
        feature_path: 特征数据文件路径
        target_path: 目标数据文件路径
        target_column: 目标列名，如果为None则尝试自动检测
        
    返回:
        Tuple[pd.DataFrame, pd.Series]: 特征DataFrame和目标Series
    """
    # 加载特征数据
    try:
        # 尝试读取 Parquet 文件
        features_df = pd.read_parquet(feature_path)
        logger.info(f"已从 Parquet 文件加载特征: {feature_path}")
    except Exception as e_parquet:
        logger.warning(f"无法将特征文件作为 Parquet 读取 ({e_parquet})，尝试作为 CSV 读取...")
        try:
            # 如果 Parquet 失败，尝试 CSV 作为后备
            features_df = pd.read_csv(feature_path, index_col=0, parse_dates=True)
            logger.info(f"已从 CSV 文件加载特征: {feature_path}")
        except Exception as e_csv:
            logger.error(f"无法将特征文件作为 Parquet 或 CSV 读取: {e_csv}")
            raise FileNotFoundError(f"无法加载特征文件: {feature_path}") from e_csv
    
    # 加载目标数据
    try:
        # 尝试读取 Parquet 文件
        targets_df = pd.read_parquet(target_path)
        logger.info(f"已从 Parquet 文件加载目标: {target_path}")
    except Exception as e_parquet:
        logger.warning(f"无法将目标文件作为 Parquet 读取 ({e_parquet})，尝试作为 CSV 读取...")
        try:
            # 如果 Parquet 失败，尝试 CSV 作为后备
            targets_df = pd.read_csv(target_path, index_col=0, parse_dates=True)
            logger.info(f"已从 CSV 文件加载目标: {target_path}")
        except Exception as e_csv:
            logger.error(f"无法将目标文件作为 Parquet 或 CSV 读取: {e_csv}")
            raise FileNotFoundError(f"无法加载目标文件: {target_path}") from e_csv
    
    # 确保索引是 DatetimeIndex
    if not isinstance(features_df.index, pd.DatetimeIndex):
        try:
            features_df.index = pd.to_datetime(features_df.index)
        except Exception as e:
            logger.warning(f"无法自动将特征索引转换为 DatetimeIndex: {e}")
            # Depending on requirements, you might want to raise an error here
    if not isinstance(targets_df.index, pd.DatetimeIndex):
        try:
            targets_df.index = pd.to_datetime(targets_df.index)
        except Exception as e:
            logger.warning(f"无法自动将目标索引转换为 DatetimeIndex: {e}")
            # Depending on requirements, you might want to raise an error here
    
    # 检查是否是跨周期特征（通过检查列名前缀）
    is_cross_timeframe = any('_' in col and col.split('_')[0] in ['1h', '4h', '1d', '1m', '5m', '15m'] for col in features_df.columns)
    if is_cross_timeframe:
        logger.info("检测到跨周期特征数据")
    
    # 确保索引对齐 - 处理时区和格式问题
    features_index = features_df.index
    targets_index = targets_df.index
    
    # 移除可能的时区信息以便更一致地比较
    if features_index.tz is not None:
        features_index = features_index.tz_localize(None)
        features_df.index = features_index
    
    if targets_index.tz is not None:
        targets_index = targets_index.tz_localize(None)
        targets_df.index = targets_index
    
    # 找出公共索引
    common_index = features_index.intersection(targets_index)
    if len(common_index) == 0:
        # 尝试将目标索引转换为特征索引的格式
        if len(features_index) > 0 and len(targets_index) > 0:
            logger.warning("找不到共同索引，尝试统一索引格式")
            # 获取索引格式示例
            feature_idx_example = str(features_index[0])
            target_idx_example = str(targets_index[0])
            logger.debug(f"特征索引格式: {feature_idx_example}")
            logger.debug(f"目标索引格式: {target_idx_example}")
            
            # 尝试转换
            try:
                # 尝试不同的日期格式
                formats = [
                    '%Y-%m-%d %H:%M:%S', 
                    '%Y-%m-%d %H:%M:%S%z', 
                    '%Y-%m-%d',
                    '%Y/%m/%d %H:%M:%S',
                    '%Y%m%d %H:%M:%S'
                ]
                
                for date_format in formats:
                    try:
                        # 如果特征和目标索引的格式不同，尝试统一它们
                        targets_df.index = pd.to_datetime(targets_df.index.astype(str), format=date_format)
                        common_index = features_index.intersection(targets_df.index)
                        if len(common_index) > 0:
                            logger.info(f"成功使用格式 {date_format} 统一索引")
                            break
                    except:
                        continue
            except Exception as e:
                logger.error(f"转换索引格式时出错: {str(e)}")
        
        if len(common_index) == 0:
            raise ValueError("特征和目标数据没有共同的索引，且无法自动调整")
    
    # 如果特征和目标时间不完全一致，可能需要调整
    if len(features_df) != len(targets_df) or not features_df.index.equals(targets_df.index):
        logger.warning(f"特征和目标数据长度或索引不一致: 特征={len(features_df)}, 目标={len(targets_df)}")
        logger.warning(f"使用它们的交集: {len(common_index)} 条记录")
    
    # 提取共同索引对应的数据
    features_df = features_df.loc[common_index]
    targets_df = targets_df.loc[common_index]
    
    # 提取目标列
    target_series = targets_df[target_column]
    
    # 检查并处理NaN值
    nan_count_features = features_df.isna().sum().sum()
    nan_count_target = target_series.isna().sum()
    
    if nan_count_features > 0:
        logger.warning(f"特征数据中有 {nan_count_features} 个缺失值")
        # 记录每列的缺失值数量
        for col in features_df.columns:
            col_nan = features_df[col].isna().sum()
            if col_nan > 0:
                logger.debug(f"  - 列 '{col}' 有 {col_nan} 个缺失值")
    
    if nan_count_target > 0:
        logger.warning(f"目标数据中有 {nan_count_target} 个缺失值")
    
    # 去除包含NaN的行
    if nan_count_features > 0 or nan_count_target > 0:
        logger.info("处理数据中的NaN值")
        
        # 处理特征中的NaN
        if nan_count_features > 0:
            logger.info("使用前向填充和后向填充处理特征中的NaN值")
            # 先用前向填充，再用后向填充
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            # 检查是否还有NaN
            remaining_nans = features_df.isna().sum().sum()
            if remaining_nans > 0:
                logger.warning(f"填充后特征中仍有 {remaining_nans} 个NaN值，将用0填充")
                features_df = features_df.fillna(0)
        
        # 处理目标中的NaN
        if nan_count_target > 0:
            # 检查目标是否全为NaN
            if nan_count_target == len(target_series):
                logger.error(f"目标列 '{target_column}' 全是NaN值，无法训练模型")
                
                # 尝试查找其他可用目标列
                other_targets = [col for col in targets_df.columns if col != target_column and col.startswith('target_')]
                if other_targets:
                    # 检查其他目标列的可用性
                    for other_col in other_targets:
                        other_nan_count = targets_df[other_col].isna().sum()
                        valid_ratio = 1 - (other_nan_count / len(targets_df))
                        if valid_ratio > 0.3:  # 如果至少30%的值有效
                            logger.warning(f"建议使用其他目标列 '{other_col}'，其有效值比例为 {valid_ratio:.2%}")
                
                # 这里有两种选择：
                # 1. 直接抛出异常中断流程
                # 2. 返回空数据框让调用方处理
                # 我们选择返回空数据框，以便调用方可以做出决策
                return pd.DataFrame(columns=features_df.columns), pd.Series(dtype=float)
            
            # 如果目标列的NaN值过多但不是全部
            elif nan_count_target > len(target_series) * 0.7:  # 如果超过70%是NaN
                logger.warning(f"目标列 '{target_column}' 中超过70%的值为NaN，可能影响模型质量")
            
            # 处理目标中的NaN：对于时间序列，前向填充通常是合理的
            logger.info("使用前向填充和后向填充处理目标中的NaN值")
            target_series = target_series.fillna(method='ffill').fillna(method='bfill')
            
            # 检查是否还有NaN
            remaining_nans = target_series.isna().sum()
            if remaining_nans > 0:
                # 如果还有NaN，使用均值填充
                mean_val = target_series[~target_series.isna()].mean()
                if pd.isna(mean_val):  # 如果均值也是NaN
                    logger.warning("无法计算目标均值，使用0填充目标中的NaN值")
                    target_series = target_series.fillna(0)
                else:
                    logger.info(f"使用均值 {mean_val:.4f} 填充目标中的NaN值")
                    target_series = target_series.fillna(mean_val)
        
        # 更新填充后的统计信息
        logger.info(f"NaN处理后的数据: 特征={features_df.shape}, 目标={target_series.shape}")
    
    logger.info(f"已加载数据: 特征={features_df.shape}, 目标={target_series.shape}")
    logger.info(f"时间范围: {features_df.index.min()} 到 {features_df.index.max()}")
    
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
    
    # 数据清洗：处理特征和目标数据中的无效值
    X_train_clean = X_train.copy()
    y_train_clean = y_train.copy()
    
    # 检查目标数据量
    if len(y_train_clean) == 0:
        raise ValueError("目标数据为空，无法训练模型")
    
    # 检查并处理目标数据中的无效值
    nan_mask = y_train_clean.isna()
    inf_mask = ~np.isfinite(y_train_clean)
    invalid_mask = nan_mask | inf_mask
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        logger.warning(f"目标数据中发现 {invalid_count} 个无效值 (NaN/无穷大)，占总数的 {invalid_count/len(y_train_clean)*100:.2f}%")
        
        # 如果无效值超过一定比例，发出警告
        invalid_ratio = invalid_count / len(y_train_clean)
        if invalid_ratio > 0.5:
            logger.warning(f"警告：目标数据中超过50%的值为无效值，可能会影响模型质量")
        
        # 计算有效值的均值和中位数，用于替换无效值
        valid_values = y_train_clean[~invalid_mask]
        if len(valid_values) > 0:
            # 如果还有有效值，使用有效值的均值替换无效值
            replacement_value = valid_values.mean()
            logger.info(f"使用有效值的均值 {replacement_value:.6f} 替换目标中的无效值")
            y_train_clean = y_train_clean.fillna(replacement_value)
            # 对于无穷大值，也用均值替换
            y_train_clean[inf_mask] = replacement_value
        else:
            # 如果没有有效值，使用0替换
            logger.warning("目标数据中没有有效值，使用0替换所有无效值")
            y_train_clean = y_train_clean.fillna(0)
            y_train_clean[inf_mask] = 0
    
    # 检查特征数据中的无效值
    feature_nan_count = X_train_clean.isna().sum().sum()
    if feature_nan_count > 0:
        logger.warning(f"特征数据中发现 {feature_nan_count} 个 NaN 值，将使用前向填充和后向填充法处理")
        # 先用前向填充，再用后向填充
        X_train_clean = X_train_clean.fillna(method='ffill').fillna(method='bfill')
        
        # 如果还有NaN（列全为NaN的情况），用0填充
        if X_train_clean.isna().sum().sum() > 0:
            logger.warning("特征数据中仍有NaN值，用0填充")
            X_train_clean = X_train_clean.fillna(0)
    
    # 检查无穷大值
    inf_feature_count = (~np.isfinite(X_train_clean)).sum().sum()
    if inf_feature_count > 0:
        logger.warning(f"特征数据中发现 {inf_feature_count} 个无穷大值，将替换为0")
        # 将无穷大替换为0
        X_train_clean = X_train_clean.replace([np.inf, -np.inf], 0)
    
    # 处理验证集（如果有）
    validation_data = None
    if X_val is not None and y_val is not None:
        X_val_clean = X_val.copy()
        y_val_clean = y_val.copy()
        
        # 检查验证目标数据中的无效值
        val_nan_mask = y_val_clean.isna()
        val_inf_mask = ~np.isfinite(y_val_clean)
        val_invalid_mask = val_nan_mask | val_inf_mask
        val_invalid_count = val_invalid_mask.sum()
        
        if val_invalid_count > 0:
            logger.warning(f"验证目标数据中发现 {val_invalid_count} 个无效值，将替换为训练集均值")
            
            # 使用训练集中有效值的均值替换
            if 'replacement_value' in locals():
                y_val_clean = y_val_clean.fillna(replacement_value)
                y_val_clean[val_inf_mask] = replacement_value
            else:
                # 如果之前没有计算替换值，使用训练集均值
                replacement_value = y_train_clean.mean()
                y_val_clean = y_val_clean.fillna(replacement_value)
                y_val_clean[val_inf_mask] = replacement_value
        
        # 处理验证特征中的缺失值（使用与训练集相同的方法）
        if X_val_clean.isna().sum().sum() > 0:
            X_val_clean = X_val_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 处理验证特征中的无穷大值
        if (~np.isfinite(X_val_clean)).sum().sum() > 0:
            X_val_clean = X_val_clean.replace([np.inf, -np.inf], 0)
        
        validation_data = (X_val_clean, y_val_clean)
    
    # 最终检查确保数据有效
    if len(X_train_clean) == 0 or len(y_train_clean) == 0:
        raise ValueError("数据清洗后训练集为空，请检查数据质量或调整清洗参数")
    
    # 输出清洗后的数据统计
    logger.info(f"数据清洗完成: 训练特征={X_train_clean.shape}, 训练目标={y_train_clean.shape}")
    if validation_data:
        logger.info(f"验证特征={validation_data[0].shape}, 验证目标={validation_data[1].shape}")
    
    # 训练模型
    train_metrics = model.train(X_train_clean, y_train_clean, validation_data)
    
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