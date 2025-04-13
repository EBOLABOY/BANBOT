"""
数据预处理模块
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.config import load_config, get_config_value

logger = get_logger(__name__)

class DataProcessor:
    """
    数据预处理器
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        初始化数据预处理器
        
        参数:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.data_config = self.config.get("data", {})
        
        # 数据路径
        self.raw_data_path = "data/raw"
        self.processed_data_path = "data/processed"
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        logger.info("数据预处理器已初始化")
    
    def list_raw_data_files(self, pattern=None):
        """
        列出原始数据文件
        
        参数:
            pattern: 文件名匹配模式
            
        返回:
            文件路径列表
        """
        files = []
        
        for filename in os.listdir(self.raw_data_path):
            if pattern is None or pattern in filename:
                filepath = os.path.join(self.raw_data_path, filename)
                if os.path.isfile(filepath):
                    files.append(filepath)
        
        logger.info(f"找到 {len(files)} 个原始数据文件")
        return files
    
    def load_data(self, filepath):
        """
        加载数据文件
        
        参数:
            filepath: 文件路径
            
        返回:
            DataFrame对象
        """
        try:
            df = pd.read_csv(filepath)
            
            # 检查是否包含timestamp列
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            
            logger.debug(f"已加载数据文件: {filepath}，共 {len(df)} 条记录")
            return df
        
        except Exception as e:
            logger.error(f"加载数据文件时出错: {e}")
            return None
    
    def check_data_quality(self, df):
        """
        检查数据质量
        
        参数:
            df: DataFrame对象
            
        返回:
            包含数据质量问题的字典
        """
        issues = {}
        
        if df is None or df.empty:
            issues["empty"] = "数据为空"
            return issues
        
        # 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            issues["missing_values"] = missing_values[missing_values > 0].to_dict()
        
        # 检查重复行
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            issues["duplicate_rows"] = duplicate_rows
        
        # 检查时间序列的连续性
        if isinstance(df.index, pd.DatetimeIndex):
            # 检查索引是否排序
            if not df.index.is_monotonic_increasing:
                issues["unsorted_index"] = "时间戳未排序"
            
            # 检查时间间隔
            time_diffs = df.index.to_series().diff().dropna()
            if not time_diffs.empty:
                # 获取最常见的时间间隔
                common_diff = time_diffs.mode()[0]
                irregular_intervals = (time_diffs != common_diff).sum()
                
                if irregular_intervals > 0:
                    issues["irregular_intervals"] = {
                        "count": irregular_intervals,
                        "percent": irregular_intervals / len(time_diffs) * 100
                    }
        
        # 检查异常值
        for column in df.select_dtypes(include=[np.number]).columns:
            # 使用IQR方法检测异常值
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            
            if outliers > 0:
                if "outliers" not in issues:
                    issues["outliers"] = {}
                
                issues["outliers"][column] = {
                    "count": outliers,
                    "percent": outliers / len(df) * 100
                }
        
        if not issues:
            logger.info("数据质量检查通过，未发现问题")
        else:
            logger.warning(f"数据质量检查发现问题: {issues}")
        
        return issues
    
    def clean_data(self, df, symbol=None, timeframe=None):
        """
        清洗数据
        
        参数:
            df: DataFrame对象
            symbol: 交易对符号
            timeframe: 时间框架
            
        返回:
            清洗后的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法清洗空数据")
            return df
        
        # 复制数据，避免修改原始数据
        cleaned_df = df.copy()
        
        # 排序索引
        if isinstance(cleaned_df.index, pd.DatetimeIndex):
            cleaned_df.sort_index(inplace=True)
            
            # 去除重复的时间戳
            cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]
        
        # 处理缺失值
        has_missing = cleaned_df.isnull().any().any()
        if has_missing:
            # 对于OHLCV数据，使用前向填充
            if all(col in cleaned_df.columns for col in ['open', 'high', 'low', 'close']):
                # 价格列使用前向填充
                price_cols = ['open', 'high', 'low', 'close']
                cleaned_df[price_cols] = cleaned_df[price_cols].fillna(method='ffill')
                
                # 交易量可能为0
                if 'volume' in cleaned_df.columns:
                    cleaned_df['volume'] = cleaned_df['volume'].fillna(0)
            else:
                # 其他数据使用前向填充
                cleaned_df = cleaned_df.fillna(method='ffill')
        
        # 处理异常值
        # 对于金融数据，我们通常不直接替换异常值，而是标记它们
        
        logger.info(f"数据清洗完成: 原始记录数 {len(df)}，清洗后记录数 {len(cleaned_df)}")
        return cleaned_df
    
    def resample_data(self, df, rule, agg_dict=None):
        """
        重采样数据到不同的时间间隔
        
        参数:
            df: DataFrame对象
            rule: 重采样规则（例如'1H', '4H', '1D'）
            agg_dict: 聚合字典，指定每列的聚合方法
            
        返回:
            重采样后的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法对空数据进行重采样")
            return df
        
        # 确保索引是日期时间类型
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("数据索引不是日期时间类型，无法进行重采样")
            return df
        
        # 默认的OHLCV聚合方法
        if agg_dict is None:
            # 检查是否为OHLCV数据
            if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                agg_dict = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
            else:
                # 对于其他数据，使用默认的聚合方法
                agg_dict = {}
                for col in df.columns:
                    if df[col].dtype == np.float64 or df[col].dtype == np.int64:
                        agg_dict[col] = 'mean'
                    else:
                        agg_dict[col] = 'last'
        
        # 执行重采样
        try:
            resampled_df = df.resample(rule).agg(agg_dict)
            
            logger.info(f"数据重采样完成: 原始记录数 {len(df)}，重采样后记录数 {len(resampled_df)}")
            return resampled_df
        
        except Exception as e:
            logger.error(f"数据重采样时出错: {e}")
            return df
    
    def process_files(self, file_list=None, clean=True, resample=False, resample_rule='1H'):
        """
        批量处理多个数据文件
        
        参数:
            file_list: 文件路径列表
            clean: 是否清洗数据
            resample: 是否重采样数据
            resample_rule: 重采样规则
            
        返回:
            成功处理的文件数量
        """
        if file_list is None:
            file_list = self.list_raw_data_files()
        
        processed_count = 0
        
        for filepath in tqdm(file_list):
            try:
                # 从文件名中提取信息
                filename = os.path.basename(filepath)
                parts = filename.split('_')
                
                symbol = parts[0] if len(parts) > 0 else None
                timeframe = parts[1] if len(parts) > 1 else None
                
                # 加载数据
                df = self.load_data(filepath)
                if df is None or df.empty:
                    continue
                
                # 检查数据质量
                issues = self.check_data_quality(df)
                
                # 清洗数据
                if clean:
                    df = self.clean_data(df, symbol, timeframe)
                
                # 重采样数据
                if resample and isinstance(df.index, pd.DatetimeIndex):
                    df = self.resample_data(df, resample_rule)
                
                # 保存处理后的数据
                processed_filename = f"processed_{filename}"
                processed_filepath = os.path.join(self.processed_data_path, processed_filename)
                
                df.to_csv(processed_filepath)
                logger.info(f"已保存处理后的数据到 {processed_filepath}")
                
                processed_count += 1
            
            except Exception as e:
                logger.error(f"处理文件 {filepath} 时出错: {e}")
        
        logger.info(f"批量处理完成，成功处理 {processed_count}/{len(file_list)} 个文件")
        return processed_count 