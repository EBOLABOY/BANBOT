"""
跨周期特征整合模块 - 用于合并不同时间周期(1h, 4h, 1d)的特征数据，将它们对齐并合并成一个特征集
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)

class CrossTimeframeFeatureIntegrator:
    """
    跨时间周期特征整合器，用于将不同时间框架的特征整合到一起
    """
    
    def __init__(self, base_timeframe: str = "1h"):
        """
        初始化跨时间周期特征整合器
        
        参数:
            base_timeframe: 基础时间框架，所有其他时间框架将对齐到这个时间框架
        """
        self.base_timeframe = base_timeframe
        self.timeframe_order = {"1m": 0, "5m": 1, "15m": 2, "30m": 3, "1h": 4, "4h": 5, "1d": 6, "1w": 7}
    
    def _validate_timeframes(self, timeframes: List[str]) -> bool:
        """
        验证时间框架列表是否有效
        
        参数:
            timeframes: 时间框架列表
            
        返回:
            是否有效
        """
        if not timeframes:
            logger.error("时间框架列表为空")
            return False
            
        # 检查时间框架是否都在支持的列表中
        for tf in timeframes:
            if tf not in self.timeframe_order:
                logger.error(f"不支持的时间框架: {tf}")
                return False
                
        # 检查是否包含基础时间框架
        if self.base_timeframe not in timeframes:
            logger.error(f"时间框架列表中必须包含基础时间框架: {self.base_timeframe}")
            return False
            
        return True
    
    def _convert_timeframe_to_minutes(self, timeframe: str) -> int:
        """
        将时间框架转换为分钟数
        
        参数:
            timeframe: 时间框架字符串，如 "1h", "4h", "1d"
            
        返回:
            对应的分钟数
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == "m":
            return value
        elif unit == "h":
            return value * 60
        elif unit == "d":
            return value * 24 * 60
        elif unit == "w":
            return value * 7 * 24 * 60
        else:
            raise ValueError(f"无法识别的时间单位: {unit}")
    
    def _load_features(self, 
                      features_dir: str, 
                      symbol: str, 
                      timeframe: str) -> pd.DataFrame:
        """
        加载特定交易对和时间框架的特征数据
        
        参数:
            features_dir: 特征目录
            symbol: 交易对名称
            timeframe: 时间框架
            
        返回:
            特征DataFrame
        """
        feature_path = os.path.join(features_dir, symbol, f"features_{timeframe}.csv")
        
        if not os.path.exists(feature_path):
            logger.error(f"特征文件不存在: {feature_path}")
            return None
            
        try:
            df = pd.read_csv(feature_path, index_col=0, parse_dates=True)
            logger.info(f"已加载 {symbol} {timeframe} 特征, 形状: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"加载特征文件时出错: {str(e)}")
            return None
    
    def _resample_to_lower_timeframe(self, 
                                    df: pd.DataFrame, 
                                    source_tf: str, 
                                    target_tf: str) -> pd.DataFrame:
        """
        将高时间框架的数据下采样到低时间框架
        
        参数:
            df: 源数据
            source_tf: 源时间框架
            target_tf: 目标时间框架
            
        返回:
            下采样后的DataFrame
        """
        if self.timeframe_order[source_tf] <= self.timeframe_order[target_tf]:
            logger.warning(f"源时间框架 {source_tf} 不高于目标时间框架 {target_tf}，无需下采样")
            return df
            
        # 获取时间框架的分钟数
        source_minutes = self._convert_timeframe_to_minutes(source_tf)
        target_minutes = self._convert_timeframe_to_minutes(target_tf)
        
        if source_minutes % target_minutes != 0:
            logger.warning(f"源时间框架 {source_tf} 不是目标时间框架 {target_tf} 的整数倍，下采样可能不准确")
        
        # 创建新的索引
        start_time = df.index.min()
        end_time = df.index.max()
        
        # 为目标时间框架创建新的时间索引
        new_index = pd.date_range(start=start_time, end=end_time, freq=target_tf)
        
        # 创建结果DataFrame
        result = pd.DataFrame(index=new_index)
        
        # 前向填充特征值
        for col in df.columns:
            # 将每个特征重采样到目标时间框架
            resampled = df[col].reindex(result.index, method='ffill')
            result[col] = resampled
            
        return result
    
    def _add_prefix_to_columns(self, 
                              df: pd.DataFrame, 
                              prefix: str) -> pd.DataFrame:
        """
        为DataFrame的列添加前缀，以便区分不同时间框架的特征
        
        参数:
            df: 特征DataFrame
            prefix: 要添加的前缀
            
        返回:
            添加前缀后的DataFrame
        """
        if not df.empty:
            df.columns = [f"{prefix}_{col}" for col in df.columns]
        return df
    
    def integrate_features(self, 
                          features_dir: str, 
                          symbol: str, 
                          timeframes: List[str], 
                          selected_features: Optional[Dict[str, List[str]]] = None,
                          output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        整合不同时间框架的特征
        
        参数:
            features_dir: 特征目录
            symbol: 交易对名称
            timeframes: 要整合的时间框架列表
            selected_features: 每个时间框架要选择的特征，格式为 {timeframe: [feature1, feature2, ...]}
            output_dir: 输出目录，如果指定则将结果保存到该目录
            
        返回:
            整合后的特征DataFrame
        """
        # 验证时间框架
        if not self._validate_timeframes(timeframes):
            return None
            
        # 按照时间框架顺序排序
        sorted_timeframes = sorted(timeframes, key=lambda x: self.timeframe_order[x])
        base_tf_idx = sorted_timeframes.index(self.base_timeframe)
        
        # 加载所有特征
        features_dfs = {}
        for tf in sorted_timeframes:
            df = self._load_features(features_dir, symbol, tf)
            if df is None:
                logger.error(f"无法加载 {symbol} {tf} 特征，跳过该时间框架")
                continue
                
            # 如果指定了选择的特征，则只保留这些特征
            if selected_features and tf in selected_features:
                selected = selected_features[tf]
                available = set(df.columns)
                to_use = list(set(selected) & available)
                
                if len(to_use) < len(selected):
                    missing = set(selected) - available
                    logger.warning(f"{tf} 时间框架缺少以下特征: {missing}")
                
                df = df[to_use]
                
            features_dfs[tf] = df
        
        # 检查是否成功加载了所有时间框架的特征
        if not features_dfs:
            logger.error("未能加载任何时间框架的特征")
            return None
            
        if self.base_timeframe not in features_dfs:
            logger.error(f"未能加载基础时间框架 {self.base_timeframe} 的特征")
            return None
            
        # 获取基础时间框架的数据
        base_df = features_dfs[self.base_timeframe]
        
        # 初始化结果DataFrame
        result_df = base_df.copy()
        
        # 为基础时间框架的特征添加前缀
        result_df = self._add_prefix_to_columns(result_df, self.base_timeframe)
        
        # 处理其他时间框架的特征
        for tf in sorted_timeframes:
            if tf == self.base_timeframe:
                continue
                
            df = features_dfs[tf]
            
            # 如果时间框架高于基础时间框架，需要下采样
            if self.timeframe_order[tf] > self.timeframe_order[self.base_timeframe]:
                df = self._resample_to_lower_timeframe(df, tf, self.base_timeframe)
            
            # 确保索引对齐
            aligned_df = df.reindex(result_df.index, method='ffill')
            
            # 为特征添加前缀
            aligned_df = self._add_prefix_to_columns(aligned_df, tf)
            
            # 合并特征
            result_df = pd.concat([result_df, aligned_df], axis=1)
        
        # 处理缺失值
        result_df = result_df.ffill().bfill()
        
        # 如果指定了输出目录，则保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建文件名，包含所有时间框架
            timeframes_str = "_".join(sorted_timeframes)
            output_file = os.path.join(output_dir, f"{symbol}_multi_tf_{timeframes_str}.csv")
            
            result_df.to_csv(output_file)
            logger.info(f"已将整合后的特征保存到 {output_file}")
        
        return result_df

def create_cross_timeframe_features(config: dict, symbols: List[str] = None, timeframes: List[str] = None):
    """
    为多个交易对创建跨周期特征
    
    参数:
        config: 配置字典
        symbols: 要处理的交易对列表，如果为None则使用配置中的所有交易对
        timeframes: 要整合的时间框架列表，如果为None则使用默认的时间框架列表
    """
    # 获取配置信息
    data_config = config.get("data", {})
    feature_config = config.get("feature_engineering", {})
    
    # 如果未指定交易对，使用配置中的目标交易对
    if not symbols:
        symbols = data_config.get("target_currencies", ["BTCUSDT"])
    
    # 如果未指定时间框架，使用默认的跨周期时间框架
    if not timeframes:
        timeframes = ["1h", "4h", "1d"]
    
    # 获取特征目录
    base_data_dir = data_config.get("data_dir", "data")
    features_dir = os.path.join(base_data_dir, "processed/features")
    output_dir = os.path.join(base_data_dir, "processed/features/cross_timeframe")
    
    # 创建跨周期特征整合器
    integrator = CrossTimeframeFeatureIntegrator(base_timeframe="1h")
    
    # 获取要选择的特征
    selected_features = feature_config.get("cross_timeframe_features", {})
    
    # 为每个交易对创建跨周期特征
    for symbol in symbols:
        logger.info(f"为交易对 {symbol} 创建跨周期特征")
        
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
                logger.info(f"成功为 {symbol} 创建跨周期特征，形状: {integrated_df.shape}")
            else:
                logger.error(f"为 {symbol} 创建跨周期特征失败")
        except Exception as e:
            logger.error(f"为 {symbol} 创建跨周期特征时出错: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())

if __name__ == "__main__":
    # 测试代码
    from src.utils.config import load_config
    
    config = load_config("config.yaml")
    create_cross_timeframe_features(config, symbols=["BTCUSDT"], timeframes=["1h", "4h", "1d"]) 