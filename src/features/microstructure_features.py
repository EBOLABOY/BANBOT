"""
市场微观结构特征计算模块
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)

class MicrostructureFeatures:
    """
    市场微观结构特征计算类，提供高频数据特征的计算
    """
    
    @staticmethod
    def calculate_bid_ask_features(df):
        """
        计算买卖盘特征
        
        参数:
            df: DataFrame对象，包含买卖盘数据
            
        返回:
            包含买卖盘特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的买卖盘特征")
            return df
        
        # 检查必要的列
        required_columns = ['bid', 'ask']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"数据缺少必要的列: {missing_columns}")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 计算买卖价差
        result_df['spread'] = result_df['ask'] - result_df['bid']
        
        # 计算相对价差（占中间价格的百分比）
        result_df['relative_spread'] = result_df['spread'] / ((result_df['bid'] + result_df['ask']) / 2) * 100
        
        # 计算中间价格
        result_df['mid_price'] = (result_df['bid'] + result_df['ask']) / 2
        
        # 计算价格压力指标（bid/ask比率）
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            result_df['bid_ask_volume_ratio'] = result_df['bid_volume'] / result_df['ask_volume']
            result_df['bid_ask_volume_imbalance'] = (result_df['bid_volume'] - result_df['ask_volume']) / (result_df['bid_volume'] + result_df['ask_volume'])
            
        logger.info("已计算买卖盘特征")
        return result_df
    
    @staticmethod
    def calculate_order_flow_features(df, window_sizes=None):
        """
        计算订单流特征
        
        参数:
            df: DataFrame对象，包含高频交易数据
            window_sizes: 窗口大小列表
            
        返回:
            包含订单流特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的订单流特征")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 默认窗口大小
        if window_sizes is None:
            window_sizes = [5, 10, 20, 50]
        
        # 计算订单流指标（如果有交易方向信息）
        if 'trade_direction' in df.columns:
            # 订单流量指标 (OFI)
            result_df['order_flow_indicator'] = result_df['volume'] * result_df['trade_direction']
            
            # 计算累积订单流
            result_df['cumulative_order_flow'] = result_df['order_flow_indicator'].cumsum()
            
            # 计算窗口订单流
            for window in window_sizes:
                result_df[f'order_flow_{window}'] = result_df['order_flow_indicator'].rolling(window).sum()
                
                # 归一化订单流
                result_df[f'normalized_order_flow_{window}'] = result_df[f'order_flow_{window}'] / result_df['volume'].rolling(window).sum()
        
        # 如果没有交易方向，但有价格变化，可以推断方向
        elif 'close' in df.columns:
            # 计算价格变化
            price_changes = df['close'].diff()
            
            # 推断交易方向（1：买入，-1：卖出，0：无变化）
            result_df['inferred_direction'] = np.sign(price_changes)
            
            # 订单流量指标 (使用推断的方向)
            result_df['inferred_order_flow'] = result_df['volume'] * result_df['inferred_direction']
            
            # 计算累积订单流
            result_df['cumulative_inferred_flow'] = result_df['inferred_order_flow'].cumsum()
            
            # 计算窗口订单流
            for window in window_sizes:
                result_df[f'inferred_flow_{window}'] = result_df['inferred_order_flow'].rolling(window).sum()
                
                # 归一化订单流
                result_df[f'norm_inferred_flow_{window}'] = result_df[f'inferred_flow_{window}'] / result_df['volume'].rolling(window).sum()
        
        logger.info("已计算订单流特征")
        return result_df
    
    @staticmethod
    def calculate_liquidity_features(df, window_sizes=None):
        """
        计算流动性特征
        
        参数:
            df: DataFrame对象，包含市场深度数据
            window_sizes: 窗口大小列表
            
        返回:
            包含流动性特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的流动性特征")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 默认窗口大小
        if window_sizes is None:
            window_sizes = [5, 10, 20, 50]
        
        # 检查必要的列
        has_market_depth = all(col in df.columns for col in ['bid', 'ask', 'bid_volume', 'ask_volume'])
        
        # 计算基本流动性指标
        if has_market_depth:
            # Amihud比率 - 衡量价格对交易量的敏感性
            result_df['mid_price'] = (result_df['bid'] + result_df['ask']) / 2
            result_df['mid_price_change'] = result_df['mid_price'].pct_change().abs()
            
            for window in window_sizes:
                # 计算Amihud比率 = 平均(|价格变化|/交易量)
                result_df[f'amihud_ratio_{window}'] = (result_df['mid_price_change'] / result_df['volume']).rolling(window).mean()
                
                # 流动性比率 = 交易量/价差
                result_df[f'liquidity_ratio_{window}'] = (result_df['volume'] / (result_df['ask'] - result_df['bid'])).rolling(window).mean()
                
                # 市场深度 = 买入量 + 卖出量
                result_df[f'market_depth_{window}'] = (result_df['bid_volume'] + result_df['ask_volume']).rolling(window).mean()
                
                # 市场韧性 = 价格变化/订单流不平衡
                if 'bid_ask_volume_imbalance' in result_df.columns:
                    result_df[f'market_resiliency_{window}'] = (result_df['mid_price_change'] / result_df['bid_ask_volume_imbalance'].abs()).rolling(window).mean()
        
        # 如果只有OHLCV数据，计算基于此的流动性指标
        elif 'close' in df.columns and 'volume' in df.columns:
            result_df['price_change'] = result_df['close'].pct_change().abs()
            
            for window in window_sizes:
                # 简化的Amihud比率
                result_df[f'simple_amihud_{window}'] = (result_df['price_change'] / result_df['volume']).rolling(window).mean()
                
                # 流动性比率（使用当日价格范围代替价差）
                if 'high' in df.columns and 'low' in df.columns:
                    result_df[f'range_liquidity_{window}'] = (result_df['volume'] / (result_df['high'] - result_df['low'])).rolling(window).mean()
        
        logger.info("已计算流动性特征")
        return result_df
    
    @staticmethod
    def calculate_volatility_clustering(df, window_sizes=None):
        """
        计算波动聚集特征
        
        参数:
            df: DataFrame对象，包含价格数据
            window_sizes: 窗口大小列表
            
        返回:
            包含波动聚集特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的波动聚集特征")
            return df
        
        # 检查必要的列
        if 'close' not in df.columns:
            logger.warning("数据缺少close列，无法计算波动聚集特征")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 默认窗口大小
        if window_sizes is None:
            window_sizes = [5, 10, 20, 50]
        
        # 计算回报率
        result_df['returns'] = result_df['close'].pct_change()
        
        # 计算绝对回报率
        result_df['abs_returns'] = result_df['returns'].abs()
        
        # 计算波动性聚集指标
        for window in window_sizes:
            # 计算窗口内的波动性自相关性
            # 使用滚动窗口内波动性的自相关性作为代理
            autocorrs = []
            
            for i in range(len(result_df) - window + 1):
                if i < window:  # 对于前window个点
                    autocorrs.append(np.nan)
                    continue
                
                abs_returns = result_df['abs_returns'].iloc[i:i+window]
                if len(abs_returns) <= 1 or abs_returns.isna().all():
                    autocorrs.append(np.nan)
                    continue
                
                # 计算滞后1阶的自相关性
                if len(abs_returns) > 1:
                    autocorr = abs_returns.autocorr(lag=1)
                    autocorrs.append(autocorr)
                else:
                    autocorrs.append(np.nan)
            
            # 填充前window个值
            autocorrs = [np.nan] * (len(result_df) - len(autocorrs)) + autocorrs
            
            # 添加特征
            result_df[f'volatility_clustering_{window}'] = autocorrs
        
        # 计算ARCH效应
        for window in window_sizes:
            # 使用滚动窗口的ARCH效应
            # ARCH效应测试基于Ljung-Box Q统计量
            # 这里使用平方回报率的自相关性作为简化的代理
            result_df[f'arch_effect_{window}'] = result_df['returns'].pow(2).rolling(window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
            )
        
        logger.info("已计算波动聚集特征")
        return result_df
    
    @staticmethod
    def calculate_price_impact(df, window_sizes=None):
        """
        计算价格冲击特征
        
        参数:
            df: DataFrame对象，包含价格和交易量数据
            window_sizes: 窗口大小列表
            
        返回:
            包含价格冲击特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的价格冲击特征")
            return df
        
        # 检查必要的列
        if 'close' not in df.columns or 'volume' not in df.columns:
            logger.warning("数据缺少close或volume列，无法计算价格冲击特征")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 默认窗口大小
        if window_sizes is None:
            window_sizes = [5, 10, 20, 50]
        
        # 计算价格变化
        result_df['price_change'] = result_df['close'].pct_change()
        
        # 计算价格冲击特征
        for window in window_sizes:
            # 计算价格冲击系数：价格变化/交易量
            result_df[f'price_impact_{window}'] = (result_df['price_change'] / result_df['volume']).rolling(window).mean()
            
            # 计算价格冲击函数：在窗口内回归价格变化与交易量的关系
            impact_coeffs = []
            
            for i in range(len(result_df)):
                if i < window:  # 对于前window个点
                    impact_coeffs.append(np.nan)
                    continue
                
                price_changes = result_df['price_change'].iloc[i-window:i]
                volumes = result_df['volume'].iloc[i-window:i]
                
                if len(price_changes) <= 1 or price_changes.isna().all() or volumes.isna().all() or (volumes == 0).all():
                    impact_coeffs.append(np.nan)
                    continue
                
                # 计算线性回归系数
                try:
                    slope, _, _, _, _ = stats.linregress(volumes, price_changes)
                    impact_coeffs.append(slope)
                except:
                    impact_coeffs.append(np.nan)
            
            # 添加特征
            result_df[f'price_impact_coef_{window}'] = impact_coeffs
        
        logger.info("已计算价格冲击特征")
        return result_df 