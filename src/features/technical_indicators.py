"""
技术指标计算模块
"""

import pandas as pd
import numpy as np
import talib
from talib import abstract
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)

class TechnicalIndicators:
    """
    技术指标计算类，提供常用技术分析指标的计算
    """
    
    @staticmethod
    def calculate_indicators(df, indicators=None, window_sizes=None):
        """
        计算技术指标
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            indicators: 要计算的指标列表
            window_sizes: 窗口大小列表或字典
            
        返回:
            包含技术指标的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的技术指标")
            return df
        
        # 确保数据包含必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"数据缺少必要的列: {missing_columns}")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 默认指标列表
        if indicators is None:
            indicators = [
                'SMA', 'EMA', 'WMA', 'MACD', 'RSI', 'STOCH', 'BBANDS', 'ATR',
                'ADX', 'CCI', 'ROC', 'OBV', 'MFI', 'TRIX', 'CMO'
            ]
        
        # 默认窗口大小
        if window_sizes is None:
            window_sizes = {
                'short': [5, 10, 20],
                'medium': [50, 100, 200],
                'long': [500]
            }
        
        # 将窗口大小转换为列表
        if isinstance(window_sizes, dict):
            all_windows = []
            for size_list in window_sizes.values():
                all_windows.extend(size_list)
            window_sizes = sorted(all_windows)
        
        # 计算移动平均线指标
        if 'SMA' in indicators:
            for window in window_sizes:
                result_df[f'SMA_{window}'] = talib.SMA(result_df['close'], timeperiod=window)
        
        # 计算指数移动平均线
        if 'EMA' in indicators:
            for window in window_sizes:
                result_df[f'EMA_{window}'] = talib.EMA(result_df['close'], timeperiod=window)
        
        # 计算加权移动平均线
        if 'WMA' in indicators:
            for window in window_sizes:
                result_df[f'WMA_{window}'] = talib.WMA(result_df['close'], timeperiod=window)
        
        # 计算MACD
        if 'MACD' in indicators:
            # 默认参数：快线=12，慢线=26，信号线=9
            macd, macd_signal, macd_hist = talib.MACD(
                result_df['close'],
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            result_df['MACD'] = macd
            result_df['MACD_signal'] = macd_signal
            result_df['MACD_hist'] = macd_hist
        
        # 计算RSI
        if 'RSI' in indicators:
            for window in [6, 14, 20]:
                result_df[f'RSI_{window}'] = talib.RSI(result_df['close'], timeperiod=window)
        
        # 计算随机振荡器
        if 'STOCH' in indicators:
            # 默认参数：K线=5，D线=3
            slowk, slowd = talib.STOCH(
                result_df['high'],
                result_df['low'],
                result_df['close'],
                fastk_period=5,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            result_df['STOCH_K'] = slowk
            result_df['STOCH_D'] = slowd
        
        # 计算布林带
        if 'BBANDS' in indicators:
            for window in [20, 50]:
                upper, middle, lower = talib.BBANDS(
                    result_df['close'],
                    timeperiod=window,
                    nbdevup=2,
                    nbdevdn=2,
                    matype=0
                )
                result_df[f'BBANDS_upper_{window}'] = upper
                result_df[f'BBANDS_middle_{window}'] = middle
                result_df[f'BBANDS_lower_{window}'] = lower
                # 计算带宽
                result_df[f'BBANDS_width_{window}'] = (upper - lower) / middle
        
        # 计算平均真实范围
        if 'ATR' in indicators:
            for window in [14, 20]:
                result_df[f'ATR_{window}'] = talib.ATR(
                    result_df['high'],
                    result_df['low'],
                    result_df['close'],
                    timeperiod=window
                )
        
        # 计算平均趋向指标
        if 'ADX' in indicators:
            for window in [14, 20]:
                result_df[f'ADX_{window}'] = talib.ADX(
                    result_df['high'],
                    result_df['low'],
                    result_df['close'],
                    timeperiod=window
                )
        
        # 计算顺势指标
        if 'CCI' in indicators:
            for window in [14, 20]:
                result_df[f'CCI_{window}'] = talib.CCI(
                    result_df['high'],
                    result_df['low'],
                    result_df['close'],
                    timeperiod=window
                )
        
        # 计算变动率
        if 'ROC' in indicators:
            for window in [10, 20]:
                result_df[f'ROC_{window}'] = talib.ROC(result_df['close'], timeperiod=window)
        
        # 计算能量潮指标
        if 'OBV' in indicators:
            result_df['OBV'] = talib.OBV(result_df['close'], result_df['volume'])
        
        # 计算资金流量指标
        if 'MFI' in indicators:
            for window in [14, 20]:
                result_df[f'MFI_{window}'] = talib.MFI(
                    result_df['high'],
                    result_df['low'],
                    result_df['close'],
                    result_df['volume'],
                    timeperiod=window
                )
        
        # 计算TRIX指标
        if 'TRIX' in indicators:
            for window in [15, 30]:
                result_df[f'TRIX_{window}'] = talib.TRIX(result_df['close'], timeperiod=window)
        
        # 计算钱德动量摆动指标
        if 'CMO' in indicators:
            for window in [14, 20]:
                result_df[f'CMO_{window}'] = talib.CMO(result_df['close'], timeperiod=window)
        
        logger.info(f"已计算 {len(indicators)} 种技术指标")
        return result_df
    
    @staticmethod
    def calculate_price_features(df):
        """
        计算基于价格的特征
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含价格特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的价格特征")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 计算价格变化
        result_df['price_change'] = result_df['close'].diff()
        
        # 计算价格变化百分比
        result_df['price_change_pct'] = result_df['close'].pct_change() * 100
        
        # 计算价格变化方向（1=上涨，0=持平，-1=下跌）
        result_df['price_direction'] = np.sign(result_df['price_change'])
        
        # 计算交易范围
        result_df['range'] = result_df['high'] - result_df['low']
        
        # 计算相对收盘位置
        # (close - low) / (high - low)，接近1表示收盘价接近最高价
        result_df['close_position'] = (result_df['close'] - result_df['low']) / result_df['range']
        
        # 计算价格与移动平均线的关系
        for window in [10, 20, 50, 200]:
            result_df[f'MA_{window}'] = talib.SMA(result_df['close'], timeperiod=window)
            # 价格相对于移动平均线的百分比距离
            result_df[f'close_to_MA_{window}_pct'] = ((result_df['close'] / result_df[f'MA_{window}']) - 1) * 100
        
        logger.info("已计算基于价格的特征")
        return result_df
    
    @staticmethod
    def calculate_volume_features(df):
        """
        计算基于交易量的特征
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含交易量特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的交易量特征")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 计算交易量变化
        result_df['volume_change'] = result_df['volume'].diff()
        
        # 计算交易量变化百分比
        result_df['volume_change_pct'] = result_df['volume'].pct_change() * 100
        
        # 计算交易量移动平均线
        for window in [5, 10, 20, 50]:
            result_df[f'volume_MA_{window}'] = talib.SMA(result_df['volume'], timeperiod=window)
            # 交易量相对于移动平均线的比率
            result_df[f'volume_ratio_{window}'] = result_df['volume'] / result_df[f'volume_MA_{window}']
        
        # 计算上涨和下跌交易量
        result_df['up_volume'] = result_df['volume'] * (result_df['close'] >= result_df['open']).astype(int)
        result_df['down_volume'] = result_df['volume'] * (result_df['close'] < result_df['open']).astype(int)
        
        # 计算买入/卖出压力比率
        result_df['volume_pressure_ratio'] = result_df['up_volume'].rolling(10).sum() / \
                                             result_df['down_volume'].rolling(10).sum()
        
        # 填充无穷大值
        result_df['volume_pressure_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        result_df['volume_pressure_ratio'].fillna(1, inplace=True)
        
        logger.info("已计算基于交易量的特征")
        return result_df
    
    @staticmethod
    def calculate_volatility_features(df):
        """
        计算波动性特征
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含波动性特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的波动性特征")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 计算对数收益率
        result_df['log_return'] = np.log(result_df['close'] / result_df['close'].shift(1))
        
        # 计算历史波动率
        for window in [5, 10, 20, 50]:
            # 标准差波动率 (标准差 * sqrt(交易日数))
            result_df[f'volatility_{window}'] = result_df['log_return'].rolling(window).std() * np.sqrt(252)
        
        # 计算Parkinson波动率
        for window in [5, 10, 20]:
            high_low_ratio = np.log(result_df['high'] / result_df['low'])
            result_df[f'parkinson_volatility_{window}'] = np.sqrt(
                high_low_ratio.pow(2).rolling(window).mean() / (4 * np.log(2)) * 252
            )
        
        # 计算振幅
        result_df['amplitude'] = (result_df['high'] - result_df['low']) / result_df['close'].shift(1) * 100
        
        # 计算真实波动范围
        result_df['true_range'] = np.maximum(
            result_df['high'] - result_df['low'],
            np.maximum(
                np.abs(result_df['high'] - result_df['close'].shift(1)),
                np.abs(result_df['low'] - result_df['close'].shift(1))
            )
        )
        
        # 计算平均真实波动范围 (ATR)
        for window in [5, 10, 14, 20]:
            result_df[f'ATR_{window}'] = talib.ATR(
                result_df['high'],
                result_df['low'],
                result_df['close'],
                timeperiod=window
            )
            # ATR百分比
            result_df[f'ATR_pct_{window}'] = result_df[f'ATR_{window}'] / result_df['close'] * 100
        
        logger.info("已计算波动性特征")
        return result_df
    
    @staticmethod
    def calculate_trend_features(df):
        """
        计算趋势特征
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含趋势特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的趋势特征")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 计算ADX指标用于判断趋势强度
        for window in [14, 20]:
            result_df[f'ADX_{window}'] = talib.ADX(
                result_df['high'],
                result_df['low'],
                result_df['close'],
                timeperiod=window
            )
        
        # 计算移动平均线斜率
        for window in [10, 20, 50, 200]:
            # 计算移动平均线
            result_df[f'MA_{window}'] = talib.SMA(result_df['close'], timeperiod=window)
            # 计算移动平均线的斜率（变化率）
            result_df[f'MA_{window}_slope'] = result_df[f'MA_{window}'].diff(5) / 5
            # 计算移动平均线的斜率方向
            result_df[f'MA_{window}_slope_direction'] = np.sign(result_df[f'MA_{window}_slope'])
        
        # 计算价格相对长期均线的位置
        result_df['price_to_200MA_ratio'] = result_df['close'] / result_df['MA_200']
        
        # 计算短期与长期移动平均线关系
        result_df['MA_ratio_50_200'] = result_df['MA_50'] / result_df['MA_200']
        result_df['MA_ratio_20_50'] = result_df['MA_20'] / result_df['MA_50']
        
        # 计算移动平均线排列状态
        result_df['MA_alignment'] = np.where(
            (result_df['MA_20'] > result_df['MA_50']) & (result_df['MA_50'] > result_df['MA_200']),
            1,  # 多头排列
            np.where(
                (result_df['MA_20'] < result_df['MA_50']) & (result_df['MA_50'] < result_df['MA_200']),
                -1,  # 空头排列
                0   # 交叉排列
            )
        )
        
        # 计算长期趋势方向（基于50周期）
        up_days = (result_df['close'] > result_df['close'].shift(1)).rolling(50).sum()
        down_days = (result_df['close'] < result_df['close'].shift(1)).rolling(50).sum()
        result_df['trend_strength_50'] = (up_days - down_days) / 50
        
        # 计算趋势一致性（收盘价偏离均线的频率）
        for window in [20, 50]:
            # 计算价格高于均线的天数比例
            result_df[f'above_MA_{window}_ratio'] = (result_df['close'] > result_df[f'MA_{window}']).rolling(window).mean()
        
        logger.info("已计算趋势特征")
        return result_df
    
    @staticmethod
    def calculate_momentum_features(df):
        """
        计算动量特征
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含动量特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的动量特征")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 计算不同时间窗口的价格变化率
        for window in [1, 3, 5, 10, 20, 60]:
            result_df[f'momentum_{window}'] = result_df['close'].pct_change(window) * 100
        
        # 计算RSI指标
        for window in [6, 14, 20]:
            result_df[f'RSI_{window}'] = talib.RSI(result_df['close'], timeperiod=window)
        
        # 计算随机振荡器
        slowk, slowd = talib.STOCH(
            result_df['high'],
            result_df['low'],
            result_df['close'],
            fastk_period=5,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        result_df['STOCH_K'] = slowk
        result_df['STOCH_D'] = slowd
        
        # 计算MACD
        macd, signal, hist = talib.MACD(
            result_df['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        result_df['MACD'] = macd
        result_df['MACD_signal'] = signal
        result_df['MACD_hist'] = hist
        
        # 计算MACD柱状图动量
        result_df['MACD_hist_momentum'] = result_df['MACD_hist'].diff()
        
        # 计算顺势指标
        for window in [14, 20]:
            result_df[f'CCI_{window}'] = talib.CCI(
                result_df['high'],
                result_df['low'],
                result_df['close'],
                timeperiod=window
            )
        
        # 计算Williams %R
        for window in [14, 20]:
            result_df[f'WILLR_{window}'] = talib.WILLR(
                result_df['high'],
                result_df['low'],
                result_df['close'],
                timeperiod=window
            )
        
        # 计算变动率
        for window in [10, 20]:
            result_df[f'ROC_{window}'] = talib.ROC(result_df['close'], timeperiod=window)
        
        # 计算钱德动量摆动指标
        for window in [14, 20]:
            result_df[f'CMO_{window}'] = talib.CMO(result_df['close'], timeperiod=window)
        
        logger.info("已计算动量特征")
        return result_df 