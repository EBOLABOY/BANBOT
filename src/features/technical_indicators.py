"""
技术指标计算模块 - 使用ta库替代talib
"""

import pandas as pd
import numpy as np
# 替换talib为ta库
import ta
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)

class TechnicalIndicators:
    """
    技术指标计算类，提供常用技术分析指标的计算
    """
    
    @staticmethod
    def tsi(close, r=25, s=13):
        """
        计算趋势强度指数 (TSI)
        
        参数:
            close: 收盘价数据
            r: 长周期
            s: 短周期
            
        返回:
            TSI值
        """
        m = close.diff()
        m1 = m.ewm(span=r, adjust=False).mean()
        m2 = m1.ewm(span=s, adjust=False).mean()
        a = m.abs()
        a1 = a.ewm(span=r, adjust=False).mean()
        a2 = a1.ewm(span=s, adjust=False).mean()
        return 100 * m2 / a2
    
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
                'ADX', 'CCI', 'ROC', 'OBV', 'MFI'
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
                result_df[f'SMA_{window}'] = ta.trend.sma_indicator(result_df['close'], window=window)
        
        # 计算指数移动平均线
        if 'EMA' in indicators:
            for window in window_sizes:
                result_df[f'EMA_{window}'] = ta.trend.ema_indicator(result_df['close'], window=window)
        
        # 计算加权移动平均线 (ta库没有WMA，用EMA代替)
        if 'WMA' in indicators:
            for window in window_sizes:
                result_df[f'WMA_{window}'] = ta.trend.ema_indicator(result_df['close'], window=window)
        
        # 计算MACD
        if 'MACD' in indicators:
            # 默认参数：快线=12，慢线=26，信号线=9
            result_df['MACD'] = ta.trend.macd(result_df['close'], window_slow=26, window_fast=12)
            result_df['MACD_signal'] = ta.trend.macd_signal(result_df['close'], window_slow=26, window_fast=12, window_sign=9)
            result_df['MACD_hist'] = ta.trend.macd_diff(result_df['close'], window_slow=26, window_fast=12, window_sign=9)
        
        # 计算RSI
        if 'RSI' in indicators:
            for window in [6, 14, 20]:
                result_df[f'RSI_{window}'] = ta.momentum.rsi(result_df['close'], window=window)
        
        # 计算随机振荡器
        if 'STOCH' in indicators:
            result_df['STOCH_K'] = ta.momentum.stoch(result_df['high'], result_df['low'], result_df['close'], window=5, smooth_window=3)
            result_df['STOCH_D'] = ta.momentum.stoch_signal(result_df['high'], result_df['low'], result_df['close'], window=5, smooth_window=3)
        
        # 计算布林带
        if 'BBANDS' in indicators:
            for window in [20, 50]:
                result_df[f'BBANDS_upper_{window}'] = ta.volatility.bollinger_hband(result_df['close'], window=window, window_dev=2)
                result_df[f'BBANDS_middle_{window}'] = ta.volatility.bollinger_mavg(result_df['close'], window=window)
                result_df[f'BBANDS_lower_{window}'] = ta.volatility.bollinger_lband(result_df['close'], window=window, window_dev=2)
                # 计算带宽
                result_df[f'BBANDS_width_{window}'] = ta.volatility.bollinger_wband(result_df['close'], window=window, window_dev=2)
        
        # 计算平均真实范围
        if 'ATR' in indicators:
            for window in [14, 20]:
                result_df[f'ATR_{window}'] = ta.volatility.average_true_range(result_df['high'], result_df['low'], result_df['close'], window=window)
        
        # 计算平均趋向指标
        if 'ADX' in indicators:
            for window in [14, 20]:
                result_df[f'ADX_{window}'] = ta.trend.adx(result_df['high'], result_df['low'], result_df['close'], window=window)
        
        # 计算顺势指标
        if 'CCI' in indicators:
            for window in [14, 20]:
                result_df[f'CCI_{window}'] = ta.trend.cci(result_df['high'], result_df['low'], result_df['close'], window=window)
        
        # 计算变动率
        if 'ROC' in indicators:
            for window in [10, 20]:
                result_df[f'ROC_{window}'] = ta.momentum.roc(result_df['close'], window=window)
        
        # 计算能量潮指标
        if 'OBV' in indicators:
            result_df['OBV'] = ta.volume.on_balance_volume(result_df['close'], result_df['volume'])
        
        # 计算资金流量指标
        if 'MFI' in indicators:
            for window in [14, 20]:
                result_df[f'MFI_{window}'] = ta.volume.money_flow_index(result_df['high'], result_df['low'], result_df['close'], result_df['volume'], window=window)
        
        logger.info(f"已计算 {len(indicators)} 种技术指标")
        return result_df
    
    @staticmethod
    def calculate_price_features(df):
        """
        计算价格相关特征
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含价格特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的价格特征")
            return df
        
        result_df = df.copy()
        
        # 计算价格变化百分比
        result_df['price_change_pct'] = result_df['close'].pct_change()
        
        # 计算价格差异指标
        result_df['price_gap'] = (result_df['high'] - result_df['low']) / result_df['low']
        
        # 计算开盘-收盘差异
        result_df['open_close_diff'] = (result_df['close'] - result_df['open']) / result_df['open']
        
        # 计算最高-最低差异
        result_df['high_low_diff'] = (result_df['high'] - result_df['low']) / result_df['low']
        
        # 计算当前价格相对近期高低点的位置
        for window in [5, 10, 20, 50]:
            # 相对高点位置
            result_df[f'price_rel_high_{window}'] = result_df['close'] / result_df['high'].rolling(window=window).max()
            # 相对低点位置
            result_df[f'price_rel_low_{window}'] = result_df['close'] / result_df['low'].rolling(window=window).min()
        
        # 计算价格波动性
        result_df['price_volatility'] = result_df['price_change_pct'].rolling(window=20).std()
        
        logger.info("已计算价格特征")
        return result_df
    
    @staticmethod
    def calculate_volume_features(df):
        """
        计算交易量相关特征
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含交易量特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的交易量特征")
            return df
        
        result_df = df.copy()
        
        # 计算交易量变化百分比
        result_df['volume_change_pct'] = result_df['volume'].pct_change()
        
        # 计算相对交易量 (相对于近期平均值)
        for window in [5, 10, 20, 50]:
            result_df[f'rel_volume_{window}'] = result_df['volume'] / result_df['volume'].rolling(window=window).mean()
        
        # 计算成交量加权平均价格
        result_df['vwap'] = (result_df['high'] + result_df['low'] + result_df['close']) / 3 * result_df['volume']
        result_df['vwap'] = result_df['vwap'].cumsum() / result_df['volume'].cumsum()
        
        # 计算价格和交易量的相关性
        for window in [10, 20]:
            temp = pd.DataFrame({
                'price': result_df['close'],
                'volume': result_df['volume']
            })
            result_df[f'price_volume_corr_{window}'] = temp['price'].rolling(window=window).corr(temp['volume'])
        
        # 计算交易量振荡器
        result_df['volume_oscillator'] = result_df['volume'].rolling(window=5).mean() / result_df['volume'].rolling(window=20).mean()
        
        # 使用ta库的交易量指标
        # 负量指标
        result_df['nvi'] = ta.volume.negative_volume_index(result_df['close'], result_df['volume'])
        
        # 价格-交易量趋势 (PVT) - 自定义实现，因为ta库中没有
        # PVT计算公式: PVT = [((CurrentClose - PreviousClose) / PreviousClose) × Volume] + PreviousPVT
        close_pct_change = result_df['close'].pct_change()
        pvt = pd.Series(index=result_df.index, dtype='float64')
        pvt.iloc[0] = 0  # 初始值
        
        for i in range(1, len(result_df)):
            if pd.notna(close_pct_change.iloc[i]):
                pvt.iloc[i] = close_pct_change.iloc[i] * result_df['volume'].iloc[i] + pvt.iloc[i-1]
            else:
                pvt.iloc[i] = pvt.iloc[i-1]
        
        result_df['pvt'] = pvt
        
        logger.info("已计算交易量特征")
        return result_df

    @staticmethod
    def calculate_volatility_features(df):
        """
        计算波动性相关特征
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含波动性特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的波动性特征")
            return df
        
        result_df = df.copy()
        
        # 计算真实范围 (True Range)
        result_df['true_range'] = ta.volatility.average_true_range(result_df['high'], result_df['low'], result_df['close'], window=1)
        
        # 计算不同窗口的真实范围平均值
        for window in [5, 10, 14, 20, 50]:
            result_df[f'atr_{window}'] = ta.volatility.average_true_range(result_df['high'], result_df['low'], result_df['close'], window=window)
        
        # 计算收益率的历史波动率
        for window in [5, 10, 20, 50]:
            result_df[f'returns_volatility_{window}'] = result_df['price_change_pct'].rolling(window=window).std()
        
        # 计算收益率的振幅 (Range)
        for window in [5, 10, 20, 50]:
            result_df[f'returns_range_{window}'] = result_df['price_change_pct'].rolling(window=window).max() - result_df['price_change_pct'].rolling(window=window).min()
        
        # 计算Garman-Klass波动率估计
        # Garman-Klass = 0.5 * ln(high/low)^2 - (2ln(2)-1) * ln(close/open)^2
        result_df['garman_klass_vol'] = (0.5 * np.log(result_df['high'] / result_df['low']) ** 2) - ((2 * np.log(2) - 1) * np.log(result_df['close'] / result_df['open']) ** 2)
        
        # 计算不同时期的Garman-Klass波动率估计
        for window in [5, 10, 20, 50]:
            result_df[f'garman_klass_vol_{window}'] = result_df['garman_klass_vol'].rolling(window=window).mean()
        
        # 计算布林带宽度
        for window in [20, 50]:
            result_df[f'bollinger_width_{window}'] = ta.volatility.bollinger_wband(result_df['close'], window=window)
        
        # 计算肯特纳通道宽度
        result_df['keltner_width'] = (ta.volatility.keltner_channel_hband(result_df['high'], result_df['low'], result_df['close'], window=20) -
                                      ta.volatility.keltner_channel_lband(result_df['high'], result_df['low'], result_df['close'], window=20)) / ta.volatility.keltner_channel_mband(result_df['high'], result_df['low'], result_df['close'], window=20)
        
        logger.info("已计算波动性特征")
        return result_df

    @staticmethod
    def calculate_trend_features(df):
        """
        计算趋势相关特征
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含趋势特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的趋势特征")
            return df
        
        result_df = df.copy()
        
        # 计算不同周期的简单移动平均线
        for window in [5, 10, 20, 50, 100, 200]:
            result_df[f'sma_{window}'] = ta.trend.sma_indicator(result_df['close'], window=window)
        
        # 计算不同周期的指数移动平均线
        for window in [5, 10, 20, 50, 100, 200]:
            result_df[f'ema_{window}'] = ta.trend.ema_indicator(result_df['close'], window=window)
        
        # 计算价格相对于移动平均线的位置
        for window in [5, 10, 20, 50, 100, 200]:
            result_df[f'price_rel_sma_{window}'] = result_df['close'] / result_df[f'sma_{window}']
            result_df[f'price_rel_ema_{window}'] = result_df['close'] / result_df[f'ema_{window}']
        
        # 计算移动平均线之间的关系
        # 短期vs中期
        result_df['sma_5_20_ratio'] = result_df['sma_5'] / result_df['sma_20']
        result_df['ema_5_20_ratio'] = result_df['ema_5'] / result_df['ema_20']
        
        # 中期vs长期
        result_df['sma_20_50_ratio'] = result_df['sma_20'] / result_df['sma_50']
        result_df['ema_20_50_ratio'] = result_df['ema_20'] / result_df['ema_50']
        
        # 计算平均趋向指标 (ADX)
        for window in [14, 20]:
            result_df[f'adx_{window}'] = ta.trend.adx(result_df['high'], result_df['low'], result_df['close'], window=window)
            result_df[f'adx_pos_{window}'] = ta.trend.adx_pos(result_df['high'], result_df['low'], result_df['close'], window=window)
            result_df[f'adx_neg_{window}'] = ta.trend.adx_neg(result_df['high'], result_df['low'], result_df['close'], window=window)
        
        # 计算趋势强度指数 (TSI)
        result_df['tsi'] = TechnicalIndicators.tsi(result_df['close'])
        
        # 计算Ichimoku云指标
        result_df['ichimoku_a'] = ta.trend.ichimoku_a(result_df['high'], result_df['low'])
        result_df['ichimoku_b'] = ta.trend.ichimoku_b(result_df['high'], result_df['low'])
        result_df['ichimoku_base'] = ta.trend.ichimoku_base_line(result_df['high'], result_df['low'])
        result_df['ichimoku_conv'] = ta.trend.ichimoku_conversion_line(result_df['high'], result_df['low'])
        
        logger.info("已计算趋势特征")
        return result_df

    @staticmethod
    def calculate_momentum_features(df):
        """
        计算动量相关特征
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含动量特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的动量特征")
            return df
        
        result_df = df.copy()
        
        # 计算不同周期的价格变动率
        for window in [1, 5, 10, 20, 50, 100]:
            result_df[f'price_change_{window}'] = result_df['close'].pct_change(periods=window)
        
        # 计算相对强弱指标
        for window in [2, 6, 14, 20]:
            result_df[f'rsi_{window}'] = ta.momentum.rsi(result_df['close'], window=window)
        
        # 计算随机振荡器
        result_df['stoch_k'] = ta.momentum.stoch(result_df['high'], result_df['low'], result_df['close'], window=14, smooth_window=3)
        result_df['stoch_d'] = ta.momentum.stoch_signal(result_df['high'], result_df['low'], result_df['close'], window=14, smooth_window=3)
        
        # 计算变动率
        for window in [10, 20, 50]:
            result_df[f'roc_{window}'] = ta.momentum.roc(result_df['close'], window=window)
        
        # 计算顺势指标
        for window in [14, 20]:
            result_df[f'cci_{window}'] = ta.trend.cci(result_df['high'], result_df['low'], result_df['close'], window=window)
        
        # 计算威廉指标
        for window in [14, 20]:
            result_df[f'williams_{window}'] = ta.momentum.williams_r(result_df['high'], result_df['low'], result_df['close'], lbp=window)
        
        # 计算动量指标
        for window in [10, 20]:
            result_df[f'momentum_{window}'] = ta.momentum.roc(result_df['close'], window=window) * 100
        
        # 计算资金流量指标
        for window in [14, 20]:
            result_df[f'mfi_{window}'] = ta.volume.money_flow_index(result_df['high'], result_df['low'], result_df['close'], result_df['volume'], window=window)
        
        # 计算终极振荡器
        result_df['ultimate_osc'] = ta.momentum.ultimate_oscillator(result_df['high'], result_df['low'], result_df['close'])
        
        # 计算动量平均值
        for window in [5, 10, 20]:
            result_df[f'tsi_{window}'] = TechnicalIndicators.tsi(result_df['close'], r=window, s=window//2)
        
        # 计算MACD指标
        result_df['macd'] = ta.trend.macd(result_df['close'])
        result_df['macd_signal'] = ta.trend.macd_signal(result_df['close'])
        result_df['macd_diff'] = ta.trend.macd_diff(result_df['close'])
        
        logger.info("已计算动量特征")
        return result_df 