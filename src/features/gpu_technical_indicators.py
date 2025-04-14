#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU加速技术指标计算模块 - 使用RAPIDS库加速计算
"""

import pandas as pd
import numpy as np
# 替换talib为ta库
import ta
import logging

from src.utils.logger import get_logger
from src.features.gpu_accelerator import get_accelerator

logger = get_logger(__name__)
gpu_accel = get_accelerator()  # 获取GPU加速器实例

class GPUTechnicalIndicators:
    """
    GPU加速的技术指标计算类，提供常用技术分析指标的计算
    """
    
    def __init__(self):
        """初始化GPU技术指标计算器"""
        self.gpu_available = gpu_accel.is_available()
        logger.info(f"GPU加速技术指标计算器初始化 - GPU可用: {self.gpu_available}")
    
    def tsi(self, close, r=25, s=13):
        """
        使用GPU加速计算趋势强度指数 (TSI)
        
        参数:
            close: 收盘价数据
            r: 长周期
            s: 短周期
            
        返回:
            TSI值
        """
        if not self.gpu_available:
            # 退回到CPU计算
            m = close.diff()
            m1 = m.ewm(span=r, adjust=False).mean()
            m2 = m1.ewm(span=s, adjust=False).mean()
            a = m.abs()
            a1 = a.ewm(span=r, adjust=False).mean()
            a2 = a1.ewm(span=s, adjust=False).mean()
            return 100 * m2 / a2
        
        try:
            # 转换到GPU
            gpu_close = gpu_accel.to_gpu(pd.DataFrame({'close': close}))['close']
            
            # GPU计算
            m = gpu_close.diff()
            m1 = m.ewm(span=r, adjust=False).mean()
            m2 = m1.ewm(span=s, adjust=False).mean()
            a = m.abs()
            a1 = a.ewm(span=r, adjust=False).mean()
            a2 = a1.ewm(span=s, adjust=False).mean()
            result = 100 * m2 / a2
            
            # 转回CPU
            return gpu_accel.to_cpu(result)
        except Exception as e:
            logger.error(f"GPU TSI计算失败，回退到CPU: {e}")
            # 退回到CPU计算
            m = close.diff()
            m1 = m.ewm(span=r, adjust=False).mean()
            m2 = m1.ewm(span=s, adjust=False).mean()
            a = m.abs()
            a1 = a.ewm(span=r, adjust=False).mean()
            a2 = a1.ewm(span=s, adjust=False).mean()
            return 100 * m2 / a2
    
    def calculate_indicators(self, df, indicators=None, window_sizes=None):
        """
        使用GPU加速计算技术指标
        
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
        
        if self.gpu_available:
            try:
                # 转换到GPU
                gpu_df = gpu_accel.to_gpu(result_df)
                
                # 计算移动平均线指标
                if 'SMA' in indicators:
                    for window in window_sizes:
                        gpu_df[f'SMA_{window}'] = gpu_df['close'].rolling(window=window).mean()
                
                # 计算指数移动平均线
                if 'EMA' in indicators:
                    for window in window_sizes:
                        gpu_df[f'EMA_{window}'] = gpu_df['close'].ewm(span=window, adjust=False).mean()
                
                # 计算加权移动平均线 (使用EMA代替)
                if 'WMA' in indicators:
                    for window in window_sizes:
                        gpu_df[f'WMA_{window}'] = gpu_df['close'].ewm(span=window, adjust=False).mean()
                
                # 转回CPU继续计算其他指标
                result_df = gpu_accel.to_cpu(gpu_df)
                logger.info("已使用GPU计算基础指标")
            except Exception as e:
                logger.error(f"GPU计算指标失败，回退到CPU: {e}")
        
        # 计算MACD (使用CPU，因为cuDF可能不支持复杂指标)
        if 'MACD' in indicators:
            # 默认参数：快线=12，慢线=26，信号线=9
            result_df['MACD'] = ta.trend.macd(result_df['close'], window_slow=26, window_fast=12)
            result_df['MACD_signal'] = ta.trend.macd_signal(result_df['close'], window_slow=26, window_fast=12, window_sign=9)
            result_df['MACD_hist'] = ta.trend.macd_diff(result_df['close'], window_slow=26, window_fast=12, window_sign=9)
        
        # 计算RSI
        if 'RSI' in indicators:
            for window in [6, 14, 20]:
                result_df[f'RSI_{window}'] = ta.momentum.rsi(result_df['close'], window=window)
        
        # 其他指标计算...与原始版本相同
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
    
    def calculate_price_features(self, df):
        """
        使用GPU加速计算价格相关特征
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含价格特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的价格特征")
            return df
        
        # GPU加速处理
        if self.gpu_available:
            try:
                # 转换到GPU
                gpu_df = gpu_accel.to_gpu(df.copy())
                
                # 计算价格变化百分比
                gpu_df['price_change_pct'] = gpu_df['close'].pct_change()
                
                # 计算价格差异指标
                gpu_df['price_gap'] = (gpu_df['high'] - gpu_df['low']) / gpu_df['low']
                
                # 计算开盘-收盘差异
                gpu_df['open_close_diff'] = (gpu_df['close'] - gpu_df['open']) / gpu_df['open']
                
                # 计算最高-最低差异
                gpu_df['high_low_diff'] = (gpu_df['high'] - gpu_df['low']) / gpu_df['low']
                
                # 计算当前价格相对近期高低点的位置
                for window in [5, 10, 20, 50]:
                    # 相对高点位置
                    gpu_df[f'price_rel_high_{window}'] = gpu_df['close'] / gpu_df['high'].rolling(window=window).max()
                    # 相对低点位置
                    gpu_df[f'price_rel_low_{window}'] = gpu_df['close'] / gpu_df['low'].rolling(window=window).min()
                
                # 计算价格波动性
                gpu_df['price_volatility'] = gpu_df['price_change_pct'].rolling(window=20).std()
                
                # 转回CPU
                result_df = gpu_accel.to_cpu(gpu_df)
                logger.info("已使用GPU计算价格特征")
                return result_df
            except Exception as e:
                logger.error(f"GPU计算价格特征失败，回退到CPU: {e}")
                # 回退到CPU计算
        
        # CPU处理逻辑
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
        
        logger.info("已使用CPU计算价格特征")
        return result_df
    
    def calculate_volume_features(self, df):
        """
        使用GPU加速计算交易量相关特征
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含交易量特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的交易量特征")
            return df
        
        # GPU加速处理
        if self.gpu_available:
            try:
                # 转换到GPU
                gpu_df = gpu_accel.to_gpu(df.copy())
                
                # 计算交易量变化百分比
                gpu_df['volume_change_pct'] = gpu_df['volume'].pct_change()
                
                # 计算相对交易量 (相对于近期平均值)
                for window in [5, 10, 20, 50]:
                    gpu_df[f'rel_volume_{window}'] = gpu_df['volume'] / gpu_df['volume'].rolling(window=window).mean()
                
                # 转回CPU继续计算其他特征
                result_df = gpu_accel.to_cpu(gpu_df)
                logger.info("已使用GPU计算基础交易量特征")
            except Exception as e:
                logger.error(f"GPU计算交易量特征失败，回退到CPU: {e}")
                result_df = df.copy()
                # 计算交易量变化百分比
                result_df['volume_change_pct'] = result_df['volume'].pct_change()
                # 计算相对交易量 (相对于近期平均值)
                for window in [5, 10, 20, 50]:
                    result_df[f'rel_volume_{window}'] = result_df['volume'] / result_df['volume'].rolling(window=window).mean()
        else:
            result_df = df.copy()
            # 计算交易量变化百分比
            result_df['volume_change_pct'] = result_df['volume'].pct_change()
            # 计算相对交易量 (相对于近期平均值)
            for window in [5, 10, 20, 50]:
                result_df[f'rel_volume_{window}'] = result_df['volume'] / result_df['volume'].rolling(window=window).mean()
        
        # 继续使用CPU计算其他交易量特征
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
        
        # 价格-交易量趋势 (PVT)
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
    
    # 其他计算方法可以按照类似方式实现:
    # calculate_volatility_features, calculate_trend_features, calculate_momentum_features 等

# 允许创建一个与原始TechnicalIndicators兼容的接口
class GpuCompatibleTechnicalIndicators:
    """兼容原始TechnicalIndicators类的接口，但支持GPU加速"""
    
    def __init__(self):
        self._gpu_indicators = GPUTechnicalIndicators()
    
    @staticmethod
    def tsi(close, r=25, s=13):
        """静态方法接口兼容"""
        _gpu_indicators = GPUTechnicalIndicators()
        return _gpu_indicators.tsi(close, r, s)
    
    @staticmethod
    def calculate_indicators(df, indicators=None, window_sizes=None):
        """静态方法接口兼容"""
        _gpu_indicators = GPUTechnicalIndicators()
        return _gpu_indicators.calculate_indicators(df, indicators, window_sizes)
    
    @staticmethod
    def calculate_price_features(df):
        """静态方法接口兼容"""
        _gpu_indicators = GPUTechnicalIndicators()
        return _gpu_indicators.calculate_price_features(df)
    
    @staticmethod
    def calculate_volume_features(df):
        """静态方法接口兼容"""
        _gpu_indicators = GPUTechnicalIndicators()
        return _gpu_indicators.calculate_volume_features(df)
    
    # 其他静态方法接口... 