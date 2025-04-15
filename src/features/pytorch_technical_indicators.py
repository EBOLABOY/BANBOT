"""
PyTorch加速技术指标计算模块 - 使用混合计算方案（GPU+CPU）加速计算
"""

import pandas as pd
import numpy as np
import torch
import logging
import time
import concurrent.futures
from functools import partial

from src.utils.logger import get_logger

logger = get_logger(__name__)

class PyTorchCompatibleTechnicalIndicators:
    """
    使用混合计算方案（GPU+CPU）的技术指标计算类，提供常用技术分析指标的计算
    """
    
    def __init__(self):
        """初始化混合计算技术指标计算器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = self.device.type == 'cuda'
        # 设置复杂特征的列表（适合GPU计算）
        self.complex_features = [
            'rolling_statistics', 'volatility_features', 'trend_features', 
            'momentum_features', 'pattern_recognition'
        ]
        # 设置简单特征的列表（适合CPU计算）
        self.simple_features = [
            'basic_price_features', 'basic_volume_features', 'single_indicators'
        ]
        # 增加CPU工作线程数，利用更多核心
        self.max_cpu_workers = 8  # 增加到8个线程
        # 设置内存优化选项
        self.torch_pin_memory = True  # 加速CPU到GPU的数据传输
        # 设置缓存大小，减少重复计算
        self.cache_size = 10  # 最多缓存10个计算结果
        # 添加监控
        self.measure_time = True  # 打开时间测量
        
        logger.info(f"混合计算技术指标计算器初始化 - GPU可用: {self.gpu_available}, 设备: {self.device}")
        logger.info(f"CPU工作线程数: {self.max_cpu_workers}")
        if self.gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"GPU: {gpu_name}, 显存: {gpu_memory:.2f}GB")
        
    def compute_features(self, df, feature_groups=None):
        """
        使用混合计算方案计算所有特征，减少数据传输
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            feature_groups: 要计算的特征组列表
            
        返回:
            包含计算特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的特征")
            return df
            
        start_time = time.time()
        result_df = df.copy()
        
        # 默认特征组
        if feature_groups is None:
            feature_groups = ['price_based', 'volume_based', 'volatility', 'trend', 'momentum']
        
        # 确定哪些特征组适合GPU计算，哪些适合CPU计算
        gpu_groups = []
        cpu_groups = []
        
        for group in feature_groups:
            if group in ['volatility', 'trend', 'momentum']:
                gpu_groups.append(group)
            else:
                cpu_groups.append(group)
        
        # 如果GPU可用，先将数据转换为GPU张量一次性完成，避免多次传输
        gpu_tensors = {}
        if self.gpu_available and gpu_groups:
            logger.info(f"使用GPU计算复杂特征组: {gpu_groups}")
            try:
                # 一次性将所有需要的列转换为GPU张量
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        gpu_tensors[col] = torch.tensor(
                            df[col].values, 
                            dtype=torch.float32, 
                            device=self.device,
                            pin_memory=self.torch_pin_memory
                        )
                # 一次性计算所有GPU特征
                gpu_features = {}
                for group in gpu_groups:
                    if group == 'volatility':
                        # 传递已有的GPU张量而不是DataFramc
                        gpu_features.update(self._gpu_calculate_volatility_features(df, gpu_tensors))
                    elif group == 'trend':
                        gpu_features.update(self._gpu_calculate_trend_features(df, gpu_tensors))
                    elif group == 'momentum':
                        gpu_features.update(self._gpu_calculate_momentum_features(df, gpu_tensors))
                
                # 将计算结果一次性添加到结果DataFrame
                for feature_name, feature_data in gpu_features.items():
                    result_df[feature_name] = feature_data
            except Exception as e:
                logger.error(f"GPU计算出错，回退到CPU: {e}")
                # 将出错的特征组添加到CPU计算
                cpu_groups.extend(gpu_groups)
                gpu_groups = []
        
        # 使用CPU并行计算简单特征
        if cpu_groups:
            logger.info(f"使用CPU计算简单特征组: {cpu_groups}")
            # 使用线程池并行计算CPU特征
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_cpu_workers) as executor:
                futures = []
                
                # 为每个CPU特征组创建future
                for group in cpu_groups:
                    if group == 'price_based':
                        futures.append(executor.submit(self._cpu_calculate_price_features, df.copy()))
                    elif group == 'volume_based':
                        futures.append(executor.submit(self._cpu_calculate_volume_features, df.copy()))
                    elif group == 'volatility':
                        futures.append(executor.submit(self._cpu_calculate_volatility_features, df.copy()))
                    elif group == 'trend':
                        futures.append(executor.submit(self._cpu_calculate_trend_features, df.copy()))
                    elif group == 'momentum':
                        futures.append(executor.submit(self._cpu_calculate_momentum_features, df.copy()))
                
                # 合并所有CPU计算结果
                for future in concurrent.futures.as_completed(futures):
                    try:
                        cpu_result = future.result()
                        # 只合并新增的特征列
                        new_cols = [col for col in cpu_result.columns if col not in df.columns]
                        result_df[new_cols] = cpu_result[new_cols]
                    except Exception as e:
                        logger.error(f"CPU特征计算出错: {str(e)}")
        
        # 清理GPU内存
        if self.gpu_available:
            torch.cuda.empty_cache()
        
        logger.info(f"特征计算完成，耗时: {time.time() - start_time:.2f}秒")
        return result_df
    
    def _cpu_calculate_price_features(self, df):
        """在CPU上计算价格相关特征"""
        from src.features.technical_indicators import TechnicalIndicators
        return TechnicalIndicators.calculate_price_features(df)
    
    def _cpu_calculate_volume_features(self, df):
        """在CPU上计算交易量相关特征"""
        from src.features.technical_indicators import TechnicalIndicators
        return TechnicalIndicators.calculate_volume_features(df)
    
    def _gpu_calculate_volatility_features(self, df, tensors=None):
        """GPU优化的波动性特征计算，复用已有的GPU张量避免重复传输"""
        features = {}
        
        # 如果没有传入张量，则创建
        if tensors is None:
            tensors = {}
            for col in ['close', 'high', 'low']:
                if col in df.columns:
                    tensors[col] = torch.tensor(df[col].values, dtype=torch.float32, device=self.device)
        
        # 获取所需的张量
        close = tensors.get('close')
        high = tensors.get('high')
        low = tensors.get('low')
        
        if close is None or high is None or low is None:
            logger.warning("计算波动性特征需要close/high/low数据，但未提供")
            return features
        
        # 计算ATR - 使用批处理加速
        tr1 = high - low
        prev_close = torch.zeros_like(close)
        prev_close[1:] = close[:-1]
        tr2 = torch.abs(high - prev_close)
        tr3 = torch.abs(low - prev_close)
        
        # 使用向量化操作计算TR
        tr = torch.max(torch.max(tr1, tr2), tr3)
        
        # 计算不同周期的ATR (平均真实范围)
        for window in [7, 14, 21]:
            # 使用2D卷积进行滑动窗口计算，比循环更高效
            atr = torch.nn.functional.avg_pool1d(
                tr.view(1, 1, -1), 
                kernel_size=window, 
                stride=1, 
                padding=window-1
            ).squeeze()[:len(close)]
            
            # 复制到结果字典
            features[f'atr_{window}'] = atr.cpu().numpy()
            
            # 计算ATR百分比 (ATR/价格)
            atr_pct = atr / close * 100
            features[f'atr_pct_{window}'] = atr_pct.cpu().numpy()
        
        return features

    def _gpu_calculate_trend_features(self, df, tensors=None):
        """GPU优化的趋势特征计算"""
        # 实现类似于_gpu_calculate_volatility_features的优化
        # ...假设的实现内容...
        features = {}
        return features

    def _gpu_calculate_momentum_features(self, df, tensors=None):
        """GPU优化的动量特征计算"""
        # 实现类似于_gpu_calculate_volatility_features的优化
        # ...假设的实现内容...
        features = {}
        return features

    def _cpu_calculate_volatility_features(self, df):
        """CPU优化的波动性特征计算"""
        from src.features.technical_indicators import TechnicalIndicators
        # 使用numba或向量化操作优化
        return TechnicalIndicators.calculate_volatility_features(df)

    def _cpu_calculate_trend_features(self, df):
        """CPU优化的趋势特征计算"""
        from src.features.technical_indicators import TechnicalIndicators
        return TechnicalIndicators.calculate_trend_features(df)

    def _cpu_calculate_momentum_features(self, df):
        """CPU优化的动量特征计算"""
        from src.features.technical_indicators import TechnicalIndicators
        return TechnicalIndicators.calculate_momentum_features(df)
    
    @staticmethod
    def tsi(close, r=25, s=13):
        """
        计算趋势强度指数 (TSI)，使用混合计算方式
        如果数据量较大，使用PyTorch加速；数据量较小则直接用CPU计算
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 如果数据量较小或没有GPU，直接用CPU计算
        if len(close) < 10000 or device.type == 'cpu':
            m = close.diff()
            m1 = m.ewm(span=r, adjust=False).mean()
            m2 = m1.ewm(span=s, adjust=False).mean()
            a = m.abs()
            a1 = a.ewm(span=r, adjust=False).mean()
            a2 = a1.ewm(span=s, adjust=False).mean()
            return 100 * m2 / a2
        
        try:
            # 转换到PyTorch张量
            close_tensor = torch.tensor(close.values, dtype=torch.float32, device=device)
            
            # 计算动量
            m = torch.zeros_like(close_tensor)
            m[1:] = close_tensor[1:] - close_tensor[:-1]  # diff
            
            # 计算EMA
            def ema(x, span):
                alpha = 2.0 / (span + 1.0)
                # 创建与x相同形状的输出张量
                output = torch.zeros_like(x)
                output[0] = x[0]  # 初始值
                
                # 计算EMA
                for i in range(1, len(x)):
                    output[i] = alpha * x[i] + (1 - alpha) * output[i-1]
                return output
            
            # 计算TSI
            m1 = ema(m, r)
            m2 = ema(m1, s)
            a = torch.abs(m)
            a1 = ema(a, r)
            a2 = ema(a1, s)
            tsi = 100.0 * m2 / a2
            
            # 转回NumPy/Pandas
            result = pd.Series(tsi.cpu().numpy(), index=close.index)
            return result
            
        except Exception as e:
            logger.error(f"PyTorch TSI计算失败，回退到CPU: {e}")
            # 回退到原始计算
            m = close.diff()
            m1 = m.ewm(span=r, adjust=False).mean()
            m2 = m1.ewm(span=s, adjust=False).mean()
            a = m.abs()
            a1 = a.ewm(span=r, adjust=False).mean()
            a2 = a1.ewm(span=s, adjust=False).mean()
            return 100 * m2 / a2
    
    @staticmethod
    def calculate_price_features(df):
        """
        计算价格相关特征，根据数据量和复杂度自动选择计算设备
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含价格特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的价格特征")
            return df
        
        # 数据量大于阈值并且有GPU才使用GPU计算
        use_gpu = len(df) > 50000 and torch.cuda.is_available()
        
        if use_gpu:
            logger.info(f"使用GPU计算价格特征 (数据量: {len(df)}行)")
            device = torch.device('cuda')
            result_df = df.copy()
            
            try:
                # 转换到PyTorch张量
                close = torch.tensor(df['close'].values, dtype=torch.float32, device=device)
                open_price = torch.tensor(df['open'].values, dtype=torch.float32, device=device)
                high = torch.tensor(df['high'].values, dtype=torch.float32, device=device)
                low = torch.tensor(df['low'].values, dtype=torch.float32, device=device)
                
                # 计算价格变化百分比
                price_change = torch.zeros_like(close)
                price_change[1:] = (close[1:] - close[:-1]) / close[:-1]
                result_df['price_change_pct'] = price_change.cpu().numpy()
                
                # 计算价格差异指标
                price_gap = (high - low) / low
                result_df['price_gap'] = price_gap.cpu().numpy()
                
                # 计算开盘-收盘差异
                open_close_diff = (close - open_price) / open_price
                result_df['open_close_diff'] = open_close_diff.cpu().numpy()
                
                # 计算最高-最低差异
                high_low_diff = (high - low) / low
                result_df['high_low_diff'] = high_low_diff.cpu().numpy()
                
                # 计算当前价格相对近期高低点的位置
                for window in [5, 10, 20, 50]:
                    # 相对高点位置
                    high_max = torch.zeros_like(high)
                    low_min = torch.zeros_like(low)
                    
                    # 计算滚动窗口的最大/最小值
                    for i in range(len(high)):
                        start_idx = max(0, i - window + 1)
                        high_max[i] = torch.max(high[start_idx:i+1])
                        low_min[i] = torch.min(low[start_idx:i+1])
                    
                    price_rel_high = close / high_max
                    price_rel_low = close / low_min
                    
                    result_df[f'price_rel_high_{window}'] = price_rel_high.cpu().numpy()
                    result_df[f'price_rel_low_{window}'] = price_rel_low.cpu().numpy()
                
                # 计算价格波动性
                price_volatility = torch.zeros_like(close)
                for i in range(20, len(close)):
                    start_idx = i - 20 + 1
                    price_volatility[i] = torch.std(price_change[start_idx:i+1])
                
                result_df['price_volatility'] = price_volatility.cpu().numpy()
                
                logger.info("已使用GPU计算价格特征")
                return result_df
                
            except Exception as e:
                logger.error(f"GPU价格特征计算失败，回退到CPU: {e}")
                # 回退到CPU计算（见下文）
                use_gpu = False
        
        # CPU计算
        if not use_gpu:
            logger.info(f"使用CPU计算价格特征 (数据量: {len(df)}行)")
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
    
    @staticmethod
    def calculate_volatility_features(df):
        """计算波动性相关特征，优先使用GPU"""
        if df is None or df.empty:
            logger.warning("无法计算空数据的波动性特征")
            return df
        
        # 对于大数据集才使用GPU加速
        use_gpu = len(df) > 10000 and torch.cuda.is_available()
        
        if use_gpu:
            logger.info(f"使用GPU计算波动性特征 (数据量: {len(df)}行)")
            device = torch.device('cuda')
            result_df = df.copy()
            
            try:
                # 转换到PyTorch张量
                close = torch.tensor(df['close'].values, dtype=torch.float32, device=device)
                high = torch.tensor(df['high'].values, dtype=torch.float32, device=device)
                low = torch.tensor(df['low'].values, dtype=torch.float32, device=device)
                
                # 计算真实范围 (True Range)
                tr1 = high - low
                prev_close = torch.zeros_like(close)
                prev_close[1:] = close[:-1]
                tr2 = torch.abs(high - prev_close)
                tr3 = torch.abs(low - prev_close)
                
                # 堆叠三个张量并获取每行的最大值
                tr_stacked = torch.stack([tr1, tr2, tr3], dim=1)
                tr = torch.max(tr_stacked, dim=1)[0]
                
                # 计算不同周期的ATR (平均真实范围)
                for window in [7, 14, 21]:
                    atr = torch.zeros_like(close)
                    for i in range(window-1, len(tr)):
                        atr[i] = torch.mean(tr[i-window+1:i+1])
                    
                    # 复制到结果DataFrame
                    result_df[f'atr_{window}'] = atr.cpu().numpy()
                    
                    # 计算ATR百分比 (ATR/价格)
                    atr_pct = atr / close * 100
                    result_df[f'atr_pct_{window}'] = atr_pct.cpu().numpy()
                
                # 计算布林带
                for window in [20, 50]:
                    # 滚动平均和标准差
                    rolling_mean = torch.zeros_like(close)
                    rolling_std = torch.zeros_like(close)
                    
                    for i in range(window-1, len(close)):
                        window_data = close[i-window+1:i+1]
                        rolling_mean[i] = torch.mean(window_data)
                        rolling_std[i] = torch.std(window_data)
                    
                    # 上轨、中轨、下轨
                    upper_band = rolling_mean + 2 * rolling_std
                    lower_band = rolling_mean - 2 * rolling_std
                    
                    # 带宽
                    band_width = (upper_band - lower_band) / rolling_mean * 100
                    
                    # 百分比B值 - 当前价格在布林带中的位置
                    percent_b = (close - lower_band) / (upper_band - lower_band)
                    
                    # 保存到结果DataFrame
                    result_df[f'bb_upper_{window}'] = upper_band.cpu().numpy()
                    result_df[f'bb_middle_{window}'] = rolling_mean.cpu().numpy()
                    result_df[f'bb_lower_{window}'] = lower_band.cpu().numpy()
                    result_df[f'bb_width_{window}'] = band_width.cpu().numpy()
                    result_df[f'bb_percent_b_{window}'] = percent_b.cpu().numpy()
                
                # 计算历史波动率 - 收盘价的滚动标准差
                for window in [5, 10, 21, 63]:
                    # 计算对数收益率
                    log_returns = torch.zeros_like(close)
                    log_returns[1:] = torch.log(close[1:] / close[:-1])
                    
                    # 计算滚动标准差
                    rolling_std = torch.zeros_like(close)
                    for i in range(window-1, len(log_returns)):
                        rolling_std[i] = torch.std(log_returns[i-window+1:i+1])
                    
                    # 年化波动率 (假设日K线，如果是其他周期请相应调整)
                    annualized_vol = rolling_std * torch.sqrt(torch.tensor(252.0, device=device))  # 252是一年的交易日数量
                    
                    # 保存到结果DataFrame
                    result_df[f'volatility_{window}'] = annualized_vol.cpu().numpy()
                
                # 计算Garman-Klass波动率估计器 (适用于OHLC数据)
                open_price = torch.tensor(df['open'].values, dtype=torch.float32, device=device)
                high_low = torch.log(high / low)
                high_low_sq = high_low ** 2
                
                # GK波动率
                close_open = torch.log(close / open_price)
                close_open_sq = close_open ** 2
                gk_vol = torch.zeros_like(close)
                factor = 0.5  # GK系数
                
                for i in range(1, len(close)):
                    gk_vol[i] = torch.sqrt(factor * high_low_sq[i] - (2 * torch.log(2) - 1) * close_open_sq[i])
                
                result_df['gk_volatility'] = gk_vol.cpu().numpy()
                
                logger.info("已使用GPU计算波动性特征")
                return result_df
                
            except Exception as e:
                logger.error(f"PyTorch波动性特征计算失败，回退到CPU: {e}")
                # 回退到CPU计算
                use_gpu = False
        
        if not use_gpu:
            logger.info(f"使用CPU计算波动性特征 (数据量: {len(df)}行)")
            from src.features.technical_indicators import TechnicalIndicators
            return TechnicalIndicators.calculate_volatility_features(df)
    
    @staticmethod
    def calculate_indicators(df, indicators=None, window_sizes=None):
        """智能选择计算设备计算各种技术指标"""
        if df is None or df.empty:
            logger.warning("无法计算空数据的技术指标")
            return df
        
        # 数据量较小直接使用CPU
        if len(df) < 10000 or not torch.cuda.is_available():
            logger.info(f"使用CPU计算技术指标 (数据量较小: {len(df)}行)")
            from src.features.technical_indicators import TechnicalIndicators
            return TechnicalIndicators.calculate_indicators(df, indicators, window_sizes)
        
        # 数据量较大且GPU可用，混合计算
        logger.info(f"使用混合计算技术指标 (数据量: {len(df)}行)")
        
        # 初始化时间测量
        start_time = time.time()
        
        # 创建CPU和GPU结果
        result_df = df.copy()
        
        # 默认指标列表
        if indicators is None:
            indicators = [
                'SMA', 'EMA', 'WMA', 'MACD', 'RSI', 'BBANDS', 'ATR',
                'ADX', 'CCI', 'ROC', 'OBV', 'MFI'
            ]
        
        # 划分GPU和CPU指标
        gpu_indicators = ['SMA', 'EMA', 'WMA', 'RSI', 'BBANDS', 'ATR', 'ROC']
        cpu_indicators = [ind for ind in indicators if ind not in gpu_indicators]
        
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
        
        # 在GPU上计算指标
        if gpu_indicators:
            try:
                device = torch.device('cuda')
                
                # 转换核心数据到GPU
                tensors = {}
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        tensors[col] = torch.tensor(df[col].values, dtype=torch.float32, device=device)
                
                close = tensors['close']
                high = tensors.get('high')
                low = tensors.get('low')
                
                # 批量计算移动平均线指标 - 使用卷积操作
                if 'SMA' in gpu_indicators:
                    for window in window_sizes:
                        # 使用卷积进行高效滑动窗口计算
                        weights = torch.ones(1, 1, window, device=device) / window
                        # 添加维度以匹配卷积输入要求
                        close_expanded = close.view(1, 1, -1)
                        # 使用卷积计算移动平均
                        padded_close = torch.nn.functional.pad(close_expanded, (window-1, 0), 'constant', 0)
                        sma = torch.nn.functional.conv1d(padded_close, weights).view(-1)
                        
                        result_df[f'SMA_{window}'] = sma.cpu().numpy()
                
                # 计算指数移动平均线 - 向量化版本
                if 'EMA' in gpu_indicators:
                    for window in window_sizes:
                        alpha = 2.0 / (window + 1.0)
                        
                        # 使用cumsum进行高效的EMA计算
                        weights = (1 - alpha) ** torch.arange(len(close)-1, -1, -1, device=device)
                        weights = weights / weights.sum()  # 归一化
                        
                        # 使用卷积操作计算EMA
                        close_padded = torch.nn.functional.pad(close.view(1, 1, -1), (len(close)-1, 0), 'constant', 0)
                        weights_reshaped = weights.view(1, 1, -1)
                        ema = torch.nn.functional.conv1d(close_padded, weights_reshaped).view(-1)

                        # 保证长度正确
                        if len(ema) < len(close):
                            padding = torch.full((len(close) - len(ema),), float('nan'), device=device)
                            ema = torch.cat([padding, ema])
                        
                        result_df[f'EMA_{window}'] = ema.cpu().numpy()
                
                # 计算RSI - 向量化版本，不使用循环
                if 'RSI' in gpu_indicators:
                    for window in [6, 14, 20]:
                        # 计算价格变化
                        delta = torch.zeros_like(close)
                        delta[1:] = close[1:] - close[:-1]
                        
                        # 分离上涨和下跌
                        gain = torch.where(delta > 0, delta, torch.zeros_like(delta))
                        loss = torch.where(delta < 0, -delta, torch.zeros_like(delta))
                        
                        # 使用卷积计算滑动窗口平均
                        weights = torch.ones(1, 1, window, device=device) / window
                        gain_expanded = gain.view(1, 1, -1)
                        loss_expanded = loss.view(1, 1, -1)
                        
                        # 添加padding以保持尺寸
                        padded_gain = torch.nn.functional.pad(gain_expanded, (window-1, 0), 'constant', 0)
                        padded_loss = torch.nn.functional.pad(loss_expanded, (window-1, 0), 'constant', 0)
                        
                        # 计算移动平均
                        avg_gain = torch.nn.functional.conv1d(padded_gain, weights).view(-1)
                        avg_loss = torch.nn.functional.conv1d(padded_loss, weights).view(-1)
                        
                        # 计算相对强度
                        rs = avg_gain / (avg_loss + 1e-10)  # 避免除以零
                        
                        # 计算RSI
                        rsi = 100 - (100 / (1 + rs))
                        
                        result_df[f'RSI_{window}'] = rsi.cpu().numpy()
                
                # 计算布林带 - 使用高效的向量化操作
                if 'BBANDS' in gpu_indicators and high is not None and low is not None:
                    for window in [20, 50]:
                        # 使用卷积计算移动平均
                        weights = torch.ones(1, 1, window, device=device) / window
                        close_expanded = close.view(1, 1, -1)
                        padded_close = torch.nn.functional.pad(close_expanded, (window-1, 0), 'constant', 0)
                        sma = torch.nn.functional.conv1d(padded_close, weights).view(-1)
                        
                        # 计算标准差 - 使用高效的向量化操作
                        squared_diff_expanded = ((close.view(1, 1, -1) - sma.view(1, 1, -1))**2).view(1, 1, -1)
                        padded_sq_diff = torch.nn.functional.pad(squared_diff_expanded, (window-1, 0), 'constant', 0)
                        variance = torch.nn.functional.conv1d(padded_sq_diff, weights).view(-1)
                        std = torch.sqrt(variance + 1e-10)  # 避免负数
                        
                        # 计算布林带边界
                        upper = sma + 2 * std
                        lower = sma - 2 * std
                        
                        # 添加到结果
                        result_df[f'BBANDS_middle_{window}'] = sma.cpu().numpy()
                        result_df[f'BBANDS_upper_{window}'] = upper.cpu().numpy()
                        result_df[f'BBANDS_lower_{window}'] = lower.cpu().numpy()
                        result_df[f'BBANDS_width_{window}'] = ((upper - lower) / sma * 100).cpu().numpy()
                
                # 计算ROC - 价格变化率
                if 'ROC' in gpu_indicators:
                    for window in [9, 14, 25]:
                        # 使用移位操作计算ROC
                        shifted_close = torch.zeros_like(close)
                        shifted_close[window:] = close[:-window]
                        
                        # 处理前window个位置
                        shifted_close[:window] = close[0]  
                        
                        # 计算ROC
                        roc = (close - shifted_close) / (shifted_close + 1e-10) * 100
                        result_df[f'ROC_{window}'] = roc.cpu().numpy()
                
                # ATR - 平均真实范围
                if 'ATR' in gpu_indicators and high is not None and low is not None:
                    # 计算真实范围 (True Range)
                    tr1 = high - low                        # 当日最高-最低
                    prev_close = torch.zeros_like(close)
                    prev_close[1:] = close[:-1]             # 前一日收盘价
                    tr2 = torch.abs(high - prev_close)      # 当日最高-前日收盘
                    tr3 = torch.abs(low - prev_close)       # 当日最低-前日收盘
                    
                    # 取三者中的最大值
                    tr = torch.max(torch.max(tr1, tr2), tr3)
                    
                    # 使用卷积计算滑动平均
                    for window in [7, 14, 21]:
                        weights = torch.ones(1, 1, window, device=device) / window
                        tr_expanded = tr.view(1, 1, -1)
                        padded_tr = torch.nn.functional.pad(tr_expanded, (window-1, 0), 'constant', 0)
                        atr = torch.nn.functional.conv1d(padded_tr, weights).view(-1)
                        
                        result_df[f'ATR_{window}'] = atr.cpu().numpy()
                
                # 清理GPU内存
                torch.cuda.empty_cache()
                
                logger.info(f"GPU指标计算完成，耗时: {time.time() - start_time:.2f}秒")
            except Exception as e:
                logger.error(f"GPU指标计算失败: {e}")
                # 发生错误时，将所有指标放入CPU计算
                cpu_indicators = indicators
        
        # 在CPU上计算其余指标
        if cpu_indicators:
            try:
                from src.features.technical_indicators import TechnicalIndicators
                cpu_start_time = time.time()
                
                # 仅使用未在GPU上计算的指标
                cpu_df = TechnicalIndicators.calculate_indicators(df, indicators=cpu_indicators, window_sizes=window_sizes)
                
                # 将CPU计算的列添加到结果中
                for col in cpu_df.columns:
                    if col not in df.columns and col not in result_df.columns:
                        result_df[col] = cpu_df[col]
                    
                logger.info(f"CPU指标计算完成，耗时: {time.time() - cpu_start_time:.2f}秒")
            except Exception as e:
                logger.error(f"CPU指标计算失败: {e}")
        
        logger.info(f"所有指标计算完成，总耗时: {time.time() - start_time:.2f}秒")
        return result_df 