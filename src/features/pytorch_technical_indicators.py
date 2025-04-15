"""
基于PyTorch的技术指标计算模块 - 使用GPU加速
"""

import torch
import pandas as pd
import numpy as np
import logging

from src.utils.logger import get_logger
from src.features.torch_utils import get_device, df_to_tensor, tensor_to_df, moving_average, ewma, correlation, rolling_window

logger = get_logger(__name__)

class PyTorchTechnicalIndicators:
    """
    使用PyTorch实现的技术指标计算类，充分利用GPU加速
    """
    
    def __init__(self):
        """
        初始化技术指标计算器
        """
        self.device = get_device()
        logger.info(f"PyTorch技术指标计算器初始化，使用设备: {self.device}")
    
    @staticmethod
    def tsi(close_tensor, r=25, s=13):
        """
        计算趋势强度指数 (TSI)，PyTorch版本
        
        参数:
            close_tensor: 收盘价张量
            r: 长周期
            s: 短周期
            
        返回:
            TSI值张量
        """
        # 计算动量（价格变化）
        m = torch.diff(close_tensor, dim=0, prepend=close_tensor[0:1])
        
        # 计算动量的EWMA
        m1 = ewma(m, span=r, adjust=False)
        m2 = ewma(m1, span=s, adjust=False)
        
        # 计算动量绝对值的EWMA
        a = torch.abs(m)
        a1 = ewma(a, span=r, adjust=False)
        a2 = ewma(a1, span=s, adjust=False)
        
        # 计算TSI
        tsi = 100 * m2 / a2
        
        # 处理边界值
        tsi = torch.where(torch.isnan(tsi) | torch.isinf(tsi), torch.zeros_like(tsi), tsi)
        
        return tsi
    
    def calculate_indicators(self, df, indicators=None, window_sizes=None):
        """
        计算技术指标，PyTorch版
        
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
        
        # 将数据转换为PyTorch张量，仅选择所需列
        tensor_dict = {}
        for col in required_columns:
            if col in df.columns:
                tensor, _ = df_to_tensor(df, [col])
                tensor_dict[col] = tensor
        
        # 计算技术指标并将结果存储在结果字典中
        result_tensors = {}
        
        # 计算移动平均线指标
        if 'SMA' in indicators:
            for window in window_sizes:
                sma = moving_average(tensor_dict['close'], window)
                result_tensors[f'SMA_{window}'] = sma
        
        # 计算指数移动平均线
        if 'EMA' in indicators:
            for window in window_sizes:
                ema = ewma(tensor_dict['close'], span=window)
                result_tensors[f'EMA_{window}'] = ema
        
        # 计算加权移动平均线
        if 'WMA' in indicators:
            for window in window_sizes:
                # 创建线性递减权重
                weights = torch.linspace(1, window, window, device=self.device)
                weights = weights / weights.sum()
                
                # 计算WMA
                wma = moving_average(tensor_dict['close'], window, weights=weights)
                result_tensors[f'WMA_{window}'] = wma
        
        # 计算MACD
        if 'MACD' in indicators:
            # 计算快线EMA (12)
            fast_ema = ewma(tensor_dict['close'], span=12)
            
            # 计算慢线EMA (26)
            slow_ema = ewma(tensor_dict['close'], span=26)
            
            # 计算MACD线
            macd_line = fast_ema - slow_ema
            
            # 计算信号线 (9日EMA)
            signal_line = ewma(macd_line, span=9)
            
            # 计算直方图 (MACD - 信号线)
            histogram = macd_line - signal_line
            
            # 存储结果
            result_tensors['MACD'] = macd_line
            result_tensors['MACD_signal'] = signal_line
            result_tensors['MACD_hist'] = histogram
        
        # 计算RSI
        if 'RSI' in indicators:
            # 计算价格变化
            price_diff = torch.diff(tensor_dict['close'], dim=0, prepend=tensor_dict['close'][0:1])
            
            for window in [6, 14, 20]:
                # 分离上涨和下跌
                up = torch.maximum(price_diff, torch.zeros_like(price_diff))
                down = torch.abs(torch.minimum(price_diff, torch.zeros_like(price_diff)))
                
                # 计算平均上涨和下跌
                avg_up = moving_average(up, window)
                avg_down = moving_average(down, window)
                
                # 计算相对强度
                rs = avg_up / (avg_down + 1e-10)  # 添加小值避免除以零
                
                # 计算RSI
                rsi = 100 - (100 / (1 + rs))
                
                # 处理边界情况
                rsi = torch.clamp(rsi, 0, 100)
                
                # 存储结果
                result_tensors[f'RSI_{window}'] = rsi
        
        # 计算随机振荡器
        if 'STOCH' in indicators:
            # 默认参数: k=5, d=3
            window_k = 5
            window_d = 3
            
            # 获取滚动窗口的最高价和最低价
            windows_high = rolling_window(tensor_dict['high'], window_k)
            windows_low = rolling_window(tensor_dict['low'], window_k)
            
            # 计算最大值和最小值
            highest_high = torch.max(windows_high, dim=1)[0]
            lowest_low = torch.min(windows_low, dim=1)[0]
            
            # 填充缺失值
            highest_high_full = torch.full_like(tensor_dict['close'], float('nan'))
            lowest_low_full = torch.full_like(tensor_dict['close'], float('nan'))
            
            highest_high_full[window_k-1:] = highest_high
            lowest_low_full[window_k-1:] = lowest_low
            
            # 复制前面的值填充前面的NaN
            for i in range(window_k-1):
                highest_high_full[i] = torch.max(tensor_dict['high'][:i+1])
                lowest_low_full[i] = torch.min(tensor_dict['low'][:i+1])
            
            # 计算随机值 %K
            stoch_k = 100 * (tensor_dict['close'] - lowest_low_full) / (highest_high_full - lowest_low_full + 1e-10)
            
            # 计算 %D (K的移动平均)
            stoch_d = moving_average(stoch_k, window_d)
            
            # 处理边界值
            stoch_k = torch.clamp(stoch_k, 0, 100)
            stoch_d = torch.clamp(stoch_d, 0, 100)
            
            # 存储结果
            result_tensors['STOCH_K'] = stoch_k
            result_tensors['STOCH_D'] = stoch_d
        
        # 计算布林带
        if 'BBANDS' in indicators:
            for window in [20, 50]:
                # 计算中轨 (SMA)
                middle_band = moving_average(tensor_dict['close'], window)
                
                # 计算标准差
                close_windows = rolling_window(tensor_dict['close'], window)
                std_dev = torch.std(close_windows, dim=1, unbiased=True)
                
                # 填充缺失值
                std_dev_full = torch.full_like(tensor_dict['close'], float('nan'))
                std_dev_full[window-1:] = std_dev
                
                # 计算上轨和下轨 (使用2个标准差)
                upper_band = middle_band + 2 * std_dev_full
                lower_band = middle_band - 2 * std_dev_full
                
                # 计算带宽
                bandwidth = (upper_band - lower_band) / middle_band
                
                # 存储结果
                result_tensors[f'BBANDS_middle_{window}'] = middle_band
                result_tensors[f'BBANDS_upper_{window}'] = upper_band
                result_tensors[f'BBANDS_lower_{window}'] = lower_band
                result_tensors[f'BBANDS_width_{window}'] = bandwidth
        
        # 计算平均真实范围
        if 'ATR' in indicators:
            # 计算真实范围
            high_low = tensor_dict['high'] - tensor_dict['low']
            high_close_prev = torch.abs(tensor_dict['high'] - torch.roll(tensor_dict['close'], 1, dims=0))
            low_close_prev = torch.abs(tensor_dict['low'] - torch.roll(tensor_dict['close'], 1, dims=0))
            
            # 第一个值需要特殊处理
            high_close_prev[0] = 0
            low_close_prev[0] = 0
            
            # 真实范围是三者的最大值
            tr = torch.maximum(high_low, torch.maximum(high_close_prev, low_close_prev))
            
            for window in [14, 20]:
                # 计算ATR (使用EMA)
                atr = ewma(tr, span=window)
                
                # 存储结果
                result_tensors[f'ATR_{window}'] = atr
        
        # 计算平均趋向指标
        if 'ADX' in indicators:
            for window in [14, 20]:
                # 计算方向移动 (+DM 和 -DM)
                high_diff = torch.diff(tensor_dict['high'], dim=0, prepend=tensor_dict['high'][0:1])
                low_diff = torch.diff(tensor_dict['low'], dim=0, prepend=tensor_dict['low'][0:1])
                
                # +DM: 如果high_diff > 0且high_diff > |low_diff|，则为high_diff，否则为0
                plus_dm = torch.where(
                    (high_diff > 0) & (high_diff > torch.abs(low_diff)),
                    high_diff,
                    torch.zeros_like(high_diff)
                )
                
                # -DM: 如果low_diff < 0且|low_diff| > high_diff，则为|low_diff|，否则为0
                minus_dm = torch.where(
                    (low_diff < 0) & (torch.abs(low_diff) > high_diff),
                    torch.abs(low_diff),
                    torch.zeros_like(low_diff)
                )
                
                # 计算ATR
                high_low = tensor_dict['high'] - tensor_dict['low']
                high_close_prev = torch.abs(tensor_dict['high'] - torch.roll(tensor_dict['close'], 1, dims=0))
                low_close_prev = torch.abs(tensor_dict['low'] - torch.roll(tensor_dict['close'], 1, dims=0))
                high_close_prev[0] = 0
                low_close_prev[0] = 0
                tr = torch.maximum(high_low, torch.maximum(high_close_prev, low_close_prev))
                
                # 计算smoothed值
                tr_ema = ewma(tr, span=window)
                plus_dm_ema = ewma(plus_dm, span=window)
                minus_dm_ema = ewma(minus_dm, span=window)
                
                # 计算+DI和-DI
                plus_di = 100 * plus_dm_ema / (tr_ema + 1e-10)
                minus_di = 100 * minus_dm_ema / (tr_ema + 1e-10)
                
                # 计算DX (方向指数)
                dx = 100 * torch.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
                
                # 计算ADX (DX的平滑移动平均)
                adx = ewma(dx, span=window)
                
                # 存储结果
                result_tensors[f'ADX_{window}'] = adx
        
        # 计算顺势指标
        if 'CCI' in indicators:
            for window in [14, 20]:
                # 计算典型价格 (TP)
                tp = (tensor_dict['high'] + tensor_dict['low'] + tensor_dict['close']) / 3
                
                # 计算TP的简单移动平均
                sma_tp = moving_average(tp, window)
                
                # 计算TP和SMA(TP)之间的平均偏差
                windows_tp = rolling_window(tp, window)
                mad = torch.mean(torch.abs(windows_tp - sma_tp.unsqueeze(1)), dim=1)
                
                # 填充缺失值
                mad_full = torch.full_like(tensor_dict['close'], float('nan'))
                mad_full[window-1:] = mad
                
                # 计算CCI
                cci = (tp - sma_tp) / (0.015 * mad_full + 1e-10)
                
                # 存储结果
                result_tensors[f'CCI_{window}'] = cci
        
        # 计算变动率
        if 'ROC' in indicators:
            for window in [10, 20]:
                # 计算n周期前的收盘价
                prev_close = torch.roll(tensor_dict['close'], shifts=window, dims=0)
                
                # 对于开始的n个值，使用第一个值
                prev_close[:window] = tensor_dict['close'][0]
                
                # 计算ROC (变动率)
                roc = 100 * (tensor_dict['close'] - prev_close) / (prev_close + 1e-10)
                
                # 存储结果
                result_tensors[f'ROC_{window}'] = roc
        
        # 计算能量潮指标
        if 'OBV' in indicators:
            # 计算价格变化方向
            close_diff = torch.diff(tensor_dict['close'], dim=0, prepend=tensor_dict['close'][0:1])
            direction = torch.sign(close_diff)
            
            # 第一个值设置为0
            direction[0] = 0
            
            # 计算OBV
            volume_direction = tensor_dict['volume'] * direction
            obv = torch.cumsum(volume_direction, dim=0)
            
            # 存储结果
            result_tensors['OBV'] = obv
        
        # 计算资金流量指标
        if 'MFI' in indicators:
            for window in [14, 20]:
                # 计算典型价格 (TP)
                tp = (tensor_dict['high'] + tensor_dict['low'] + tensor_dict['close']) / 3
                
                # 计算价格变化方向
                tp_diff = torch.diff(tp, dim=0, prepend=tp[0:1])
                
                # 计算正资金流和负资金流
                positive_flow = torch.where(tp_diff >= 0, tp * tensor_dict['volume'], torch.zeros_like(tp))
                negative_flow = torch.where(tp_diff < 0, tp * tensor_dict['volume'], torch.zeros_like(tp))
                
                # 计算n周期的正负资金流总和
                positive_flow_sum = moving_average(positive_flow, window) * window
                negative_flow_sum = moving_average(negative_flow, window) * window
                
                # 计算资金流量比率
                money_flow_ratio = positive_flow_sum / (negative_flow_sum + 1e-10)
                
                # 计算资金流量指数
                mfi = 100 - (100 / (1 + money_flow_ratio))
                
                # 处理边界值
                mfi = torch.clamp(mfi, 0, 100)
                
                # 存储结果
                result_tensors[f'MFI_{window}'] = mfi
        
        # 将计算结果添加到DataFrame
        for name, tensor in result_tensors.items():
            # 确保张量是2D的
            if tensor.dim() == 1:
                tensor = tensor.view(-1, 1)
            
            # 转换回NumPy数组
            if tensor.is_cuda:
                numpy_data = tensor.cpu().numpy()
            else:
                numpy_data = tensor.numpy()
            
            # 添加到结果DataFrame
            result_df[name] = numpy_data
        
        logger.info(f"已计算 {len(indicators)} 种技术指标")
        return result_df
    
    def calculate_price_features(self, df):
        """
        计算价格相关特征，PyTorch版
        
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
        
        # 将数据转换为PyTorch张量
        tensor_dict = {}
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                tensor, _ = df_to_tensor(df, [col])
                tensor_dict[col] = tensor
        
        # 存储计算结果
        result_tensors = {}
        
        # 计算价格变化百分比
        close = tensor_dict['close']
        prev_close = torch.roll(close, shifts=1, dims=0)
        prev_close[0] = close[0]  # 第一个值设置为本身
        
        price_change_pct = (close - prev_close) / (prev_close + 1e-10)
        result_tensors['price_change_pct'] = price_change_pct
        
        # 计算价格差异指标
        high = tensor_dict['high']
        low = tensor_dict['low']
        open_price = tensor_dict['open']
        
        price_gap = (high - low) / (low + 1e-10)
        result_tensors['price_gap'] = price_gap
        
        # 计算开盘-收盘差异
        open_close_diff = (close - open_price) / (open_price + 1e-10)
        result_tensors['open_close_diff'] = open_close_diff
        
        # 计算最高-最低差异
        high_low_diff = (high - low) / (low + 1e-10)
        result_tensors['high_low_diff'] = high_low_diff
        
        # 计算当前价格相对近期高低点的位置
        for window in [5, 10, 20, 50]:
            # 创建滚动窗口
            high_windows = rolling_window(high, window)
            low_windows = rolling_window(low, window)
            
            # 计算窗口内的最大值和最小值
            high_max = torch.max(high_windows, dim=1)[0]
            low_min = torch.min(low_windows, dim=1)[0]
            
            # 填充开头的NaN值
            high_max_full = torch.zeros_like(close)
            low_min_full = torch.zeros_like(close)
            
            # 将计算得到的值复制到完整张量中
            high_max_full[window-1:] = high_max
            low_min_full[window-1:] = low_min
            
            # 对于开头的值，使用阶段性数据
            for i in range(window-1):
                if i == 0:
                    high_max_full[i] = high[i]
                    low_min_full[i] = low[i]
                else:
                    high_max_full[i] = torch.max(high[:i+1])
                    low_min_full[i] = torch.min(low[:i+1])
            
            # 计算相对位置
            price_rel_high = close / (high_max_full + 1e-10)
            price_rel_low = close / (low_min_full + 1e-10)
            
            # 存储结果
            result_tensors[f'price_rel_high_{window}'] = price_rel_high
            result_tensors[f'price_rel_low_{window}'] = price_rel_low
        
        # 计算价格波动性 (20天标准差)
        windows = rolling_window(price_change_pct, 20)
        volatility = torch.std(windows, dim=1, unbiased=True)
        
        # 填充缺失值
        volatility_full = torch.zeros_like(close)
        volatility_full[19:] = volatility  # 20-1
        
        # 存储结果
        result_tensors['price_volatility'] = volatility_full
        
        # 将计算结果添加到DataFrame
        for name, tensor in result_tensors.items():
            # 确保张量是2D的
            if tensor.dim() == 1:
                tensor = tensor.view(-1, 1)
            
            # 转换回NumPy数组
            if tensor.is_cuda:
                numpy_data = tensor.cpu().numpy()
            else:
                numpy_data = tensor.numpy()
            
            # 添加到结果DataFrame
            result_df[name] = numpy_data.flatten()
        
        logger.info("已计算价格特征")
        return result_df
                
    def calculate_volume_features(self, df):
        """
        计算交易量相关特征，PyTorch版
        
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
        
        # 将数据转换为PyTorch张量
        tensor_dict = {}
        for col in ['high', 'low', 'close', 'volume']:
            if col in df.columns:
                tensor, _ = df_to_tensor(df, [col])
                tensor_dict[col] = tensor
        
        # 存储计算结果
        result_tensors = {}
        
        # 获取主要数据
        volume = tensor_dict['volume']
        close = tensor_dict['close']
        high = tensor_dict['high']
        low = tensor_dict['low']
        
        # 计算交易量变化百分比
        prev_volume = torch.roll(volume, shifts=1, dims=0)
        prev_volume[0] = volume[0]  # 第一个值设置为本身
        
        volume_change_pct = (volume - prev_volume) / (prev_volume + 1e-10)
        result_tensors['volume_change_pct'] = volume_change_pct
        
        # 计算相对交易量 (相对于近期平均值)
        for window in [5, 10, 20, 50]:
            volume_ma = moving_average(volume, window)
            rel_volume = volume / (volume_ma + 1e-10)
            
            # 存储结果
            result_tensors[f'rel_volume_{window}'] = rel_volume
        
        # 计算成交量加权平均价格 (VWAP)
        typical_price = (high + low + close) / 3
        weighted_price = typical_price * volume
        
        cum_weighted_price = torch.cumsum(weighted_price, dim=0)
        cum_volume = torch.cumsum(volume, dim=0)
        
        vwap = cum_weighted_price / (cum_volume + 1e-10)
        result_tensors['vwap'] = vwap
        
        # 计算价格和交易量的相关性
        for window in [10, 20]:
            price_volume_corr = correlation(close, volume, window)
            result_tensors[f'price_volume_corr_{window}'] = price_volume_corr
        
        # 计算交易量振荡器
        volume_ma5 = moving_average(volume, 5)
        volume_ma20 = moving_average(volume, 20)
        
        volume_oscillator = volume_ma5 / (volume_ma20 + 1e-10)
        result_tensors['volume_oscillator'] = volume_oscillator
        
        # 计算负量指标 (NVI)
        # 初始化为1
        nvi = torch.ones_like(volume)
        
        # 计算价格变化百分比
        close_pct_change = (close - torch.roll(close, shifts=1, dims=0)) / (torch.roll(close, shifts=1, dims=0) + 1e-10)
        close_pct_change[0] = 0  # 第一个值设置为0
        
        # 计算交易量变化
        volume_decrease = volume < torch.roll(volume, shifts=1, dims=0)
        volume_decrease[0] = False  # 第一个值设置为False
        
        # 当交易量减少时，NVI = 前一天NVI * (1 + 价格变化百分比)
        for i in range(1, len(nvi)):
            if volume_decrease[i]:
                nvi[i] = nvi[i-1] * (1 + close_pct_change[i])
            else:
                nvi[i] = nvi[i-1]
        
        result_tensors['nvi'] = nvi
        
        # 将计算结果添加到DataFrame
        for name, tensor in result_tensors.items():
            # 确保张量是2D的
            if tensor.dim() == 1:
                tensor = tensor.view(-1, 1)
            
            # 转换回NumPy数组
            if tensor.is_cuda:
                numpy_data = tensor.cpu().numpy()
            else:
                numpy_data = tensor.numpy()
            
            # 添加到结果DataFrame
            result_df[name] = numpy_data.flatten()
        
        logger.info("已计算交易量特征")
        return result_df
    
    def calculate_volatility_features(self, df):
        """
        计算波动性相关特征，PyTorch版
        
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
        
        # 将数据转换为PyTorch张量
        tensor_dict = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                tensor, _ = df_to_tensor(df, [col])
                tensor_dict[col] = tensor
        
        # 存储计算结果
        result_tensors = {}
        
        # 获取主要数据
        close = tensor_dict['close']
        high = tensor_dict['high']
        low = tensor_dict['low']
        
        # 计算日内波动性
        intraday_volatility = (high - low) / (close + 1e-10)
        result_tensors['intraday_volatility'] = intraday_volatility
        
        # 计算不同窗口的历史波动率
        returns = torch.log(close / torch.roll(close, shifts=1, dims=0))
        returns[0] = 0  # 第一个值设置为0
        
        for window in [5, 10, 20, 50]:
            # 计算标准差
            returns_windows = rolling_window(returns, window)
            std_dev = torch.std(returns_windows, dim=1, unbiased=True)
            
            # 填充缺失值
            std_dev_full = torch.zeros_like(close)
            std_dev_full[window-1:] = std_dev
            
            # 年化波动率 (假设一年252个交易日)
            annualized_vol = std_dev_full * torch.sqrt(torch.tensor(252.0, device=self.device))
            
            # 存储结果
            result_tensors[f'volatility_{window}d'] = annualized_vol
        
        # 计算真实范围
        high_low = high - low
        high_close_prev = torch.abs(high - torch.roll(close, shifts=1, dims=0))
        low_close_prev = torch.abs(low - torch.roll(close, shifts=1, dims=0))
        
        # 第一个值处理
        high_close_prev[0] = high_low[0]
        low_close_prev[0] = high_low[0]
        
        # 真实范围是三者的最大值
        tr = torch.maximum(high_low, torch.maximum(high_close_prev, low_close_prev))
        result_tensors['tr'] = tr
        
        # 计算不同窗口的ATR
        for window in [5, 14, 20]:
            atr = moving_average(tr, window)
            result_tensors[f'atr_{window}'] = atr
            
            # 计算相对ATR (ATR/收盘价)
            atr_pct = atr / (close + 1e-10)
            result_tensors[f'atr_pct_{window}'] = atr_pct
        
        # 计算布林带波动率指标
        for window in [20, 50]:
            # 计算中轨 (SMA)
            middle_band = moving_average(close, window)
            
            # 计算标准差
            close_windows = rolling_window(close, window)
            std_dev = torch.std(close_windows, dim=1, unbiased=True)
            
            # 填充缺失值
            std_dev_full = torch.zeros_like(close)
            std_dev_full[window-1:] = std_dev
            
            # 计算布林带宽度 (波动率指标)
            bb_width = 2 * std_dev_full / (middle_band + 1e-10)
            result_tensors[f'bb_width_{window}'] = bb_width
            
            # 计算收盘价在布林带中的位置 (0-100)
            upper_band = middle_band + 2 * std_dev_full
            lower_band = middle_band - 2 * std_dev_full
            
            bb_pos = (close - lower_band) / ((upper_band - lower_band) + 1e-10) * 100
            bb_pos = torch.clamp(bb_pos, 0, 100)
            result_tensors[f'bb_pos_{window}'] = bb_pos
        
        # 将计算结果添加到DataFrame
        for name, tensor in result_tensors.items():
            # 确保张量是2D的
            if tensor.dim() == 1:
                tensor = tensor.view(-1, 1)
            
            # 转换回NumPy数组
            if tensor.is_cuda:
                numpy_data = tensor.cpu().numpy()
            else:
                numpy_data = tensor.numpy()
            
            # 添加到结果DataFrame
            result_df[name] = numpy_data.flatten()
        
        logger.info("已计算波动性特征")
        return result_df
                
    def calculate_trend_features(self, df):
        """
        计算趋势相关特征，PyTorch版
        
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
        
        # 将数据转换为PyTorch张量
        tensor_dict = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                tensor, _ = df_to_tensor(df, [col])
                tensor_dict[col] = tensor
        
        # 存储计算结果
        result_tensors = {}
        close = tensor_dict['close']
        
        # 计算价格变化率
        price_change = (close - torch.roll(close, shifts=1, dims=0)) / (torch.roll(close, shifts=1, dims=0) + 1e-10)
        price_change[0] = 0  # 第一个值设置为0
        
        # 计算价格相对于不同周期移动平均线的位置
        for window in [10, 20, 50, 100, 200]:
            # 计算SMA
            sma = moving_average(close, window)
            
            # 价格相对于SMA的百分比
            price_rel_sma = close / (sma + 1e-10) - 1
            result_tensors[f'price_rel_sma_{window}'] = price_rel_sma
        
        # 计算移动平均线趋势方向和强度
        for window in [20, 50]:
            # 当前SMA
            sma = moving_average(close, window)
            
            # 计算SMA的变化率
            sma_change = (sma - torch.roll(sma, shifts=5, dims=0)) / (torch.roll(sma, shifts=5, dims=0) + 1e-10)
            sma_change[:5] = 0  # 前5个值设置为0
            
            # 计算SMA的方向 (上升/下降)
            sma_direction = torch.sign(sma_change)
            
            # 存储结果
            result_tensors[f'sma_{window}_change'] = sma_change
            result_tensors[f'sma_{window}_direction'] = sma_direction
        
        # 计算多周期EMA交叉信号
        fast_period = 20
        slow_period = 50
        
        # 计算快速和慢速EMA
        fast_ema = ewma(close, span=fast_period)
        slow_ema = ewma(close, span=slow_period)
        
        # 计算EMA差值和交叉信号
        ema_diff = fast_ema - slow_ema
        ema_cross = torch.sign(ema_diff)  # 1 表示快线在上，-1 表示慢线在上
        
        # 计算交叉变化 (交叉点)
        ema_cross_change = ema_cross - torch.roll(ema_cross, shifts=1, dims=0)
        ema_cross_change[0] = 0  # 第一个值设置为0
        
        # 存储结果
        result_tensors['ema_diff'] = ema_diff
        result_tensors['ema_cross'] = ema_cross
        result_tensors['ema_cross_change'] = ema_cross_change
        
        # 计算趋势强度指数 (TSI)
        tsi_tensor = self.tsi(close, r=25, s=13)
        result_tensors['tsi'] = tsi_tensor
        
        # 计算ADX趋势强度
        for window in [14]:
            # 计算方向移动 (+DM 和 -DM)
            high = tensor_dict['high']
            low = tensor_dict['low']
            
            high_diff = torch.diff(high, dim=0, prepend=high[0:1])
            low_diff = torch.diff(low, dim=0, prepend=low[0:1])
            
            # +DM: 如果high_diff > 0且high_diff > |low_diff|，则为high_diff，否则为0
            plus_dm = torch.where(
                (high_diff > 0) & (high_diff > torch.abs(low_diff)),
                high_diff,
                torch.zeros_like(high_diff)
            )
            
            # -DM: 如果low_diff < 0且|low_diff| > high_diff，则为|low_diff|，否则为0
            minus_dm = torch.where(
                (low_diff < 0) & (torch.abs(low_diff) > high_diff),
                torch.abs(low_diff),
                torch.zeros_like(low_diff)
            )
            
            # 计算ATR
            high_low = high - low
            high_close_prev = torch.abs(high - torch.roll(close, shifts=1, dims=0))
            low_close_prev = torch.abs(low - torch.roll(close, shifts=1, dims=0))
            high_close_prev[0] = 0
            low_close_prev[0] = 0
            tr = torch.maximum(high_low, torch.maximum(high_close_prev, low_close_prev))
            
            # 计算smoothed值
            tr_ema = ewma(tr, span=window)
            plus_dm_ema = ewma(plus_dm, span=window)
            minus_dm_ema = ewma(minus_dm, span=window)
            
            # 计算+DI和-DI
            plus_di = 100 * plus_dm_ema / (tr_ema + 1e-10)
            minus_di = 100 * minus_dm_ema / (tr_ema + 1e-10)
            
            # 计算DX (方向指数)
            dx = 100 * torch.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            
            # 计算ADX (DX的平滑移动平均)
            adx = ewma(dx, span=window)
            
            # 存储结果
            result_tensors[f'adx_{window}'] = adx
            result_tensors['plus_di'] = plus_di
            result_tensors['minus_di'] = minus_di
            
            # 计算趋势方向 (ADX > 25 且 +DI > -DI 为强劲上升趋势，ADX > 25 且 -DI > +DI 为强劲下降趋势)
            trend_strength = torch.where(adx > 25, torch.ones_like(adx), torch.zeros_like(adx))
            trend_direction = torch.sign(plus_di - minus_di)
            
            strong_trend = trend_strength * trend_direction
            result_tensors['trend_strength'] = trend_strength
            result_tensors['trend_direction'] = trend_direction
            result_tensors['strong_trend'] = strong_trend
        
        # 将计算结果添加到DataFrame
        for name, tensor in result_tensors.items():
            # 确保张量是2D的
            if tensor.dim() == 1:
                tensor = tensor.view(-1, 1)
            
            # 转换回NumPy数组
            if tensor.is_cuda:
                numpy_data = tensor.cpu().numpy()
            else:
                numpy_data = tensor.numpy()
            
            # 添加到结果DataFrame
            result_df[name] = numpy_data.flatten()
        
        logger.info("已计算趋势特征")
        return result_df
                
    def calculate_momentum_features(self, df):
        """
        计算动量相关特征，PyTorch版
        
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
        
        # 将数据转换为PyTorch张量
        tensor_dict = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                tensor, _ = df_to_tensor(df, [col])
                tensor_dict[col] = tensor
        
        # 存储计算结果
        result_tensors = {}
        close = tensor_dict['close']
        
        # 计算不同周期的价格变化率 (ROC)
        for period in [1, 3, 5, 10, 20, 60]:
            # 计算n周期前的收盘价
            prev_close = torch.roll(close, shifts=period, dims=0)
            
            # 对于开始的n个值，使用第一个值
            prev_close[:period] = close[0]
            
            # 计算ROC (变动率)
            roc = (close - prev_close) / (prev_close + 1e-10)
            
            # 存储结果
            result_tensors[f'roc_{period}'] = roc
        
        # 计算RSI动量指标
        for window in [6, 14]:
            # 计算价格变化
            price_diff = torch.diff(close, dim=0, prepend=close[0:1])
            
            # 分离上涨和下跌
            up = torch.maximum(price_diff, torch.zeros_like(price_diff))
            down = torch.abs(torch.minimum(price_diff, torch.zeros_like(price_diff)))
            
            # 计算平均上涨和下跌
            avg_up = moving_average(up, window)
            avg_down = moving_average(down, window)
                        
            # 计算相对强度
            rs = avg_up / (avg_down + 1e-10)
                        
            # 计算RSI
            rsi = 100 - (100 / (1 + rs))
                        
            # 处理边界情况
            rsi = torch.clamp(rsi, 0, 100)
            
            # 存储结果
            result_tensors[f'rsi_{window}'] = rsi
        
        # 计算MACD动量指标
        fast_period = 12
        slow_period = 26
        signal_period = 9
        
        # 计算EMA
        fast_ema = ewma(close, span=fast_period)
        slow_ema = ewma(close, span=slow_period)
        
        # 计算MACD线
        macd_line = fast_ema - slow_ema
        
        # 计算信号线
        signal_line = ewma(macd_line, span=signal_period)
        
        # 计算MACD直方图
        histogram = macd_line - signal_line
        
        # 计算MACD变化率及其趋势
        histogram_change = histogram - torch.roll(histogram, shifts=1, dims=0)
        histogram_change[0] = 0  # 第一个值设置为0
        
        histogram_direction = torch.sign(histogram)
        histogram_trend = torch.sign(histogram_change)
        
        # 存储结果
        result_tensors['macd_line'] = macd_line
        result_tensors['macd_signal'] = signal_line
        result_tensors['macd_hist'] = histogram
        result_tensors['macd_hist_change'] = histogram_change
        result_tensors['macd_hist_direction'] = histogram_direction
        result_tensors['macd_hist_trend'] = histogram_trend
        
        # 计算随机振荡器
        window_k = 14
        window_d = 3
        
        # 获取滚动窗口的最高价和最低价
        high = tensor_dict['high']
        low = tensor_dict['low']
        windows_high = rolling_window(high, window_k)
        windows_low = rolling_window(low, window_k)
        
        # 计算最大值和最小值
        highest_high = torch.max(windows_high, dim=1)[0]
        lowest_low = torch.min(windows_low, dim=1)[0]
        
        # 填充缺失值
        highest_high_full = torch.zeros_like(close)
        lowest_low_full = torch.zeros_like(close)
        
        highest_high_full[window_k-1:] = highest_high
        lowest_low_full[window_k-1:] = lowest_low
        
        # 对于开头的数据，计算阶段性结果
        for i in range(window_k-1):
            highest_high_full[i] = torch.max(high[:i+1])
            lowest_low_full[i] = torch.min(low[:i+1])
        
        # 计算随机值 %K
        stoch_k = 100 * (close - lowest_low_full) / (highest_high_full - lowest_low_full + 1e-10)
        
        # 计算 %D (K的移动平均)
        stoch_d = moving_average(stoch_k, window_d)
        
        # 处理边界值
        stoch_k = torch.clamp(stoch_k, 0, 100)
        stoch_d = torch.clamp(stoch_d, 0, 100)
        
        # 计算随机振荡器金叉、死叉
        stoch_cross = stoch_k - stoch_d
        stoch_cross_direction = torch.sign(stoch_cross)
        
        # 存储结果
        result_tensors['stoch_k'] = stoch_k
        result_tensors['stoch_d'] = stoch_d
        result_tensors['stoch_cross'] = stoch_cross
        result_tensors['stoch_cross_direction'] = stoch_cross_direction
        
        # 将计算结果添加到DataFrame
        for name, tensor in result_tensors.items():
            # 确保张量是2D的
            if tensor.dim() == 1:
                tensor = tensor.view(-1, 1)
            
            # 转换回NumPy数组
            if tensor.is_cuda:
                numpy_data = tensor.cpu().numpy()
            else:
                numpy_data = tensor.numpy()
            
            # 添加到结果DataFrame
            result_df[name] = numpy_data.flatten()
        
        logger.info("已计算动量特征")
        return result_df


# 为了与原始TechnicalIndicators类兼容，创建别名类
class PyTorchCompatibleTechnicalIndicators(PyTorchTechnicalIndicators):
    """与原始TechnicalIndicators接口兼容的PyTorch实现"""
    pass 