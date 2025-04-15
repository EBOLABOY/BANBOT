"""
基于PyTorch的技术指标计算模块 - 使用GPU加速
"""

import torch
import pandas as pd
import numpy as np
import logging
import time # For timing

from src.utils.logger import get_logger
# Ensure torch_utils exists and functions are correctly defined
from src.features.torch_utils import get_device, df_to_tensor, tensor_to_df, moving_average, ewma, correlation, rolling_window

logger = get_logger(__name__)

def ensure_2d(tensor):
    """确保张量为二维[N, 1]"""
    if tensor.dim() == 1:
        return tensor.unsqueeze(1)
    return tensor

def match_shape(a, b):
    # 兼容保留，实际全程用[N, 1]后很少用到
    a = ensure_2d(a)
    b = ensure_2d(b)
    if a.shape == b.shape:
        return a
    if a.shape[0] == b.shape[0] and a.shape[1] == 1 and b.shape[1] == 1:
        return a
    raise ValueError(f"match_shape: 不能匹配形状 {a.shape} 和 {b.shape}")

class PyTorchTechnicalIndicators:
    """
    使用PyTorch实现的技术指标计算类，使用CPU计算
    Includes both DataFrame-based methods (for compatibility/ease of use)
    and tensor-based methods (for optimized data flow).
    """
    
    def __init__(self):
        """
        初始化技术指标计算器
        """
        self.device = get_device()  # 已修改为只返回CPU
        logger.info(f"PyTorch技术指标计算器初始化，使用CPU设备")
        self.default_windows = [5, 10, 20, 30, 50] # 添加默认窗口大小列表

    # --- Helper Functions ---

    def _add_results_to_df(self, result_df, result_tensors):
        """ Helper function to add computed tensors back to DataFrame """
        if not isinstance(result_df, pd.DataFrame):
            logger.error("无效的 result_df 传递给 _add_results_to_df")
            return pd.DataFrame() # Return empty df

        if not isinstance(result_tensors, dict):
            logger.error("无效的 result_tensors 传递给 _add_results_to_df")
            return result_df # Return original df

        for name, tensor in result_tensors.items():
            if not isinstance(tensor, torch.Tensor):
                logger.warning(f"跳过非张量结果: {name}")
                continue

            # Ensure tensor is 1D/2D for conversion
            if tensor.dim() > 2:
                logger.warning(f"张量 {name} 维度 > 2，无法添加到 DataFrame，跳过。")
                continue
            if tensor.dim() == 2 and tensor.shape[1] != 1:
                logger.warning(f"张量 {name} 是多列 2D 张量，将展平为 1D，请检查是否符合预期。")
                tensor = tensor.flatten() # Flatten if it's multi-column 2D
            elif tensor.dim() == 2:
                 tensor = tensor.squeeze() # Make 1D if it's [N, 1]

            try:
                numpy_data = tensor.numpy()

                # Handle potential length mismatch if df index is used
                target_index = result_df.index
                if len(numpy_data) == len(target_index):
                     result_df[name] = pd.Series(numpy_data, index=target_index)
                elif len(numpy_data) < len(target_index):
                     # Pad with NaNs if tensor is shorter (e.g., due to diff)
                     padded_data = np.full(len(target_index), np.nan)
                     # Align based on assumption calculation produces result for end of sequence
                     padded_data[-len(numpy_data):] = numpy_data
                     result_df[name] = pd.Series(padded_data, index=target_index)
                else:
                     # Truncate if tensor is longer (less common)
                     result_df[name] = pd.Series(numpy_data[:len(target_index)], index=target_index)
            except Exception as e:
                 logger.error(f"将张量 {name} 添加到 DataFrame 时出错: {e}")

        return result_df

    # --- Static Methods (like TSI) ---
    @staticmethod
    def tsi(close_tensor, r=25, s=13):
        """
        计算趋势强度指数 (TSI)，PyTorch版本

        参数:
            close_tensor: 收盘价张量 (1D or [N,1])
            r: 长周期
            s: 短周期

        返回:
            TSI值张量 (same shape as input)
        """
        if not isinstance(close_tensor, torch.Tensor):
             close_tensor = torch.tensor(close_tensor, dtype=torch.float32, device=get_device())
        if close_tensor.dim() == 1:
            close_tensor = close_tensor.unsqueeze(1) # Ensure 2D for ewma/diff
        if close_tensor.shape[0] == 0: # Handle empty tensor
            return torch.empty_like(close_tensor)


        # Calculate momentum (price change)
        # Prepend with first value to keep same dimension after diff
        m = torch.diff(close_tensor, dim=0, prepend=close_tensor[0:1])

        # Calculate EMA of momentum
        m1 = ewma(m, span=r, adjust=False)
        m2 = ewma(m1, span=s, adjust=False)

        # Calculate EMA of absolute momentum
        a = torch.abs(m)
        a1 = ewma(a, span=r, adjust=False)
        a2 = ewma(a1, span=s, adjust=False)

        # Calculate TSI, add epsilon to denominator
        tsi = 100 * m2 / (a2 + 1e-10)

        # Handle NaN/Inf resulting from calculation or division by near-zero
        tsi = torch.where(torch.isnan(tsi) | torch.isinf(tsi), torch.zeros_like(tsi), tsi)

        # Return in the original shape (if input was 1D)
        if close_tensor.shape[1] == 1 and tsi.shape[1] == 1:
            return tsi.squeeze(1)
        return tsi


    # --- Tensor-based Calculation Methods ---

    def calculate_price_features_tensor(self, tensor_dict):
        """
        计算价格相关特征，纯张量操作

        参数:
            tensor_dict: 包含输入张量的字典 ('open', 'high', 'low', 'close')

        返回:
            包含价格特征张量的字典
        """
        required_keys = ['open', 'high', 'low', 'close']
        if not all(key in tensor_dict for key in required_keys):
            logger.warning("价格特征张量计算缺少必要的输入张量")
            return {}

        close = ensure_2d(tensor_dict['close'])
        high = ensure_2d(tensor_dict['high'])
        low = ensure_2d(tensor_dict['low'])
        open_price = ensure_2d(tensor_dict['open'])

        if not all(t.shape[0] == close.shape[0] for t in [high, low, open_price]):
            logger.error("价格特征输入张量长度不匹配")
            return {}
        if close.shape[0] == 0: return {} # Handle empty

        result_tensors = {}

        # Price change percentage
        prev_close = torch.roll(close, shifts=1, dims=0)
        prev_close[0] = float('nan')
        price_change_pct = (close - prev_close) / (prev_close + 1e-10)
        result_tensors['price_change_pct'] = price_change_pct

        # Price gap (High-Low Range / Low)
        price_gap = (high - low) / (low + 1e-10)
        result_tensors['price_gap'] = price_gap
        
        # Open-Close difference / Open
        open_close_diff = (close - open_price) / (open_price + 1e-10)
        result_tensors['open_close_diff'] = open_close_diff
        
        # High-Low difference / Low (same as price_gap, maybe intended diff calculation?)
        high_low_diff = (high - low) / (low + 1e-10)
        result_tensors['high_low_diff'] = high_low_diff
        
        # Price relative to recent high/low
        for window in [5, 10, 20, 50]:
            if window > 0 and window <= len(close):
                high_windows = rolling_window(high, window)
                low_windows = rolling_window(low, window)
                # 降维，rolling_window后立即 squeeze(1)
                if high_windows.dim() == 3 and high_windows.shape[1] == 1:
                    high_windows = high_windows.squeeze(1)
                if low_windows.dim() == 3 and low_windows.shape[1] == 1:
                    low_windows = low_windows.squeeze(1)
                if high_windows.shape[0] == 0 or low_windows.shape[0] == 0:
                    logger.debug(f"价格相对位置窗口大小 {window} 无效或大于数据长度")
                    continue
                high_max = torch.max(high_windows, dim=1, keepdim=True)[0]
                low_min = torch.min(low_windows, dim=1, keepdim=True)[0]
                high_max_full = torch.full_like(close, float('nan'))
                low_min_full = torch.full_like(close, float('nan'))
                high_max_full[window-1:] = high_max
                low_min_full[window-1:] = low_min
                for i in range(window-1):
                    if i+1 <= len(high):
                        high_max_full[i] = torch.max(high[:i+1])
                        low_min_full[i] = torch.min(low[:i+1])
                price_rel_high = close / (high_max_full + 1e-10)
                price_rel_low = close / (low_min_full + 1e-10)
                result_tensors[f'price_rel_high_{window}'] = price_rel_high
                result_tensors[f'price_rel_low_{window}'] = price_rel_low

        # Price volatility (rolling std dev of pct change)
        volatility_window = 20
        if volatility_window > 0 and volatility_window <= len(price_change_pct):
            price_change_pct_nonan = torch.where(torch.isnan(price_change_pct), torch.zeros_like(price_change_pct), price_change_pct)
            windows = rolling_window(price_change_pct_nonan, volatility_window)
            # 降维，rolling_window后立即 squeeze(1)
            if windows.dim() == 3 and windows.shape[1] == 1:
                windows = windows.squeeze(1)
            if windows.shape[0] > 0:
                volatility = torch.std(windows, dim=1, unbiased=True, keepdim=True)
                volatility_full = torch.full_like(close, float('nan'))
                volatility_full[volatility_window-1:] = volatility
                result_tensors['price_volatility'] = volatility_full
            else:
                logger.debug(f"无法为波动率生成滚动窗口 {volatility_window}")
        else:
            logger.debug(f"价格波动率窗口大小 {volatility_window} 无效或大于数据长度")

        return result_tensors

    def calculate_volume_features_tensor(self, tensor_dict):
        """
        计算交易量相关特征，纯张量操作

        参数:
            tensor_dict: 包含输入张量的字典 ('high', 'low', 'close', 'volume')

        返回:
            包含交易量特征张量的字典
        """
        required_keys = ['high', 'low', 'close', 'volume']
        if not all(key in tensor_dict for key in required_keys):
            logger.warning("交易量特征张量计算缺少必要的输入张量")
            return {}

        volume = ensure_2d(tensor_dict['volume'])
        close = ensure_2d(tensor_dict['close'])
        high = ensure_2d(tensor_dict['high'])
        low = ensure_2d(tensor_dict['low'])

        if not all(t.shape[0] == volume.shape[0] for t in [close, high, low]):
            logger.error("交易量特征输入张量长度不匹配")
            return {}
        if volume.shape[0] == 0: return {} # Handle empty

        result_tensors = {}

        # Volume change percentage
        prev_volume = torch.roll(volume, shifts=1, dims=0)
        prev_volume[0] = float('nan')
        volume_change_pct = (volume - prev_volume) / (prev_volume + 1e-10)
        result_tensors['volume_change_pct'] = volume_change_pct

        # Relative volume (compared to recent average)
        for window in [5, 10, 20, 50]:
            if window > 0 and window <= len(volume):
                volume_ma = moving_average(volume, window)
                # 降维，moving_average后立即 squeeze(1)
                if volume_ma.dim() == 2 and volume_ma.shape[1] == 1:
                    rel_volume = volume / (volume_ma + 1e-10)
                else:
                    rel_volume = volume / (volume_ma.squeeze(1) + 1e-10)
                result_tensors[f'rel_volume_{window}'] = rel_volume
            else:
                logger.debug(f"相对交易量窗口大小 {window} 无效或大于数据长度")

        # Volume Weighted Average Price (VWAP) - Cumulative for the batch
        typical_price = (high + low + close) / 3
        weighted_price = typical_price * volume
        cum_weighted_price = torch.cumsum(weighted_price, dim=0)
        cum_volume = torch.cumsum(volume, dim=0)
        vwap = cum_weighted_price / (cum_volume + 1e-10)
        result_tensors['vwap'] = vwap

        # Price-volume correlation
        for window in [10, 20]:
            if window > 0 and window <= len(close):
                price_volume_corr = correlation(close, volume, window)
                result_tensors[f'price_volume_corr_{window}'] = price_volume_corr
            else:
                 logger.debug(f"价格-交易量相关性窗口大小 {window} 无效或大于数据长度")

        # Volume oscillator
        short_window = 5
        long_window = 20
        if long_window > 0 and long_window <= len(volume) and short_window > 0:
             volume_ma_short = moving_average(volume, short_window)
             volume_ma_long = moving_average(volume, long_window)
             volume_oscillator = volume_ma_short / (volume_ma_long + 1e-10)
             result_tensors['volume_oscillator'] = volume_oscillator
        else:
             logger.debug("交易量振荡器窗口大小无效")

        # Negative Volume Index (NVI)
        nvi = torch.ones_like(volume) * 1000 # Initialize
        # Recalculate close_pct_change if not available
        if 'price_change_pct' in tensor_dict: # Reuse if possible
            close_pct_change = tensor_dict['price_change_pct']
        else:
            prev_close_nvi = torch.roll(close, shifts=1, dims=0); prev_close_nvi[0]=float('nan')
            close_pct_change = (close - prev_close_nvi) / (prev_close_nvi + 1e-10)
            close_pct_change[0] = 0 # Override first NaN with 0 for NVI calc

        volume_decrease = volume < torch.roll(volume, shifts=1, dims=0)
        volume_decrease[0] = False # First value doesn't decrease

        # Iterative calculation (can be slow for long series)
        # Consider optimized implementation if performance critical
        for i in range(1, len(nvi)):
            if volume_decrease[i]:
                # Handle potential NaN in close_pct_change for the first few points
                change_factor = (1 + close_pct_change[i]) if not torch.isnan(close_pct_change[i]) else 1.0
                nvi[i] = nvi[i-1] * change_factor
            else:
                nvi[i] = nvi[i-1]
        result_tensors['nvi'] = nvi

        return result_tensors

    def calculate_volatility_features_tensor(self, tensor_dict, windows=None):
        """
        计算基于波动率的技术指标

        参数:
            tensor_dict: 包含'close'和可选的'high'和'low'价格的张量字典
            windows: 窗口大小列表，如果为None则使用默认窗口大小

        返回:
            包含波动率特征的张量字典
        """
        try:
            if not isinstance(tensor_dict, dict):
                raise ValueError("tensor_dict必须是字典类型")

            if 'close' not in tensor_dict:
                raise ValueError("tensor_dict必须包含'close'键")

            close = ensure_2d(tensor_dict['close'])
            
            # 确保张量是2D的 [batch_size, features]
            if close.dim() == 1:
                close = close.unsqueeze(1)
            
            windows = windows or self.default_windows
            result_tensors = {}

            # 对每个窗口计算标准差
            for window in windows:
                if window <= 1:
                    continue
                    
                # 计算滚动标准差
                rolling_std = self._calculate_rolling_std(close, window)
                if rolling_std is not None:
                    col_name = f'std_{window}'
                    result_tensors[col_name] = rolling_std
                
                # 计算滚动均值
                rolling_mean = self._calculate_rolling_mean(close, window)
                if rolling_mean is not None and rolling_std is not None:
                    # 计算变异系数 (CV) = std/mean
                    # 使用非零掩码避免除零错误
                    non_zero_mask = rolling_mean != 0
                    cv = torch.zeros_like(rolling_mean)
                    cv[non_zero_mask] = rolling_std[non_zero_mask] / rolling_mean[non_zero_mask]
                    
                    col_name = f'cv_{window}'
                    result_tensors[col_name] = cv
                    
                    # 计算标准化波动率 = std/price
                    logger.debug(f"Volatility: Before match_shape for norm_vol - close shape: {close.shape}, rolling_std shape: {rolling_std.shape}")
                    close_expanded = match_shape(close, rolling_std)
                    non_zero_mask = close_expanded != 0
                    norm_vol = torch.zeros_like(rolling_std)
                    norm_vol[non_zero_mask] = rolling_std[non_zero_mask] / close_expanded[non_zero_mask]
                    
                    col_name = f'norm_vol_{window}'
                    result_tensors[col_name] = norm_vol

            # 计算ATR (如果高低价格可用)
            if 'high' in tensor_dict and 'low' in tensor_dict:
                high = ensure_2d(tensor_dict['high'])
                low = ensure_2d(tensor_dict['low'])
                
                # 确保张量是2D的
                if high.dim() == 1:
                    high = high.unsqueeze(1)
                if low.dim() == 1:
                    low = low.unsqueeze(1)
                
                # 计算真实范围 (TR)
                close_shift = torch.cat([torch.full((1, close.size(1)), float('nan'), device=close.device), close[:-1]], dim=0)
                
                # TR = max(high - low, |high - close_prev|, |low - close_prev|)
                tr1 = high - low  # 当前高低价差
                
                # 创建掩码过滤无效数据
                valid_mask = ~torch.isnan(close_shift)
                tr2 = torch.abs(high - close_shift)
                tr3 = torch.abs(low - close_shift)
                
                # 计算TR
                tr = torch.maximum(tr1, torch.maximum(tr2, tr3))
                
                # 对每个窗口计算ATR
                for window in windows:
                    if window <= 1:
                        continue
                    
                    atr = self._calculate_rolling_mean(tr, window)
                    if atr is not None:
                        col_name = f'atr_{window}'
                        result_tensors[col_name] = atr
                        
                        # 计算标准化ATR
                        logger.debug(f"Volatility: Before match_shape for norm_atr - close shape: {close.shape}, atr shape: {atr.shape}")
                        non_zero_mask = close != 0
                        close_expanded = match_shape(close, atr)
                        mask = non_zero_mask & ~torch.isnan(atr)
                        norm_atr = torch.zeros_like(atr)
                        norm_atr[mask] = atr[mask] / close_expanded[mask]
                        
                        col_name = f'norm_atr_{window}'
                        result_tensors[col_name] = norm_atr

            return result_tensors

        except Exception as e:
            logger.error(f"计算波动率特征时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def calculate_trend_features_tensor(self, tensor_dict, windows=None):
        """
        计算趋势相关特征

        参数:
            tensor_dict: 包含'close'价格的张量字典
            windows: 窗口大小列表，如果为None则使用默认窗口大小

        返回:
            包含趋势特征的张量字典
        """
        try:
            if not isinstance(tensor_dict, dict):
                raise ValueError("tensor_dict必须是字典类型")

            if 'close' not in tensor_dict:
                raise ValueError("tensor_dict必须包含'close'键")

            close = ensure_2d(tensor_dict['close'])
            
            # 确保张量是2D的 [batch_size, features]
            if close.dim() == 1:
                close = close.unsqueeze(1)
            
            windows = windows or self.default_windows
            result_tensors = {}

            # 计算移动平均线
            for window in windows:
                if window <= 1:
                    continue
                
                # 计算简单移动平均线 (SMA)
                sma = self._calculate_rolling_mean(close, window)
                if sma is not None:
                    col_name = f'sma_{window}'
                    result_tensors[col_name] = sma
                    
                    # 计算价格相对于移动平均线的百分比变化
                    logger.debug(f"Trend: Before match_shape for close_to_ma - close shape: {close.shape}, sma shape: {sma.shape}")
                    non_zero_mask = sma != 0
                    close_expanded = match_shape(close, sma)
                    close_to_ma = torch.zeros_like(sma)
                    close_to_ma[non_zero_mask] = (close_expanded[non_zero_mask] / sma[non_zero_mask]) - 1.0
                    
                    col_name = f'close_to_ma_{window}'
                    result_tensors[col_name] = close_to_ma
                    
                    # 计算移动平均线的斜率 (变化率)
                    if sma.size(0) > 1:
                        ma_slope = torch.zeros_like(sma)
                        ma_slope[1:] = (sma[1:] - sma[:-1]) / sma[:-1].clamp(min=1e-8)
                        ma_slope[0] = 0  # 第一个值无法计算斜率
                        
                        col_name = f'ma_slope_{window}'
                        result_tensors[col_name] = ma_slope
            
            # 计算价格动量
            for window in windows:
                if window <= 1 or close.size(0) <= window:
                    continue
                
                # 计算过去n期的价格变化率
                shifted_close = torch.cat([torch.full((window, close.size(1)), float('nan'), device=close.device), close[:-window]], dim=0)
                # 降维，shifted_close后立即 squeeze(1)
                if shifted_close.dim() == 3 and shifted_close.shape[1] == 1:
                    shifted_close = shifted_close.squeeze(1)
                non_zero_mask = shifted_close != 0
                momentum = torch.zeros_like(close)
                momentum[non_zero_mask] = (close[non_zero_mask] / shifted_close[non_zero_mask]) - 1.0
                
                col_name = f'momentum_{window}'
                result_tensors[col_name] = momentum
                
                # 计算历史最高价和最低价
                if window > 1:
                    rolling_max = self._calculate_rolling_max(close, window)
                    rolling_min = self._calculate_rolling_min(close, window)
                    
                    if rolling_max is not None and rolling_min is not None:
                        # 计算价格位置百分比 (PPO) = (close - min) / (max - min)
                        price_range = rolling_max - rolling_min
                        non_zero_range = price_range > 0
                        ppo = torch.zeros_like(close)
                        logger.debug(f"Trend: Before match_shape for ppo - close shape: {close.shape}, rolling_min shape: {rolling_min.shape}")
                        close_expanded = match_shape(close, rolling_min)
                        logger.debug(f"Trend: Before match_shape for ppo - rolling_min shape: {rolling_min.shape}, close shape: {close.shape}")
                        min_expanded = match_shape(rolling_min, close)
                        ppo[non_zero_range] = (close_expanded[non_zero_range] - min_expanded[non_zero_range]) / price_range[non_zero_range]
                        
                        col_name = f'ppo_{window}'
                        result_tensors[col_name] = ppo
            
            return result_tensors
            
        except Exception as e:
            logger.error(f"计算趋势特征时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def calculate_momentum_features_tensor(self, tensor_dict, windows=None):
        """
        计算动量相关特征，纯张量操作
        
        参数:
            tensor_dict: 包含输入张量的字典 ('close')
            windows: 可选，计算特征的窗口大小列表，默认为[5, 10, 20, 30]
            
        返回:
            包含动量特征张量的字典
        """
        result_tensors = {}
        
        # 设置默认窗口
        if windows is None:
            windows = [5, 10, 20, 30]
            
        # 检查所需字段
        if 'close' not in tensor_dict:
            logger.warning("计算动量特征缺少 'close' 键")
            return result_tensors
            
        try:
            # 获取收盘价并确保张量格式正确
            close = ensure_2d(tensor_dict['close'])
            
            # 计算RSI
            try:
                for window in windows:
                    # 计算价格变化
                    price_diff = close - close.roll(1, 0)
                    # 将第一个差值设为0
                    price_diff[0] = 0
                    
                    # 分离正负变化
                    gains = torch.where(price_diff > 0, price_diff, torch.zeros_like(price_diff))
                    losses = torch.abs(torch.where(price_diff < 0, price_diff, torch.zeros_like(price_diff)))
                    
                    # 计算平均增益和平均损失
                    avg_gain = self._calculate_rolling_mean(gains, window)
                    avg_loss = self._calculate_rolling_mean(losses, window)
                    
                    # 计算相对强度和RSI
                    if avg_gain is not None and avg_loss is not None:
                        rs = torch.where(
                            avg_loss > 1e-10,
                            avg_gain / avg_loss,
                            torch.ones_like(avg_gain) * 100
                        )
                        rsi = 100 - (100 / (1 + rs))
                        result_tensors[f'rsi_{window}'] = rsi
            except Exception as e:
                logger.error(f"计算RSI时发生错误: {str(e)}")
            
            # MACD指标
            try:
                # MACD参数
                fast = 12
                slow = 26
                signal = 9
                
                # 计算指数移动平均
                ema_fast = self._calculate_ewma(close, span=fast)
                ema_slow = self._calculate_ewma(close, span=slow)
                
                if ema_fast is not None and ema_slow is not None:
                    # 计算MACD线
                    macd_line = ema_fast - ema_slow
                    # 计算信号线
                    signal_line = self._calculate_ewma(macd_line, span=signal)
                    
                    if signal_line is not None:
                        # 计算直方图
                        histogram = macd_line - signal_line
                        
                        result_tensors['macd_line'] = macd_line
                        result_tensors['macd_signal'] = signal_line
                        result_tensors['macd_histogram'] = histogram
            except Exception as e:
                logger.error(f"计算MACD时发生错误: {str(e)}")
                
            # 动量震荡器
            try:
                for window in windows:
                    momentum = close - close.roll(window, 0)
                    normalized_momentum = momentum / close.roll(window, 0)
                    result_tensors[f'momentum_oscillator_{window}'] = normalized_momentum
            except Exception as e:
                logger.error(f"计算动量震荡器时发生错误: {str(e)}")
                
            # Williams %R
            try:
                for window in windows:
                    high = self._calculate_rolling_max(close, window)
                    low = self._calculate_rolling_min(close, window)
                    
                    if high is not None and low is not None:
                        denom = high - low
                        williams_r = torch.where(
                            denom > 1e-10,
                            -100 * ((high - close) / denom),
                            torch.zeros_like(denom)
                        )
                        result_tensors[f'williams_r_{window}'] = williams_r
            except Exception as e:
                logger.error(f"计算Williams %R时发生错误: {str(e)}")
                
        except Exception as e:
            logger.error(f"计算动量特征时发生错误: {str(e)}")
            
        return result_tensors

    def _calculate_rolling_std(self, tensor, window):
        """
        计算张量的滚动标准差
        
        参数:
            tensor: 输入张量 [batch_size, features]
            window: 窗口大小
            
        返回:
            滚动标准差张量，与输入形状相同
        """
        try:
            # 确保输入是2D的
            original_dim = tensor.dim()
            if original_dim == 1:
                tensor = tensor.unsqueeze(1)
            
            batch_size, features = tensor.shape
            
            # 使用unfold操作获取滚动窗口
            # unfold 返回 [N-window_size+1, C, window_size]
            if batch_size >= window:
                windows = tensor.unfold(0, window, 1)
                
                # 计算每个窗口的标准差，对窗口维度 (dim=2) 计算
                # 结果形状为 [batch_size-window+1, features]
                stds = torch.std(windows, dim=2)
                
                # 为前window-1个元素填充NaN值
                padding = torch.full((window-1, features), float('nan'), 
                                    device=self.device, dtype=tensor.dtype)  # 使用CPU
                result = torch.cat([padding, stds], dim=0)
                
                # 恢复原始维度
                if original_dim == 1:
                    result = result.squeeze(1)
                
                return result
            else:
                logger.warning(f"滚动标准差计算窗口大小 {window} 大于批次大小 {batch_size}")
                return None
        except Exception as e:
            logger.error(f"计算滚动标准差时出错: {str(e)}")
            return None
            
    def _calculate_rolling_max(self, tensor, window):
        """
        计算张量的滚动最大值
        
        参数:
            tensor: 输入张量 [batch_size, features]
            window: 窗口大小
            
        返回:
            滚动最大值张量，与输入形状相同
        """
        try:
            # 确保输入是2D的
            original_dim = tensor.dim()
            if original_dim == 1:
                tensor = tensor.unsqueeze(1)
                
            batch_size, features = tensor.shape
            
            if batch_size >= window:
                # 使用unfold操作获取滚动窗口
                # unfold 返回 [N-window_size+1, C, window_size]
                windows = tensor.unfold(0, window, 1)
                
                # 计算每个窗口的最大值，对窗口维度 (dim=2) 计算
                max_values = torch.max(windows, dim=2)[0]
                
                # 为前window-1个元素填充NaN值
                padding = torch.full((window-1, features), float('nan'), 
                                    device=self.device, dtype=tensor.dtype)  # 使用CPU
                result = torch.cat([padding, max_values], dim=0)
                
                # 恢复原始维度
                if original_dim == 1:
                    result = result.squeeze(1)
                
                return result
            else:
                logger.warning(f"滚动最大值计算窗口大小 {window} 大于批次大小 {batch_size}")
                return None
        except Exception as e:
            logger.error(f"计算滚动最大值时出错: {str(e)}")
            return None
            
    def _calculate_rolling_min(self, tensor, window):
        """
        计算张量的滚动最小值
        
        参数:
            tensor: 输入张量 [batch_size, features]
            window: 窗口大小
            
        返回:
            滚动最小值张量，与输入形状相同
        """
        try:
            # 确保输入是2D的
            original_dim = tensor.dim()
            if original_dim == 1:
                tensor = tensor.unsqueeze(1)
                
            batch_size, features = tensor.shape
            
            if batch_size >= window:
                # 使用unfold操作获取滚动窗口
                # unfold 返回 [N-window_size+1, C, window_size]
                windows = tensor.unfold(0, window, 1)
                
                # 计算每个窗口的最小值，对窗口维度 (dim=2) 计算
                min_values = torch.min(windows, dim=2)[0]
                
                # 为前window-1个元素填充NaN值
                padding = torch.full((window-1, features), float('nan'), 
                                    device=self.device, dtype=tensor.dtype)  # 使用CPU
                result = torch.cat([padding, min_values], dim=0)
                
                # 恢复原始维度
                if original_dim == 1:
                    result = result.squeeze(1)
                
                return result
            else:
                logger.warning(f"滚动最小值计算窗口大小 {window} 大于批次大小 {batch_size}")
                return None
        except Exception as e:
            logger.error(f"计算滚动最小值时出错: {str(e)}")
            return None
            
    def _calculate_rolling_mean(self, tensor, window):
        """
        计算张量的滚动平均值
        参数:
            tensor: 输入张量，可以是1D [N] 或 2D [N, C]
            window: 窗口大小
        返回:
            滚动平均值张量
        """
        try:
            # 检查张量维度
            original_dim = tensor.dim()
            # 确保张量是2D [N, C]
            if original_dim == 1:
                tensor = tensor.unsqueeze(1)
            n_samples, n_features = tensor.shape
            
            if n_samples >= window:
                # 获取滚动窗口 [N-window+1, C, window]
                windows = tensor.unfold(0, window, 1)
                
                # 对窗口维度 (dim=2) 计算平均值
                # 结果形状 [N-window+1, C]
                mean_values = torch.nanmean(windows, dim=2)
                
                # 创建结果张量 [N, C]
                padding = torch.full((window-1, n_features), float('nan'), 
                                     device=self.device, dtype=tensor.dtype)  # 使用CPU
                result = torch.cat([padding, mean_values], dim=0)
                
                # 恢复原始维度
                if original_dim == 1:
                    result = result.squeeze(1)
                return result
            else:
                logger.warning(f"无法为 {window} 窗口生成滚动平均值，样本数不足")
                # 返回与输入形状相同的NaN张量
                error_result = torch.full((n_samples, n_features), float('nan'), device=self.device)  # 使用CPU
                if original_dim == 1:
                    error_result = error_result.squeeze(1)
                return error_result
        except Exception as e:
            logger.error(f"计算滚动平均值时出错: {str(e)}")
            # 返回与输入形状相同的NaN张量
            n_samples, n_features = tensor.shape if tensor.dim() == 2 else (tensor.shape[0], 1)
            error_result = torch.full((n_samples, n_features), float('nan'), device=self.device)  # 使用CPU
            if original_dim == 1:
                error_result = error_result.squeeze(1)
            return error_result

    def _calculate_ewma(self, tensor, span):
        """
        计算张量的指数移动平均值
        
        参数:
            tensor: 输入张量，可以是1D [N] 或 2D [N, C]
            span: 时间窗口大小
            
        返回:
            指数移动平均值张量
        """
        try:
            # 检查张量维度
            original_shape = tensor.shape
            original_dim = tensor.dim()
            
            # 确保张量是1D的
            if original_dim > 1:
                tensor = tensor.squeeze()
                logger.debug(f"将张量从 {original_shape} 转换为 {tensor.shape}")
            
            # 计算指数移动平均值
            ema = ewma(tensor, span=span, adjust=False)
            
            # 创建结果张量
            result = torch.full_like(tensor, float('nan'), device=self.device)  # 使用CPU
            result[span-1:] = ema
            
            # 恢复原始形状
            if original_dim > 1:
                result = result.view(original_shape)
            
            # 降维，rolling_window后立即 squeeze(1)
            if result.dim() == 3 and result.shape[1] == 1:
                result = result.squeeze(1)
            
            return result
        except Exception as e:
            logger.error(f"计算指数移动平均值时出错: {str(e)}")
            return torch.full_like(tensor, float('nan'), device=self.device)  # 使用CPU

    # --- Original DataFrame-based Methods (kept for compatibility or potential CPU fallback) ---

    def calculate_indicators(self, df, indicators=None, window_sizes=None):
        """
        计算技术指标，PyTorch版 (DataFrame based)
        参数:
            df: DataFrame对象，包含OHLCV数据
            indicators: 要计算的指标列表，默认为None表示计算所有指标
            window_sizes: 窗口大小字典，用于覆盖默认窗口大小
        返回:
            包含技术指标的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的技术指标")
            return df
        
        # 提前初始化result_df，确保在所有代码路径上result_df都已定义
        result_df = df.copy()
        start_time = time.time()
        
        try:
            # 查找数据中的OHLCV列
            ohlcv_candidates = {
                'open': ['open', 'o', 'open_price'],
                'high': ['high', 'h', 'high_price'],
                'low': ['low', 'l', 'low_price'],
                'close': ['close', 'c', 'close_price', 'adj_close', 'adjusted_close'],
                'volume': ['volume', 'v', 'vol', 'volume_traded']
            }
            # 查找实际的列名
            actual_cols = {}
            cols_found = []
            for standard_col, candidates in ohlcv_candidates.items():
                for col in df.columns:
                    if col.lower() in [c.lower() for c in candidates]:
                        actual_cols[standard_col] = col
                        cols_found.append(standard_col)
                        break
            # 记录找到的列
            if cols_found:
                logger.debug(f"找到的OHLCV列: {cols_found}")
            else:
                logger.warning("未找到任何OHLCV列，将使用原始列名")
            # 将DataFrame转换为张量字典
            tensor_dict = df_to_tensor(df)
            # 将找到的列映射到标准名称
            mapped_tensor_dict = {}
            for standard_col, actual_col in actual_cols.items():
                if actual_col in tensor_dict:
                    mapped_tensor_dict[standard_col] = tensor_dict[actual_col]
                elif actual_col.lower() in tensor_dict:
                    mapped_tensor_dict[standard_col] = tensor_dict[actual_col.lower()]
            # 如果映射的字典为空，使用原始的张量字典
            if not mapped_tensor_dict:
                logger.warning("映射后的张量字典为空，使用原始张量字典")
                for standard_col in ['open', 'high', 'low', 'close', 'volume']:
                    if standard_col in tensor_dict:
                        mapped_tensor_dict[standard_col] = tensor_dict[standard_col]
                    elif standard_col.lower() in tensor_dict:
                        mapped_tensor_dict[standard_col] = tensor_dict[standard_col.lower()]
            # 检查是否有必要的键
            required_keys = ['open', 'high', 'low', 'close']
            missing_keys = [key for key in required_keys if key not in mapped_tensor_dict]
            if missing_keys:
                logger.warning(f"张量字典缺少必要的键: {missing_keys}，某些特征可能无法计算")
                for key in missing_keys:
                    for alt_key in tensor_dict.keys():
                        if key in alt_key.lower():
                            mapped_tensor_dict[key] = tensor_dict[alt_key]
                            logger.debug(f"使用 {alt_key} 替代 {key}")
                            break
            all_indicators = ['price', 'volume', 'volatility', 'trend', 'momentum']
            if indicators is None:
                indicators = all_indicators
            elif isinstance(indicators, list) and indicators and isinstance(indicators[0], str):
                traditional_indicators = ['MACD', 'RSI', 'STOCH', 'BBANDS', 'ATR', 'ADX', 'CCI']
                if any(ind.upper() in traditional_indicators for ind in indicators):
                    indicator_map = {
                        'MACD': 'momentum', 'RSI': 'momentum', 'STOCH': 'momentum',
                        'BBANDS': 'volatility', 'ATR': 'volatility',
                        'ADX': 'trend', 'CCI': 'trend'
                    }
                    converted_indicators = set()
                    for ind in indicators:
                        mapped_ind = indicator_map.get(ind.upper())
                        if mapped_ind:
                            converted_indicators.add(mapped_ind)
                    if converted_indicators:
                        indicators = list(converted_indicators)
                    else:
                        indicators = all_indicators
            result_tensors = {}
            if 'price' in indicators and all(k in mapped_tensor_dict for k in ['open', 'high', 'low', 'close']):
                logger.debug("计算价格特征...")
                price_features = self.calculate_price_features_tensor(mapped_tensor_dict)
                result_tensors.update(price_features)
            if 'volume' in indicators and 'volume' in mapped_tensor_dict and all(k in mapped_tensor_dict for k in ['high', 'low', 'close']):
                logger.debug("计算交易量特征...")
                volume_features = self.calculate_volume_features_tensor(mapped_tensor_dict)
                result_tensors.update(volume_features)
            if 'volatility' in indicators and all(k in mapped_tensor_dict for k in ['high', 'low', 'close']):
                logger.debug("计算波动性特征...")
                volatility_features = self.calculate_volatility_features_tensor(mapped_tensor_dict)
                result_tensors.update(volatility_features)
            if 'trend' in indicators and all(k in mapped_tensor_dict for k in ['high', 'low', 'close']):
                logger.debug("计算趋势特征...")
                trend_features = self.calculate_trend_features_tensor(mapped_tensor_dict)
                result_tensors.update(trend_features)
            if 'momentum' in indicators and all(k in mapped_tensor_dict for k in ['high', 'low', 'close']):
                logger.debug("计算动量特征...")
                momentum_features = self.calculate_momentum_features_tensor(mapped_tensor_dict)
                result_tensors.update(momentum_features)
            if result_tensors:
                result_df = self._add_results_to_df(result_df, result_tensors)
            end_time = time.time()
            calculation_time = end_time - start_time
            logger.info(f"已计算技术指标 (PyTorch版): {', '.join(indicators)} (耗时: {calculation_time:.2f}秒)")
            return result_df
        except Exception as e:
            logger.error(f"计算技术指标时出错: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            # 此处不需要检查result_df是否已定义，因为已在try前初始化
            return result_df
    
    def calculate_price_features(self, df):
        """
        计算价格相关特征 (DataFrame based)
        
        参数:
            df: DataFrame对象，包含OHLCV数据
            
        返回:
            包含价格特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的价格特征")
            return df
        
        # 创建副本，并备份原始DataFrame
        result_df = df.copy()
        original_df = df.copy()
        
        try:
            # 将DataFrame转换为张量字典，并映射列名
            try:
                tensor_dict = df_to_tensor(df)
            except Exception as e:
                logger.error(f"DataFrame转张量时出错: {str(e)}")
                return original_df # 转换失败则返回原始数据
            
            # 映射张量字典中的键名
            try:
                mapped_tensor_dict = {}
                # 尝试匹配常见的列命名
                for col in df.columns:
                    col_lower = col.lower()
                    if 'open' in col_lower:
                        mapped_tensor_dict['open'] = tensor_dict[col]
                    elif 'high' in col_lower:
                        mapped_tensor_dict['high'] = tensor_dict[col]
                    elif 'low' in col_lower:
                        mapped_tensor_dict['low'] = tensor_dict[col]
                    elif 'close' in col_lower:
                        mapped_tensor_dict['close'] = tensor_dict[col]
                    elif 'volume' in col_lower:
                        mapped_tensor_dict['volume'] = tensor_dict[col]
            except Exception as e:
                logger.error(f"映射张量键名时出错: {str(e)}")
                return original_df # 映射失败则返回原始数据
            
            # 使用张量方法计算价格特征
            try:
                result_tensors = self.calculate_price_features_tensor(mapped_tensor_dict)
                if not result_tensors:
                    logger.warning("价格特征张量计算返回空结果")
                    return original_df # 张量计算失败返回原始数据
            except Exception as e:
                logger.error(f"计算价格特征张量时出错: {str(e)}")
                return original_df # 张量计算失败返回原始数据
            
            # 将计算结果添加回DataFrame
            try:
                result_df = self._add_results_to_df(result_df, result_tensors)
                if result_df is None or result_df.empty:
                     logger.warning("添加特征到DataFrame后结果为空")
                     return original_df # 添加失败返回原始数据
            except Exception as e:
                logger.error(f"添加价格特征到DataFrame时出错: {str(e)}")
                return original_df # 添加失败返回原始数据
            
            logger.info("已计算价格特征 (PyTorch版)")
            return result_df
            
        except Exception as e:
            logger.error(f"计算价格特征过程中发生意外错误: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            # 兜底返回原始DataFrame副本
            return original_df
    
    def calculate_volume_features(self, df):
        """
        计算交易量相关特征 (DataFrame based)
        
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
        
        # 将DataFrame转换为张量字典，并映射列名
        tensor_dict = df_to_tensor(df)
        
        # 映射张量字典中的键名
        mapped_tensor_dict = {}
        # 尝试匹配常见的列命名
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                mapped_tensor_dict['open'] = tensor_dict[col]
            elif 'high' in col_lower:
                mapped_tensor_dict['high'] = tensor_dict[col]
            elif 'low' in col_lower:
                mapped_tensor_dict['low'] = tensor_dict[col]
            elif 'close' in col_lower:
                mapped_tensor_dict['close'] = tensor_dict[col]
            elif 'volume' in col_lower:
                mapped_tensor_dict['volume'] = tensor_dict[col]
        
        # 使用张量方法计算交易量特征
        result_tensors = self.calculate_volume_features_tensor(mapped_tensor_dict)
        
        # 将计算结果添加回DataFrame
        result_df = self._add_results_to_df(result_df, result_tensors)
        
        logger.info("已计算交易量特征 (PyTorch版)")
        return result_df
    
    def calculate_volatility_features(self, df):
        """
        计算波动性相关特征 (DataFrame based)
        
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
            
        # 将DataFrame转换为张量字典，并映射列名
        tensor_dict = df_to_tensor(df)
        
        # 映射张量字典中的键名
        mapped_tensor_dict = {}
        # 尝试匹配常见的列命名
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                mapped_tensor_dict['open'] = tensor_dict[col]
            elif 'high' in col_lower:
                mapped_tensor_dict['high'] = tensor_dict[col]
            elif 'low' in col_lower:
                mapped_tensor_dict['low'] = tensor_dict[col]
            elif 'close' in col_lower:
                mapped_tensor_dict['close'] = tensor_dict[col]
            elif 'volume' in col_lower:
                mapped_tensor_dict['volume'] = tensor_dict[col]
        
        # 使用张量方法计算波动性特征
        result_tensors = self.calculate_volatility_features_tensor(mapped_tensor_dict)
        
        # 将计算结果添加回DataFrame
        result_df = self._add_results_to_df(result_df, result_tensors)
        
        logger.info("已计算波动性特征 (PyTorch版)")
        return result_df
    
    def calculate_trend_features(self, df):
        """
        计算趋势相关特征 (DataFrame based)
        
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
        
        # 将DataFrame转换为张量字典，并映射列名
        tensor_dict = df_to_tensor(df)
        
        # 映射张量字典中的键名
        mapped_tensor_dict = {}
        # 尝试匹配常见的列命名
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                mapped_tensor_dict['open'] = tensor_dict[col]
            elif 'high' in col_lower:
                mapped_tensor_dict['high'] = tensor_dict[col]
            elif 'low' in col_lower:
                mapped_tensor_dict['low'] = tensor_dict[col]
            elif 'close' in col_lower:
                mapped_tensor_dict['close'] = tensor_dict[col]
            elif 'volume' in col_lower:
                mapped_tensor_dict['volume'] = tensor_dict[col]
        
        # 使用张量方法计算趋势特征
        result_tensors = self.calculate_trend_features_tensor(mapped_tensor_dict)
        
        # 将计算结果添加回DataFrame
        result_df = self._add_results_to_df(result_df, result_tensors)
        
        logger.info("已计算趋势特征 (PyTorch版)")
        return result_df
    
    def calculate_momentum_features(self, df):
        """
        计算动量相关特征 (DataFrame based)
        
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
        
        # 将DataFrame转换为张量字典，并映射列名
        tensor_dict = df_to_tensor(df)
        
        # 映射张量字典中的键名
        mapped_tensor_dict = {}
        # 尝试匹配常见的列命名
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                mapped_tensor_dict['open'] = tensor_dict[col]
            elif 'high' in col_lower:
                mapped_tensor_dict['high'] = tensor_dict[col]
            elif 'low' in col_lower:
                mapped_tensor_dict['low'] = tensor_dict[col]
            elif 'close' in col_lower:
                mapped_tensor_dict['close'] = tensor_dict[col]
            elif 'volume' in col_lower:
                mapped_tensor_dict['volume'] = tensor_dict[col]
        
        # 使用张量方法计算动量特征
        result_tensors = self.calculate_momentum_features_tensor(mapped_tensor_dict)
        
        # 将计算结果添加回DataFrame
        result_df = self._add_results_to_df(result_df, result_tensors)
        
        logger.info("已计算动量特征 (PyTorch版)")
        return result_df
                

# 将 PyTorchCompatibleTechnicalIndicators 定义为全局类，而不是内部类，这样可以直接替换原始类
class PyTorchCompatibleTechnicalIndicators(PyTorchTechnicalIndicators):
    """与原始TechnicalIndicators接口兼容的PyTorch实现"""
    
    def __init__(self):
        super().__init__()
        logger.info("初始化 PyTorch 兼容的技术指标类 - 使用CPU计算")
        
    # 静态方法模拟原始 TechnicalIndicators 的静态方法
    @staticmethod
    def tsi(close, r=25, s=13):
        """静态方法接口兼容，使用 PyTorch 版本的 TSI 计算"""
        return PyTorchTechnicalIndicators.tsi(close, r, s)
    
    @staticmethod
    def calculate_indicators(df, indicators=None, window_sizes=None):
        """静态方法接口兼容，使用 PyTorch 版本的计算"""
        instance = PyTorchTechnicalIndicators()
        
        if df is None or df.empty:
            logger.warning("无法计算空数据的技术指标")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        try:
            # 将DataFrame转换为张量字典
            tensor_dict = df_to_tensor(df)
            
            # 将找到的列映射到标准名称
            mapped_tensor_dict = {}
            # 尝试匹配常见的列命名
            for col in df.columns:
                col_lower = col.lower()
                if 'open' in col_lower:
                    mapped_tensor_dict['open'] = tensor_dict[col]
                elif 'high' in col_lower:
                    mapped_tensor_dict['high'] = tensor_dict[col]
                elif 'low' in col_lower:
                    mapped_tensor_dict['low'] = tensor_dict[col]
                elif 'close' in col_lower:
                    mapped_tensor_dict['close'] = tensor_dict[col]
                elif 'volume' in col_lower:
                    mapped_tensor_dict['volume'] = tensor_dict[col]
            
            # 如果传入了具体的技术指标名称（如'MACD', 'RSI'等），需要转换
            # 因为我们的新实现使用了分类（'price', 'volume', 'volatility', 'trend', 'momentum'）
            if indicators is not None and isinstance(indicators, list) and len(indicators) > 0 and isinstance(indicators[0], str):
                # 检查第一个元素，如果不是我们的分类之一，则进行映射
                first_elem = indicators[0].lower()
                if first_elem not in ['price', 'volume', 'volatility', 'trend', 'momentum']:
                    # 创建映射，转换传统指标名称为新的分类
                    indicators_mapping = {
                        'MACD': 'momentum',
                        'RSI': 'momentum',
                        'STOCH': 'momentum',
                        'BBANDS': 'volatility',
                        'ATR': 'volatility',
                        'ADX': 'trend',
                        'CCI': 'trend'
                    }
                    
                    # 将指标名称转换为分类
                    needed_groups = set()
                    for ind in indicators:
                        if ind.upper() in indicators_mapping:
                            needed_groups.add(indicators_mapping[ind.upper()])
                    
                    # 如果有分类，使用这些分类
                    if needed_groups:
                        indicators = list(needed_groups)
                    else:
                        # 如果没有匹配的分类，使用所有分类
                        indicators = ['price', 'volume', 'volatility', 'trend', 'momentum']
            
            return instance.calculate_indicators(df, indicators, window_sizes)
            
        except Exception as e:
            logger.error(f"PyTorch兼容层计算技术指标时出错: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return result_df
    
    @staticmethod
    def calculate_price_features(df):
        """静态方法接口兼容，使用 PyTorch 版本的计算"""
        instance = PyTorchTechnicalIndicators()
        return instance.calculate_price_features(df)
    
    @staticmethod
    def calculate_volume_features(df):
        """静态方法接口兼容，使用 PyTorch 版本的计算"""
        instance = PyTorchTechnicalIndicators()
        return instance.calculate_volume_features(df)
    
    @staticmethod
    def calculate_volatility_features(df):
        """静态方法接口兼容，使用 PyTorch 版本的计算"""
        instance = PyTorchTechnicalIndicators()
        return instance.calculate_volatility_features(df)
    
    @staticmethod
    def calculate_trend_features(df):
        """静态方法接口兼容，使用 PyTorch 版本的计算"""
        instance = PyTorchTechnicalIndicators()
        return instance.calculate_trend_features(df)
    
    @staticmethod
    def calculate_momentum_features(df):
        """静态方法接口兼容，使用 PyTorch 版本的计算"""
        instance = PyTorchTechnicalIndicators()
        return instance.calculate_momentum_features(df) 