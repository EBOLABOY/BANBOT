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

class PyTorchTechnicalIndicators:
    """
    使用PyTorch实现的技术指标计算类，充分利用GPU加速
    Includes both DataFrame-based methods (for compatibility/ease of use)
    and tensor-based methods (for optimized data flow).
    """
    
    def __init__(self):
        """
        初始化技术指标计算器
        """
        self.device = get_device()
        logger.info(f"PyTorch技术指标计算器初始化，使用设备: {self.device}")

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

            # Ensure tensor is on CPU and 1D/2D for conversion
            if tensor.is_cuda:
                tensor = tensor.cpu()
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

        close = tensor_dict['close']
        high = tensor_dict['high']
        low = tensor_dict['low']
        open_price = tensor_dict['open']

        if not all(t.shape[0] == close.shape[0] for t in [high, low, open_price]):
            logger.error("价格特征输入张量长度不匹配")
            return {}
        if close.shape[0] == 0: return {} # Handle empty

        result_tensors = {}

        # Price change percentage
        prev_close = torch.roll(close, shifts=1, dims=0)
        prev_close[0] = float('nan') # First value has no pct change
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
                # Ensure window calculation didn't return empty
                if high_windows.shape[0] == 0 or low_windows.shape[0] == 0: continue

                high_max = torch.max(high_windows, dim=1)[0]
                low_min = torch.min(low_windows, dim=1)[0]

                high_max_full = torch.full_like(close, float('nan'))
                low_min_full = torch.full_like(close, float('nan'))

                high_max_full[window-1:] = high_max
                low_min_full[window-1:] = low_min

                # Fill initial NaNs by calculating max/min over the available window
                for i in range(window-1):
                     if i+1 <= len(high):
                         high_max_full[i] = torch.max(high[:i+1])
                         low_min_full[i] = torch.min(low[:i+1])

                price_rel_high = close / (high_max_full + 1e-10)
                price_rel_low = close / (low_min_full + 1e-10)
                result_tensors[f'price_rel_high_{window}'] = price_rel_high
                result_tensors[f'price_rel_low_{window}'] = price_rel_low
            else:
                 logger.debug(f"价格相对位置窗口大小 {window} 无效或大于数据长度") # Debug level

        # Price volatility (rolling std dev of pct change)
        volatility_window = 20
        if volatility_window > 0 and volatility_window <= len(price_change_pct):
             # Need to handle NaNs in price_change_pct for std calculation
             price_change_pct_nonan = torch.where(torch.isnan(price_change_pct), torch.zeros_like(price_change_pct), price_change_pct)
             windows = rolling_window(price_change_pct_nonan, volatility_window)
             if windows.shape[0] > 0: # Ensure windows are generated
                 volatility = torch.std(windows, dim=1, unbiased=True)
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

        volume = tensor_dict['volume']
        close = tensor_dict['close']
        high = tensor_dict['high']
        low = tensor_dict['low']

        if not all(t.shape[0] == volume.shape[0] for t in [close, high, low]):
            logger.error("交易量特征输入张量长度不匹配")
            return {}
        if volume.shape[0] == 0: return {} # Handle empty

        result_tensors = {}

        # Volume change percentage
        prev_volume = torch.roll(volume, shifts=1, dims=0)
        prev_volume[0] = float('nan') # First value has no previous
        volume_change_pct = (volume - prev_volume) / (prev_volume + 1e-10)
        result_tensors['volume_change_pct'] = volume_change_pct

        # Relative volume (compared to recent average)
        for window in [5, 10, 20, 50]:
            if window > 0 and window <= len(volume):
                volume_ma = moving_average(volume, window)
                rel_volume = volume / (volume_ma + 1e-10)
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

        # Iterative calculation (can be slow on GPU for long series)
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

    def calculate_volatility_features_tensor(self, tensor_dict):
        """
        计算波动性相关特征，纯张量操作

        参数:
            tensor_dict: 包含输入张量的字典 ('high', 'low', 'close')

        返回:
            包含波动性特征张量的字典
        """
        required_keys = ['high', 'low', 'close']
        if not all(key in tensor_dict for key in required_keys):
            logger.warning("波动性特征张量计算缺少必要的输入张量")
            return {}

        close = tensor_dict['close']
        high = tensor_dict['high']
        low = tensor_dict['low']

        if not all(t.shape[0] == close.shape[0] for t in [high, low]):
            logger.error("波动性特征输入张量长度不匹配")
            return {}
        if close.shape[0] == 0: return {} # Handle empty

        result_tensors = {}

        # Intraday volatility (High-Low Range / Close)
        intraday_volatility = (high - low) / (close + 1e-10)
        result_tensors['intraday_volatility'] = intraday_volatility

        # Historical volatility (std dev of log returns)
        log_returns = torch.log(close / (torch.roll(close, shifts=1, dims=0) + 1e-10)) # Add epsilon
        log_returns[0] = 0 # Set first return to 0

        for window in [5, 10, 20, 50]:
            if window > 0 and window <= len(log_returns):
                returns_windows = rolling_window(log_returns, window)
                if returns_windows.shape[0] > 0:
                     std_dev = torch.std(returns_windows, dim=1, unbiased=True)
                     std_dev_full = torch.full_like(close, float('nan'))
                     std_dev_full[window-1:] = std_dev
                     annualization_factor = 252.0 # Adjust as needed
                     annualized_vol = std_dev_full * torch.sqrt(torch.tensor(annualization_factor, device=self.device))
                     result_tensors[f'volatility_{window}d'] = annualized_vol
                else:
                     logger.debug(f"无法为历史波动率生成滚动窗口 {window}")
            else:
                logger.debug(f"历史波动率窗口大小 {window} 无效或大于数据长度")

        # True Range (TR)
        high_low_tr = high - low
        high_close_prev_tr = torch.abs(high - torch.roll(close, shifts=1, dims=0))
        low_close_prev_tr = torch.abs(low - torch.roll(close, shifts=1, dims=0))
        # Handle first element correctly
        high_close_prev_tr[0] = high_low_tr[0] if len(high_low_tr)>0 else float('nan')
        low_close_prev_tr[0] = high_low_tr[0] if len(high_low_tr)>0 else float('nan')

        tr = torch.maximum(high_low_tr, torch.maximum(high_close_prev_tr, low_close_prev_tr))
        result_tensors['tr'] = tr

        # Average True Range (ATR)
        for window in [5, 14, 20]:
            if window > 0 and window <= len(tr):
                # Use EWMA for standard ATR smoothing
                # Handle potential NaNs in TR before EWMA
                tr_nonan = torch.where(torch.isnan(tr), torch.zeros_like(tr), tr)
                atr = ewma(tr_nonan, span=window)
                result_tensors[f'atr_{window}'] = atr
                # Relative ATR (ATR / Close)
                atr_pct = atr / (close + 1e-10)
                result_tensors[f'atr_pct_{window}'] = atr_pct
            else:
                logger.debug(f"ATR 计算窗口大小 {window} 无效或大于数据长度")

        # Bollinger Band based volatility indicators
        for window in [20, 50]:
            if window > 0 and window <= len(close):
                middle_band_vol = moving_average(close, window)
                close_windows_vol = rolling_window(close, window)
                if close_windows_vol.shape[0] > 0:
                     std_dev_vol = torch.std(close_windows_vol, dim=1, unbiased=True)
                     std_dev_full_vol = torch.full_like(close, float('nan'))
                     std_dev_full_vol[window-1:] = std_dev_vol

                     upper_band_vol = middle_band_vol + 2 * std_dev_full_vol
                     lower_band_vol = middle_band_vol - 2 * std_dev_full_vol

                     # Bollinger Band Width (%)
                     bb_width = (upper_band_vol - lower_band_vol) / (middle_band_vol + 1e-10) * 100
                     result_tensors[f'bb_width_pct_{window}'] = bb_width

                     # %B Indicator (Price position relative to bands)
                     bb_pos = (close - lower_band_vol) / ((upper_band_vol - lower_band_vol) + 1e-10)
                     bb_pos = torch.clamp(bb_pos, 0, 1) # Scale 0 to 1
                     result_tensors[f'bb_pos_{window}'] = bb_pos
                else:
                     logger.debug(f"无法为BBands波动率生成滚动窗口 {window}")
            else:
                 logger.debug(f"BBands 波动率窗口大小 {window} 无效或大于数据长度")

        return result_tensors

    def calculate_trend_features_tensor(self, tensor_dict):
        """
        计算趋势相关特征，纯张量操作

        参数:
            tensor_dict: 包含输入张量的字典 ('open', 'high', 'low', 'close', Optional['volume'])

        返回:
            包含趋势特征张量的字典
        """
        required_keys = ['open', 'high', 'low', 'close'] # Base requirements
        if not all(key in tensor_dict for key in required_keys):
            logger.warning("趋势特征张量计算缺少必要的输入张量 (OHLC)")
            return {}

        close = tensor_dict['close']
        high = tensor_dict['high']
        low = tensor_dict['low']
        # open_price = tensor_dict['open'] # Not used here currently
        # volume = tensor_dict.get('volume') # Needed for ADX

        if close.shape[0] == 0: return {} # Handle empty

        result_tensors = {}

        # Price relative to moving averages
        for window in [10, 20, 50, 100, 200]:
            if window > 0 and window <= len(close):
                sma = moving_average(close, window)
                price_rel_sma = (close / (sma + 1e-10)) - 1 # As percentage difference
                result_tensors[f'price_rel_sma_{window}'] = price_rel_sma
            else:
                 logger.debug(f"价格相对SMA窗口大小 {window} 无效")

        # Moving average slope/direction
        ma_slope_period = 5
        for window in [20, 50]:
            if window > 0 and window + ma_slope_period <= len(close):
                sma = moving_average(close, window)
                sma_change = (sma - torch.roll(sma, shifts=ma_slope_period, dims=0)) / (torch.roll(sma, shifts=ma_slope_period, dims=0) + 1e-10)
                sma_change[:window+ma_slope_period-1] = float('nan')
                sma_direction = torch.sign(sma_change)
                result_tensors[f'sma_{window}_slope_{ma_slope_period}'] = sma_change
                result_tensors[f'sma_{window}_direction'] = sma_direction
            else:
                 logger.debug(f"SMA 斜率窗口大小 {window} 或周期 {ma_slope_period} 无效")

        # Calculate EMA crossover
        fast_period = 20
        slow_period = 50
        if slow_period > fast_period and slow_period <= len(close):
            fast_ema = ewma(close, span=fast_period)
            slow_ema = ewma(close, span=slow_period)
            ema_diff = fast_ema - slow_ema
            ema_cross = torch.sign(ema_diff)
            ema_cross_change = torch.diff(ema_cross, dim=0, prepend=ema_cross[0:1])
            result_tensors['ema_diff'] = ema_diff
            result_tensors['ema_cross_signal'] = ema_cross
            result_tensors['ema_crossover'] = ema_cross_change
        else:
             logger.debug("EMA 交叉周期无效")

        # Calculate Trend Strength Index (TSI)
        tsi_tensor = self.tsi(close, r=25, s=13)
        result_tensors['tsi'] = tsi_tensor

        # Calculate ADX trend strength and direction
        adx_window = 14
        if high is not None and low is not None and adx_window <= len(close)-1:
             high_diff_adx = torch.diff(high, dim=0, prepend=high[0:1])
             low_diff_adx = torch.diff(low, dim=0, prepend=low[0:1])
             plus_dm_adx = torch.where((high_diff_adx > 0) & (high_diff_adx > torch.abs(low_diff_adx)), high_diff_adx, torch.zeros_like(high_diff_adx))
             minus_dm_adx = torch.where((low_diff_adx < 0) & (torch.abs(low_diff_adx) > high_diff_adx), torch.abs(low_diff_adx), torch.zeros_like(low_diff_adx))
             high_low_adx = high - low
             high_close_prev_adx = torch.abs(high - torch.roll(close, shifts=1, dims=0))
             low_close_prev_adx = torch.abs(low - torch.roll(close, shifts=1, dims=0))
             high_close_prev_adx[0]=0
             low_close_prev_adx[0]=0
             tr_adx = torch.maximum(high_low_adx, torch.maximum(high_close_prev_adx, low_close_prev_adx))
             tr_ema_adx = ewma(tr_adx, span=adx_window)
             plus_dm_ema_adx = ewma(plus_dm_adx, span=adx_window)
             minus_dm_ema_adx = ewma(minus_dm_adx, span=adx_window)
             plus_di = 100 * plus_dm_ema_adx / (tr_ema_adx + 1e-10)
             minus_di = 100 * minus_dm_ema_adx / (tr_ema_adx + 1e-10)
             dx = 100 * torch.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
             dx = torch.where(torch.isnan(dx) | torch.isinf(dx), torch.zeros_like(dx), dx)
             adx = ewma(dx, span=adx_window)

             result_tensors[f'adx_{adx_window}'] = adx
             result_tensors['plus_di'] = plus_di
             result_tensors['minus_di'] = minus_di
             trend_strength = torch.where(adx > 25, torch.ones_like(adx), torch.zeros_like(adx))
             trend_direction = torch.sign(plus_di - minus_di)
             strong_trend = trend_strength * trend_direction
             result_tensors['trend_strength'] = trend_strength
             result_tensors['trend_direction'] = trend_direction
             result_tensors['strong_trend'] = strong_trend
        else:
             logger.warning("无法计算ADX趋势强度/方向，缺少数据或列")

        return result_tensors

    def calculate_momentum_features_tensor(self, tensor_dict):
        """
        计算动量相关特征，纯张量操作
        
        参数:
            tensor_dict: 包含输入张量的字典 ('open', 'high', 'low', 'close', Optional['volume'])
            
        返回:
            包含动量特征张量的字典
        """
        required_keys = ['open', 'high', 'low', 'close'] # Volume optional
        if not all(key in tensor_dict for key in required_keys):
            logger.warning("动量特征张量计算缺少必要的输入张量")
            return {}

        # Get tensors
        close = tensor_dict['close']
        high = tensor_dict['high']
        low = tensor_dict['low']
        # open_price = tensor_dict['open']
        # volume = tensor_dict.get('volume')

        # Store computed tensors
        result_tensors = {}

        # Calculate Rate of Change (ROC)
        for period in [1, 3, 5, 10, 20, 60]:
            if period > 0 and period <= len(close):
                prev_close = torch.roll(close, shifts=period, dims=0)
                prev_close[:period] = float('nan')
                roc = (close - prev_close) / (prev_close + 1e-10) * 100
                result_tensors[f'roc_{period}'] = roc
            else:
                 logger.warning(f"ROC 周期 {period} 无效")

        # Calculate RSI momentum indicator
        price_diff_rsi = torch.diff(close, dim=0, prepend=close[0:1])
        for window in [6, 14]:
            if window > 0 and window <= len(price_diff_rsi):
                up_rsi = torch.maximum(price_diff_rsi, torch.zeros_like(price_diff_rsi))
                down_rsi = torch.abs(torch.minimum(price_diff_rsi, torch.zeros_like(price_diff_rsi)))
                avg_up_rsi = moving_average(up_rsi, window)
                avg_down_rsi = moving_average(down_rsi, window)
                rs_rsi = avg_up_rsi / (avg_down_rsi + 1e-10)
                rsi = 100 - (100 / (1 + rs_rsi))
                rsi = torch.clamp(rsi, 0, 100)
                result_tensors[f'rsi_{window}'] = rsi
            else:
                logger.warning(f"RSI 动量窗口 {window} 无效")

        # Calculate MACD momentum indicators
        fast_p = 12; slow_p = 26; signal_p = 9
        if slow_p <= len(close):
             fast_ema_macd = ewma(close, span=fast_p)
             slow_ema_macd = ewma(close, span=slow_p)
             macd_line = fast_ema_macd - slow_ema_macd
             if signal_p <= len(macd_line):
                  signal_line = ewma(macd_line, span=signal_p)
                  histogram = macd_line - signal_line
                  histogram_change = torch.diff(histogram, dim=0, prepend=histogram[0:1])
                  histogram_direction = torch.sign(histogram)
                  histogram_trend = torch.sign(histogram_change)
                  result_tensors['macd_line'] = macd_line
                  result_tensors['macd_signal'] = signal_line
                  result_tensors['macd_hist'] = histogram
                  result_tensors['macd_hist_change'] = histogram_change
                  result_tensors['macd_hist_direction'] = histogram_direction
                  result_tensors['macd_hist_trend'] = histogram_trend
             else:
                  logger.warning("无法计算 MACD 动量 - 信号周期")
        else:
             logger.warning("无法计算 MACD 动量 - 慢周期")

        # Calculate Stochastic Oscillator momentum indicators
        window_k_stoch = 14
        window_d_stoch = 3
        if high is not None and low is not None and window_k_stoch <= len(close):
            windows_high_stoch = rolling_window(high, window_k_stoch)
            windows_low_stoch = rolling_window(low, window_k_stoch)
            highest_high_stoch = torch.max(windows_high_stoch, dim=1)[0]
            lowest_low_stoch = torch.min(windows_low_stoch, dim=1)[0]
            highest_high_full_stoch = torch.full_like(close, float('nan'))
            lowest_low_full_stoch = torch.full_like(close, float('nan'))
            if len(highest_high_stoch) > 0 : highest_high_full_stoch[window_k_stoch-1:] = highest_high_stoch
            if len(lowest_low_stoch) > 0 : lowest_low_full_stoch[window_k_stoch-1:] = lowest_low_stoch
            for i in range(window_k_stoch-1):
                if i+1 <= len(high):
                     highest_high_full_stoch[i] = torch.max(high[:i+1])
                     lowest_low_full_stoch[i] = torch.min(low[:i+1])
            stoch_k = 100 * (close - lowest_low_full_stoch) / (highest_high_full_stoch - lowest_low_full_stoch + 1e-10)
            stoch_k = torch.where(torch.isnan(stoch_k) | torch.isinf(stoch_k), torch.zeros_like(stoch_k), stoch_k)
            stoch_k = torch.clamp(stoch_k, 0, 100)

            if window_d_stoch <= len(stoch_k):
                 stoch_d = moving_average(stoch_k, window_d_stoch)
                 stoch_d = torch.clamp(stoch_d, 0, 100)
                 stoch_cross = stoch_k - stoch_d
                 stoch_cross_signal = torch.sign(stoch_cross)
                 result_tensors['stoch_k'] = stoch_k
                 result_tensors['stoch_d'] = stoch_d
                 result_tensors['stoch_cross_diff'] = stoch_cross
                 result_tensors['stoch_cross_signal'] = stoch_cross_signal
            else:
                 logger.warning("无法计算 Stochastic %D")
        else:
            logger.warning("无法计算 Stochastic %K")

        return result_tensors

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
            
        start_time = time.time()
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
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
                    # 尝试使用小写名称
                    mapped_tensor_dict[standard_col] = tensor_dict[actual_col.lower()]
            
            # 如果映射的字典为空，使用原始的张量字典
            if not mapped_tensor_dict:
                logger.warning("映射后的张量字典为空，使用原始张量字典")
                # 尝试直接从原始字典中查找OHLCV列
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
                # 尝试从原始张量字典中复制必要的键
                for key in missing_keys:
                    # 使用可能的替代键
                    for alt_key in tensor_dict.keys():
                        if key in alt_key.lower():
                            mapped_tensor_dict[key] = tensor_dict[alt_key]
                            logger.debug(f"使用 {alt_key} 替代 {key}")
                            break
            
            # 根据indicators参数决定计算哪些指标
            all_indicators = ['price', 'volume', 'volatility', 'trend', 'momentum']
            if indicators is None:
                indicators = all_indicators
            elif isinstance(indicators, list) and indicators and isinstance(indicators[0], str):
                # 检查是否是传统的指标名称
                traditional_indicators = ['MACD', 'RSI', 'STOCH', 'BBANDS', 'ATR', 'ADX', 'CCI']
                if any(ind.upper() in traditional_indicators for ind in indicators):
                    # 创建映射
                    indicator_map = {
                        'MACD': 'momentum', 'RSI': 'momentum', 'STOCH': 'momentum',
                        'BBANDS': 'volatility', 'ATR': 'volatility',
                        'ADX': 'trend', 'CCI': 'trend'
                    }
                    # 转换为新的分类
                    converted_indicators = set()
                    for ind in indicators:
                        mapped_ind = indicator_map.get(ind.upper())
                        if mapped_ind:
                            converted_indicators.add(mapped_ind)
                    
                    # 如果有转换后的指标，使用它们
                    if converted_indicators:
                        indicators = list(converted_indicators)
                    else:
                        # 如果没有映射成功，使用所有分类
                        indicators = all_indicators
            
            # 计算各类特征
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
            
            # 将计算的特征结果添加回DataFrame
            if result_tensors:
                result_df = self._add_results_to_df(result_df, result_tensors)
                
            # 性能日志
            end_time = time.time()
            calculation_time = end_time - start_time
            logger.info(f"已计算技术指标 (PyTorch版): {', '.join(indicators)} (耗时: {calculation_time:.2f}秒)")
            return result_df
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
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
        
        # 使用张量方法计算价格特征
        result_tensors = self.calculate_price_features_tensor(mapped_tensor_dict)
        
        # 将计算结果添加回DataFrame
        result_df = self._add_results_to_df(result_df, result_tensors)
        
        logger.info("已计算价格特征 (PyTorch版)")
        return result_df
    
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
        logger.info("初始化 PyTorch 兼容的技术指标类 - 使用GPU加速")
        
    # 静态方法模拟原始 TechnicalIndicators 的静态方法
    @staticmethod
    def tsi(close, r=25, s=13):
        """静态方法接口兼容，使用 PyTorch 版本的 TSI 计算"""
        return PyTorchTechnicalIndicators.tsi(close, r, s)
    
    @staticmethod
    def calculate_indicators(df, indicators=None, window_sizes=None):
        """静态方法接口兼容，使用 PyTorch 版本的计算"""
        instance = PyTorchTechnicalIndicators()
        
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