"""
PyTorch 工具函数模块 - 用于GPU加速
"""

import torch
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 全局设备变量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device():
    """
    获取当前设备（CPU或GPU）
    """
    return DEVICE

def set_device(device_str=None):
    """
    设置全局设备
    
    参数:
        device_str: 设备字符串，如 'cuda:0', 'cpu' 等
    """
    global DEVICE
    if device_str is None:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(device_str)
    return DEVICE

def df_to_tensor(df, columns=None):
    """
    将DataFrame转换为PyTorch张量字典
    
    参数:
        df: pandas DataFrame
        columns: 要转换的列名列表，如果为None则转换所有数值列
        
    返回:
        tensor_dict - 包含每列对应张量的字典
    """
    if df is None or df.empty:
        logger.warning("df_to_tensor: 输入DataFrame为空")
        return {}
    
    # 创建一个字典，存储每列对应的张量
    tensor_dict = {}
    
    try:
        if columns is None:
            # 只选择数值类型的列
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols
            logger.debug(f"df_to_tensor: 自动选择的数值列: {columns}")
        
        # 规范化列名 (小写处理)
        col_mapping = {}
        for col in columns:
            col_mapping[col.lower()] = col
        
        # 匹配常见的OHLCV列名
        standard_cols = {
            'open': ['open', 'o', 'open_price', 'opening_price'],
            'high': ['high', 'h', 'high_price', 'highest_price'],
            'low': ['low', 'l', 'low_price', 'lowest_price'],
            'close': ['close', 'c', 'close_price', 'closing_price'],
            'volume': ['volume', 'v', 'vol', 'quantity']
        }
        
        # 尝试标准化列名
        for standard_col, variants in standard_cols.items():
            for variant in variants:
                if variant in col_mapping:
                    tensor_dict[standard_col] = torch.tensor(
                        df[col_mapping[variant]].values.astype(np.float32), 
                        dtype=torch.float32, 
                        device=DEVICE
                    )
                    break
        
        # 针对每一列创建张量 (包括非标准列名)
        for col in columns:
            # 如果已经在标准化列中处理过，则跳过
            if col.lower() in [variant for variants in standard_cols.values() for variant in variants]:
                if any(col.lower() == variant for variant in variants for standard_col, variants in standard_cols.items() if standard_col in tensor_dict):
                    continue
            
            # 从DataFrame中提取数值列
            try:
                data_np = df[col].values.astype(np.float32)
                
                # 转换为PyTorch张量
                tensor_dict[col] = torch.tensor(data_np, dtype=torch.float32, device=DEVICE)
            except Exception as e:
                logger.warning(f"df_to_tensor: 无法转换列 {col} 为张量: {str(e)}")
    
    except Exception as e:
        logger.error(f"df_to_tensor: 处理DataFrame时出错: {str(e)}")
    
    # 检查输出
    if not tensor_dict:
        logger.warning("df_to_tensor: 未能创建任何张量，返回空字典")
    else:
        logger.debug(f"df_to_tensor: 成功创建张量字典，键名: {list(tensor_dict.keys())}")
    
    return tensor_dict

def tensor_to_df(tensor, columns, index=None):
    """
    将PyTorch张量转换回DataFrame
    
    参数:
        tensor: PyTorch张量
        columns: 列名列表
        index: 索引值，如时间戳列表
        
    返回:
        pandas DataFrame
    """
    # 确保张量在CPU上
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 转换为NumPy数组
    data_np = tensor.numpy()
    
    # 创建DataFrame
    return pd.DataFrame(data=data_np, columns=columns, index=index)

def rolling_window(tensor, window_size):
    """
    创建滚动窗口视图，用于计算移动平均等
    
    参数:
        tensor: 形状为 [N, C] 的张量，其中 N 是样本数，C 是特征数
        window_size: 窗口大小
        
    返回:
        形状为 [N-window_size+1, window_size, C] 的张量
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(1)
    
    shape = tensor.shape
    
    # 使用 unfold 创建窗口视图
    return tensor.unfold(0, window_size, 1)

def moving_average(tensor, window_size, weights=None):
    """
    计算移动平均
    
    参数:
        tensor: 输入张量
        window_size: 窗口大小
        weights: 权重张量，形状为 [window_size]，如果为None则计算简单移动平均
        
    返回:
        移动平均结果张量
    """
    # 确保张量在GPU上
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor, dtype=torch.float32, device=DEVICE)
    
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(1)
    
    # 创建滚动窗口视图
    windows = rolling_window(tensor, window_size)
    
    # 处理NaN值
    nan_mask = torch.isnan(windows)
    windows = torch.where(nan_mask, torch.zeros_like(windows), windows)
    
    # 计算移动平均
    if weights is not None:
        # 确保权重在GPU上
        weights = weights.to(DEVICE)
        # 加权移动平均
        ma = torch.sum(windows * weights.view(1, -1, 1), dim=1) / torch.sum(weights)
    else:
        # 简单移动平均
        ma = torch.nanmean(windows, dim=1)
    
    # 填充缺失的值（窗口前的部分）
    result = torch.full((tensor.shape[0], tensor.shape[1]), float('nan'), device=DEVICE)
    result[window_size-1:] = ma
    
    return result

def ewma(tensor, span, adjust=False):
    """
    计算指数加权移动平均 (EWMA)
    
    参数:
        tensor: 输入张量
        span: 跨度参数
        adjust: 是否调整权重衰减
        
    返回:
        EWMA结果张量
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(1)
    
    # 计算EWMA的alpha参数
    alpha = 2.0 / (span + 1.0)
    
    # 初始化结果
    result = torch.zeros_like(tensor)
    result[0] = tensor[0]
    
    # 迭代计算EWMA
    for i in range(1, len(tensor)):
        # 处理NaN值
        mask = ~torch.isnan(tensor[i])
        if adjust:
            # 调整衰减
            prev_val = torch.where(mask, result[i-1], torch.zeros_like(result[i-1]))
            result[i] = torch.where(mask, alpha * tensor[i] + (1 - alpha) * prev_val, result[i-1])
        else:
            # 标准EWMA
            result[i] = torch.where(mask, alpha * tensor[i] + (1 - alpha) * result[i-1], result[i-1])
    
    return result

def correlation(x, y, window_size):
    """
    计算两个张量的滚动相关系数
    
    参数:
        x: 第一个输入张量
        y: 第二个输入张量
        window_size: 窗口大小
        
    返回:
        滚动相关系数张量
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    
    # 创建滚动窗口
    x_windows = rolling_window(x, window_size)
    y_windows = rolling_window(y, window_size)
    
    # 计算窗口内的平均值
    x_mean = torch.nanmean(x_windows, dim=1, keepdim=True)
    y_mean = torch.nanmean(y_windows, dim=1, keepdim=True)
    
    # 计算协方差
    cov = torch.nanmean((x_windows - x_mean) * (y_windows - y_mean), dim=1)
    
    # 计算标准差
    x_std = torch.sqrt(torch.nanmean((x_windows - x_mean)**2, dim=1))
    y_std = torch.sqrt(torch.nanmean((y_windows - y_mean)**2, dim=1))
    
    # 计算相关系数
    corr = cov / (x_std * y_std)
    
    # 处理边界情况
    corr = torch.where(torch.isnan(corr) | torch.isinf(corr), torch.zeros_like(corr), corr)
    
    # 填充缺失的值
    result = torch.full((x.shape[0], x.shape[1]), float('nan'), device=DEVICE)
    result[window_size-1:] = corr
    
    return result 