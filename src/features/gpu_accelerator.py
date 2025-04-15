#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU加速器模块 - 用于加速特征计算过程，支持 PyTorch 和 RAPIDS
"""

import logging
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
logger = get_logger(__name__)

class GPUAccelerator:
    """GPU加速计算类，负责将CPU计算转换为GPU计算，支持 PyTorch 和 RAPIDS"""
    
    def __init__(self):
        """初始化GPU加速器，优先检测 PyTorch GPU 可用性，然后检查 RAPIDS 可用性"""
        self.gpu_available = False
        self.cudf = None
        self.cupy = None
        self.torch = None
        self.backend = None  # 'pytorch' 或 'rapids'
        
        # 首先尝试导入 PyTorch
        try:
            import torch
            self.torch = torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.backend = 'pytorch'
                logger.info(f"GPU加速可用 - PyTorch CUDA已成功加载 (设备: {torch.cuda.get_device_name(0)})")
                # 已找到 PyTorch，不再继续检查 RAPIDS
                return
            else:
                logger.warning("PyTorch已安装但CUDA不可用")
        except ImportError:
            logger.warning("PyTorch库不可用")
        
        # 如果 PyTorch 不可用或没有 CUDA，尝试 RAPIDS
        try:
            # 尝试导入GPU库
            import cudf
            import cupy
            self.cudf = cudf
            self.cupy = cupy
            self.gpu_available = True
            self.backend = 'rapids'
            logger.info("GPU加速可用 - RAPIDS库已成功加载")
        except ImportError:
            logger.warning("GPU加速不可用 - 既未找到PyTorch CUDA也未找到RAPIDS库，将使用CPU计算")
    
    def is_available(self):
        """检查GPU加速是否可用"""
        return self.gpu_available
    
    def get_backend(self):
        """获取当前使用的后端类型"""
        return self.backend
    
    def to_gpu(self, df):
        """将pandas DataFrame转换为GPU数据结构"""
        if not self.gpu_available:
            return df
        
        try:
            if self.backend == 'pytorch':
                # 对于 PyTorch，我们将DataFrame的值转换为PyTorch张量
                # 注意：这里只转换数值列，保持索引和其他非数值列不变
                numeric_cols = df.select_dtypes(include=['number']).columns
                result = df.copy()
                
                if len(numeric_cols) > 0:
                    tensor_dict = {}
                    for col in numeric_cols:
                        tensor_dict[col] = self.torch.tensor(df[col].values, 
                                                            dtype=self.torch.float32, 
                                                            device='cuda')
                    return {'df': result, 'tensors': tensor_dict, 'numeric_cols': numeric_cols}
                return {'df': result, 'tensors': {}, 'numeric_cols': []}
                
            elif self.backend == 'rapids':
                return self.cudf.DataFrame.from_pandas(df)
                
        except Exception as e:
            logger.error(f"转换到GPU失败: {e}")
            
        return df
    
    def to_cpu(self, gpu_data):
        """将GPU数据结构转换回pandas DataFrame"""
        if not self.gpu_available:
            return gpu_data
        
        try:
            if self.backend == 'pytorch':
                # 对于 PyTorch，将所有张量转换回 NumPy 数组并更新 DataFrame
                if isinstance(gpu_data, dict) and 'df' in gpu_data and 'tensors' in gpu_data:
                    result_df = gpu_data['df'].copy()
                    for col, tensor in gpu_data['tensors'].items():
                        if tensor is not None:
                            result_df[col] = tensor.cpu().numpy()
                    return result_df
                return gpu_data  # 如果不是预期的格式，原样返回
                
            elif self.backend == 'rapids':
                # 检查是否为cuDF DataFrame
                if hasattr(gpu_data, 'to_pandas'):
                    return gpu_data.to_pandas()
                
        except Exception as e:
            logger.error(f"转换到CPU失败: {e}")
            
        return gpu_data
    
    def rolling_mean(self, series, window):
        """使用GPU加速计算滚动平均"""
        if not self.gpu_available:
            return series.rolling(window).mean()
        
        try:
            if self.backend == 'pytorch':
                # 将数据转换为 PyTorch 张量
                device = self.torch.device('cuda')
                tensor = self.torch.tensor(series.values, dtype=self.torch.float32, device=device)
                result = self.torch.zeros_like(tensor)
                
                # 计算滚动平均
                for i in range(len(tensor)):
                    if i < window - 1:
                        # 前 window-1 个位置设为 NaN
                        result[i] = float('nan')
                    else:
                        # 滚动窗口平均值
                        result[i] = self.torch.mean(tensor[i-window+1:i+1])
                
                # 转回 pandas Series
                return pd.Series(result.cpu().numpy(), index=series.index)
                
            elif self.backend == 'rapids':
                # 假设series已经是GPU series
                return series.rolling(window).mean()
                
        except Exception as e:
            logger.error(f"GPU滚动平均计算失败: {e}")
            
        return series.rolling(window).mean()
    
    def rolling_std(self, series, window):
        """使用GPU加速计算滚动标准差"""
        if not self.gpu_available:
            return series.rolling(window).std()
        
        try:
            if self.backend == 'pytorch':
                # 将数据转换为 PyTorch 张量
                device = self.torch.device('cuda')
                tensor = self.torch.tensor(series.values, dtype=self.torch.float32, device=device)
                result = self.torch.zeros_like(tensor)
                
                # 计算滚动标准差
                for i in range(len(tensor)):
                    if i < window - 1:
                        # 前 window-1 个位置设为 NaN
                        result[i] = float('nan')
                    else:
                        # 滚动窗口标准差
                        result[i] = self.torch.std(tensor[i-window+1:i+1])
                
                # 转回 pandas Series
                return pd.Series(result.cpu().numpy(), index=series.index)
                
            elif self.backend == 'rapids':
                return series.rolling(window).std()
                
        except Exception as e:
            logger.error(f"GPU滚动标准差计算失败: {e}")
            
        return series.rolling(window).std()
    
    def correlation(self, x, y, window):
        """使用GPU加速计算滚动相关系数"""
        if not self.gpu_available:
            return x.rolling(window).corr(y)
        
        try:
            if self.backend == 'pytorch':
                # 将数据转换为 PyTorch 张量
                device = self.torch.device('cuda')
                x_tensor = self.torch.tensor(x.values, dtype=self.torch.float32, device=device)
                y_tensor = self.torch.tensor(y.values, dtype=self.torch.float32, device=device)
                result = self.torch.zeros_like(x_tensor)
                
                # 计算滚动相关系数
                for i in range(len(x_tensor)):
                    if i < window - 1:
                        # 前 window-1 个位置设为 NaN
                        result[i] = float('nan')
                    else:
                        # 获取当前窗口数据
                        x_window = x_tensor[i-window+1:i+1]
                        y_window = y_tensor[i-window+1:i+1]
                        
                        # 计算相关系数
                        x_mean = self.torch.mean(x_window)
                        y_mean = self.torch.mean(y_window)
                        x_std = self.torch.std(x_window)
                        y_std = self.torch.std(y_window)
                        
                        if x_std == 0 or y_std == 0:
                            result[i] = float('nan')
                        else:
                            # 计算协方差和相关系数
                            covar = self.torch.mean((x_window - x_mean) * (y_window - y_mean))
                            result[i] = covar / (x_std * y_std)
                
                # 转回 pandas Series
                return pd.Series(result.cpu().numpy(), index=x.index)
                
            elif self.backend == 'rapids':
                return x.rolling(window).corr(y)
                
        except Exception as e:
            logger.error(f"GPU相关系数计算失败: {e}")
            
        return x.rolling(window).corr(y)
            
    def exponential_moving_average(self, series, span):
        """使用GPU加速计算指数移动平均"""
        if not self.gpu_available:
            return series.ewm(span=span, adjust=False).mean()
        
        try:
            if self.backend == 'pytorch':
                # 将数据转换为 PyTorch 张量
                device = self.torch.device('cuda')
                tensor = self.torch.tensor(series.values, dtype=self.torch.float32, device=device)
                result = self.torch.zeros_like(tensor)
                
                # 计算EMA
                alpha = 2.0 / (span + 1.0)
                result[0] = tensor[0]  # 初始值
                
                for i in range(1, len(tensor)):
                    result[i] = alpha * tensor[i] + (1 - alpha) * result[i-1]
                
                # 转回 pandas Series
                return pd.Series(result.cpu().numpy(), index=series.index)
                
            elif self.backend == 'rapids':
                return series.ewm(span=span, adjust=False).mean()
                
        except Exception as e:
            logger.error(f"GPU指数移动平均计算失败: {e}")
            
        return series.ewm(span=span, adjust=False).mean()

# 单例模式，确保整个应用只有一个GPU加速器实例
_accelerator = None

def get_accelerator():
    """获取GPU加速器单例"""
    global _accelerator
    if _accelerator is None:
        _accelerator = GPUAccelerator()
    return _accelerator 