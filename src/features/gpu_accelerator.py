#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU加速器模块 - 用于加速特征计算过程
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class GPUAccelerator:
    """GPU加速计算类，负责将CPU计算转换为GPU计算"""
    
    def __init__(self):
        """初始化GPU加速器，检测GPU可用性"""
        self.gpu_available = False
        self.cudf = None
        self.cupy = None
        
        try:
            # 尝试导入GPU库
            import cudf
            import cupy
            self.cudf = cudf
            self.cupy = cupy
            self.gpu_available = True
            logger.info("GPU加速可用 - RAPIDS库已成功加载")
        except ImportError:
            logger.warning("GPU加速不可用 - 未找到RAPIDS库，将使用CPU计算")
    
    def is_available(self):
        """检查GPU加速是否可用"""
        return self.gpu_available
    
    def to_gpu(self, df):
        """将pandas DataFrame转换为GPU DataFrame"""
        if not self.gpu_available:
            return df
        
        try:
            return self.cudf.DataFrame.from_pandas(df)
        except Exception as e:
            logger.error(f"转换到GPU失败: {e}")
            return df
    
    def to_cpu(self, gpu_df):
        """将GPU DataFrame转换回pandas DataFrame"""
        if not self.gpu_available:
            return gpu_df
        
        try:
            # 检查是否为cuDF DataFrame
            if hasattr(gpu_df, 'to_pandas'):
                return gpu_df.to_pandas()
            return gpu_df
        except Exception as e:
            logger.error(f"转换到CPU失败: {e}")
            return gpu_df
    
    def rolling_mean(self, series, window):
        """使用GPU加速计算滚动平均"""
        if not self.gpu_available:
            return series.rolling(window).mean()
        
        try:
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
            return series.rolling(window).std()
        except Exception as e:
            logger.error(f"GPU滚动标准差计算失败: {e}")
            return series.rolling(window).std()
    
    def correlation(self, x, y, window):
        """使用GPU加速计算滚动相关系数"""
        if not self.gpu_available:
            return x.rolling(window).corr(y)
        
        try:
            return x.rolling(window).corr(y)
        except Exception as e:
            logger.error(f"GPU相关系数计算失败: {e}")
            return x.rolling(window).corr(y)
            
    def exponential_moving_average(self, series, span):
        """使用GPU加速计算指数移动平均"""
        if not self.gpu_available:
            return series.ewm(span=span, adjust=False).mean()
        
        try:
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