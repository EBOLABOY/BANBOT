"""
数据管理器模块
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.config import load_config, get_config_value
from .data_collector import DataCollector
from .data_processor import DataProcessor

logger = get_logger(__name__)

class DataManager:
    """
    数据管理器，用于集中管理和合并来自不同来源的数据
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        初始化数据管理器
        
        参数:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.data_config = self.config.get("data", {})
        
        # 初始化数据收集器和处理器
        self.collector = DataCollector(config_path)
        self.processor = DataProcessor(config_path)
        
        # 数据路径
        self.raw_data_path = "data/raw"
        self.processed_data_path = "data/processed"
        self.merged_data_path = "data/processed/merged"
        os.makedirs(self.merged_data_path, exist_ok=True)
        
        # 目标货币和时间框架
        self.target_currencies = self.data_config.get("target_currencies", ["BTC"])
        self.base_currency = self.data_config.get("base_currency", "USDT")
        self.timeframes = self.data_config.get("timeframes", [1, 5, 15, 60, 240, 1440])
        
        # 时间框架映射（分钟 -> CCXT格式）
        self.timeframe_map = {
            1: "1m",
            5: "5m",
            15: "15m",
            30: "30m",
            60: "1h",
            120: "2h",
            240: "4h",
            360: "6h",
            480: "8h",
            720: "12h",
            1440: "1d",
            10080: "1w",
        }
        
        logger.info("数据管理器已初始化")
    
    def collect_and_process_data(self, start_date=None, end_date=None, parallel=True):
        """
        收集并处理数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            parallel: 是否并行处理
            
        返回:
            是否成功的布尔值
        """
        try:
            # 收集历史数据
            logger.info("开始收集历史数据")
            success_count = self.collector.collect_historical_data(parallel=parallel)
            
            if success_count == 0:
                logger.warning("没有成功收集到任何数据")
                return False
            
            # 处理数据
            logger.info("开始处理收集到的数据")
            raw_files = self.processor.list_raw_data_files()
            processed_count = self.processor.process_files(raw_files, clean=True, resample=False)
            
            if processed_count == 0:
                logger.warning("没有成功处理到任何数据")
                return False
            
            logger.info("数据收集和处理完成")
            return True
        
        except Exception as e:
            logger.error(f"数据收集和处理时出错: {e}")
            return False
    
    def get_data_file_list(self, symbol=None, timeframe=None):
        """
        获取数据文件列表
        
        参数:
            symbol: 交易对符号
            timeframe: 时间框架字符串
            
        返回:
            文件路径列表
        """
        pattern = ""
        
        if symbol:
            pattern += f"{symbol}_"
        
        if timeframe:
            pattern += f"{timeframe}_"
        
        # 查找已处理的文件
        processed_files = glob.glob(f"{self.processed_data_path}/processed_*{pattern}*.csv")
        
        if not processed_files:
            # 如果没有处理过的文件，查找原始文件
            raw_files = glob.glob(f"{self.raw_data_path}/*{pattern}*.csv")
            return raw_files
        
        return processed_files
    
    def load_symbol_data(self, symbol, timeframe, start_date=None, end_date=None):
        """
        加载指定交易对和时间框架的数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间框架（分钟或字符串）
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            DataFrame对象
        """
        # 将时间框架转换为字符串格式
        if isinstance(timeframe, int) and timeframe in self.timeframe_map:
            timeframe_str = self.timeframe_map[timeframe]
        else:
            timeframe_str = timeframe
        
        # 获取数据文件列表
        files = self.get_data_file_list(symbol, timeframe_str)
        
        if not files:
            logger.warning(f"找不到 {symbol} 的 {timeframe_str} 数据文件")
            return None
        
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                
                # 尝试设置时间索引
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                
                # 筛选日期范围
                if start_date:
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date)
                    df = df[df.index >= start_date]
                
                if end_date:
                    if isinstance(end_date, str):
                        end_date = pd.to_datetime(end_date)
                    df = df[df.index <= end_date]
                
                dfs.append(df)
            
            except Exception as e:
                logger.error(f"加载数据文件 {file} 时出错: {e}")
        
        if not dfs:
            logger.warning(f"无法加载 {symbol} 的 {timeframe_str} 数据")
            return None
        
        # 合并数据框
        if len(dfs) == 1:
            result_df = dfs[0]
        else:
            # 合并多个数据框，并按索引排序
            result_df = pd.concat(dfs)
            result_df = result_df[~result_df.index.duplicated(keep='first')]
            result_df.sort_index(inplace=True)
        
        logger.info(f"已加载 {symbol} 的 {timeframe_str} 数据，共 {len(result_df)} 条记录")
        return result_df
    
    def merge_data_sources(self, start_date=None, end_date=None):
        """
        合并多个数据源
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            合并后的数据集字典
        """
        merged_data = {}
        
        # 为每个货币和时间框架组合创建合并数据
        symbols = [f"{currency}{self.base_currency}" for currency in self.target_currencies]
        
        for symbol in tqdm(symbols, desc="合并数据"):
            merged_data[symbol] = {}
            
            for timeframe in self.timeframes:
                if timeframe in self.timeframe_map:
                    timeframe_str = self.timeframe_map[timeframe]
                    
                    # 加载数据
                    df = self.load_symbol_data(symbol, timeframe_str, start_date, end_date)
                    
                    if df is not None and not df.empty:
                        # 保存合并后的数据
                        merged_data[symbol][timeframe_str] = df
                        
                        # 保存到文件
                        start_date_str = "all" if start_date is None else pd.to_datetime(start_date).strftime("%Y%m%d")
                        end_date_str = "now" if end_date is None else pd.to_datetime(end_date).strftime("%Y%m%d")
                        
                        filename = f"{symbol}_{timeframe_str}_{start_date_str}_{end_date_str}.csv"
                        filepath = os.path.join(self.merged_data_path, filename)
                        
                        df.to_csv(filepath)
                        logger.info(f"已合并并保存 {symbol} 的 {timeframe_str} 数据到 {filepath}，共 {len(df)} 条记录")
        
        logger.info(f"数据合并完成，共有 {len(merged_data)} 个交易对的数据")
        return merged_data
    
    def get_latest_data(self, symbol, timeframe, n_periods=1):
        """
        获取最新的n条数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间框架
            n_periods: 获取的周期数
            
        返回:
            最新的数据DataFrame
        """
        df = self.load_symbol_data(symbol, timeframe)
        
        if df is None or df.empty:
            logger.warning(f"找不到 {symbol} 的 {timeframe} 数据")
            return None
        
        # 返回最后n条记录
        return df.iloc[-n_periods:]
    
    def update_data(self, symbols=None, timeframes=None):
        """
        更新现有数据，添加最新的市场数据
        
        参数:
            symbols: 要更新的交易对列表
            timeframes: 要更新的时间框架列表
            
        返回:
            成功更新的数据项数量
        """
        if symbols is None:
            symbols = [f"{currency}{self.base_currency}" for currency in self.target_currencies]
        
        if timeframes is None:
            timeframes = [self.timeframe_map[tf] for tf in self.timeframes if tf in self.timeframe_map]
        
        # 收集当前市场数据
        current_data = self.collector.collect_current_data()
        
        update_count = 0
        
        # 更新每个数据文件
        for symbol in symbols:
            if symbol not in current_data:
                logger.warning(f"没有 {symbol} 的最新数据")
                continue
            
            symbol_data = current_data[symbol]
            
            for timeframe in timeframes:
                try:
                    # 加载现有数据
                    df = self.load_symbol_data(symbol, timeframe)
                    
                    if df is None:
                        logger.warning(f"找不到 {symbol} 的 {timeframe} 现有数据，无法更新")
                        continue
                    
                    # 提取当前价格和时间
                    ticker_data = None
                    for key, value in symbol_data.items():
                        if "ticker" in key:
                            ticker_data = value
                            break
                    
                    if ticker_data is None:
                        logger.warning(f"没有 {symbol} 的交易行情数据")
                        continue
                    
                    # 创建新的数据行
                    now = datetime.now()
                    new_row = pd.DataFrame({
                        "timestamp": [now],
                        "open": [float(ticker_data.get("last", 0))],
                        "high": [float(ticker_data.get("last", 0))],
                        "low": [float(ticker_data.get("last", 0))],
                        "close": [float(ticker_data.get("last", 0))],
                        "volume": [0.0]  # 无法获取当前周期的交易量
                    })
                    
                    # 设置时间索引
                    new_row["timestamp"] = pd.to_datetime(new_row["timestamp"])
                    new_row.set_index("timestamp", inplace=True)
                    
                    # 追加新行到现有数据
                    df = pd.concat([df, new_row])
                    
                    # 去除重复的时间戳
                    df = df[~df.index.duplicated(keep='last')]
                    df.sort_index(inplace=True)
                    
                    # 保存更新后的数据
                    start_date_str = df.index.min().strftime("%Y%m%d")
                    end_date_str = df.index.max().strftime("%Y%m%d")
                    
                    filename = f"{symbol}_{timeframe}_{start_date_str}_{end_date_str}.csv"
                    filepath = os.path.join(self.merged_data_path, filename)
                    
                    df.to_csv(filepath)
                    logger.info(f"已更新并保存 {symbol} 的 {timeframe} 数据到 {filepath}，共 {len(df)} 条记录")
                    
                    update_count += 1
                
                except Exception as e:
                    logger.error(f"更新 {symbol} 的 {timeframe} 数据时出错: {e}")
        
        logger.info(f"数据更新完成，共成功更新 {update_count} 项数据")
        return update_count 