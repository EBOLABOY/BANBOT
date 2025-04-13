"""
数据收集器模块
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import concurrent.futures

from ..utils.logger import get_logger
from ..utils.config import load_config, get_config_value
from .exchange_api import BinanceAPI, CoinbaseAPI

logger = get_logger(__name__)

class DataCollector:
    """
    加密货币数据收集器
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        初始化数据收集器
        
        参数:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.data_config = self.config.get("data", {})
        
        # 初始化交易所API
        self.exchange_apis = {}
        self._init_exchange_apis()
        
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
        
        # 数据存储路径
        self.raw_data_path = "data/raw"
        os.makedirs(self.raw_data_path, exist_ok=True)
        
        logger.info("数据收集器已初始化")
    
    def _init_exchange_apis(self):
        """
        初始化交易所API列表
        """
        sources = self.data_config.get("sources", [])
        
        for source in sources:
            name = source.get("name", "").lower()
            api_key = source.get("api_key", None)
            api_secret = source.get("api_secret", None)
            
            try:
                if name == "binance":
                    self.exchange_apis[name] = BinanceAPI(api_key, api_secret)
                elif name == "coinbase":
                    self.exchange_apis[name] = CoinbaseAPI(api_key, api_secret)
                else:
                    logger.warning(f"不支持的交易所: {name}")
            except Exception as e:
                logger.error(f"初始化 {name} API时出错: {e}")
        
        if not self.exchange_apis:
            logger.warning("没有成功初始化任何交易所API")
    
    def get_historical_data(self, symbol, timeframe, start_date, end_date=None, exchange_name=None):
        """
        获取历史K线数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间框架（分钟）
            start_date: 开始日期
            end_date: 结束日期（默认为当前时间）
            exchange_name: 指定交易所名称（默认使用优先级最高的）
            
        返回:
            DataFrame包含OHLCV数据
        """
        if not self.exchange_apis:
            logger.error("没有可用的交易所API")
            return None
        
        # 默认使用第一个可用的交易所
        if exchange_name is None:
            exchange_name = list(self.exchange_apis.keys())[0]
        
        if exchange_name not in self.exchange_apis:
            logger.error(f"交易所 {exchange_name} 不可用")
            return None
        
        # 转换时间框架为CCXT格式
        if timeframe in self.timeframe_map:
            timeframe_str = self.timeframe_map[timeframe]
        else:
            logger.error(f"不支持的时间框架: {timeframe}")
            return None
        
        # 设置结束日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # 获取数据
            api = self.exchange_apis[exchange_name]
            df = api.get_historical_data(symbol, timeframe_str, start_date)
            
            # 过滤日期范围
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            df = df[df.index <= end_date]
            
            return df
        
        except Exception as e:
            logger.error(f"获取 {symbol} 的历史数据时出错: {e}")
            return None
    
    def collect_historical_data(self, parallel=True, max_workers=4):
        """
        收集所有目标货币的历史数据
        
        参数:
            parallel: 是否并行收集
            max_workers: 最大工作线程数
            
        返回:
            成功收集的数据计数
        """
        historical_config = self.data_config.get("historical", {})
        start_date = historical_config.get("start_date", "2022-01-01")
        end_date = historical_config.get("end_date", None)
        
        symbols = [f"{currency}{self.base_currency}" for currency in self.target_currencies]
        tasks = []
        
        for symbol in symbols:
            for timeframe in self.timeframes:
                if timeframe in self.timeframe_map:
                    tasks.append((symbol, timeframe))
        
        logger.info(f"开始收集历史数据，共 {len(tasks)} 个任务")
        
        results = []
        if parallel and len(tasks) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for symbol, timeframe in tasks:
                    future = executor.submit(
                        self._collect_and_save_data, symbol, timeframe, start_date, end_date
                    )
                    futures.append(future)
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            for symbol, timeframe in tqdm(tasks):
                result = self._collect_and_save_data(symbol, timeframe, start_date, end_date)
                if result:
                    results.append(result)
        
        success_count = len(results)
        logger.info(f"历史数据收集完成，成功: {success_count}/{len(tasks)}")
        
        return success_count
    
    def _collect_and_save_data(self, symbol, timeframe, start_date, end_date):
        """
        收集并保存单个交易对和时间框架的数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间框架（分钟）
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            成功则返回(symbol, timeframe)，否则返回None
        """
        try:
            # 尝试从所有可用交易所获取数据
            df = None
            for exchange_name in self.exchange_apis:
                try:
                    df = self.get_historical_data(
                        symbol, timeframe, start_date, end_date, exchange_name
                    )
                    if df is not None and not df.empty:
                        break
                except Exception as e:
                    logger.warning(f"从 {exchange_name} 获取 {symbol} 数据时出错: {e}")
            
            if df is None or df.empty:
                logger.warning(f"无法获取 {symbol} 的 {timeframe} 分钟数据")
                return None
            
            # 保存数据
            timeframe_str = self.timeframe_map[timeframe]
            filename = f"{symbol}_{timeframe_str}_{start_date.replace('-', '')}.csv"
            filepath = os.path.join(self.raw_data_path, filename)
            
            df.to_csv(filepath)
            logger.info(f"已保存 {symbol} 的 {timeframe_str} 数据到 {filepath}，共 {len(df)} 条记录")
            
            return (symbol, timeframe)
        
        except Exception as e:
            logger.error(f"收集并保存 {symbol} 的 {timeframe} 分钟数据时出错: {e}")
            return None
    
    def collect_current_data(self):
        """
        收集当前市场数据（行情和订单簿）
        
        返回:
            包含当前数据的字典
        """
        symbols = [f"{currency}{self.base_currency}" for currency in self.target_currencies]
        current_data = {}
        
        for symbol in symbols:
            current_data[symbol] = {}
            
            # 从所有可用交易所获取数据
            for exchange_name, api in self.exchange_apis.items():
                try:
                    # 获取行情
                    ticker = api.get_ticker(symbol)
                    if ticker:
                        current_data[symbol][f"{exchange_name}_ticker"] = ticker
                    
                    # 获取订单簿
                    order_book = api.get_order_book(symbol)
                    if order_book:
                        current_data[symbol][f"{exchange_name}_order_book"] = order_book
                
                except Exception as e:
                    logger.warning(f"从 {exchange_name} 获取 {symbol} 当前数据时出错: {e}")
        
        # 保存当前数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"current_data_{timestamp}.csv"
        filepath = os.path.join(self.raw_data_path, filename)
        
        # 提取关键信息并保存为CSV
        data_rows = []
        for symbol, data in current_data.items():
            for source, values in data.items():
                if "ticker" in source:
                    row = {
                        "symbol": symbol,
                        "source": source,
                        "timestamp": datetime.now(),
                        "last_price": values.get("last", None),
                        "bid": values.get("bid", None),
                        "ask": values.get("ask", None),
                        "volume_24h": values.get("volume", None),
                    }
                    data_rows.append(row)
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            df.to_csv(filepath, index=False)
            logger.info(f"已保存当前市场数据到 {filepath}，共 {len(df)} 条记录")
        
        return current_data 