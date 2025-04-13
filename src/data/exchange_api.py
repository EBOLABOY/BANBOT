"""
交易所API接口模块
"""

import time
from abc import ABC, abstractmethod
import pandas as pd
import logging
import ccxt
from datetime import datetime, timedelta

from ..utils.logger import get_logger

logger = get_logger(__name__)

class ExchangeAPI(ABC):
    """
    交易所API抽象基类
    """
    
    def __init__(self, api_key=None, api_secret=None):
        """
        初始化交易所API接口
        
        参数:
            api_key: API密钥
            api_secret: API密钥
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange_name = "unknown"
    
    @abstractmethod
    def get_historical_data(self, symbol, timeframe, since, limit=None):
        """
        获取历史K线数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间框架
            since: 开始时间戳
            limit: 返回记录的最大数量
            
        返回:
            DataFrame包含OHLCV数据
        """
        pass
    
    @abstractmethod
    def get_ticker(self, symbol):
        """
        获取当前市场行情
        
        参数:
            symbol: 交易对符号
            
        返回:
            字典包含当前价格信息
        """
        pass
    
    @abstractmethod
    def get_order_book(self, symbol, limit=None):
        """
        获取市场深度数据
        
        参数:
            symbol: 交易对符号
            limit: 返回深度的最大数量
            
        返回:
            字典包含买卖盘数据
        """
        pass


class CCXTExchangeAPI(ExchangeAPI):
    """
    使用CCXT库实现的通用交易所API
    """
    
    def __init__(self, exchange_id, api_key=None, api_secret=None):
        """
        初始化CCXT交易所API
        
        参数:
            exchange_id: 交易所ID (binance, coinbase等)
            api_key: API密钥
            api_secret: API密钥
        """
        super().__init__(api_key, api_secret)
        
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        self.exchange_name = exchange_id
        logger.info(f"已初始化 {exchange_id} 交易所API")
    
    def get_historical_data(self, symbol, timeframe, since, limit=1000):
        """
        获取历史K线数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间框架 (1m, 5m, 15m, 1h, 4h, 1d等)
            since: 开始时间戳或日期字符串
            limit: 返回记录的最大数量
            
        返回:
            DataFrame包含OHLCV数据
        """
        try:
            # 处理日期字符串
            if isinstance(since, str):
                since = int(datetime.strptime(since, "%Y-%m-%d").timestamp() * 1000)
            
            # 获取历史数据
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"已从 {self.exchange_name} 获取 {symbol} 的 {timeframe} 历史数据，共 {len(df)} 条记录")
            return df
        
        except Exception as e:
            logger.error(f"获取历史数据时出错: {e}")
            raise
    
    def get_ticker(self, symbol):
        """
        获取当前市场行情
        
        参数:
            symbol: 交易对符号
            
        返回:
            字典包含当前价格信息
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.debug(f"已获取 {symbol} 的当前行情")
            return ticker
        except Exception as e:
            logger.error(f"获取当前行情时出错: {e}")
            raise
    
    def get_order_book(self, symbol, limit=20):
        """
        获取市场深度数据
        
        参数:
            symbol: 交易对符号
            limit: 返回深度的最大数量
            
        返回:
            字典包含买卖盘数据
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            logger.debug(f"已获取 {symbol} 的市场深度数据，深度为 {limit}")
            return order_book
        except Exception as e:
            logger.error(f"获取市场深度数据时出错: {e}")
            raise
    
    def get_exchange_info(self):
        """
        获取交易所信息
        
        返回:
            交易所信息字典
        """
        try:
            markets = self.exchange.load_markets()
            return {
                'name': self.exchange_name,
                'symbols': list(markets.keys()),
                'timeframes': list(self.exchange.timeframes.keys()) if hasattr(self.exchange, 'timeframes') else []
            }
        except Exception as e:
            logger.error(f"获取交易所信息时出错: {e}")
            raise


class BinanceAPI(CCXTExchangeAPI):
    """
    针对Binance交易所的API实现
    """
    
    def __init__(self, api_key=None, api_secret=None):
        """
        初始化Binance API
        
        参数:
            api_key: API密钥
            api_secret: API密钥
        """
        super().__init__('binance', api_key, api_secret)
    
    def get_funding_rate(self, symbol):
        """
        获取资金费率数据（期货特有）
        
        参数:
            symbol: 交易对符号
            
        返回:
            资金费率数据字典
        """
        try:
            # 确保使用期货市场
            if 'options' in self.exchange.options:
                old_type = self.exchange.options.get('defaultType', 'spot')
                self.exchange.options['defaultType'] = 'future'
            
            funding_rate = self.exchange.fetch_funding_rate(symbol)
            
            # 恢复原来的市场类型
            if 'options' in self.exchange.options and 'old_type' in locals():
                self.exchange.options['defaultType'] = old_type
            
            logger.debug(f"已获取 {symbol} 的资金费率数据")
            return funding_rate
        except Exception as e:
            logger.error(f"获取资金费率数据时出错: {e}")
            # 恢复原来的市场类型以防异常
            if 'options' in self.exchange.options and 'old_type' in locals():
                self.exchange.options['defaultType'] = old_type
            raise


class CoinbaseAPI(CCXTExchangeAPI):
    """
    针对Coinbase交易所的API实现
    """
    
    def __init__(self, api_key=None, api_secret=None):
        """
        初始化Coinbase API
        
        参数:
            api_key: API密钥
            api_secret: API密钥
        """
        super().__init__('coinbasepro', api_key, api_secret) 