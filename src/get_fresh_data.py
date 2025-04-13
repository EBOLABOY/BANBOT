"""
直接使用币安API获取新鲜数据的脚本
"""

import os
import sys
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import ccxt
import time
from tqdm import tqdm
import concurrent.futures

# 加载.env文件
load_dotenv()

# 获取API密钥
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

# 定义目标货币和时间框架
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
START_DATE = "2022-01-01"  # 数据开始日期
LIMIT = 1000  # 每次请求的记录数
RAW_DATA_DIR = "data/raw"  # 原始数据存储目录

def get_historical_data(exchange, symbol, timeframe, since, limit=1000):
    """
    获取历史K线数据
    
    参数:
        exchange: 交易所对象
        symbol: 交易对
        timeframe: 时间框架
        since: 开始时间戳
        limit: 每次请求的记录数
    """
    try:
        # 获取历史数据
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        
        # 转换为DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"已从Binance获取 {symbol} 的 {timeframe} 历史数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"获取历史数据时出错: {str(e)}")
        return None

def save_data_to_csv(df, symbol, timeframe, start_date):
    """
    保存数据到CSV文件
    
    参数:
        df: 数据DataFrame
        symbol: 交易对
        timeframe: 时间框架
        start_date: 开始日期
    """
    try:
        if df is None or df.empty:
            print(f"没有可保存的数据: {symbol}_{timeframe}")
            return False
            
        # 创建文件名
        date_suffix = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
        filename = f"{symbol}_{timeframe}_{date_suffix}.csv"
        filepath = os.path.join(RAW_DATA_DIR, filename)
        
        # 确保目录存在
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        
        # 保存到CSV
        df.to_csv(filepath)
        print(f"已保存 {symbol} 的 {timeframe} 数据到 {filepath}，共 {len(df)} 条记录")
        return True
    except Exception as e:
        print(f"保存数据时出错: {str(e)}")
        return False

def collect_data_for_symbol_timeframe(symbol, timeframe):
    """
    收集单个交易对和时间框架的数据
    """
    try:
        # 初始化交易所对象
        exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # 计算开始时间
        since = int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp() * 1000)
        
        # 获取历史数据
        df = get_historical_data(exchange, symbol, timeframe, since, LIMIT)
        
        # 保存数据
        if df is not None:
            save_data_to_csv(df, symbol, timeframe, START_DATE)
            return symbol, timeframe, True
        else:
            return symbol, timeframe, False
    except Exception as e:
        print(f"收集 {symbol} {timeframe} 数据时发生错误: {str(e)}")
        return symbol, timeframe, False

def main():
    """主函数"""
    print(f"开始从Binance获取数据，使用API密钥: {api_key[:5]}...{api_key[-5:]}")
    
    # 确保目录存在
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # 创建任务列表
    tasks = []
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            tasks.append((symbol, timeframe))
    
    print(f"开始收集 {len(tasks)} 个数据集...")
    
    # 并行收集数据
    successful = 0
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for symbol, timeframe in tasks:
            future = executor.submit(collect_data_for_symbol_timeframe, symbol, timeframe)
            futures.append(future)
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            symbol, timeframe, success = future.result()
            if success:
                successful += 1
            else:
                failed += 1
    
    print(f"数据收集完成: 成功 {successful}/{len(tasks)}, 失败 {failed}/{len(tasks)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 