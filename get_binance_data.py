from binance.client import Client
import pandas as pd
from dotenv import load_dotenv
import os
import time
from datetime import datetime, timedelta

# 加载 .env 文件
load_dotenv()

# 获取API密钥
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

# 初始化币安客户端
client = Client(api_key, api_secret)

# --- 配置参数 ---
symbol = "BTCUSDT"  # 现货交易对，用于K线
futures_symbol = "BTCUSDT" # 合约交易对，用于资金费率 (如果需要)
interval = "1h"  # K线数据间隔
start_str = "1 Jan, 2021"  # 开始时间
end_str = None  # 结束时间 (None表示到现在)
output_dir = "data/processed" # 输出目录
# 修改输出文件名，明确不包含持仓量
output_filename = f"{output_dir}/btc_with_funding_rate_features.csv"

# --- 辅助函数：分批获取数据，处理API限制 ---
def get_paginated_data(fetch_func, symbol, start_time_ms, end_time_ms, limit=1000, delay_seconds=1, **kwargs):
    all_data = []
    current_start = start_time_ms
    while current_start < end_time_ms:
        print(f"  获取数据从 {pd.to_datetime(current_start, unit='ms')}...")
        try:
            # 修正：确保kwargs能正确传递给fetch_func
            data = fetch_func(symbol=symbol, startTime=current_start, limit=limit, **kwargs)
            if not data:
                break # 没有更多数据
            all_data.extend(data)
            # 更新下一次请求的开始时间
            # K线和合约数据的时间戳通常是开盘时间，需要基于最后一条数据的时间来推进
            # 检查 data[-1] 的类型以获取时间戳
            if isinstance(data[-1], dict) and 'timestamp' in data[-1]:
                last_timestamp = data[-1]['timestamp']
            elif isinstance(data[-1], dict) and 'fundingTime' in data[-1]: # 资金费率用 fundingTime
                last_timestamp = data[-1]['fundingTime']
            elif isinstance(data[-1], list) and len(data[-1]) > 0:
                last_timestamp = data[-1][0] # K线是列表，第一个元素是时间戳
            else:
                print("警告：无法确定最后一条数据的时间戳，可能导致获取中断。")
                break
                
            current_start = last_timestamp + 1 # 移动到下一条记录的时间
            
            # 如果获取的数据少于limit，说明已经到了末尾附近
            if len(data) < limit:
                print("  获取的数据量小于limit，可能已到达数据末尾。")
                break
                
        except Exception as e:
            print(f"    获取数据时出错: {e}. 等待 {delay_seconds} 秒后重试...")
            time.sleep(delay_seconds)
        # 稍微增加请求间隔，避免过于频繁
        time.sleep(max(0.2, delay_seconds / 2)) # 尊重API频率限制
    return all_data

# --- 1. 获取K线数据 (OHLCV + Taker Volume) ---
print(f"开始获取 {symbol} K线数据 ({interval}) 从 {start_str}...")
try:
    # 转换开始和结束时间为毫秒时间戳
    start_ts = int(client._get_earliest_valid_timestamp(symbol, interval))
    if start_str:
        # 尝试更健壮的时间解析
        try:
            start_dt = pd.to_datetime(start_str)
            start_ts = int(start_dt.timestamp() * 1000)
        except ValueError:
            print(f"错误：无法解析开始时间字符串 '{start_str}'. 请使用 'YYYY-MM-DD' 或类似格式。")
            exit()
            
    end_ts = int(datetime.now().timestamp() * 1000)
    if end_str:
        try:
            end_dt = pd.to_datetime(end_str)
            end_ts = int(end_dt.timestamp() * 1000)
        except ValueError:
            print(f"错误：无法解析结束时间字符串 '{end_str}'. 请使用 'YYYY-MM-DD' 或类似格式。")
            exit()
    
    klines = []
    current_start = start_ts
    while current_start < end_ts:
        limit = 1000 # 每次最多获取1000条
        print(f"  获取K线数据从 {pd.to_datetime(current_start, unit='ms')}...")
        try:
            # 确保结束时间不会超过当前时间
            # batch_end_ts = min(end_ts, int(datetime.now().timestamp() * 1000)) # 暂时移除，因为endTime不受支持
            fetched_klines = client.get_historical_klines(symbol, interval, current_start, limit=limit)
            if not fetched_klines:
                print("  未获取到新的K线数据，结束获取。")
                break # 没有更多数据
            klines.extend(fetched_klines)
            
            # 更新下一次请求的开始时间
            # K线数据返回的是[开盘时间,...], 下次请求从最后一条的开盘时间+间隔开始
            interval_map = {'m': 60*1000, 'h': 60*60*1000, 'd': 24*60*60*1000}
            interval_ms = int(interval[:-1]) * interval_map[interval[-1]]
            next_start_time = fetched_klines[-1][0] + interval_ms
            
            # 如果计算出的下一个开始时间超过了结束时间，就停止
            if next_start_time >= end_ts:
                break
                
            current_start = next_start_time
            
            # 如果获取的数据少于limit，不一定意味着结束，继续尝试直到时间范围完成
            # if len(fetched_klines) < limit:
            #     print("  获取的K线数据少于limit，可能是最后批次。")
            #     break 
                
        except Exception as e:
            print(f"    获取K线数据时出错: {e}. 等待1秒后重试...")
            time.sleep(1)
        time.sleep(0.5) # 轻微延迟

    print(f"成功获取 {len(klines)} 条K线数据")
    
    if not klines:
        print("错误：未能获取任何K线数据，请检查参数和API连接。")
        exit()

    # 转换为Pandas DataFrame
    kline_df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    kline_df['timestamp'] = pd.to_datetime(kline_df['timestamp'], unit='ms')
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                     'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    kline_df[numeric_columns] = kline_df[numeric_columns].apply(pd.to_numeric)
    
    # 重命名taker volume列以匹配模型预期
    kline_df.rename(columns={'taker_buy_base_asset_volume': 'taker_buy_volume', 
                             'taker_buy_quote_asset_volume': 'taker_buy_quote_volume'}, inplace=True)
    # 计算taker sell volume
    kline_df['taker_sell_volume'] = kline_df['volume'] - kline_df['taker_buy_volume']
    
    # 选择需要的列
    kline_df = kline_df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
                         'number_of_trades', 'taker_buy_volume', 'taker_sell_volume']]
    
    # 去重，保留第一个出现的记录 (防止API可能返回重叠数据)
    kline_df = kline_df.drop_duplicates(subset=['timestamp'], keep='first')
    kline_df = kline_df.sort_values('timestamp').reset_index(drop=True)
    
    print("K线数据处理完成.")
    
except Exception as e:
    print(f"获取K线数据过程中发生严重错误: {e}")
    exit()

# --- 2. 获取资金费率历史 (Funding Rate History) ---
print(f"\n开始获取 {futures_symbol} 资金费率历史...")
funding_rate_data = []
try:
    # 注意：资金费率通常是每8小时一次，与1h K线频率不同
    # 获取资金费率历史的API是 client.futures_funding_rate
    funding_rates_raw = get_paginated_data(
        client.futures_funding_rate, 
        symbol=futures_symbol, 
        start_time_ms=start_ts, 
        end_time_ms=end_ts,
        limit=1000 # API限制
    )
    
    if funding_rates_raw:
        funding_rate_df = pd.DataFrame(funding_rates_raw)
        funding_rate_df['fundingTime'] = pd.to_datetime(funding_rate_df['fundingTime'], unit='ms')
        funding_rate_df['fundingRate'] = pd.to_numeric(funding_rate_df['fundingRate'])
        # 重命名并选择列
        funding_rate_df = funding_rate_df[['fundingTime', 'fundingRate']].rename(columns={'fundingTime': 'timestamp', 'fundingRate': 'funding_rate'})
        # 去重并排序
        funding_rate_df = funding_rate_df.drop_duplicates(subset=['timestamp'], keep='first')
        funding_rate_df = funding_rate_df.sort_values('timestamp').reset_index(drop=True)
        print(f"成功获取 {len(funding_rate_df)} 条资金费率数据")
    else:
        print("未能获取资金费率数据，可能API不支持或无数据。")
        funding_rate_df = pd.DataFrame(columns=['timestamp', 'funding_rate']) # 创建空DataFrame

except Exception as e:
    print(f"获取资金费率时出错: {e}. 继续执行...")
    funding_rate_df = pd.DataFrame(columns=['timestamp', 'funding_rate']) # 创建空DataFrame

# --- 3. 合并数据 --- (移除持仓量部分)
print("\n开始合并K线和资金费率数据...")

# 将K线数据设为主DataFrame
merged_df = kline_df.copy()

# 合并资金费率数据
# 由于资金费率频率低（通常8小时），我们需要将其填充到每小时
# 使用 merge_asof 进行前向填充，匹配到最近的资金费率时间点
if not funding_rate_df.empty:
    # 确保两个DataFrame都已排序
    merged_df = merged_df.sort_values('timestamp')
    funding_rate_df = funding_rate_df.sort_values('timestamp')
    
    merged_df = pd.merge_asof(
        merged_df, 
        funding_rate_df, 
        on='timestamp', 
        direction='backward' # 向后查找匹配（使用之前的费率）
    )
    print("资金费率数据已合并.")
else:
    merged_df['funding_rate'] = pd.NA # 如果没获取到，填充NA
    print("未找到资金费率数据进行合并。")

# 填充合并后可能产生的NaN值 (例如开始时没有历史费率)
# 使用ffill填充，表示沿用上一时刻的值
if 'funding_rate' in merged_df.columns:
    initial_nan_count = merged_df['funding_rate'].isna().sum()
    merged_df['funding_rate'] = merged_df['funding_rate'].ffill()
    filled_nan_count = initial_nan_count - merged_df['funding_rate'].isna().sum()
    print(f"使用前向填充填补了 {filled_nan_count} 个资金费率的NaN值。")
    # 如果开头仍有NaN，可以用0或其他合理值填充，或删除这些行
    if merged_df['funding_rate'].isna().any():
        print("警告：数据开头部分仍存在资金费率NaN值，将用0填充。")
        merged_df['funding_rate'].fillna(0, inplace=True)

# --- 4. 保存最终合并的数据 ---
print("\n数据合并完成.")

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 保存到CSV文件
merged_df.to_csv(output_filename, index=False)
print(f"所有合并后的数据已保存到 {output_filename}")

# 查看最终数据的前几行和信息
print("\n最终合并数据前5行:")
print(merged_df.head())
print("\n最终合并数据信息:")
merged_df.info()
print(f"\n总行数: {len(merged_df)}")
print(f"日期范围: {merged_df['timestamp'].min()} 到 {merged_df['timestamp'].max()}")
print("\n数据获取和合并流程结束.") 