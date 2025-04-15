"""
直接使用币安API获取新鲜数据的脚本
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import ccxt
import time
from tqdm import tqdm
import concurrent.futures
import glob
from dateutil.relativedelta import relativedelta
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import re

# 加载.env文件
load_dotenv()

# 获取API密钥
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

# 定义目标货币和时间框架
SYMBOLS = ["BTCUSDT"]  # 只获取BTCUSDT
TIMEFRAMES = ["1m"]    # 只获取1分钟数据
START_DATE = "2021-01-01"  # 从2021年开始
LIMIT = 1000  # 每次请求的记录数（币安API最大限制）
RAW_DATA_DIR = "data/raw"  # 原始数据存储目录

def get_historical_data(exchange, symbol, timeframe, since, limit=1000):
    """
    获取历史K线数据，支持获取长时间范围的数据（自动分批请求）
    
    参数:
        exchange: 交易所对象
        symbol: 交易对
        timeframe: 时间框架
        since: 开始时间戳
        limit: 每次请求的记录数
    """
    all_data = []
    current_since = since
    now = int(datetime.now().timestamp() * 1000)  # 当前时间戳
    
    print(f"开始获取 {symbol} 的 {timeframe} 历史数据...")
    
    # 获取起始时间和结束时间
    start_time = datetime.fromtimestamp(since/1000).strftime('%Y-%m-%d')
    end_time = datetime.fromtimestamp(now/1000).strftime('%Y-%m-%d')
    print(f"时间范围: {start_time} 到 {end_time}")
    
    # 使用tqdm显示进度条
    pbar = tqdm(total=None)  # 初始化进度条，不确定总长度
    
    try:
        while current_since < now:
            # 添加延迟，避免API限制
            time.sleep(0.5)
            
            # 获取历史数据
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_since, limit)
            
            if not ohlcv or len(ohlcv) == 0:
                print("没有更多数据，停止获取")
                break
                
            # 添加到总数据中
            all_data.extend(ohlcv)
            
            # 更新进度条
            pbar.update(len(ohlcv))
            pbar.set_description(f"已获取: {len(all_data)}")
            
            # 更新下一批数据的起始时间
            last_timestamp = ohlcv[-1][0]
            if last_timestamp <= current_since:
                print("时间戳没有推进，停止获取")
                break
                
            current_since = last_timestamp + 1  # 下一条数据的时间戳
        
        pbar.close()
        
        # 转换为DataFrame
        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 删除重复的行
            df = df.drop_duplicates(subset=['timestamp'])
            
            # 设置索引
            df.set_index('timestamp', inplace=True)
            
            print(f"已从Binance获取 {symbol} 的 {timeframe} 历史数据，共 {len(df)} 条记录")
            return df
        else:
            print("没有获取到任何数据")
            return None
    except Exception as e:
        print(f"获取历史数据时出错: {str(e)}")
        return None
    finally:
        pbar.close()

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

def sync_exchange_time(exchange):
    """
    同步本地时间与交易所服务器时间
    
    参数:
        exchange: 交易所对象
    
    返回:
        int: 本地时间与服务器时间的差值（毫秒）
    """
    try:
        # 获取服务器时间
        server_time = exchange.fetch_time()
        # 获取本地时间
        local_time = int(time.time() * 1000)
        # 计算时间差
        time_diff = local_time - server_time
        
        print(f"本地时间与服务器时间差: {time_diff} ms")
        
        # 如果时间差超过500毫秒，发出警告
        if abs(time_diff) > 500:
            print("警告: 本地时间与服务器时间差异较大，可能导致API请求失败")
            
        return time_diff
    except Exception as e:
        print(f"同步交易所时间时出错: {str(e)}")
        return 0

def create_date_ranges(start_date, end_date, months_per_chunk=1):
    """
    将给定的日期范围分割为按月的时间块
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        months_per_chunk: 每个时间块的月数

    Returns:
        日期范围列表 [(start_date1, end_date1), (start_date2, end_date2), ...]
    """
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    
    date_ranges = []
    current_start = start_date_obj
    
    while current_start < end_date_obj:
        # 计算当前块的结束日期
        if months_per_chunk == 1:
            # 如果每块1个月，则移动到下个月的同一天
            current_end = current_start.replace(day=1)
            if current_end.month == 12:
                next_month = current_end.replace(year=current_end.year + 1, month=1)
            else:
                next_month = current_end.replace(month=current_end.month + 1)
            
            # 获取下一个月的最后一天作为当前块的结束日期
            next_month_lastday = (next_month.replace(day=28) + timedelta(days=4))
            next_month_lastday = next_month_lastday - timedelta(days=next_month_lastday.day)
            current_end = min(next_month_lastday, end_date_obj)
        else:
            # 多月块，添加指定的月数
            months_to_add = months_per_chunk
            current_end = current_start
            
            for _ in range(months_to_add):
                if current_end.month == 12:
                    current_end = current_end.replace(year=current_end.year + 1, month=1)
                else:
                    current_end = current_end.replace(month=current_end.month + 1)
            
            # 确保不超过结束日期
            current_end = min(current_end, end_date_obj)
        
        date_ranges.append((
            current_start.strftime("%Y-%m-%d"),
            current_end.strftime("%Y-%m-%d")
        ))
        
        # 更新下一个块的开始日期
        current_start = current_end + timedelta(days=1)
    
    return date_ranges

def collect_data_for_symbol_timeframe(exchange, symbol, timeframe, start_date, end_date, months_per_chunk=3):
    """
    为特定交易对和时间框架收集历史数据，分块处理以减少内存使用
    
    参数:
        exchange: 交易所对象
        symbol: 交易对
        timeframe: 时间框架
        start_date: 开始日期，格式为 "YYYY-MM-DD"
        end_date: 结束日期，格式为 "YYYY-MM-DD"
        months_per_chunk: 每个处理块的月数
    
    返回:
        成功收集数据返回True，否则返回False
    """
    print(f"开始为 {symbol} {timeframe} 收集从 {start_date} 到 {end_date} 的历史数据")
    
    # 生成日期范围列表
    date_ranges = create_date_ranges(start_date, end_date, months_per_chunk)
    print(f"将时间范围划分为 {len(date_ranges)} 个块进行处理")
    
    # 创建进度文件路径
    progress_file = os.path.join(RAW_DATA_DIR, f"{symbol}_{timeframe}_progress.json")
    
    # 初始化或加载进度记录
    completed_ranges = []
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                completed_ranges = progress_data.get('completed_ranges', [])
            print(f"加载了进度记录，已完成 {len(completed_ranges)} 个日期范围")
        except Exception as e:
            print(f"读取进度文件时出错: {str(e)}")
    
    # 确定需要处理的日期范围
    ranges_to_process = [r for r in date_ranges if r not in completed_ranges]
    print(f"需要处理 {len(ranges_to_process)} 个日期范围，已完成 {len(completed_ranges)} 个")
    
    if not ranges_to_process:
        print(f"所有 {symbol} {timeframe} 的数据已收集完成")
        return True
    
    # 最终数据文件路径
    output_file = os.path.join(RAW_DATA_DIR, f"{symbol}_{timeframe}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv")
    
    # 存储每个块的数据
    all_dfs = []
    
    # 已有的完整数据文件
    existing_full_file = os.path.join(RAW_DATA_DIR, f"{symbol}_{timeframe}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv")
    if os.path.exists(existing_full_file):
        try:
            existing_df = pd.read_csv(existing_full_file, index_col=0, parse_dates=True)
            if not existing_df.empty:
                all_dfs.append(existing_df)
                print(f"加载了现有的完整数据文件，包含 {len(existing_df)} 条记录")
        except Exception as e:
            print(f"读取现有数据文件时出错: {str(e)}")
    
    try:
        # 处理每个日期范围
        for i, (chunk_start, chunk_end) in enumerate(ranges_to_process):
            print(f"处理第 {i+1}/{len(ranges_to_process)} 个日期块: {chunk_start} 到 {chunk_end}")
            
            # 收集特定日期范围的数据
            df = get_historical_data_for_period(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                since=chunk_start,
                until=chunk_end
            )
            
            if df is not None and not df.empty:
                all_dfs.append(df)
                
                # 更新进度记录
                completed_ranges.append((chunk_start, chunk_end))
                with open(progress_file, 'w') as f:
                    json.dump({'completed_ranges': completed_ranges}, f)
                print(f"更新了进度记录，已完成 {len(completed_ranges)}/{len(date_ranges)} 个范围")
                
                # 每完成一个块，尝试合并和保存当前所有数据
                if all_dfs:
                    try:
                        # 合并所有数据帧
                        merged_df = pd.concat(all_dfs)
                        
                        # 删除重复项并排序
                        merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
                        merged_df = merged_df.sort_index()
                        
                        # 保存合并后的数据
                        merged_df.to_csv(output_file)
                        print(f"已保存合并数据到 {output_file}，共 {len(merged_df)} 条记录")
                        
                        # 释放内存
                        del merged_df
                        import gc
                        gc.collect()
                    except Exception as e:
                        print(f"保存合并数据时出错: {str(e)}")
            else:
                print(f"没有获取到 {chunk_start} 到 {chunk_end} 的数据，跳过此日期范围")
            
            # 每次获取完一个时间块后，释放可能的大量内存
            del df
            import gc
            gc.collect()
        
        # 处理完所有日期范围后，最终合并并保存
        if all_dfs:
            try:
                # 合并所有数据帧
                final_df = pd.concat(all_dfs)
                
                # 删除重复项并排序
                final_df = final_df[~final_df.index.duplicated(keep='last')]
                final_df = final_df.sort_index()
                
                # 保存最终数据
                final_df.to_csv(output_file)
                print(f"已完成并保存 {symbol} {timeframe} 的所有数据到 {output_file}，共 {len(final_df)} 条记录")
                
                # 清理进度文件
                if os.path.exists(progress_file):
                    os.remove(progress_file)
                    print(f"已删除进度文件 {progress_file}")
                
                return True
            except Exception as e:
                print(f"保存最终数据时出错: {str(e)}")
                return False
        else:
            print(f"没有收集到任何 {symbol} {timeframe} 的数据")
            return False
    
    except KeyboardInterrupt:
        print("\n用户中断操作，保存当前进度...")
        # 保存当前进度
        with open(progress_file, 'w') as f:
            json.dump({'completed_ranges': completed_ranges}, f)
        print(f"已保存进度到 {progress_file}")
        return False
    
    except Exception as e:
        print(f"收集数据时出错: {str(e)}")
        # 保存当前进度
        with open(progress_file, 'w') as f:
            json.dump({'completed_ranges': completed_ranges}, f)
        print(f"已保存进度到 {progress_file}")
        return False

def get_historical_data_for_period(exchange, symbol, timeframe, since, until):
    """
    获取指定时间段的历史数据
    
    Args:
        exchange: CCXT交易所对象
        symbol: 交易对，例如 'BTC/USDT'
        timeframe: 时间框架，例如 '1h', '1d'
        since: 开始日期 (YYYY-MM-DD)
        until: 结束日期 (YYYY-MM-DD)
    
    Returns:
        包含历史数据的DataFrame
    """
    try:
        # 转换日期字符串为timestamp (毫秒)
        since_ts = int(datetime.strptime(since, "%Y-%m-%d").timestamp() * 1000)
        until_ts = int(datetime.strptime(until, "%Y-%m-%d").timestamp() * 1000)
        
        print(f"获取 {symbol} 的 {timeframe} 数据，从 {since} 到 {until}")
        
        # 使用get_historical_data函数获取数据
        df = get_historical_data(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            since=since_ts,
            until=until_ts
        )
        
        if df is not None and not df.empty:
            print(f"成功获取 {len(df)} 条 {symbol} {timeframe} 数据，从 {since} 到 {until}")
            return df
        else:
            print(f"没有获取到 {symbol} {timeframe} 数据，从 {since} 到 {until}")
            return None
    
    except Exception as e:
        print(f"获取 {symbol} {timeframe} 数据时出错: {str(e)}")
        return None

def collect_data_for_symbol_timeframe_chunk(exchange, symbol, timeframe, start_date, end_date, output_file, low_memory=False):
    """
    为特定交易对、时间框架和日期范围收集数据，并保存到指定文件
    
    Args:
        exchange: 交易所对象
        symbol: 交易对名称
        timeframe: 时间框架
        start_date: 开始日期
        end_date: 结束日期
        output_file: 输出文件路径
        low_memory: 是否启用低内存模式
    
    Returns:
        成功返回True，失败返回False
    """
    try:
        # 获取该时间范围的数据
        df = get_historical_data_for_period(
            exchange=exchange, 
            symbol=symbol, 
            timeframe=timeframe, 
            since=start_date, 
            until=end_date
        )
        
        if df is None or df.empty:
            print(f"警告: 没有获取到 {symbol} {timeframe} 数据 ({start_date} 到 {end_date})")
            return False
        
        # 保存数据
        # 提取日期部分用于文件名
        start_date_str = start_date.replace('-', '')
        end_date_str = end_date.replace('-', '')
        chunk_file = f"{symbol}_{timeframe}_{start_date_str}_{end_date_str}.csv"
        chunk_path = os.path.join(RAW_DATA_DIR, chunk_file)
        
        # 保存CSV文件
        df.to_csv(chunk_path)
        print(f"已将 {symbol} {timeframe} 数据保存到 {chunk_path}，共 {len(df)} 条记录")
        
        # 如果在低内存模式下，清除DataFrame以释放内存
        if low_memory:
            del df
            import gc
            gc.collect()
        
        return True
    
    except Exception as e:
        print(f"处理 {symbol} {timeframe} ({start_date} 到 {end_date}) 数据时出错: {str(e)}")
        return False

def merge_timeframe_files(symbol, timeframes):
    """
    合并同一个交易对不同时间范围的数据文件
    
    Args:
        symbol: 交易对，例如 'BTCUSDT'
        timeframes: 时间框架列表，例如 ['1h', '1d']
    """
    for timeframe in timeframes:
        # 查找该交易对和时间框架的所有文件
        pattern = f"{symbol}_{timeframe}_*.csv"
        files = glob.glob(os.path.join(RAW_DATA_DIR, pattern))
        
        if not files:
            print(f"没有找到 {symbol} {timeframe} 的数据文件")
            continue
        
        print(f"合并 {symbol} {timeframe} 的 {len(files)} 个数据文件")
        
        # 读取所有文件并合并
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file, index_col=0)
                dfs.append(df)
                print(f"已读取文件 {file}，包含 {len(df)} 条记录")
            except Exception as e:
                print(f"读取文件 {file} 时出错: {str(e)}")
        
        if not dfs:
            print(f"没有成功读取任何 {symbol} {timeframe} 的数据文件")
            continue
        
        # 合并所有数据框
        merged_df = pd.concat(dfs)
        
        # 按时间戳排序并去重
        merged_df = merged_df.sort_values('timestamp').drop_duplicates(subset='timestamp')
        
        # 提取文件名中的日期部分来命名合并文件
        # 使用第一个文件名中的日期
        first_file = os.path.basename(files[0])
        match = re.search(r'_(\d{8})_', first_file)
        date_suffix = match.group(1) if match else "merged"
        
        # 保存合并的数据
        merged_file = os.path.join(RAW_DATA_DIR, f"{symbol}_{timeframe}_{date_suffix}.csv")
        merged_df.to_csv(merged_file)
        print(f"已将 {symbol} {timeframe} 的数据合并并保存到 {merged_file}，共 {len(merged_df)} 条记录")
        
        # 可选：删除中间文件
        # for file in files:
        #     os.remove(file)
        # print(f"已删除 {len(files)} 个中间文件")

def main():
    """主函数，用于从API获取数据并保存到文件"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='获取加密货币历史数据')
    parser.add_argument('--start_date', type=str, default='2021-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), 
                        help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT'], 
                        help='要获取的交易对列表')
    parser.add_argument('--timeframes', nargs='+', default=['1d', '4h', '1h'], 
                        help='要获取的时间框架列表')
    parser.add_argument('--months_per_chunk', type=int, default=1, 
                        help='每个数据块的月数（低配置系统建议设为1）')
    parser.add_argument('--max_workers', type=int, default=2, 
                        help='最大并行工作线程数（低配置系统建议设为2）')
    parser.add_argument('--low_memory', action='store_true', 
                        help='启用低内存模式，减少内存使用')
    parser.add_argument('--force', action='store_true', 
                        help='强制重新获取已存在的数据文件')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # 初始化交易所
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True  # 自动调整时间差异
        }
    })
    
    # 测试交易所连接
    if not test_exchange_connection(exchange):
        print("无法连接到交易所。请检查API凭据和网络连接。")
        return
    
    print("交易所连接成功！")
    
    # 转换标准交易对格式 (BTCUSDT -> BTC/USDT)
    symbols_ccxt = [f"{symbol[:-4]}/{symbol[-4:]}" if symbol.endswith('USDT') else symbol 
                   for symbol in args.symbols]
    
    # 创建日期范围列表
    date_ranges = create_date_ranges(args.start_date, args.end_date, args.months_per_chunk)
    print(f"已将时间范围分割为 {len(date_ranges)} 个块")
    
    # 创建任务列表
    tasks = []
    
    for symbol, symbol_ccxt in zip(args.symbols, symbols_ccxt):
        for timeframe in args.timeframes:
            for start_date, end_date in date_ranges:
                # 构建输出文件路径
                start_date_str = start_date.replace('-', '')
                end_date_str = end_date.replace('-', '')
                output_file = os.path.join(RAW_DATA_DIR, f"{symbol}_{timeframe}_{start_date_str}_{end_date_str}.csv")
                
                # 检查文件是否已存在
                if os.path.exists(output_file) and not args.force:
                    print(f"文件 {output_file} 已存在，跳过（使用 --force 强制重新获取）")
                    continue
                
                # 添加到任务列表
                tasks.append((symbol, symbol_ccxt, timeframe, start_date, end_date, output_file))
    
    if not tasks:
        print("没有需要获取的数据。所有文件都已存在或没有符合条件的交易对/时间框架。")
        # 如果没有新任务，但仍需要合并文件，则执行合并
        for symbol in args.symbols:
            merge_timeframe_files(symbol, args.timeframes)
        return
    
    print(f"需要获取 {len(tasks)} 个数据块")
    
    # 使用线程池执行任务
    results = []
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(tasks))) as executor:
        futures = []
        
        for symbol, symbol_ccxt, timeframe, start_date, end_date, output_file in tasks:
            future = executor.submit(
                collect_data_for_symbol_timeframe_chunk,
                exchange, symbol_ccxt, timeframe, start_date, end_date, output_file, args.low_memory
            )
            futures.append(future)
        
        # 收集结果
        for future in as_completed(futures):
            results.append(future.result())
    
    # 统计成功和失败的任务
    success_count = sum(1 for result in results if result)
    fail_count = len(results) - success_count
    
    print(f"数据获取完成。成功: {success_count}, 失败: {fail_count}")
    
    # 合并每个交易对每个时间框架的文件
    for symbol in args.symbols:
        merge_timeframe_files(symbol, args.timeframes)
    
    print("所有操作完成！")

if __name__ == "__main__":
    main() 