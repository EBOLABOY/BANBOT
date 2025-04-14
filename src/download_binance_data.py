#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从Binance数据仓库下载历史K线数据
https://data.binance.vision/
"""

import os
import re
import sys
import time
import zipfile
import hashlib
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 数据目录配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
TEMP_DIR = os.path.join(BASE_DIR, 'data', 'temp')
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Binance数据URL配置
BASE_URL = "https://data.binance.vision/data"

def download_file(url, dest_folder, filename, retry=3):
    """
    下载文件并显示进度条
    
    Args:
        url: 文件URL
        dest_folder: 目标文件夹
        filename: 文件名
        retry: 重试次数
    
    Returns:
        下载的文件路径或None（如果下载失败）
    """
    dest_path = os.path.join(dest_folder, filename)
    
    # 如果文件已存在，检查是否完整
    if os.path.exists(dest_path):
        print(f"文件已存在: {dest_path}")
        return dest_path
    
    attempts = 0
    while attempts < retry:
        try:
            print(f"下载 {url} 到 {dest_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB
            
            # 使用tqdm创建进度条
            with open(dest_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    bar.update(len(data))
            
            return dest_path
        
        except requests.exceptions.RequestException as e:
            print(f"下载失败 ({attempts+1}/{retry}): {e}")
            attempts += 1
            time.sleep(5)  # 等待5秒后重试
    
    print(f"下载 {url} 失败，已达到最大重试次数")
    return None

def verify_checksum(file_path, checksum_path):
    """
    验证文件的SHA256校验和
    
    Args:
        file_path: 文件路径
        checksum_path: 校验和文件路径
    
    Returns:
        验证成功返回True，失败返回False
    """
    try:
        # 读取校验和文件
        with open(checksum_path, 'r') as f:
            expected_checksum = f.read().strip().split()[0]
        
        # 计算文件校验和
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        calculated_checksum = sha256_hash.hexdigest()
        
        # 验证校验和
        if expected_checksum == calculated_checksum:
            print(f"校验和验证成功: {os.path.basename(file_path)}")
            return True
        else:
            print(f"校验和验证失败: {os.path.basename(file_path)}")
            print(f"预期: {expected_checksum}")
            print(f"实际: {calculated_checksum}")
            return False
    
    except Exception as e:
        print(f"校验和验证出错: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """
    解压ZIP文件
    
    Args:
        zip_path: ZIP文件路径
        extract_to: 解压目标目录
    
    Returns:
        解压的CSV文件路径列表或空列表（如果解压失败）
    """
    try:
        print(f"解压 {zip_path} 到 {extract_to}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # 返回解压的CSV文件路径列表
        csv_files = [os.path.join(extract_to, f) for f in os.listdir(extract_to) 
                    if f.endswith('.csv') and os.path.isfile(os.path.join(extract_to, f))]
        
        return csv_files
    
    except Exception as e:
        print(f"解压 {zip_path} 失败: {e}")
        return []

def parse_csv(csv_path):
    """
    解析CSV文件到DataFrame
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        pandas DataFrame或None（如果解析失败）
    """
    try:
        # Binance K线CSV格式: open_time,open,high,low,close,volume,close_time,quote_volume,...
        df = pd.read_csv(csv_path, names=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'count', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # 转换时间戳为日期时间
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 仅保留必要的列
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # 转换价格和成交量为浮点数
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df
    
    except Exception as e:
        print(f"解析 {csv_path} 失败: {e}")
        return None

def process_month_data(symbol, interval, year, month, output_dir, temp_dir, verify=True):
    """
    处理单个月的数据：下载、验证、解压和解析
    
    Args:
        symbol: 交易对，例如 'BTCUSDT'
        interval: 时间间隔，例如 '1m'
        year: 年份，例如 2021
        month: 月份，例如 1
        output_dir: 输出目录
        temp_dir: 临时目录
        verify: 是否验证校验和
    
    Returns:
        成功返回解析后的DataFrame，失败返回None
    """
    # 构建文件名和URL
    ym_str = f"{year}-{month:02d}"
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    checksum_filename = f"{filename}.CHECKSUM"
    
    # 构建URL
    url_prefix = f"{BASE_URL}/spot/monthly/klines/{symbol}/{interval}"
    file_url = f"{url_prefix}/{filename}"
    checksum_url = f"{url_prefix}/{checksum_filename}"
    
    # 创建每月临时目录
    month_temp_dir = os.path.join(temp_dir, f"{symbol}_{interval}_{year}_{month:02d}")
    os.makedirs(month_temp_dir, exist_ok=True)
    
    try:
        # 1. 下载ZIP文件
        zip_path = download_file(file_url, temp_dir, filename)
        if not zip_path:
            return None
        
        # 2. 下载并验证校验和
        if verify:
            checksum_path = download_file(checksum_url, temp_dir, checksum_filename)
            if checksum_path and not verify_checksum(zip_path, checksum_path):
                print(f"校验和验证失败，跳过 {filename}")
                return None
        
        # 3. 解压ZIP文件
        csv_files = extract_zip(zip_path, month_temp_dir)
        if not csv_files:
            return None
        
        # 4. 解析CSV文件
        dfs = []
        for csv_file in csv_files:
            df = parse_csv(csv_file)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            return None
        
        # 5. 合并DataFrame
        combined_df = pd.concat(dfs)
        
        # 6. 排序并去重
        combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset='timestamp')
        
        # 7. 保存月度数据
        output_path = os.path.join(output_dir, f"{symbol}_{interval}_{year}_{month:02d}.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"已保存 {symbol} {interval} {year}-{month:02d} 数据，共 {len(combined_df)} 条记录")
        
        return combined_df
    
    except Exception as e:
        print(f"处理 {symbol} {interval} {year}-{month:02d} 数据失败: {e}")
        return None
    
    finally:
        # 清理临时文件
        try:
            import shutil
            if os.path.exists(month_temp_dir):
                shutil.rmtree(month_temp_dir)
        except Exception as e:
            print(f"清理临时文件失败: {e}")

def download_historical_data(symbol, interval, start_date, end_date, max_workers=4, verify=True):
    """
    下载指定交易对和时间范围的历史数据
    
    Args:
        symbol: 交易对，例如 'BTCUSDT'
        interval: 时间间隔，例如 '1m'
        start_date: 开始日期，'YYYY-MM-DD'格式
        end_date: 结束日期，'YYYY-MM-DD'格式
        max_workers: 最大工作线程数
        verify: 是否验证校验和
    
    Returns:
        合并后的完整数据DataFrame
    """
    # 解析日期范围
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # 生成年月列表
    months = []
    current_dt = start_dt
    while current_dt <= end_dt:
        months.append((current_dt.year, current_dt.month))
        # 移动到下个月
        if current_dt.month == 12:
            current_dt = datetime(current_dt.year + 1, 1, 1)
        else:
            current_dt = datetime(current_dt.year, current_dt.month + 1, 1)
    
    print(f"将下载 {symbol} {interval} 从 {start_date} 到 {end_date} 的数据，共 {len(months)} 个月")
    
    # 创建输出目录
    symbol_dir = os.path.join(RAW_DATA_DIR, f"{symbol}_{interval}")
    os.makedirs(symbol_dir, exist_ok=True)
    
    # 使用线程池并行下载数据
    monthly_dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for year, month in months:
            future = executor.submit(
                process_month_data,
                symbol, interval, year, month, symbol_dir, TEMP_DIR, verify
            )
            futures.append((future, year, month))
        
        # 收集结果
        for future, year, month in tqdm(futures, desc="处理月度数据"):
            try:
                df = future.result()
                if df is not None:
                    monthly_dfs.append(df)
                    print(f"已处理 {year}-{month:02d} 数据，共 {len(df)} 条记录")
                else:
                    print(f"处理 {year}-{month:02d} 数据失败")
            except Exception as e:
                print(f"获取 {year}-{month:02d} 的结果时出错: {e}")
    
    if not monthly_dfs:
        print("没有下载到任何数据")
        return None
    
    # 合并所有月度数据
    print(f"合并 {len(monthly_dfs)} 个月的数据...")
    all_data = pd.concat(monthly_dfs)
    
    # 排序并去重
    all_data = all_data.sort_values('timestamp').drop_duplicates(subset='timestamp')
    
    # 保存完整数据
    output_path = os.path.join(RAW_DATA_DIR, f"{symbol}_{interval}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv")
    all_data.to_csv(output_path, index=False)
    print(f"已保存完整数据到 {output_path}，共 {len(all_data)} 条记录")
    
    return all_data

def main():
    """主函数，解析命令行参数并下载数据"""
    parser = argparse.ArgumentParser(description='从Binance数据仓库下载历史K线数据')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='交易对，例如 BTCUSDT')
    parser.add_argument('--interval', type=str, default='1m', help='K线间隔，例如 1m, 5m, 15m, 1h, 4h, 1d')
    parser.add_argument('--start_date', type=str, default='2021-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--max_workers', type=int, default=4, help='最大并行工作线程数')
    parser.add_argument('--no_verify', action='store_true', help='跳过校验和验证')
    parser.add_argument('--clear_temp', action='store_true', help='下载完成后清理临时文件')
    
    args = parser.parse_args()
    
    try:
        # 下载并处理数据
        download_historical_data(
            args.symbol,
            args.interval,
            args.start_date,
            args.end_date,
            args.max_workers,
            not args.no_verify
        )
        
        # 清理临时文件
        if args.clear_temp and os.path.exists(TEMP_DIR):
            import shutil
            shutil.rmtree(TEMP_DIR)
            print(f"已清理临时目录 {TEMP_DIR}")
        
        print("数据下载和处理完成！")
    
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 