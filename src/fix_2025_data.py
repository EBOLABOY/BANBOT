#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复2025年数据解析问题
针对"Out of bounds nanosecond timestamp"错误
"""

import os
import sys
import glob
import pandas as pd
import zipfile
import requests
from datetime import datetime
import shutil

# 数据目录配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
TEMP_DIR = os.path.join(BASE_DIR, 'data', 'temp')

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
    
    # 如果文件已存在，直接返回路径
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
            
            # 写入文件
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB
                    f.write(chunk)
            
            return dest_path
        
        except requests.exceptions.RequestException as e:
            print(f"下载失败 ({attempts+1}/{retry}): {str(e)}")
            attempts += 1
    
    print(f"下载 {url} 失败，已达到最大重试次数")
    return None

def fix_timestamp_parsing(csv_file):
    """
    修复CSV文件中的时间戳解析问题
    
    Args:
        csv_file: CSV文件路径
    
    Returns:
        修复后的DataFrame或None
    """
    try:
        print(f"修复文件: {csv_file}")
        
        # 读取CSV时不自动转换时间戳
        df = pd.read_csv(csv_file, names=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'count', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # 将时间戳转换为日期时间字符串，而非datetime对象
        # 这样避免pandas的时间戳边界限制
        df['timestamp'] = df['timestamp'].apply(
            lambda x: datetime.fromtimestamp(int(x)/1000).strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # 仅保留必要的列
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # 转换价格和成交量为浮点数
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df
    
    except Exception as e:
        print(f"修复文件 {csv_file} 失败: {str(e)}")
        return None

def process_zip_file(zip_path, month):
    """
    处理ZIP文件，解压并修复其中的CSV文件
    
    Args:
        zip_path: ZIP文件路径
        month: 月份 (如 '01', '02', '03')
    
    Returns:
        修复后的DataFrame或None
    """
    try:
        # 创建临时目录
        temp_dir = os.path.join(TEMP_DIR, f"BTCUSDT_1m_2025_{month}_tmp")
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"解压 {zip_path} 到 {temp_dir}")
        
        # 解压ZIP文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # 查找解压后的CSV文件
        csv_files = glob.glob(os.path.join(temp_dir, "*.csv"))
        
        if not csv_files:
            print(f"在 {zip_path} 中未找到CSV文件")
            return None
        
        # 处理每个CSV文件
        dfs = []
        for csv_file in csv_files:
            df = fix_timestamp_parsing(csv_file)
            if df is not None:
                dfs.append(df)
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        if not dfs:
            return None
        
        # 合并所有DataFrame
        merged_df = pd.concat(dfs)
        
        # 排序并去重
        merged_df = merged_df.sort_values('timestamp').drop_duplicates(subset='timestamp')
        
        # 保存到RAW_DATA_DIR
        output_path = os.path.join(RAW_DATA_DIR, f"BTCUSDT_1m_2025_{month}.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"已将2025-{month}数据保存到 {output_path}，共 {len(merged_df)} 条记录")
        
        return merged_df
    
    except Exception as e:
        print(f"处理ZIP文件 {zip_path} 失败: {str(e)}")
        return None

def process_2025_data():
    """处理2025年的数据"""
    # 尝试从URL直接下载2025年数据
    months = ['01', '02', '03']
    
    # 存储所有成功处理的数据
    all_dfs = []
    
    for month in months:
        # 构建文件名
        filename = f"BTCUSDT-1m-2025-{month}.zip"
        
        # 尝试从原始数据目录查找文件
        zip_path = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.exists(zip_path):
            # 如果不存在，则尝试下载
            url = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/{filename}"
            zip_path = download_file(url, TEMP_DIR, filename)
        
        if zip_path and os.path.exists(zip_path):
            df = process_zip_file(zip_path, month)
            if df is not None:
                all_dfs.append(df)
    
    # 合并所有修复的数据
    if all_dfs:
        print("合并所有2025年数据...")
        merged_2025_df = pd.concat(all_dfs)
        merged_2025_df = merged_2025_df.sort_values('timestamp').drop_duplicates(subset='timestamp')
        
        # 查找2021-2024年的主数据文件
        main_files = [f for f in glob.glob(os.path.join(RAW_DATA_DIR, "BTCUSDT_1m_*.csv")) 
                     if "2025" not in f and "fixed" not in f]
        
        if main_files:
            # 找到主要数据文件
            main_file = sorted(main_files)[-1]
            print(f"找到主数据文件: {main_file}")
            
            # 加载主数据
            main_df = pd.read_csv(main_file)
            
            # 合并数据
            combined_df = pd.concat([main_df, merged_2025_df])
            combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset='timestamp')
            
            # 保存完整数据
            output_path = os.path.join(RAW_DATA_DIR, f"BTCUSDT_1m_20210101_20250414_fixed.csv")
            combined_df.to_csv(output_path, index=False)
            print(f"已保存完整数据到 {output_path}，共 {len(combined_df)} 条记录")
        else:
            # 如果没有主数据文件，只保存2025年数据
            output_path = os.path.join(RAW_DATA_DIR, f"BTCUSDT_1m_2025_merged.csv")
            merged_2025_df.to_csv(output_path, index=False)
            print(f"已保存2025年数据到 {output_path}，共 {len(merged_2025_df)} 条记录")
    else:
        print("没有成功处理任何2025年数据")

def download_latest_data():
    """下载最新的数据（2025年1-3月）"""
    # 下载2025年1-3月数据
    months = ['01', '02', '03']
    
    for month in months:
        # 构建文件名
        filename = f"BTCUSDT-1m-2025-{month}.zip"
        
        # 构建URL
        url = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/{filename}"
        print(f"尝试下载: {url}")
        
        zip_path = download_file(url, TEMP_DIR, filename)
        if zip_path:
            # 如果下载成功，处理ZIP文件
            df = process_zip_file(zip_path, month)
            if df is not None:
                print(f"成功处理2025-{month}月数据，获取了 {len(df)} 条记录")
    
    print("2025年数据下载尝试完成")

def main():
    """主函数"""
    # 确保目录存在
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    print("开始处理2025年数据...")
    
    # 下载并处理2025年1-3月数据
    download_latest_data()
    
    # 合并处理所有已下载数据
    process_2025_data()
    
    print("2025年数据处理完成！")

if __name__ == "__main__":
    main() 