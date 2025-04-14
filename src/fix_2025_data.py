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
from datetime import datetime
import shutil

# 数据目录配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
TEMP_DIR = os.path.join(BASE_DIR, 'data', 'temp')

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

def process_2025_data():
    """处理2025年的数据"""
    # 查找所有2025年的目录
    pattern = os.path.join(TEMP_DIR, "BTCUSDT_1m_2025_*")
    dirs = glob.glob(pattern)
    
    if not dirs:
        print("未找到2025年数据目录")
        return
    
    print(f"找到 {len(dirs)} 个2025年数据目录")
    
    # 处理每个目录
    all_dfs = []
    for dir_path in dirs:
        month = os.path.basename(dir_path).split('_')[-1]
        
        # 找到目录中的CSV文件
        csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
        if not csv_files:
            print(f"目录 {dir_path} 中没有CSV文件")
            continue
        
        # 修复每个CSV文件
        for csv_file in csv_files:
            df = fix_timestamp_parsing(csv_file)
            if df is not None:
                # 保存修复后的数据
                output_path = os.path.join(RAW_DATA_DIR, f"BTCUSDT_1m_2025_{month}.csv")
                df.to_csv(output_path, index=False)
                print(f"已将修复后的数据保存到 {output_path}，共 {len(df)} 条记录")
                all_dfs.append(df)
    
    # 合并所有修复的数据
    if all_dfs:
        merged_df = pd.concat(all_dfs)
        # 时间戳是字符串格式，按字符串排序
        merged_df = merged_df.sort_values('timestamp').drop_duplicates(subset='timestamp')
        
        # 获取主数据文件
        main_files = glob.glob(os.path.join(RAW_DATA_DIR, "BTCUSDT_1m_*.csv"))
        if main_files:
            # 找到主要的数据文件
            main_file = sorted(main_files)[-1]
            print(f"找到主数据文件: {main_file}")
            
            # 加载主数据
            main_df = pd.read_csv(main_file)
            
            # 合并数据
            combined_df = pd.concat([main_df, merged_df])
            # 排序并去重
            combined_df = combined_df.sort_values('timestamp').drop_duplicates(subset='timestamp')
            
            # 保存完整数据
            output_path = os.path.join(RAW_DATA_DIR, f"BTCUSDT_1m_20210101_20250414_fixed.csv")
            combined_df.to_csv(output_path, index=False)
            print(f"已保存完整数据到 {output_path}，共 {len(combined_df)} 条记录")
        else:
            print("未找到主数据文件")
    else:
        print("没有成功修复任何2025年数据")

def download_april_2025():
    """手动获取2025年4月数据（部分月份）"""
    try:
        from src.download_binance_data import download_file, extract_zip, verify_checksum
        
        # 创建临时目录
        temp_dir = os.path.join(TEMP_DIR, "BTCUSDT_1m_2025_04")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 下载当日数据而非月度数据
        # 构建今天的日期字符串
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 币安提供当日数据的URL格式
        filename = f"BTCUSDT-1m-{today}.zip"
        checksum_filename = f"{filename}.CHECKSUM"
        
        # 构建URL（使用每日数据而非月度数据）
        url_prefix = f"https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m"
        file_url = f"{url_prefix}/{filename}"
        checksum_url = f"{url_prefix}/{checksum_filename}"
        
        print(f"尝试下载当日数据: {file_url}")
        
        # 下载ZIP文件
        zip_path = download_file(file_url, TEMP_DIR, filename)
        if not zip_path:
            print("下载当日数据失败")
            return
        
        # 下载并验证校验和
        checksum_path = download_file(checksum_url, TEMP_DIR, checksum_filename)
        if checksum_path and not verify_checksum(zip_path, checksum_path):
            print(f"校验和验证失败，跳过当日数据")
            return
        
        # 解压ZIP文件
        csv_files = extract_zip(zip_path, temp_dir)
        if not csv_files:
            print("解压当日数据失败")
            return
        
        # 使用修复方法处理
        for csv_file in csv_files:
            df = fix_timestamp_parsing(csv_file)
            if df is not None:
                # 保存修复后的数据
                output_path = os.path.join(RAW_DATA_DIR, f"BTCUSDT_1m_2025_04_partial.csv")
                df.to_csv(output_path, index=False)
                print(f"已将当日数据保存到 {output_path}，共 {len(df)} 条记录")
        
        print("当日数据处理完成")
    
    except Exception as e:
        print(f"处理当日数据失败: {e}")

def main():
    """主函数"""
    # 确保输出目录存在
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 处理已下载的2025年数据
    process_2025_data()
    
    # 尝试获取2025年4月的部分数据（当日数据）
    download_april_2025()
    
    print("修复完成！")

if __name__ == "__main__":
    main() 