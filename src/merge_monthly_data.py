#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并月度加密货币数据文件
将目录下所有月度数据文件合并为一个完整的CSV文件
"""

import os
import glob
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import sys

# 数据目录配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

def merge_monthly_files(symbol, interval, start_year=2021, start_month=1, end_year=2025, end_month=3):
    """
    合并指定时间范围内的月度数据文件
    
    Args:
        symbol: 交易对，例如 'BTCUSDT'
        interval: 时间间隔，例如 '1m'
        start_year: 开始年份
        start_month: 开始月份
        end_year: 结束年份
        end_month: 结束月份
    
    Returns:
        合并后的DataFrame
    """
    # 构建目录路径
    data_dir = os.path.join(RAW_DATA_DIR, f"{symbol}_{interval}")
    
    # 如果指定目录不存在，尝试在RAW_DATA_DIR中直接查找
    if not os.path.exists(data_dir):
        print(f"目录 {data_dir} 不存在，将在 {RAW_DATA_DIR} 中查找文件")
        data_dir = RAW_DATA_DIR
    
    # 构建文件模式
    file_pattern = os.path.join(data_dir, f"{symbol}_{interval}_*.csv")
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print(f"未找到任何匹配 {file_pattern} 的文件")
        return None
    
    print(f"找到 {len(all_files)} 个数据文件")
    
    # 过滤文件，只保留指定时间范围内的
    filtered_files = []
    for file in all_files:
        # 尝试从文件名提取年月信息
        try:
            # 假设文件名格式为 BTCUSDT_1m_2021_01.csv
            parts = os.path.basename(file).split('_')
            if len(parts) >= 4:
                year = int(parts[-2])
                month = int(parts[-1].split('.')[0])
                
                # 检查是否在指定范围内
                if (year > start_year or (year == start_year and month >= start_month)) and \
                   (year < end_year or (year == end_year and month <= end_month)):
                    filtered_files.append((file, year, month))
        except Exception as e:
            print(f"无法从文件 {file} 提取日期信息: {e}")
    
    if not filtered_files:
        print(f"没有找到指定时间范围内的文件")
        return None
    
    # 按年月排序
    filtered_files.sort(key=lambda x: (x[1], x[2]))
    print(f"要合并的文件数量: {len(filtered_files)}")
    
    # 读取并合并所有文件
    all_dfs = []
    total_rows = 0
    
    print("开始读取文件...")
    for file_info in tqdm(filtered_files):
        file, year, month = file_info
        try:
            df = pd.read_csv(file)
            # 如果timestamp列不是日期时间格式，尝试转换
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except:
                    # 如果无法转换，保持原样
                    pass
            
            rows = len(df)
            total_rows += rows
            all_dfs.append(df)
            print(f"读取文件 {file}，包含 {rows} 条记录")
        except Exception as e:
            print(f"读取文件 {file} 失败: {e}")
    
    if not all_dfs:
        print("没有成功读取任何文件")
        return None
    
    print(f"合并 {len(all_dfs)} 个数据文件...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # 排序和去重
    print("对合并后的数据进行排序和去重...")
    try:
        # 如果timestamp是日期时间格式，按时间排序
        if pd.api.types.is_datetime64_any_dtype(merged_df['timestamp']):
            merged_df = merged_df.sort_values('timestamp')
        else:  # 否则按字符串排序
            merged_df = merged_df.sort_values('timestamp')
    except:
        # 如果排序失败，跳过
        print("排序失败，跳过排序步骤")
    
    # 去重
    merged_df = merged_df.drop_duplicates(subset='timestamp')
    
    # 检查合并后的行数
    print(f"合并前总行数: {total_rows}")
    print(f"合并后行数: {len(merged_df)}")
    print(f"删除了 {total_rows - len(merged_df)} 条重复记录")
    
    return merged_df

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='合并月度加密货币数据文件')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='交易对，例如 BTCUSDT')
    parser.add_argument('--interval', type=str, default='1m', help='时间间隔，例如 1m, 1h, 1d')
    parser.add_argument('--start_year', type=int, default=2021, help='开始年份')
    parser.add_argument('--start_month', type=int, default=1, help='开始月份')
    parser.add_argument('--end_year', type=int, default=2025, help='结束年份')
    parser.add_argument('--end_month', type=int, default=3, help='结束月份')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径，默认为 symbol_interval_startdate_enddate.csv')
    
    args = parser.parse_args()
    
    # 合并文件
    merged_df = merge_monthly_files(
        args.symbol,
        args.interval,
        args.start_year,
        args.start_month,
        args.end_year,
        args.end_month
    )
    
    if merged_df is None:
        print("合并失败，未获取到有效数据")
        sys.exit(1)
    
    # 构建输出文件路径
    if args.output:
        output_path = args.output
    else:
        start_date = f"{args.start_year}{args.start_month:02d}01"
        end_date = f"{args.end_year}{args.end_month:02d}01"
        output_path = os.path.join(RAW_DATA_DIR, f"{args.symbol}_{args.interval}_{start_date}_{end_date}.csv")
    
    # 保存合并后的数据
    merged_df.to_csv(output_path, index=False)
    print(f"已将合并后的数据保存到 {output_path}，共 {len(merged_df)} 条记录")

if __name__ == "__main__":
    main() 