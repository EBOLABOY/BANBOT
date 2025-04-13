"""
数据收集与预处理主脚本
"""

import argparse
import sys
import os
import traceback
from datetime import datetime, timedelta
import glob

from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config
from src.data.data_manager import DataManager

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="加密货币数据收集与预处理工具")
    
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径")
    
    parser.add_argument("--mode", type=str, default="all",
                        choices=["collect", "process", "all", "update"],
                        help="运行模式：仅收集，仅处理，全部，或更新")
    
    parser.add_argument("--start_date", type=str,
                        help="开始日期（格式：YYYY-MM-DD）")
    
    parser.add_argument("--end_date", type=str,
                        help="结束日期（格式：YYYY-MM-DD）")
    
    parser.add_argument("--parallel", action="store_true",
                        help="启用并行处理")
    
    parser.add_argument("--max_workers", type=int, default=4,
                        help="并行处理的最大工作线程数")
    
    parser.add_argument("--symbol", type=str,
                        help="指定交易对（例如：BTCUSDT）")
    
    parser.add_argument("--timeframe", type=str,
                        help="指定时间框架（例如：1h, 4h, 1d）")
    
    parser.add_argument("--force_collect", action="store_true",
                        help="强制重新收集数据，即使本地已有数据")
    
    return parser.parse_args()

def check_data_exists(symbol, timeframe, date_suffix, raw_data_dir="data/raw"):
    """
    检查指定的数据文件是否已存在
    
    参数:
        symbol: 交易对名称
        timeframe: 时间框架
        date_suffix: 日期后缀，如20220101
        raw_data_dir: 原始数据存储目录
        
    返回:
        bool: 如果文件存在返回True，否则返回False
    """
    file_pattern = f"{raw_data_dir}/{symbol}_{timeframe}_{date_suffix}.csv"
    return len(glob.glob(file_pattern)) > 0

def get_required_data_files(config, start_date=None, end_date=None, symbol=None, timeframe=None):
    """
    获取需要收集的数据文件列表
    
    参数:
        config: 配置信息
        start_date: 开始日期
        end_date: 结束日期
        symbol: 指定交易对
        timeframe: 指定时间框架
        
    返回:
        list: 包含(symbol, timeframe)元组的列表
    """
    # 读取配置文件中的交易对和时间框架
    if not config:
        return []
        
    symbols = [symbol] if symbol else config.get("data", {}).get("symbols", [])
    timeframes = [timeframe] if timeframe else config.get("data", {}).get("timeframes", [])
    
    # 生成需要的文件列表
    required_files = []
    for sym in symbols:
        for tf in timeframes:
            required_files.append((sym, tf))
    
    return required_files

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    setup_logging(args.config)
    logger = get_logger(__name__)
    
    try:
        logger.info("启动数据收集与预处理工具")
        
        # 创建数据管理器
        data_manager = DataManager(args.config)
        config = load_config(args.config)
        
        # 确定日期后缀
        date_suffix = "20220101"  # 默认日期后缀，根据您的实际需求调整
        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            date_suffix = start_date.strftime("%Y%m%d")
        
        # 根据运行模式执行相应操作
        if args.mode in ["collect", "all"]:
            logger.info("开始收集数据...")
            
            # 获取需要收集的数据文件列表
            required_files = get_required_data_files(
                config, args.start_date, args.end_date, args.symbol, args.timeframe
            )
            
            # 检查哪些文件已存在，哪些需要收集
            files_to_collect = []
            skipped_files = []
            
            for symbol, timeframe in required_files:
                if args.force_collect or not check_data_exists(symbol, timeframe, date_suffix):
                    files_to_collect.append((symbol, timeframe))
                else:
                    skipped_files.append((symbol, timeframe))
            
            if skipped_files:
                logger.info(f"跳过已存在的 {len(skipped_files)} 个数据文件: {skipped_files}")
            
            # 如果有需要收集的文件，则进行收集
            if files_to_collect:
                logger.info(f"开始收集 {len(files_to_collect)} 个数据文件")
                
                if args.symbol:
                    # 收集指定交易对的数据
                    success = data_manager.collector.collect_historical_data(
                        parallel=args.parallel,
                        max_workers=args.max_workers
                    )
                    logger.info(f"数据收集{'成功' if success else '失败'}")
                else:
                    # 收集所有目标货币的数据
                    success = data_manager.collect_and_process_data(
                        start_date=args.start_date,
                        end_date=args.end_date,
                        parallel=args.parallel
                    )
                    logger.info(f"数据收集{'成功' if success else '失败'}")
            else:
                logger.info("所有需要的数据文件已存在，无需重新收集")
        
        if args.mode in ["process", "all"]:
            logger.info("开始处理数据...")
            
            # 获取原始数据文件
            if args.symbol or args.timeframe:
                pattern = ""
                if args.symbol:
                    pattern += args.symbol
                if args.timeframe:
                    pattern += f"_{args.timeframe}"
                
                raw_files = data_manager.processor.list_raw_data_files(pattern)
            else:
                raw_files = data_manager.processor.list_raw_data_files()
            
            # 处理数据
            processed_count = data_manager.processor.process_files(
                raw_files, clean=True, resample=False
            )
            
            logger.info(f"数据处理完成，成功处理 {processed_count}/{len(raw_files)} 个文件")
            
            # 合并数据源
            logger.info("开始合并数据源...")
            merged_data = data_manager.merge_data_sources(
                start_date=args.start_date,
                end_date=args.end_date
            )
            
            logger.info(f"数据合并完成，共有 {len(merged_data)} 个交易对的数据")
        
        if args.mode == "update":
            logger.info("开始更新数据...")
            
            symbols = [args.symbol] if args.symbol else None
            timeframes = [args.timeframe] if args.timeframe else None
            
            update_count = data_manager.update_data(symbols, timeframes)
            
            logger.info(f"数据更新完成，共更新 {update_count} 项数据")
        
        logger.info("数据收集与预处理工具运行完成")
        return 0
    
    except Exception as e:
        logger.error(f"运行过程中出错: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 