"""
数据收集与预处理主脚本
"""

import argparse
import sys
import os
import traceback
from datetime import datetime, timedelta

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
    
    return parser.parse_args()

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
        
        # 根据运行模式执行相应操作
        if args.mode in ["collect", "all"]:
            logger.info("开始收集数据...")
            if args.symbol:
                # 收集指定交易对的数据
                symbols = [args.symbol]
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