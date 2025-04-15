"""
基于特征重要性的特征选择脚本
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import json
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config
from src.models.model_training import load_data, prepare_train_test_data
from src.models.base_model import BaseModel
from src.models.traditional_models import XGBoostModel

logger = get_logger(__name__)

class FeatureImportanceSelector:
    """
    基于特征重要性的特征选择器
    """
    def __init__(
        self,
        model_path: str,
        feature_file: str,
        target_file: str,
        target_column: str,
        output_dir: str = "data/processed/features/selected",
        top_n: int = 20,
        threshold: float = 0.0,
    ):
        """
        初始化特征选择器
        
        参数:
            model_path: 已训练模型的路径
            feature_file: 特征文件路径
            target_file: 目标文件路径
            target_column: 目标列名
            output_dir: 输出目录
            top_n: 选择的顶部特征数量
            threshold: 最小特征重要性阈值
        """
        self.model_path = model_path
        self.feature_file = feature_file
        self.target_file = target_file
        self.target_column = target_column
        self.output_dir = output_dir
        self.top_n = top_n
        self.threshold = threshold
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型
        try:
            self.model = self._load_model(model_path)
            logger.info(f"已加载模型: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
        
        # 加载数据
        try:
            self.X, self.y = load_data(feature_file, target_file, target_column)
            logger.info(f"数据加载完成: X={self.X.shape}, y={self.y.shape}")
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def _load_model(self, model_path: str) -> BaseModel:
        """
        加载模型
        
        参数:
            model_path: 模型文件路径
            
        返回:
            BaseModel: 加载的模型
        """
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        elif model_path.endswith('.json'):
            # 如果是元数据文件，尝试查找对应的模型文件
            with open(model_path, 'r') as f:
                metadata = json.load(f)
                
            # 构建可能的模型文件路径
            model_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path).replace('_metadata.json', '.pkl')
            full_model_path = os.path.join(model_dir, model_name)
            
            if os.path.exists(full_model_path):
                with open(full_model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
            else:
                # 如果找不到模型文件，根据元数据重新创建模型
                logger.warning(f"找不到完整模型文件，尝试根据元数据重新创建")
                
                # 获取模型类型和参数
                model_type = metadata.get('model_type', 'xgboost')
                model_params = metadata.get('model_params', {})
                
                # 创建模型（目前只支持XGBoost）
                if model_type == 'xgboost':
                    model = XGBoostModel(
                        name=metadata.get('name', 'xgboost'),
                        model_params=model_params,
                        prediction_horizon=metadata.get('prediction_horizon', 60),
                        target_type=metadata.get('target_type', 'price_change_pct')
                    )
                    return model
                else:
                    raise ValueError(f"不支持的模型类型: {model_type}")
        else:
            raise ValueError(f"不支持的模型文件格式: {model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        返回:
            Dict[str, float]: 特征名到重要性的映射
        """
        # 检查模型是否有特征重要性方法
        if not hasattr(self.model, 'get_feature_importance') or not callable(self.model.get_feature_importance):
            logger.error("模型不支持特征重要性")
            return {}
        
        # 获取特征重要性
        importance = self.model.get_feature_importance()
        
        if not importance:
            logger.warning("无法获取特征重要性")
            return {}
        
        return importance
    
    def select_features(self) -> Tuple[List[str], Dict[str, float]]:
        """
        选择最重要的特征
        
        返回:
            Tuple[List[str], Dict[str, float]]: 选定的特征列表和它们的重要性
        """
        # 获取特征重要性
        importances = self.get_feature_importance()
        
        if not importances:
            logger.warning("没有获取到特征重要性，无法选择特征")
            return [], {}
        
        # 按重要性排序
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        
        # 应用阈值过滤
        if self.threshold > 0:
            sorted_importances = [(name, imp) for name, imp in sorted_importances if imp >= self.threshold]
        
        # 选择前N个特征
        top_features = sorted_importances[:self.top_n] if self.top_n > 0 else sorted_importances
        
        # 提取特征名称和重要性
        selected_features = [name for name, _ in top_features]
        selected_importances = {name: imp for name, imp in top_features}
        
        logger.info(f"已选择 {len(selected_features)} 个重要特征")
        
        return selected_features, selected_importances
    
    def save_selected_features(self, selected_features: List[str], importances: Dict[str, float]) -> str:
        """
        保存选定的特征
        
        参数:
            selected_features: 选定的特征列表
            importances: 特征重要性
            
        返回:
            str: 特征列表文件路径
        """
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建特征列表文件路径
        feature_list_file = os.path.join(self.output_dir, f"selected_features_{timestamp}.txt")
        
        # 写入特征列表
        with open(feature_list_file, 'w') as f:
            f.write("# 特征重要性排序\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 模型文件: {self.model_path}\n")
            f.write(f"# 特征文件: {self.feature_file}\n")
            f.write(f"# 目标文件: {self.target_file}\n")
            f.write(f"# 目标列名: {self.target_column}\n")
            f.write(f"# 特征数量: {len(selected_features)}\n")
            f.write("\n")
            
            for i, feature in enumerate(selected_features):
                f.write(f"{feature},{importances.get(feature, 0):.6f}\n")
        
        logger.info(f"已将选定特征列表保存至: {feature_list_file}")
        
        return feature_list_file
    
    def create_selected_dataset(self, selected_features: List[str]) -> Tuple[str, str]:
        """
        创建仅包含选定特征的数据集
        
        参数:
            selected_features: 选定的特征列表
            
        返回:
            Tuple[str, str]: 特征文件路径和目标文件路径
        """
        # 确保模型中的特征名称与数据集列名匹配
        valid_features = [f for f in selected_features if f in self.X.columns]
        
        if len(valid_features) != len(selected_features):
            logger.warning(f"有 {len(selected_features) - len(valid_features)} 个特征在数据集中不存在")
        
        if not valid_features:
            logger.error("没有有效特征，无法创建数据集")
            return "", ""
        
        # 选择特征子集
        X_selected = self.X[valid_features]
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建特征文件路径
        selected_feature_file = os.path.join(self.output_dir, f"features_selected_{timestamp}.csv")
        selected_target_file = os.path.join(self.output_dir, f"targets_selected_{timestamp}.csv")
        
        # 保存特征数据
        X_selected.to_csv(selected_feature_file)
        logger.info(f"已将选定特征数据保存至: {selected_feature_file}")
        
        # 保存目标数据
        self.y.to_frame().to_csv(selected_target_file)
        logger.info(f"已将目标数据保存至: {selected_target_file}")
        
        return selected_feature_file, selected_target_file
    
    def plot_importance(self, selected_features: List[str], importances: Dict[str, float]) -> str:
        """
        绘制特征重要性图
        
        参数:
            selected_features: 选定的特征列表
            importances: 特征重要性
            
        返回:
            str: 图片文件路径
        """
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 排序并绘制
        feature_importance = [(f, importances[f]) for f in selected_features]
        feature_importance.sort(key=lambda x: x[1])
        
        features = [f[0] for f in feature_importance]
        importance_values = [f[1] for f in feature_importance]
        
        # 绘制水平条形图
        plt.barh(range(len(features)), importance_values, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('特征重要性')
        plt.title('特征重要性排序')
        plt.tight_layout()
        
        # 保存图形
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.output_dir, f"feature_importance_{timestamp}.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        
        logger.info(f"已将特征重要性图保存至: {plot_file}")
        
        return plot_file
    
    def run(self) -> Tuple[List[str], str, str]:
        """
        运行特征选择流程
        
        返回:
            Tuple[List[str], str, str]: 选定的特征列表、特征文件路径和目标文件路径
        """
        # 选择特征
        selected_features, importances = self.select_features()
        
        if not selected_features:
            logger.warning("没有选择到特征")
            return [], "", ""
        
        # 保存特征列表
        feature_list_file = self.save_selected_features(selected_features, importances)
        
        # 创建数据集
        feature_file, target_file = self.create_selected_dataset(selected_features)
        
        # 绘制特征重要性图
        try:
            plot_file = self.plot_importance(selected_features, importances)
            logger.info(f"特征重要性图已保存至: {plot_file}")
        except Exception as e:
            logger.warning(f"绘制特征重要性图失败: {str(e)}")
        
        # 输出结果摘要
        logger.info("特征选择完成")
        logger.info(f"选择了 {len(selected_features)} 个特征")
        logger.info(f"特征列表已保存至: {feature_list_file}")
        logger.info(f"特征数据已保存至: {feature_file}")
        logger.info(f"目标数据已保存至: {target_file}")
        
        # 显示特征和重要性
        for i, feature in enumerate(selected_features[:20]):  # 只显示前20个
            logger.info(f"  {i+1:2d}. {feature}: {importances.get(feature, 0):.4f}")
        
        return selected_features, feature_file, target_file

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="基于特征重要性的特征选择工具")
    
    # 主要参数
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="已训练模型的路径")
    
    # 数据参数
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对")
    parser.add_argument("--timeframe", type=str, default="1h", help="时间框架")
    parser.add_argument("--feature_file", type=str, help="特征文件路径，如不指定则根据symbol和timeframe自动生成")
    parser.add_argument("--target_file", type=str, help="目标文件路径，如不指定则根据symbol和timeframe自动生成")
    parser.add_argument("--target_column", type=str, default="target_pct_60", help="目标列名")
    parser.add_argument("--feature_type", type=str, default="standard", 
                      choices=["standard", "cross_timeframe"], 
                      help="特征类型：标准特征或跨周期特征")
    
    # 特征选择参数
    parser.add_argument("--top_n", type=int, default=20, help="选择的顶部特征数量")
    parser.add_argument("--threshold", type=float, default=0.0, help="最小特征重要性阈值")
    parser.add_argument("--output_dir", type=str, default="data/processed/features/selected", help="输出目录")
    parser.add_argument("--train_model", action="store_true", help="使用选定特征训练新模型")
    
    # 其他参数
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(config_path=args.config, default_level=log_level)
    
    logger.info("开始基于特征重要性的特征选择")
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 确定特征和目标文件路径
        if args.feature_file is None:
            if args.feature_type == "standard":
                # 自动生成标准特征文件路径
                feature_dir = os.path.join("data/processed/features", args.symbol)
                args.feature_file = os.path.join(feature_dir, f"features_{args.timeframe}.csv")
            elif args.feature_type == "cross_timeframe":
                # 自动生成跨周期特征文件路径
                feature_dir = "data/processed/features/cross_timeframe"
                timeframes = ["1h", "4h", "1d"]  # 默认跨周期组合
                timeframes_str = "_".join(timeframes)
                args.feature_file = os.path.join(feature_dir, f"{args.symbol}_multi_tf_{timeframes_str}.csv")
                logger.info(f"使用跨周期特征: {timeframes_str}")
            
        if args.target_file is None:
            # 自动生成目标文件路径（目标文件总是基于基础时间框架）
            target_dir = os.path.join("data/processed/features", args.symbol)
            args.target_file = os.path.join(target_dir, f"targets_{args.timeframe}.csv")
        
        logger.info(f"模型文件: {args.model_path}")
        logger.info(f"特征文件: {args.feature_file}")
        logger.info(f"目标文件: {args.target_file}")
        logger.info(f"目标列名: {args.target_column}")
        
        # 创建特征选择器
        selector = FeatureImportanceSelector(
            model_path=args.model_path,
            feature_file=args.feature_file,
            target_file=args.target_file,
            target_column=args.target_column,
            output_dir=args.output_dir,
            top_n=args.top_n,
            threshold=args.threshold
        )
        
        # 执行特征选择
        selected_features, feature_file, target_file = selector.run()
        
        # 如果需要，使用选定特征训练新模型
        if args.train_model and feature_file and target_file:
            logger.info("使用选定特征训练新模型")
            
            # 构建训练命令
            cmd = [
                "python -m src.model_training_main",
                f"--feature_file={feature_file}",
                f"--target_file={target_file}",
                f"--target_column={args.target_column}",
                f"--symbol={args.symbol}",
                f"--timeframe={args.timeframe}",
                f"--feature_type={args.feature_type}",
                "--model_type=xgboost",
                "--target_type=price_change_pct",
                "--horizon=60",
                "--time_series_split"
            ]
            
            # 执行命令
            command = " ".join(cmd)
            logger.info(f"执行命令: {command}")
            
            import subprocess
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("模型训练成功")
                logger.info(result.stdout)
            else:
                logger.error("模型训练失败")
                logger.error(result.stderr)
        
        logger.info("特征选择完成")
        return 0
    
    except Exception as e:
        logger.error(f"特征选择失败: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 