"""
模型调优主脚本

提供命令行接口，用于执行模型正则化参数优化、特征选择和模型集成任务
"""

import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config
from src.model_optimization import ModelOptimizer
from src.model_ensemble import ModelEnsemble, PredictionSmoother
from src.feature_selector import FeatureSelector, retrain_with_selected_features

# 设置日志
logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型调优工具")
    
    # 通用参数
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["optimize", "select_features", "ensemble", "smooth"],
                        help="运行模式: 参数优化, 特征选择, 模型集成, 预测平滑")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--verbose", action="store_true", help="是否显示详细日志")
    
    # 数据参数
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对")
    parser.add_argument("--timeframe", type=str, default="1h", help="时间框架")
    parser.add_argument("--X_file", type=str, help="特征文件路径")
    parser.add_argument("--y_file", type=str, help="目标文件路径")
    parser.add_argument("--X_val_file", type=str, help="验证特征文件路径")
    parser.add_argument("--y_val_file", type=str, help="验证目标文件路径")
    parser.add_argument("--X_test_file", type=str, help="测试特征文件路径")
    parser.add_argument("--y_test_file", type=str, help="测试目标文件路径")
    
    # 参数优化参数
    parser.add_argument("--search_method", type=str, default="random", 
                        choices=["grid", "random"],
                        help="参数搜索方法")
    parser.add_argument("--n_iter", type=int, default=20, help="随机搜索迭代次数")
    parser.add_argument("--cv", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--scoring", type=str, default="neg_root_mean_squared_error", 
                        help="评分标准")
    parser.add_argument("--early_stopping_rounds", type=int, default=50, 
                        help="早停轮数")
    parser.add_argument("--n_estimators", type=int, default=1000, help="决策树数量")
    
    # 特征选择参数
    parser.add_argument("--model_path", type=str, help="模型路径（用于特征选择）")
    parser.add_argument("--selection_method", type=str, default="importance", 
                        choices=["importance", "shap"],
                        help="特征选择方法")
    parser.add_argument("--n_features", type=int, default=20, help="要选择的特征数量")
    parser.add_argument("--retrain", action="store_true", 
                        help="是否使用选择的特征重新训练模型")
    
    # 模型集成参数
    parser.add_argument("--model_paths", type=str, nargs="+", help="模型路径列表")
    parser.add_argument("--ensemble_method", type=str, default="average", 
                        choices=["average", "weighted", "median"],
                        help="集成方法")
    parser.add_argument("--weights", type=float, nargs="+", 
                        help="模型权重（用于加权集成）")
    parser.add_argument("--model_names", type=str, nargs="+", help="模型名称列表")
    
    # 预测平滑参数
    parser.add_argument("--window_size", type=int, default=5, help="平滑窗口大小")
    parser.add_argument("--smooth_method", type=str, default="moving_avg", 
                        choices=["moving_avg", "ewm", "median"],
                        help="平滑方法")
    parser.add_argument("--predictions_file", type=str, 
                        help="预测文件路径（用于预测平滑）")
    
    return parser.parse_args()


def setup_output_dir(args, mode):
    """设置输出目录"""
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 根据模式设置默认输出目录
        base_dir = "data/results"
        if mode == "optimize":
            output_dir = f"{base_dir}/optimization"
        elif mode == "select_features":
            output_dir = f"{base_dir}/feature_selection"
        elif mode == "ensemble":
            output_dir = f"{base_dir}/ensemble"
        elif mode == "smooth":
            output_dir = f"{base_dir}/smoothed"
        else:
            output_dir = base_dir
    
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    return output_dir


def load_data_files(args):
    """加载数据文件"""
    # 设置文件路径
    if not args.X_file and args.symbol and args.timeframe:
        feature_dir = os.path.join("data/processed/features", args.symbol)
        args.X_file = os.path.join(feature_dir, f"features_{args.timeframe}.csv")
    
    if not args.y_file and args.symbol and args.timeframe:
        target_dir = os.path.join("data/processed/features", args.symbol)
        args.y_file = os.path.join(target_dir, f"targets_{args.timeframe}.csv")
    
    # 加载数据
    logger.info(f"加载特征文件: {args.X_file}")
    X = pd.read_csv(args.X_file, index_col=0)
    
    logger.info(f"加载目标文件: {args.y_file}")
    y = pd.read_csv(args.y_file, index_col=0).iloc[:, 0]
    
    # 加载验证集（如果有）
    X_val, y_val = None, None
    if args.X_val_file and args.y_val_file:
        logger.info(f"加载验证特征文件: {args.X_val_file}")
        X_val = pd.read_csv(args.X_val_file, index_col=0)
        
        logger.info(f"加载验证目标文件: {args.y_val_file}")
        y_val = pd.read_csv(args.y_val_file, index_col=0).iloc[:, 0]
    
    # 加载测试集（如果有）
    X_test, y_test = None, None
    if args.X_test_file and args.y_test_file:
        logger.info(f"加载测试特征文件: {args.X_test_file}")
        X_test = pd.read_csv(args.X_test_file, index_col=0)
        
        logger.info(f"加载测试目标文件: {args.y_test_file}")
        y_test = pd.read_csv(args.y_test_file, index_col=0).iloc[:, 0]
    
    return X, y, X_val, y_val, X_test, y_test


def run_optimization(args, output_dir):
    """运行参数优化"""
    logger.info("开始参数优化")
    
    # 加载数据
    X, y, X_val, y_val, X_test, y_test = load_data_files(args)
    
    # 创建优化器
    optimizer = ModelOptimizer(
        search_method=args.search_method,
        n_iter=args.n_iter,
        cv=args.cv,
        scoring=args.scoring,
        random_state=42,
        verbose=int(args.verbose),
        output_dir=output_dir
    )
    
    # 执行优化
    best_params, best_model = optimizer.optimize_xgboost(
        X_train=X,
        y_train=y,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=args.early_stopping_rounds,
        n_estimators=args.n_estimators
    )
    
    # 保存结果
    model_path = optimizer.save_optimization_results(
        model=best_model,
        best_params=best_params,
        X_test=X_test,
        y_test=y_test,
        model_name=f"xgboost_{args.symbol}_{args.timeframe}",
        feature_names=X.columns.tolist()
    )
    
    logger.info(f"参数优化完成，模型已保存至: {model_path}")
    
    return model_path


def run_feature_selection(args, output_dir):
    """运行特征选择"""
    logger.info("开始特征选择")
    
    # 检查模型路径
    if not args.model_path:
        logger.error("请提供模型路径(--model_path)")
        return None
    
    # 加载数据
    X, y, X_val, y_val, X_test, y_test = load_data_files(args)
    
    # 创建特征选择器
    selector = FeatureSelector(
        model_path=args.model_path,
        method=args.selection_method,
        n_features=args.n_features,
        output_dir=output_dir,
        verbose=args.verbose
    )
    
    # 选择特征
    selected_features = selector.select_features(X, y)
    
    # 保存特征
    selector.save_selected_features()
    
    # 绘制特征重要性
    plot_path = os.path.join(
        output_dir, 
        f"feature_importance_{args.selection_method}_{args.n_features}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    selector.plot_feature_importance(plot_path)
    
    # 如果需要重新训练
    if args.retrain:
        new_model_path = retrain_with_selected_features(
            model_path=args.model_path,
            X_train=X,
            y_train=y,
            selected_features=selected_features,
            X_val=X_val,
            y_val=y_val,
            early_stopping_rounds=args.early_stopping_rounds,
            output_dir=os.path.join(output_dir, "models"),
            verbose=args.verbose
        )
        logger.info(f"重新训练完成，模型已保存至: {new_model_path}")
        return new_model_path
    
    return None


def run_ensemble(args, output_dir):
    """运行模型集成"""
    logger.info("开始模型集成")
    
    # 检查模型路径
    if not args.model_paths or len(args.model_paths) < 2:
        logger.error("请提供至少两个模型路径(--model_paths)")
        return None
    
    # 加载数据
    X, y, X_val, y_val, X_test, y_test = load_data_files(args)
    
    # 创建模型集成
    ensemble = ModelEnsemble(
        model_paths=args.model_paths,
        ensemble_method=args.ensemble_method,
        weights=args.weights,
        model_names=args.model_names
    )
    
    # 评估模型
    metrics = ensemble.evaluate(
        X=X_test if X_test is not None else X,
        y=y_test if y_test is not None else y
    )
    
    logger.info("集成模型评估指标:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.6f}")
    
    # 绘制预测图
    plot_path = os.path.join(
        output_dir, 
        f"ensemble_predictions_{args.ensemble_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    
    ensemble.plot_predictions(
        X=X_test if X_test is not None else X,
        y=y_test if y_test is not None else y,
        title=f"集成预测 ({args.ensemble_method})",
        save_path=plot_path
    )
    
    # 保存权重
    weights_path = os.path.join(
        output_dir,
        f"ensemble_weights_{args.ensemble_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    ensemble.save_weights(weights_path)
    
    logger.info(f"模型集成完成")
    return ensemble


def run_smooth(args, output_dir):
    """运行预测平滑"""
    logger.info("开始预测平滑")
    
    # 检查预测文件
    if not args.predictions_file:
        logger.error("请提供预测文件路径(--predictions_file)")
        return None
    
    # 加载预测
    logger.info(f"加载预测文件: {args.predictions_file}")
    pred_df = pd.read_csv(args.predictions_file, index_col=0)
    
    if len(pred_df.columns) > 1:
        logger.warning(f"预测文件包含多列，使用第一列")
    
    predictions = pred_df.iloc[:, 0]
    
    # 创建平滑处理器
    smoother = PredictionSmoother(
        window_size=args.window_size,
        method=args.smooth_method
    )
    
    # 平滑预测
    smoothed = smoother.smooth(predictions)
    
    # 创建平滑预测数据框
    smoothed_df = pd.DataFrame({
        'original': predictions,
        'smoothed': smoothed
    }, index=predictions.index)
    
    # 保存平滑预测
    output_file = os.path.join(
        output_dir,
        f"smoothed_predictions_{args.smooth_method}_{args.window_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    smoothed_df.to_csv(output_file)
    logger.info(f"平滑预测已保存至: {output_file}")
    
    # 绘制对比图
    plot_file = os.path.join(
        output_dir,
        f"smoothed_predictions_{args.smooth_method}_{args.window_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    
    smoother.plot_comparison(
        original=predictions,
        smoothed=smoothed,
        title=f"预测平滑结果 ({args.smooth_method}, 窗口大小: {args.window_size})",
        save_path=plot_file
    )
    
    logger.info(f"预测平滑完成")
    return smoothed_df


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(config_path=args.config, default_level=log_level)
    
    # 打印版本信息
    logger.info("模型调优工具 v1.0.0")
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 设置输出目录
        output_dir = setup_output_dir(args, args.mode)
        
        # 根据模式执行对应功能
        if args.mode == "optimize":
            model_path = run_optimization(args, output_dir)
            if model_path:
                logger.info(f"优化模型保存路径: {model_path}")
                
        elif args.mode == "select_features":
            model_path = run_feature_selection(args, output_dir)
            if model_path:
                logger.info(f"特征选择后的模型保存路径: {model_path}")
                
        elif args.mode == "ensemble":
            ensemble = run_ensemble(args, output_dir)
            if ensemble:
                logger.info(f"模型集成完成")
                
        elif args.mode == "smooth":
            smoothed_df = run_smooth(args, output_dir)
            if smoothed_df is not None:
                logger.info(f"预测平滑完成")
        
        logger.info("模型调优任务完成")
        return 0
    
    except Exception as e:
        logger.error(f"模型调优失败: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 