"""
模型调优功能测试脚本

用于测试模型优化、特征选择和模型集成的功能
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
from datetime import datetime
from xgboost.callback import EarlyStopping

from src.model_optimization import ModelOptimizer
from src.feature_selector import FeatureSelector
from src.model_ensemble import ModelEnsemble, PredictionSmoother

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_test_data():
    """
    准备测试数据
    """
    logger.info("准备测试数据")
    
    # 检查BTCUSDT特征数据是否存在
    symbol = "BTCUSDT"
    timeframe = "1h"
    feature_dir = os.path.join("data/processed/features", symbol)
    feature_file = os.path.join(feature_dir, f"features_{timeframe}.csv")
    target_file = os.path.join(feature_dir, f"targets_{timeframe}.csv")
    
    if not os.path.exists(feature_file) or not os.path.exists(target_file):
        logger.error(f"特征文件不存在: {feature_file} 或 {target_file}")
        logger.info("请先运行特征工程脚本生成特征")
        return None, None, None, None, None, None
    
    # 加载特征和目标
    logger.info(f"加载特征文件: {feature_file}")
    X = pd.read_csv(feature_file, index_col=0)
    
    logger.info(f"加载目标文件: {target_file}")
    y = pd.read_csv(target_file, index_col=0).iloc[:, 0]
    
    # 移除NaN值
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    logger.info(f"加载了 {len(X)} 条有效数据记录")
    
    # 按时间分割数据（假设索引是时间戳）
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15
    
    # 获取数据点数量
    n = len(X)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    # 分割数据
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    logger.info(f"数据分割完成: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_baseline_model(X_train, y_train, X_val, y_val, output_dir="data/results/models"):
    """
    训练基准模型
    """
    logger.info("训练基准XGBoost模型")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建基准模型，将early_stopping_rounds移到这里
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=10  # 早停参数在此处设置
    )
    
    # 训练模型 - fit方法只接收eval_set
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 保存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(output_dir, f"baseline_xgb_{timestamp}.joblib")
    joblib.dump(model, model_path)
    
    logger.info(f"基准模型已保存至: {model_path}")
    
    return model_path


def test_model_optimization(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    测试模型优化功能
    """
    logger.info("测试模型优化功能")
    
    output_dir = "data/results/test/optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建优化器
    optimizer = ModelOptimizer(
        search_method="random",
        n_iter=5,  # 设置较小的值以加快测试速度
        cv=3,  # 设置较小的值以加快测试速度
        scoring="neg_root_mean_squared_error",
        random_state=42,
        verbose=1,
        output_dir=output_dir
    )
    
    # 执行优化
    best_params, best_model = optimizer.optimize_xgboost(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=10,
        n_estimators=100  # 设置较小的值以加快测试速度
    )
    
    # 保存结果
    model_path = optimizer.save_optimization_results(
        model=best_model,
        best_params=best_params,
        X_test=X_test,
        y_test=y_test,
        model_name="xgboost_optimized",
        feature_names=X_train.columns.tolist()
    )
    
    logger.info(f"优化模型已保存至: {model_path}")
    return model_path


def test_feature_selection(model_path, X_train, y_train, X_val, y_val):
    """
    测试特征选择功能
    """
    logger.info("测试特征选择功能")
    
    output_dir = "data/results/test/feature_selection"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建特征选择器
    selector = FeatureSelector(
        model_path=model_path,
        method="importance",
        n_features=10,  # 选择前10个特征
        output_dir=output_dir,
        verbose=True
    )
    
    # 选择特征
    selected_features = selector.select_features(X_train, y_train)
    
    # 保存特征
    selector.save_selected_features()
    
    # 绘制特征重要性
    plot_path = os.path.join(output_dir, "feature_importance.png")
    selector.plot_feature_importance(plot_path)
    
    # 使用选定的特征重新训练模型
    from src.feature_selector import retrain_with_selected_features
    
    # 注意：retrain_with_selected_features函数内部也需要使用新的XGBoost API
    # 但由于该函数在另一个文件中，我们只能在这里添加日志提示
    logger.info("注意：如果retrain_with_selected_features函数使用旧的XGBoost API，可能会出错")
    
    new_model_path = retrain_with_selected_features(
        model_path=model_path,
        X_train=X_train,
        y_train=y_train,
        selected_features=selected_features,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=10,
        output_dir=os.path.join(output_dir, "models"),
        verbose=True
    )
    
    logger.info(f"使用选定特征重新训练的模型已保存至: {new_model_path}")
    return new_model_path, selected_features


def test_model_ensemble(model_paths, X_test, y_test):
    """
    测试模型集成功能
    """
    logger.info("测试模型集成功能")
    
    output_dir = "data/results/test/ensemble"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型集成
    ensemble = ModelEnsemble(
        model_paths=model_paths,
        ensemble_method="average",
        model_names=["model_" + str(i) for i in range(len(model_paths))]
    )
    
    # 评估模型
    metrics = ensemble.evaluate(X_test, y_test)
    
    logger.info("集成模型评估指标:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.6f}")
    
    # 绘制预测图
    plot_path = os.path.join(output_dir, "ensemble_predictions.png")
    ensemble.plot_predictions(
        X=X_test,
        y=y_test,
        title="集成预测对比",
        save_path=plot_path
    )
    
    # 测试预测平滑
    smoother = PredictionSmoother(window_size=5, method="moving_avg")
    
    # 获取原始预测
    predictions = ensemble.predict(X_test)
    
    # 平滑预测
    smoothed = smoother.smooth(predictions)
    
    # 绘制对比图
    smooth_plot_path = os.path.join(output_dir, "smoothed_predictions.png")
    smoother.plot_comparison(
        original=predictions,
        smoothed=smoothed,
        targets=y_test,
        title="原始预测与平滑预测对比",
        save_path=smooth_plot_path
    )
    
    logger.info(f"集成预测图已保存至: {plot_path}")
    logger.info(f"平滑预测图已保存至: {smooth_plot_path}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="模型调优功能测试")
    parser.add_argument("--skip_optimization", action="store_true", help="跳过模型优化测试")
    parser.add_argument("--skip_feature_selection", action="store_true", help="跳过特征选择测试")
    parser.add_argument("--skip_ensemble", action="store_true", help="跳过模型集成测试")
    parser.add_argument("--existing_model", type=str, help="使用已有模型路径进行测试")
    
    args = parser.parse_args()
    
    logger.info("开始测试模型调优功能")
    
    # 准备数据
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_test_data()
    
    if X_train is None:
        return
    
    # 如果没有提供已有模型，训练基准模型
    model_paths = []
    if args.existing_model:
        baseline_model_path = args.existing_model
        logger.info(f"使用已有模型: {baseline_model_path}")
        model_paths.append(baseline_model_path)
    else:
        baseline_model_path = train_baseline_model(X_train, y_train, X_val, y_val)
        model_paths.append(baseline_model_path)
    
    # 测试模型优化
    optimized_model_path = None
    if not args.skip_optimization:
        optimized_model_path = test_model_optimization(X_train, y_train, X_val, y_val, X_test, y_test)
        model_paths.append(optimized_model_path)
    
    # 测试特征选择
    if not args.skip_feature_selection:
        # 使用基准模型或优化后的模型
        model_path_for_selection = optimized_model_path if optimized_model_path else baseline_model_path
        selected_model_path, selected_features = test_feature_selection(
            model_path_for_selection, X_train, y_train, X_val, y_val
        )
        model_paths.append(selected_model_path)
    
    # 测试模型集成
    if not args.skip_ensemble and len(model_paths) > 1:
        test_model_ensemble(model_paths, X_test, y_test)
    
    logger.info("模型调优功能测试完成")


if __name__ == "__main__":
    main() 