"""
集成预测器脚本
用于集成多个模型的预测结果或对预测结果进行平滑处理
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

from src.utils.logger import get_logger
from src.models.model_factory import ModelFactory

logger = get_logger(__name__)

class EnsemblePredictor:
    """
    集成预测器
    用于集成多个模型的预测结果或对预测结果进行平滑处理
    """
    
    def __init__(
        self,
        ensemble_method: str = "average",  # 'average', 'weighted', 'stacking', 'voting', 'moving_avg'
        model_paths: List[str] = None,
        weights: List[float] = None,
        window_size: int = 5,  # 用于移动平均
        output_dir: str = "outputs/ensemble",
        meta_model_type: str = "lightgbm",  # 用于堆叠集成
        plot_results: bool = True
    ):
        """
        初始化集成预测器
        
        参数:
            ensemble_method: 集成方法 ('average', 'weighted', 'stacking', 'voting', 'moving_avg')
            model_paths: 模型文件路径列表
            weights: 权重列表 (用于加权平均集成)
            window_size: 窗口大小 (用于移动平均)
            output_dir: 输出目录
            meta_model_type: 元模型类型 (用于堆叠集成)
            plot_results: 是否绘制结果图
        """
        self.ensemble_method = ensemble_method
        self.model_paths = model_paths or []
        self.weights = weights
        self.window_size = window_size
        self.output_dir = output_dir
        self.meta_model_type = meta_model_type
        self.plot_results = plot_results
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化模型列表和元模型
        self.models = []
        self.meta_model = None
        
        # 如果提供了模型路径，加载模型
        if model_paths:
            self.load_models()
            
        # 验证权重
        if weights is not None and ensemble_method == "weighted":
            self._validate_weights()
    
    def _validate_weights(self):
        """
        验证权重列表
        """
        if self.weights is None:
            return
        
        # 验证权重数量
        if len(self.weights) != len(self.models) and len(self.models) > 0:
            logger.warning(f"权重数量 ({len(self.weights)}) 与模型数量 ({len(self.models)}) 不匹配，将使用平均权重")
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        # 验证权重总和
        weight_sum = sum(self.weights)
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"权重总和 ({weight_sum}) 不为1，将进行归一化")
            self.weights = [w / weight_sum for w in self.weights]
    
    def load_models(self):
        """
        加载模型列表
        """
        self.models = []
        
        for path in self.model_paths:
            try:
                model = pickle.load(open(path, 'rb'))
                self.models.append(model)
                logger.info(f"已加载模型: {path}")
            except Exception as e:
                logger.error(f"加载模型失败 {path}: {e}")
        
        # 如果没有成功加载任何模型，报错
        if not self.models:
            logger.error("未成功加载任何模型")
            
        # 验证权重
        if self.weights is not None and self.ensemble_method == "weighted":
            self._validate_weights()
    
    def add_model(self, model_path: str):
        """
        添加单个模型
        
        参数:
            model_path: 模型文件路径
        """
        try:
            model = pickle.load(open(model_path, 'rb'))
            self.models.append(model)
            self.model_paths.append(model_path)
            logger.info(f"已添加模型: {model_path}")
            
            # 如果使用加权平均，更新权重
            if self.ensemble_method == "weighted":
                if self.weights is None:
                    self.weights = [1.0 / len(self.models)] * len(self.models)
                else:
                    # 为新模型添加平均权重
                    weight_sum = sum(self.weights)
                    new_weight = 1.0 - weight_sum
                    if new_weight <= 0:
                        # 重新平均化权重
                        self.weights = [1.0 / len(self.models)] * len(self.models)
                    else:
                        self.weights.append(new_weight)
                
                # 验证权重
                self._validate_weights()
                
        except Exception as e:
            logger.error(f"添加模型失败 {model_path}: {e}")
    
    def predict_with_single_model(self, model, X: pd.DataFrame) -> np.ndarray:
        """
        使用单个模型进行预测
        
        参数:
            model: 已加载的模型
            X: 特征数据
            
        返回:
            预测结果数组
        """
        try:
            if hasattr(model, 'predict'):
                # 如果是我们的ModelWrapper对象
                return model.predict(X)
            elif hasattr(model, 'predict_proba'):
                # 如果是分类器且具有predict_proba方法
                return model.predict_proba(X)[:, 1]
            else:
                # 否则使用普通predict
                return model.predict(X)
        except Exception as e:
            logger.error(f"单模型预测失败: {e}")
            return np.zeros(len(X))
    
    def apply_moving_average(self, predictions: np.ndarray) -> np.ndarray:
        """
        对预测结果应用移动平均
        
        参数:
            predictions: 预测结果数组
            
        返回:
            移动平均后的预测结果
        """
        if self.window_size <= 1:
            return predictions
        
        smoothed = np.copy(predictions)
        
        # 将预测结果转为DataFrame以便使用rolling
        pred_df = pd.DataFrame({'pred': smoothed})
        smoothed = pred_df['pred'].rolling(window=self.window_size, min_periods=1).mean().values
        
        return smoothed
    
    def predict_average(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用平均集成方法预测
        
        参数:
            X: 特征数据
            
        返回:
            集成预测结果
        """
        if not self.models:
            logger.error("没有可用的模型进行预测")
            return np.zeros(len(X))
        
        # 获取每个模型的预测
        predictions = []
        for model in self.models:
            pred = self.predict_with_single_model(model, X)
            predictions.append(pred)
        
        # 计算平均值
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def predict_weighted(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用加权平均集成方法预测
        
        参数:
            X: 特征数据
            
        返回:
            集成预测结果
        """
        if not self.models:
            logger.error("没有可用的模型进行预测")
            return np.zeros(len(X))
        
        # 确保权重正确
        if self.weights is None or len(self.weights) != len(self.models):
            logger.warning("权重不匹配，使用平均权重")
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        # 获取每个模型的预测
        predictions = []
        for i, model in enumerate(self.models):
            pred = self.predict_with_single_model(model, X)
            # 应用权重
            pred = pred * self.weights[i]
            predictions.append(pred)
        
        # 计算加权和
        ensemble_pred = np.sum(predictions, axis=0)
        return ensemble_pred
    
    def predict_voting(self, X: pd.DataFrame, threshold: float = 0.0) -> np.ndarray:
        """
        使用投票集成方法预测 (适用于二分类或方向预测)
        
        参数:
            X: 特征数据
            threshold: 分类阈值
            
        返回:
            集成预测结果
        """
        if not self.models:
            logger.error("没有可用的模型进行预测")
            return np.zeros(len(X))
        
        # 获取每个模型的预测
        predictions = []
        for model in self.models:
            pred = self.predict_with_single_model(model, X)
            # 转换为二分类预测 (1 或 -1)
            binary_pred = np.where(pred > threshold, 1, -1)
            predictions.append(binary_pred)
        
        # 计算投票结果
        votes = np.sum(predictions, axis=0)
        # 转换为最终预测 (1 或 -1)
        ensemble_pred = np.where(votes > 0, 1, -1)
        
        return ensemble_pred
    
    def train_stacking_model(self, X: pd.DataFrame, y: pd.Series):
        """
        训练堆叠集成的元模型
        
        参数:
            X: 特征数据
            y: 目标变量
        """
        if not self.models:
            logger.error("没有可用的基础模型进行堆叠集成")
            return
        
        logger.info("训练堆叠集成元模型")
        
        # 获取每个模型的预测作为元特征
        meta_features = np.column_stack([
            self.predict_with_single_model(model, X)
            for model in self.models
        ])
        
        # 创建元模型
        model_factory = ModelFactory()
        self.meta_model = model_factory.create_model(
            model_type=self.meta_model_type
        )
        
        # 训练元模型
        self.meta_model.train(pd.DataFrame(meta_features), y)
        
        # 保存元模型
        meta_model_path = os.path.join(self.output_dir, f"stacking_meta_model_{self.meta_model_type}.pkl")
        self.meta_model.save(meta_model_path)
        logger.info(f"堆叠集成元模型已保存至: {meta_model_path}")
    
    def predict_stacking(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用堆叠集成方法预测
        
        参数:
            X: 特征数据
            
        返回:
            集成预测结果
        """
        if not self.models:
            logger.error("没有可用的基础模型进行堆叠集成")
            return np.zeros(len(X))
        
        if self.meta_model is None:
            logger.error("元模型未训练，无法进行堆叠集成预测")
            return np.zeros(len(X))
        
        # 获取每个模型的预测作为元特征
        meta_features = np.column_stack([
            self.predict_with_single_model(model, X)
            for model in self.models
        ])
        
        # 使用元模型预测
        ensemble_pred = self.meta_model.predict(pd.DataFrame(meta_features))
        
        return ensemble_pred
    
    def predict(self, X: pd.DataFrame, apply_ma: bool = False) -> np.ndarray:
        """
        根据指定的集成方法进行预测
        
        参数:
            X: 特征数据
            apply_ma: 是否应用移动平均平滑化
            
        返回:
            集成预测结果
        """
        if not self.models:
            logger.error("没有可用的模型进行预测")
            return np.zeros(len(X))
        
        # 根据集成方法选择预测函数
        if self.ensemble_method == "average":
            predictions = self.predict_average(X)
        elif self.ensemble_method == "weighted":
            predictions = self.predict_weighted(X)
        elif self.ensemble_method == "voting":
            predictions = self.predict_voting(X)
        elif self.ensemble_method == "stacking":
            if self.meta_model is None:
                logger.warning("元模型未训练，使用平均集成替代")
                predictions = self.predict_average(X)
            else:
                predictions = self.predict_stacking(X)
        elif self.ensemble_method == "moving_avg":
            # 直接对第一个模型的预测应用移动平均
            if self.models:
                predictions = self.predict_with_single_model(self.models[0], X)
            else:
                predictions = np.zeros(len(X))
        else:
            logger.error(f"不支持的集成方法: {self.ensemble_method}，使用平均集成替代")
            predictions = self.predict_average(X)
        
        # 如果需要应用移动平均或者集成方法本身是移动平均
        if apply_ma or self.ensemble_method == "moving_avg":
            predictions = self.apply_moving_average(predictions)
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        评估集成预测结果
        
        参数:
            X: 特征数据
            y: 实际目标值
            
        返回:
            评估指标字典
        """
        if not self.models:
            logger.error("没有可用的模型进行评估")
            return {}
        
        # 获取集成预测
        ensemble_pred = self.predict(X)
        
        # 计算评估指标
        metrics = {
            'mse': mean_squared_error(y, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y, ensemble_pred)),
            'mae': mean_absolute_error(y, ensemble_pred),
            'r2': r2_score(y, ensemble_pred)
        }
        
        # 计算各个单模型的指标
        individual_metrics = []
        for i, model in enumerate(self.models):
            pred = self.predict_with_single_model(model, X)
            mse = mean_squared_error(y, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, pred)
            r2 = r2_score(y, pred)
            
            model_name = f"model_{i+1}"
            individual_metrics.append({
                'model': model_name,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
        
        # 输出评估结果
        logger.info(f"集成模型 ({self.ensemble_method}) 评估指标:")
        logger.info(f"MSE: {metrics['mse']:.6f}, RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}, R2: {metrics['r2']:.6f}")
        
        for metric in individual_metrics:
            logger.info(f"单模型 {metric['model']} 评估指标:")
            logger.info(f"MSE: {metric['mse']:.6f}, RMSE: {metric['rmse']:.6f}, MAE: {metric['mae']:.6f}, R2: {metric['r2']:.6f}")
        
        # 如果需要绘图，绘制评估结果
        if self.plot_results:
            self.plot_evaluation(y, ensemble_pred, individual_metrics)
        
        # 保存评估结果
        result = {
            'ensemble_method': self.ensemble_method,
            'window_size': self.window_size if self.ensemble_method == "moving_avg" else None,
            'ensemble_metrics': metrics,
            'individual_metrics': individual_metrics,
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        result_path = os.path.join(self.output_dir, f"ensemble_evaluation_{self.ensemble_method}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
        
        logger.info(f"评估结果已保存至: {result_path}")
        
        return metrics
    
    def plot_evaluation(self, y_true: pd.Series, y_pred: np.ndarray, individual_metrics: List[Dict[str, Any]]):
        """
        绘制评估结果可视化图表
        
        参数:
            y_true: 实际目标值
            y_pred: 集成预测值
            individual_metrics: 各个单模型的评估指标
        """
        # 创建结果目录
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. 绘制预测值与实际值对比图
        plt.figure(figsize=(14, 7))
        # 子图1: 预测与实际值的时间序列
        plt.subplot(1, 2, 1)
        plt.plot(y_true.values, 'b-', label='实际值')
        plt.plot(y_pred, 'r-', label='集成预测')
        plt.title(f'预测值与实际值对比 ({self.ensemble_method}集成)')
        plt.xlabel('样本索引')
        plt.ylabel('目标值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 预测值与实际值的散点图
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
        plt.title('预测值与实际值的散点图')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.join(plots_dir, f"prediction_comparison_{self.ensemble_method}.png")
        plt.savefig(fig_path)
        plt.close()
        
        # 2. 绘制各个模型的性能对比图
        if individual_metrics:
            metrics_df = pd.DataFrame(individual_metrics)
            metrics_df.set_index('model', inplace=True)
            
            # 添加集成模型结果
            ensemble_row = pd.DataFrame({
                'mse': [mean_squared_error(y_true, y_pred)],
                'rmse': [np.sqrt(mean_squared_error(y_true, y_pred))],
                'mae': [mean_absolute_error(y_true, y_pred)],
                'r2': [r2_score(y_true, y_pred)]
            }, index=['Ensemble'])
            
            metrics_df = pd.concat([metrics_df, ensemble_row])
            
            # 绘制柱状图比较各个模型的性能
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # MSE
            axes[0, 0].bar(metrics_df.index, metrics_df['mse'])
            axes[0, 0].set_title('均方误差 (MSE)')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].tick_params(axis='x', rotation=45)
            for i, v in enumerate(metrics_df['mse']):
                axes[0, 0].text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=8)
            
            # RMSE
            axes[0, 1].bar(metrics_df.index, metrics_df['rmse'])
            axes[0, 1].set_title('均方根误差 (RMSE)')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].tick_params(axis='x', rotation=45)
            for i, v in enumerate(metrics_df['rmse']):
                axes[0, 1].text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=8)
            
            # MAE
            axes[1, 0].bar(metrics_df.index, metrics_df['mae'])
            axes[1, 0].set_title('平均绝对误差 (MAE)')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].tick_params(axis='x', rotation=45)
            for i, v in enumerate(metrics_df['mae']):
                axes[1, 0].text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=8)
            
            # R2
            axes[1, 1].bar(metrics_df.index, metrics_df['r2'])
            axes[1, 1].set_title('决定系数 (R²)')
            axes[1, 1].set_ylabel('R²')
            axes[1, 1].tick_params(axis='x', rotation=45)
            for i, v in enumerate(metrics_df['r2']):
                axes[1, 1].text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            fig_path = os.path.join(plots_dir, f"model_performance_comparison_{self.ensemble_method}.png")
            plt.savefig(fig_path)
            plt.close()

def parse_args():
    """
    解析命令行参数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="集成预测和平滑处理")
    
    parser.add_argument("--ensemble_method", type=str, default="average",
                      help="集成方法 (average, weighted, stacking, voting, moving_avg)")
    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                      help="模型文件路径列表")
    parser.add_argument("--weights", type=float, nargs='+',
                      help="权重列表 (用于加权平均集成)")
    parser.add_argument("--window_size", type=int, default=5,
                      help="窗口大小 (用于移动平均)")
    parser.add_argument("--output_dir", type=str, default="outputs/ensemble",
                      help="输出目录")
    parser.add_argument("--meta_model_type", type=str, default="lightgbm",
                      help="元模型类型 (用于堆叠集成)")
    parser.add_argument("--no_plot", action="store_true",
                      help="不绘制结果图")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                      help="数据目录")
    parser.add_argument("--symbols", type=str, default="BTCUSDT",
                      help="交易对符号，多个符号用逗号分隔")
    parser.add_argument("--timeframe", type=str, default="1h",
                      help="时间框架")
    parser.add_argument("--target_type", type=str, default="price_change_pct",
                      help="目标类型")
    parser.add_argument("--horizon", type=int, default=60,
                      help="预测时间范围")
    parser.add_argument("--eval_only", action="store_true",
                      help="仅评估，不训练堆叠模型")
    parser.add_argument("--apply_ma", action="store_true",
                      help="应用移动平均平滑")
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 设置日志
    logger.info(f"开始集成预测，使用{args.ensemble_method}方法")
    
    # 创建集成预测器
    predictor = EnsemblePredictor(
        ensemble_method=args.ensemble_method,
        model_paths=args.model_paths,
        weights=args.weights,
        window_size=args.window_size,
        output_dir=args.output_dir,
        meta_model_type=args.meta_model_type,
        plot_results=not args.no_plot
    )
    
    # 加载数据用于评估
    from src.data.data_loader import DataLoader
    
    symbols = args.symbols.split(",")
    
    data_loader = DataLoader(
        data_dir=args.data_dir,
        symbols=symbols,
        timeframes=[args.timeframe],
        target_type=args.target_type,
        target_horizon=args.horizon
    )
    
    # 获取特征和目标数据
    X_train, y_train, X_test, y_test = data_loader.load_train_test_data(test_size=0.2)
    
    if X_train is None or y_train is None or X_test is None or y_test is None:
        logger.error("加载数据失败")
        return
    
    logger.info(f"加载数据完成，训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    
    # 如果是堆叠集成且不是仅评估模式，训练元模型
    if args.ensemble_method == "stacking" and not args.eval_only:
        predictor.train_stacking_model(X_train, y_train)
    
    # 评估集成模型
    metrics = predictor.evaluate(X_test, y_test)
    
    logger.info("集成预测完成")

if __name__ == "__main__":
    main() 