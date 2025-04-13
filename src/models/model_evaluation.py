"""
模型评估模块 - 用于评估和比较不同模型的性能
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple, Optional, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from datetime import datetime

from src.utils.logger import get_logger
from src.models.base_model import BaseModel

logger = get_logger(__name__)


class ModelEvaluator:
    """
    模型评估器类，用于评估和比较模型性能
    """
    
    def __init__(self, 
                 output_dir: str = "models/evaluation",
                 save_results: bool = True):
        """
        初始化模型评估器
        
        参数:
            output_dir (str): 评估结果输出目录
            save_results (bool): 是否保存评估结果
        """
        self.output_dir = output_dir
        self.save_results = save_results
        
        # 创建输出目录
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_regression_model(self, 
                                 model: BaseModel, 
                                 X_test: Union[pd.DataFrame, np.ndarray], 
                                 y_test: Union[pd.Series, np.ndarray],
                                 model_name: str = None) -> Dict[str, float]:
        """
        评估回归模型性能
        
        参数:
            model: 待评估的模型
            X_test: 测试特征数据
            y_test: 测试目标数据
            model_name: 模型名称（可选）
            
        返回:
            Dict[str, float]: 包含各项评估指标的字典
        """
        # 确保模型已训练
        if not model.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        # 获取模型名称
        model_name = model_name or model.name
        
        # 进行预测
        y_pred = model.predict(X_test)
        
        # 转换为numpy数组以进行评估，同时确保没有无效值
        if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
            y_test_np = y_test.values.ravel()
        else:
            y_test_np = y_test.ravel() if hasattr(y_test, 'ravel') else np.array(y_test)
        
        # 检查是否有无效值
        invalid_mask = np.isnan(y_test_np) | ~np.isfinite(y_test_np)
        if invalid_mask.any():
            logger.warning(f"评估前发现{invalid_mask.sum()}个无效值，将在计算指标时忽略这些值")
            # 创建有效值的掩码
            valid_mask = ~invalid_mask
            y_test_valid = y_test_np[valid_mask]
            y_pred_valid = y_pred[valid_mask]
        else:
            y_test_valid = y_test_np
            y_pred_valid = y_pred
        
        # 如果没有有效值，则返回默认指标
        if len(y_test_valid) == 0:
            logger.error("评估数据中没有有效值，无法计算指标")
            return {"rmse": 0.0, "mae": 0.0, "r2": 0.0}
            
        # 计算回归指标
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test_valid, y_pred_valid))),
            "mae": float(mean_absolute_error(y_test_valid, y_pred_valid)),
            "r2": float(r2_score(y_test_valid, y_pred_valid))
        }
        
        # 方向准确率（对于价格变化预测）
        if model.target_type == "price_change_pct":
            direction_actual = np.sign(y_test_valid)
            direction_pred = np.sign(y_pred_valid)
            metrics["direction_accuracy"] = float(accuracy_score(direction_actual, direction_pred))
        
        # 输出评估结果
        logger.info(f"模型 {model_name} 评估结果：")
        for metric, value in metrics.items():
            logger.info(f"  - {metric}: {value:.4f}")
        
        # 如果需要保存结果
        if self.save_results:
            self._save_evaluation_results(metrics, model, model_name, "regression")
            self._plot_regression_results(y_test_valid, y_pred_valid, model_name)
        
        return metrics
    
    def evaluate_classification_model(self, 
                                     model: BaseModel, 
                                     X_test: Union[pd.DataFrame, np.ndarray], 
                                     y_test: Union[pd.Series, np.ndarray],
                                     model_name: str = None) -> Dict[str, float]:
        """
        评估分类模型性能
        
        参数:
            model: 待评估的模型
            X_test: 测试特征数据
            y_test: 测试目标数据
            model_name: 模型名称（可选）
            
        返回:
            Dict[str, float]: 包含各项评估指标的字典
        """
        # 确保模型已训练
        if not model.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        # 获取模型名称
        model_name = model_name or model.name
        
        # 进行预测
        y_pred = model.predict(X_test)
        
        # 转换为numpy数组以进行评估，同时确保没有无效值
        if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
            y_test_np = y_test.values.ravel()
        else:
            y_test_np = y_test.ravel() if hasattr(y_test, 'ravel') else np.array(y_test)
        
        # 检查是否有无效值
        invalid_mask = np.isnan(y_test_np) | ~np.isfinite(y_test_np)
        if invalid_mask.any():
            logger.warning(f"评估前发现{invalid_mask.sum()}个无效值，将在计算指标时忽略这些值")
            # 创建有效值的掩码
            valid_mask = ~invalid_mask
            y_test_valid = y_test_np[valid_mask]
            y_pred_valid = y_pred[valid_mask]
        else:
            y_test_valid = y_test_np
            y_pred_valid = y_pred
        
        # 如果没有有效值，则返回默认指标
        if len(y_test_valid) == 0:
            logger.error("评估数据中没有有效值，无法计算指标")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        # 计算分类指标
        metrics = {
            "accuracy": float(accuracy_score(y_test_valid, y_pred_valid)),
            "precision": float(precision_score(y_test_valid, y_pred_valid, average="weighted")),
            "recall": float(recall_score(y_test_valid, y_pred_valid, average="weighted")),
            "f1": float(f1_score(y_test_valid, y_pred_valid, average="weighted"))
        }
        
        # 如果模型支持概率预测，计算AUC
        try:
            y_proba = model.predict_proba(X_test)
            if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                # 多分类情况
                pass  # 暂不处理多分类AUC
            else:
                # 二分类情况
                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 1]
                if invalid_mask.any():
                    y_proba_valid = y_proba[valid_mask]
                else:
                    y_proba_valid = y_proba
                metrics["auc"] = float(roc_auc_score(y_test_valid, y_proba_valid))
        except (NotImplementedError, AttributeError, ValueError) as e:
            # 如果模型不支持概率预测，或者其他错误，则忽略AUC
            logger.warning(f"无法计算AUC: {str(e)}")
            
        # 输出评估结果
        logger.info(f"模型 {model_name} 评估结果：")
        for metric, value in metrics.items():
            logger.info(f"  - {metric}: {value:.4f}")
        
        # 如果需要保存结果
        if self.save_results:
            self._save_evaluation_results(metrics, model, model_name, "classification")
            self._plot_confusion_matrix(y_test_valid, y_pred_valid, model_name)
        
        return metrics
    
    def evaluate_model(self, 
                      model: BaseModel, 
                      X_test: Union[pd.DataFrame, np.ndarray], 
                      y_test: Union[pd.Series, np.ndarray],
                      model_name: str = None) -> Dict[str, float]:
        """
        根据目标类型评估模型性能
        
        参数:
            model: 待评估的模型
            X_test: 测试特征数据
            y_test: 测试目标数据
            model_name: 模型名称（可选）
            
        返回:
            Dict[str, float]: 包含各项评估指标的字典
        """
        # 数据清洗：处理测试数据中的无效值
        logger.info(f"评估模型前进行数据清洗，原始数据形状：X_test={X_test.shape}, y_test={y_test.shape}")
        
        # 处理特征数据
        if isinstance(X_test, pd.DataFrame):
            X_test_clean = X_test.copy()
            # 检查并处理NaN值
            if X_test_clean.isna().sum().sum() > 0:
                logger.warning(f"测试特征数据中发现NaN值，使用前向填充和后向填充处理")
                X_test_clean = X_test_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # 检查并处理无穷大值
            inf_mask = ~np.isfinite(X_test_clean)
            if inf_mask.values.any():
                logger.warning(f"测试特征数据中发现无穷大值，将替换为0")
                X_test_clean = X_test_clean.replace([np.inf, -np.inf], 0)
        else:  # numpy array
            X_test_clean = X_test.copy()
            # 处理NaN值
            if np.isnan(X_test_clean).any():
                logger.warning(f"测试特征数据中发现NaN值，将替换为0")
                X_test_clean[np.isnan(X_test_clean)] = 0
            
            # 处理无穷大值
            inf_mask = ~np.isfinite(X_test_clean)
            if inf_mask.any():
                logger.warning(f"测试特征数据中发现无穷大值，将替换为0")
                X_test_clean[inf_mask] = 0
        
        # 处理目标数据
        if isinstance(y_test, pd.Series):
            y_test_clean = y_test.copy()
            # 检查并处理NaN和无穷大值
            nan_mask = y_test_clean.isna()
            inf_mask = ~np.isfinite(y_test_clean)
            invalid_mask = nan_mask | inf_mask
            
            if invalid_mask.any():
                invalid_count = invalid_mask.sum()
                logger.warning(f"测试目标数据中发现{invalid_count}个无效值，将替换为0")
                
                # 替换无效值
                y_test_clean = y_test_clean.fillna(0)
                y_test_clean[inf_mask] = 0
        else:  # numpy array
            y_test_clean = y_test.copy()
            # 处理NaN值
            nan_mask = np.isnan(y_test_clean)
            if nan_mask.any():
                logger.warning(f"测试目标数据中发现NaN值，将替换为0")
                y_test_clean[nan_mask] = 0
            
            # 处理无穷大值
            inf_mask = ~np.isfinite(y_test_clean)
            if inf_mask.any():
                logger.warning(f"测试目标数据中发现无穷大值，将替换为0")
                y_test_clean[inf_mask] = 0
        
        logger.info(f"数据清洗完成，清洗后数据形状：X_test={X_test_clean.shape}, y_test={y_test_clean.shape}")
        
        # 根据目标类型选择评估方法
        if model.target_type == "direction":
            return self.evaluate_classification_model(model, X_test_clean, y_test_clean, model_name)
        else:
            return self.evaluate_regression_model(model, X_test_clean, y_test_clean, model_name)
    
    def compare_models(self, 
                      models: List[BaseModel], 
                      X_test: Union[pd.DataFrame, np.ndarray], 
                      y_test: Union[pd.Series, np.ndarray],
                      model_names: List[str] = None) -> pd.DataFrame:
        """
        比较多个模型的性能
        
        参数:
            models: 待比较的模型列表
            X_test: 测试特征数据
            y_test: 测试目标数据
            model_names: 模型名称列表（可选）
            
        返回:
            pd.DataFrame: 包含各模型评估指标的DataFrame
        """
        # 初始化结果字典
        results = {}
        
        # 评估每个模型
        for i, model in enumerate(models):
            # 获取模型名称
            model_name = None
            if model_names and i < len(model_names):
                model_name = model_names[i]
            else:
                model_name = model.name
            
            # 评估模型
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            # 添加到结果字典
            results[model_name] = metrics
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results).T
        
        # 如果需要保存结果
        if self.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存比较结果为CSV
            output_path = os.path.join(self.output_dir, f"model_comparison_{timestamp}.csv")
            results_df.to_csv(output_path)
            logger.info(f"模型比较结果已保存至 {output_path}")
            
            # 绘制比较图表
            self._plot_model_comparison(results_df)
        
        return results_df
    
    def _save_evaluation_results(self, 
                                metrics: Dict[str, float], 
                                model: BaseModel, 
                                model_name: str,
                                eval_type: str) -> None:
        """
        保存评估结果
        
        参数:
            metrics: 评估指标字典
            model: 评估的模型
            model_name: 模型名称
            eval_type: 评估类型（regression或classification）
        """
        # 创建模型评估目录
        model_eval_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_eval_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 构建结果字典
        result = {
            "model_name": model_name,
            "model_type": model.__class__.__name__,
            "evaluation_type": eval_type,
            "prediction_horizon": model.prediction_horizon,
            "target_type": model.target_type,
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        # 保存结果为JSON
        output_path = os.path.join(model_eval_dir, f"evaluation_{timestamp}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"评估结果已保存至 {output_path}")
    
    def _plot_regression_results(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray, 
                                model_name: str) -> None:
        """
        绘制回归结果散点图
        
        参数:
            y_true: 实际值
            y_pred: 预测值
            model_name: 模型名称
        """
        # 创建模型评估目录
        model_eval_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_eval_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([-1, 1], [-1, 1], 'r--')  # 对角线
        plt.xlim(min(y_true), max(y_true))
        plt.ylim(min(y_pred), max(y_pred))
        plt.title(f"{model_name} - 预测值 vs 实际值")
        plt.xlabel("实际值")
        plt.ylabel("预测值")
        plt.grid(True)
        
        # 添加RMSE和R²文本
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        plt.annotate(f"RMSE: {rmse:.4f}\nR²: {r2:.4f}", 
                    xy=(0.05, 0.95), xycoords='axes fraction', 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # 保存图表
        output_path = os.path.join(model_eval_dir, f"regression_plot_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"回归结果图表已保存至 {output_path}")
    
    def _plot_confusion_matrix(self, 
                              y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              model_name: str) -> None:
        """
        绘制混淆矩阵
        
        参数:
            y_true: 实际类别
            y_pred: 预测类别
            model_name: 模型名称
        """
        # 创建模型评估目录
        model_eval_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_eval_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 创建图表
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{model_name} - 混淆矩阵")
        plt.colorbar()
        
        # 添加刻度标签
        classes = np.unique(np.concatenate((y_true, y_pred)))
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # 添加数值标签
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('实际类别')
        plt.xlabel('预测类别')
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(model_eval_dir, f"confusion_matrix_{timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"混淆矩阵已保存至 {output_path}")
        
        # 输出分类报告
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # 保存分类报告
        report_path = os.path.join(model_eval_dir, f"classification_report_{timestamp}.csv")
        report_df.to_csv(report_path)
        
        logger.info(f"分类报告已保存至 {report_path}")
    
    def _plot_model_comparison(self, results_df: pd.DataFrame) -> None:
        """
        绘制模型比较图表
        
        参数:
            results_df: 包含各模型评估指标的DataFrame
        """
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确定评估类型（回归或分类）
        is_regression = 'rmse' in results_df.columns
        
        # 选择要绘制的指标
        if is_regression:
            metrics_to_plot = ['rmse', 'mae', 'r2']
            if 'direction_accuracy' in results_df.columns:
                metrics_to_plot.append('direction_accuracy')
        else:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            if 'auc' in results_df.columns:
                metrics_to_plot.append('auc')
        
        # 过滤有效的指标
        metrics_to_plot = [m for m in metrics_to_plot if m in results_df.columns]
        
        # 为每个指标创建条形图
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            ax = results_df[metric].sort_values().plot(kind='barh')
            plt.title(f"模型比较 - {metric}")
            plt.xlabel(metric)
            plt.ylabel("模型")
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # 在条形上添加数值标签
            for i, v in enumerate(results_df[metric].sort_values()):
                ax.text(v, i, f"{v:.4f}", va='center')
            
            # 保存图表
            output_path = os.path.join(self.output_dir, f"model_comparison_{metric}_{timestamp}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"模型比较图表 ({metric}) 已保存至 {output_path}")
        
        # 创建雷达图（如果有足够的指标）
        if len(metrics_to_plot) >= 3:
            self._plot_radar_chart(results_df, metrics_to_plot, timestamp) 