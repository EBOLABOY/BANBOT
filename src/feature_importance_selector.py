"""
基于特征重要性的特征选择脚本
用于从训练好的模型中提取特征重要性并进行特征筛选
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from src.utils.logger import get_logger
from src.data.data_loader import DataLoader
from src.models.model_factory import ModelFactory

logger = get_logger(__name__)

class FeatureImportanceSelector:
    """
    基于特征重要性的特征选择器
    用于从训练好的模型中提取特征重要性并进行特征筛选
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        model_path: Optional[str] = None,
        importance_threshold: float = 0.01,
        top_n_features: Optional[int] = 20,
        output_dir: str = "outputs/feature_selection",
        plot_results: bool = True
    ):
        """
        初始化特征选择器
        
        参数:
            model_type: 模型类型 ("xgboost", "lightgbm" 等)
            model_path: 预训练模型路径 (如果提供，将加载此模型)
            importance_threshold: 特征重要性阈值，低于此值的特征将被过滤
            top_n_features: 保留的顶部特征数量 (如果为None，则使用阈值)
            output_dir: 输出目录
            plot_results: 是否绘制结果图
        """
        self.model_type = model_type
        self.model_path = model_path
        self.importance_threshold = importance_threshold
        self.top_n_features = top_n_features
        self.output_dir = output_dir
        self.plot_results = plot_results
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化模型和特征重要性
        self.model = None
        self.feature_importance = None
        self.selected_features = []
        
        # 如果提供了模型路径，加载模型
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        加载预训练模型
        
        参数:
            model_path: 模型文件路径
        """
        try:
            self.model = pickle.load(open(model_path, 'rb'))
            logger.info(f"已加载模型: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.model = None
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, params: Dict[str, Any] = None):
        """
        训练特征选择模型
        
        参数:
            X: 特征数据
            y: 目标变量
            params: 模型参数 (可选)
        """
        logger.info(f"训练{self.model_type}模型用于特征选择")
        
        # 创建模型
        model_factory = ModelFactory()
        self.model = model_factory.create_model(
            model_type=self.model_type,
            params=params
        )
        
        # 训练模型
        self.model.train(X, y)
        
        # 保存模型
        model_path = os.path.join(self.output_dir, f"{self.model_type}_for_feature_selection.pkl")
        self.model.save(model_path)
        logger.info(f"特征选择模型已保存至: {model_path}")
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        获取特征重要性
        
        参数:
            X: 特征数据 (用于获取特征名称)
            
        返回:
            特征重要性DataFrame
        """
        if self.model is None:
            logger.error("模型未加载或训练，无法获取特征重要性")
            return None
        
        try:
            # 获取原始特征重要性
            feature_names = X.columns.tolist()
            importance = self.model.get_feature_importance()
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            
            # 按重要性降序排列
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # 计算规范化重要性和累积重要性
            total_importance = importance_df['importance'].sum()
            importance_df['importance_normalized'] = importance_df['importance'] / total_importance
            importance_df['importance_cumulative'] = importance_df['importance_normalized'].cumsum()
            
            self.feature_importance = importance_df
            logger.info(f"已获取{len(feature_names)}个特征的重要性")
            
            return importance_df
            
        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return None
    
    def select_features(self, X: pd.DataFrame, y: pd.Series = None) -> List[str]:
        """
        基于特征重要性选择特征
        
        参数:
            X: 特征数据
            y: 目标变量 (仅用于训练新模型时使用)
            
        返回:
            选中的特征列表
        """
        # 如果模型为None且y不为None，训练模型
        if self.model is None and y is not None:
            self.train_model(X, y)
        
        # 获取特征重要性
        if self.feature_importance is None:
            self.get_feature_importance(X)
            
        if self.feature_importance is None:
            logger.error("无法获取特征重要性，无法选择特征")
            return []
        
        # 基于阈值或Top N选择特征
        if self.top_n_features is not None:
            # 选择Top N特征
            self.selected_features = self.feature_importance['feature'].head(self.top_n_features).tolist()
            logger.info(f"已选择Top {self.top_n_features}特征")
        else:
            # 基于阈值选择特征
            self.selected_features = self.feature_importance[
                self.feature_importance['importance_normalized'] >= self.importance_threshold
            ]['feature'].tolist()
            logger.info(f"已选择{len(self.selected_features)}个重要性阈值 >= {self.importance_threshold} 的特征")
        
        # 保存选中的特征
        self.save_selected_features()
        
        # 如果需要绘图，绘制特征重要性图
        if self.plot_results:
            self.plot_feature_importance()
        
        return self.selected_features
    
    def save_selected_features(self):
        """
        保存选中的特征
        """
        if not self.selected_features:
            logger.warning("没有选中的特征可保存")
            return
        
        # 创建结果字典
        result = {
            'model_type': self.model_type,
            'selection_method': 'top_n' if self.top_n_features is not None else 'threshold',
            'importance_threshold': self.importance_threshold if self.top_n_features is None else None,
            'top_n_features': self.top_n_features,
            'n_selected_features': len(self.selected_features),
            'selected_features': self.selected_features,
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存为JSON
        result_path = os.path.join(self.output_dir, f"selected_features_{self.model_type}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"选中的特征已保存至: {result_path}")
        
        # 如果有特征重要性，保存为CSV
        if self.feature_importance is not None:
            # 标记选中的特征
            self.feature_importance['selected'] = self.feature_importance['feature'].isin(self.selected_features)
            
            importance_path = os.path.join(self.output_dir, f"feature_importance_{self.model_type}.csv")
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"特征重要性已保存至: {importance_path}")
    
    def plot_feature_importance(self, max_features: int = 30):
        """
        绘制特征重要性图
        
        参数:
            max_features: 图表中显示的最大特征数
        """
        if self.feature_importance is None:
            logger.warning("无特征重要性数据，无法绘图")
            return
        
        # 创建特征重要性条形图
        plt.figure(figsize=(12, max(8, min(30, len(self.selected_features)) / 2)))
        
        # 获取要显示的特征数量
        n_features = min(max_features, len(self.feature_importance))
        
        # 获取显示的特征子集
        plot_df = self.feature_importance.head(n_features).copy()
        
        # 标记选中的特征
        plot_df['selected'] = plot_df['feature'].isin(self.selected_features)
        
        # 设置颜色映射
        colors = ['#ff7f0e' if selected else '#1f77b4' for selected in plot_df['selected']]
        
        # 绘制条形图
        ax = sns.barplot(x='importance_normalized', y='feature', data=plot_df, 
                       palette=colors, orient='h')
        
        # 添加值标签
        for i, v in enumerate(plot_df['importance_normalized']):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        # 设置标题和标签
        plt.title(f'Top {n_features} 特征重要性 ({self.model_type})')
        plt.xlabel('规范化重要性')
        plt.ylabel('特征名称')
        plt.tight_layout()
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff7f0e', label='选中的特征'),
            Patch(facecolor='#1f77b4', label='未选中的特征')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # 保存图表
        fig_path = os.path.join(self.output_dir, f"feature_importance_{self.model_type}.png")
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"特征重要性图表已保存至: {fig_path}")
        
        # 绘制累积重要性图
        plt.figure(figsize=(10, 6))
        
        # 计算每个特征的累积贡献
        x = range(1, len(self.feature_importance) + 1)
        y = self.feature_importance['importance_cumulative'].values
        
        plt.plot(x, y, 'o-')
        
        # 标记选中的特征数量
        if self.top_n_features is not None:
            plt.axvline(x=self.top_n_features, color='r', linestyle=':', alpha=0.7)
            plt.text(self.top_n_features + 1, 0.5, f'Top {self.top_n_features} 特征', 
                    rotation=90, va='center')
        
        # 标记累积重要性阈值
        if self.top_n_features is None:
            # 找到重要性阈值对应的特征数量
            threshold_idx = (self.feature_importance['importance_normalized'] >= self.importance_threshold).sum()
            cumulative_at_threshold = self.feature_importance['importance_cumulative'].iloc[threshold_idx - 1]
            
            plt.axvline(x=threshold_idx, color='r', linestyle=':', alpha=0.7)
            plt.axhline(y=cumulative_at_threshold, color='r', linestyle=':', alpha=0.7)
            plt.text(threshold_idx + 1, 0.5, f'阈值 = {self.importance_threshold}\n特征数 = {threshold_idx}', 
                    rotation=90, va='center')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('特征数量')
        plt.ylabel('累积重要性')
        plt.title('特征累积重要性')
        
        # 保存图表
        fig_path = os.path.join(self.output_dir, f"cumulative_importance_{self.model_type}.png")
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"累积重要性图表已保存至: {fig_path}")
    
    def evaluate_feature_subsets(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        """
        评估不同数量特征子集的性能
        
        参数:
            X: 特征数据
            y: 目标变量
            cv: 交叉验证折数
        """
        if self.feature_importance is None:
            logger.warning("无特征重要性数据，无法评估特征子集")
            return
        
        logger.info("开始评估不同特征子集的性能")
        
        # 特征重要性按降序排列
        sorted_features = self.feature_importance['feature'].tolist()
        
        # 生成特征子集大小
        feature_counts = []
        total_features = len(sorted_features)
        
        # 特征数量的间隔
        if total_features <= 10:
            step = 1
        elif total_features <= 50:
            step = 5
        elif total_features <= 100:
            step = 10
        else:
            step = 20
        
        # 生成特征数量序列
        for i in range(step, total_features + 1, step):
            feature_counts.append(i)
        
        # 确保最大特征数量在列表中
        if total_features not in feature_counts:
            feature_counts.append(total_features)
            
        # 评估不同特征子集的性能
        results = []
        
        for n_features in feature_counts:
            # 获取特征子集
            feature_subset = sorted_features[:n_features]
            X_subset = X[feature_subset]
            
            # 创建模型
            model_factory = ModelFactory()
            model = model_factory.create_model(model_type=self.model_type)
            
            # 计算交叉验证分数
            try:
                scores = cross_val_score(
                    model, X_subset, y, 
                    cv=cv, 
                    scoring='neg_mean_squared_error'
                )
                mean_score = -np.mean(scores)  # 转换为正MSE
                std_score = np.std(scores)
                
                results.append({
                    'n_features': n_features,
                    'mean_mse': mean_score,
                    'std_mse': std_score,
                    'features': feature_subset
                })
                
                logger.info(f"特征数量: {n_features}, 平均MSE: {mean_score:.4f} ± {std_score:.4f}")
                
            except Exception as e:
                logger.error(f"评估特征子集 (n={n_features}) 失败: {e}")
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存为CSV
        results_path = os.path.join(self.output_dir, f"feature_subset_evaluation_{self.model_type}.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"特征子集评估结果已保存至: {results_path}")
        
        # 绘制结果
        if self.plot_results:
            plt.figure(figsize=(10, 6))
            
            # 绘制主曲线
            plt.errorbar(
                results_df['n_features'], 
                results_df['mean_mse'], 
                yerr=results_df['std_mse'],
                fmt='o-', 
                capsize=5,
                label='MSE (均值 ± 标准差)'
            )
            
            # 找到MSE最低的特征数量
            best_idx = results_df['mean_mse'].idxmin()
            best_n_features = results_df.loc[best_idx, 'n_features']
            best_score = results_df.loc[best_idx, 'mean_mse']
            
            # 标记最佳特征数量
            plt.scatter([best_n_features], [best_score], color='r', s=100, 
                       label=f'最佳特征数: {best_n_features}')
            plt.axvline(x=best_n_features, color='r', linestyle=':', alpha=0.5)
            
            # 标记选中的特征数量
            if self.top_n_features is not None:
                plt.axvline(x=self.top_n_features, color='g', linestyle='--', alpha=0.5,
                           label=f'选中的特征数: {self.top_n_features}')
            
            plt.grid(True, alpha=0.3)
            plt.xlabel('特征数量')
            plt.ylabel('均方误差 (MSE)')
            plt.title('不同特征子集大小的性能比较')
            plt.legend()
            
            # 保存图表
            fig_path = os.path.join(self.output_dir, f"feature_subset_performance_{self.model_type}.png")
            plt.savefig(fig_path)
            plt.close()
            logger.info(f"特征子集性能图表已保存至: {fig_path}")
        
        # 返回最佳特征子集
        best_idx = results_df['mean_mse'].idxmin()
        best_n_features = results_df.loc[best_idx, 'n_features']
        best_features = results_df.loc[best_idx, 'features']
        
        return {
            'best_n_features': best_n_features,
            'best_features': best_features,
            'results': results_df
        }
    
    def apply_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        应用特征选择到数据集
        
        参数:
            X: 特征数据
            
        返回:
            选中特征的数据
        """
        if not self.selected_features:
            logger.warning("没有选中的特征可应用")
            return X
        
        # 检查所有选中的特征是否都在X中
        missing_features = [f for f in self.selected_features if f not in X.columns]
        if missing_features:
            logger.warning(f"数据中缺少以下选中的特征: {missing_features}")
            # 仅使用存在的特征
            valid_features = [f for f in self.selected_features if f in X.columns]
            logger.info(f"仅应用{len(valid_features)}/{len(self.selected_features)}个有效特征")
            return X[valid_features]
        
        # 应用特征选择
        X_selected = X[self.selected_features]
        logger.info(f"已应用特征选择，从{X.shape[1]}个特征中选择了{len(self.selected_features)}个特征")
        
        return X_selected

def parse_args():
    """
    解析命令行参数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="基于特征重要性的特征选择")
    
    parser.add_argument("--model_type", type=str, default="xgboost",
                      help="模型类型 (目前支持 xgboost, lightgbm)")
    parser.add_argument("--model_path", type=str, default=None,
                      help="预训练模型路径 (如果提供，将加载此模型)")
    parser.add_argument("--importance_threshold", type=float, default=0.01,
                      help="特征重要性阈值，低于此值的特征将被过滤")
    parser.add_argument("--top_n_features", type=int, default=None,
                      help="保留的顶部特征数量")
    parser.add_argument("--output_dir", type=str, default="outputs/feature_selection",
                      help="输出目录")
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
    parser.add_argument("--evaluate_subsets", action="store_true",
                      help="是否评估不同特征子集的性能")
    parser.add_argument("--cv_folds", type=int, default=5,
                      help="交叉验证折数 (用于评估特征子集)")
    parser.add_argument("--no_plot", action="store_true",
                      help="不绘制结果图")
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 设置日志
    logger.info("开始基于特征重要性的特征选择")
    
    # 加载数据
    symbols = args.symbols.split(",")
    
    data_loader = DataLoader(
        data_dir=args.data_dir,
        symbols=symbols,
        timeframes=[args.timeframe],
        target_type=args.target_type,
        target_horizon=args.horizon
    )
    
    # 获取特征和目标数据
    X, y = data_loader.load_train_data()
    
    if X is None or y is None:
        logger.error("加载数据失败")
        return
    
    logger.info(f"加载数据完成，特征形状: {X.shape}, 目标形状: {y.shape}")
    
    # 创建特征选择器
    selector = FeatureImportanceSelector(
        model_type=args.model_type,
        model_path=args.model_path,
        importance_threshold=args.importance_threshold,
        top_n_features=args.top_n_features,
        output_dir=args.output_dir,
        plot_results=not args.no_plot
    )
    
    # 选择特征
    selected_features = selector.select_features(X, y)
    
    logger.info(f"已选择{len(selected_features)}个特征: {selected_features}")
    
    # 如果需要评估不同特征子集的性能
    if args.evaluate_subsets:
        subset_results = selector.evaluate_feature_subsets(X, y, cv=args.cv_folds)
        if subset_results:
            logger.info(f"最佳特征数量: {subset_results['best_n_features']}")
    
    logger.info("基于特征重要性的特征选择完成")

if __name__ == "__main__":
    main() 