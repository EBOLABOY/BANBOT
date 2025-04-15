"""
基于特征重要性的特征选择脚本
用于根据模型的特征重要性进行二次筛选
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import train_test_split
import joblib

from src.utils.logger import get_logger
from src.data.data_loader import DataLoader
from src.models.model_factory import ModelFactory

logger = get_logger(__name__)

class FeatureImportanceSelector:
    """
    基于特征重要性的特征选择器
    """
    def __init__(
        self,
        model_type: str = "xgboost",
        model_path: Optional[str] = None,
        importance_threshold: float = 0.01,
        top_n_features: Optional[int] = None,
        output_dir: str = "outputs/feature_selection",
        plot_results: bool = True
    ):
        """
        初始化特征选择器
        
        参数:
            model_type: 模型类型
            model_path: 预训练模型路径，如果提供则加载
            importance_threshold: 特征重要性阈值，低于此值的特征将被过滤
            top_n_features: 选择前N个重要特征，优先级高于threshold
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
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """
        加载预训练模型
        
        参数:
            model_path: 模型文件路径
        """
        logger.info(f"从 {model_path} 加载预训练模型")
        try:
            self.model = joblib.load(model_path)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            self.model = None
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame = None, y_val: pd.Series = None,
                   params: Dict[str, Any] = None):
        """
        训练模型以获取特征重要性
        
        参数:
            X_train: 训练集特征
            y_train: 训练集目标
            X_val: 验证集特征 (可选)
            y_val: 验证集目标 (可选)
            params: 模型参数 (可选)
        """
        logger.info(f"训练{self.model_type}模型以获取特征重要性")
        
        # 使用模型工厂创建模型
        model_factory = ModelFactory()
        self.model = model_factory.create_model(
            model_type=self.model_type,
            params=params
        )
        
        # 训练模型
        if X_val is not None and y_val is not None:
            self.model.train(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            self.model.train(X_train, y_train)
        
        logger.info("模型训练完成")
        
        # 保存模型
        model_path = os.path.join(self.output_dir, f"{self.model_type}_for_feature_selection.pkl")
        self.model.save(model_path)
        logger.info(f"模型已保存至 {model_path}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        返回:
            特征重要性DataFrame，包含特征名称和重要性得分
        """
        if self.model is None:
            logger.error("模型未加载或训练，无法获取特征重要性")
            return None
        
        try:
            # 获取特征重要性
            importance = self.model.get_feature_importance()
            
            if importance is None or len(importance) == 0:
                logger.error("无法获取特征重要性")
                return None
            
            # 转换为DataFrame
            importance_df = pd.DataFrame({
                'feature': list(importance.keys()),
                'importance': list(importance.values())
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
            
            # 计算归一化的重要性
            importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()
            
            # 计算累积重要性
            importance_df['importance_cumulative'] = importance_df['importance_normalized'].cumsum()
            
            self.feature_importance = importance_df
            logger.info(f"已获取{len(importance_df)}个特征的重要性得分")
            
            return importance_df
        
        except Exception as e:
            logger.error(f"获取特征重要性时出错: {e}")
            return None
    
    def select_features(self, X: pd.DataFrame = None) -> List[str]:
        """
        基于特征重要性选择特征
        
        参数:
            X: 包含所有特征的DataFrame (可选)，用于辅助选择
            
        返回:
            选中的特征列表
        """
        # 获取特征重要性 (如果尚未获取)
        if self.feature_importance is None:
            self.get_feature_importance()
        
        if self.feature_importance is None or len(self.feature_importance) == 0:
            logger.error("无特征重要性信息，无法进行特征选择")
            return []
        
        # 选择特征
        if self.top_n_features is not None and self.top_n_features > 0:
            # 基于top-N方法选择
            top_n = min(self.top_n_features, len(self.feature_importance))
            selected = self.feature_importance.head(top_n)['feature'].tolist()
            logger.info(f"基于Top-{top_n}方法选择了{len(selected)}个特征")
        else:
            # 基于阈值选择
            selected = self.feature_importance[
                self.feature_importance['importance_normalized'] >= self.importance_threshold
            ]['feature'].tolist()
            logger.info(f"基于阈值(>={self.importance_threshold})选择了{len(selected)}个特征")
        
        self.selected_features = selected
        
        # 保存选中的特征
        self._save_selected_features()
        
        # 绘制结果
        if self.plot_results:
            self._plot_feature_importance()
        
        return selected
    
    def _save_selected_features(self):
        """
        保存选中的特征
        """
        if not self.selected_features:
            logger.warning("没有选中的特征可保存")
            return
        
        # 保存为JSON
        features_path = os.path.join(self.output_dir, "selected_features.json")
        
        with open(features_path, 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'selection_method': 'feature_importance',
                'importance_threshold': self.importance_threshold,
                'top_n_features': self.top_n_features,
                'n_selected_features': len(self.selected_features),
                'selected_features': self.selected_features,
                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        logger.info(f"选中的特征已保存至: {features_path}")
        
        # 保存特征重要性
        if self.feature_importance is not None:
            importance_path = os.path.join(self.output_dir, "feature_importance.csv")
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"特征重要性已保存至: {importance_path}")
    
    def _plot_feature_importance(self):
        """
        绘制特征重要性图表
        """
        if self.feature_importance is None or len(self.feature_importance) == 0:
            logger.warning("无特征重要性数据，无法绘图")
            return
        
        plt.figure(figsize=(12, 10))
        
        # 获取要绘制的特征数量 (最多显示前30个)
        n_features = min(30, len(self.feature_importance))
        plot_data = self.feature_importance.head(n_features).copy()
        
        # 将特征名称按照重要性排序
        plot_data = plot_data.sort_values('importance', ascending=True)
        
        # 绘制条形图
        plt.barh(plot_data['feature'], plot_data['importance'])
        plt.xlabel('特征重要性')
        plt.ylabel('特征名称')
        plt.title(f'Top {n_features} 特征重要性')
        plt.tight_layout()
        
        # 保存图表
        fig_path = os.path.join(self.output_dir, "feature_importance_bar.png")
        plt.savefig(fig_path)
        plt.close()
        
        # 绘制累积重要性图
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.feature_importance) + 1), 
                self.feature_importance['importance_cumulative'], 
                marker='o', linestyle='-')
        
        plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% 重要性')
        plt.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90% 重要性')
        
        # 找出达到80%和90%重要性所需的特征数量
        n_80 = self.feature_importance[self.feature_importance['importance_cumulative'] >= 0.8].index[0] + 1
        n_90 = self.feature_importance[self.feature_importance['importance_cumulative'] >= 0.9].index[0] + 1
        
        plt.axvline(x=n_80, color='r', linestyle=':', alpha=0.5)
        plt.axvline(x=n_90, color='g', linestyle=':', alpha=0.5)
        
        plt.xlabel('特征数量')
        plt.ylabel('累积重要性')
        plt.title('特征累积重要性分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加文本标注
        plt.text(n_80 + 1, 0.8, f'达到80%: {n_80}个特征', 
                verticalalignment='center')
        plt.text(n_90 + 1, 0.9, f'达到90%: {n_90}个特征', 
                verticalalignment='center')
        
        plt.tight_layout()
        
        # 保存图表
        fig_path = os.path.join(self.output_dir, "feature_importance_cumulative.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"特征重要性图表已保存至: {self.output_dir}")
    
    def evaluate_feature_subsets(self, X: pd.DataFrame, y: pd.Series, 
                               feature_percentiles: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
                               test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        评估不同特征子集的性能
        
        参数:
            X: 全部特征数据
            y: 目标变量
            feature_percentiles: 特征比例列表，用于测试不同数量的特征
            test_size: 测试集比例
            random_state: 随机种子
            
        返回:
            不同特征集的评估结果
        """
        if self.feature_importance is None:
            logger.error("请先获取特征重要性")
            return None
        
        logger.info(f"评估{len(feature_percentiles)}个不同大小的特征子集")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 获取所有特征，按重要性排序
        all_features = self.feature_importance['feature'].tolist()
        
        results = []
        
        for percentile in feature_percentiles:
            # 计算特征数量
            n_features = max(1, int(len(all_features) * percentile))
            subset_features = all_features[:n_features]
            
            logger.info(f"评估使用{n_features}个特征 ({percentile*100:.0f}%)的子集")
            
            # 创建模型
            model_factory = ModelFactory()
            model = model_factory.create_model(
                model_type=self.model_type
            )
            
            # 训练模型
            model.train(X_train[subset_features], y_train)
            
            # 评估模型
            train_score = model.evaluate(X_train[subset_features], y_train)
            test_score = model.evaluate(X_test[subset_features], y_test)
            
            # 记录结果
            result = {
                'percentile': percentile,
                'n_features': n_features,
                'features': subset_features,
                'train_score': train_score,
                'test_score': test_score,
                'train_test_gap': train_score - test_score
            }
            
            results.append(result)
            
            logger.info(f"子集结果: 训练分数={train_score:.4f}, 测试分数={test_score:.4f}, 差距={train_score-test_score:.4f}")
        
        # 保存结果
        eval_path = os.path.join(self.output_dir, "feature_subset_evaluation.json")
        with open(eval_path, 'w') as f:
            # 无法直接序列化DataFrame列表，转换为列表
            for result in results:
                result['features'] = result['features'][:10] + ['...'] if len(result['features']) > 10 else result['features']
            
            json.dump(results, f, indent=2)
        
        logger.info(f"特征子集评估结果已保存至: {eval_path}")
        
        # 绘制评估结果
        self._plot_subset_evaluation(results)
        
        return results
    
    def _plot_subset_evaluation(self, results: List[Dict[str, Any]]):
        """
        绘制特征子集评估结果
        
        参数:
            results: 评估结果列表
        """
        if not results:
            logger.warning("没有评估结果可绘制")
            return
        
        # 提取数据
        n_features = [r['n_features'] for r in results]
        train_scores = [r['train_score'] for r in results]
        test_scores = [r['test_score'] for r in results]
        gaps = [r['train_test_gap'] for r in results]
        
        # 绘制性能图
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(n_features, train_scores, 'o-', label='训练集分数')
        plt.plot(n_features, test_scores, 's-', label='测试集分数')
        plt.xlabel('特征数量')
        plt.ylabel('模型分数')
        plt.title('特征数量与模型性能关系')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制过拟合程度图
        plt.subplot(1, 2, 2)
        plt.plot(n_features, gaps, 'o-', color='red')
        plt.xlabel('特征数量')
        plt.ylabel('训练集-测试集分数差距')
        plt.title('特征数量与过拟合程度关系')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        fig_path = os.path.join(self.output_dir, "feature_subset_performance.png")
        plt.savefig(fig_path)
        plt.close()
        
        logger.info(f"特征子集评估图表已保存至: {fig_path}")
    
    def apply_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        应用特征选择到数据集
        
        参数:
            X: 输入特征数据
            
        返回:
            仅包含选中特征的DataFrame
        """
        if not self.selected_features:
            logger.error("没有选中的特征可应用")
            return None
        
        available_features = [f for f in self.selected_features if f in X.columns]
        
        if len(available_features) < len(self.selected_features):
            missing = set(self.selected_features) - set(available_features)
            logger.warning(f"数据中缺少{len(missing)}个选中的特征: {missing}")
        
        logger.info(f"应用特征选择，从{X.shape[1]}个特征中选择了{len(available_features)}个特征")
        return X[available_features]

def parse_args():
    """
    解析命令行参数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="基于特征重要性的特征选择")
    
    parser.add_argument("--model_type", type=str, default="xgboost",
                       help="模型类型 (目前支持 xgboost, lightgbm)")
    parser.add_argument("--model_path", type=str, default=None,
                       help="预训练模型路径 (可选)")
    parser.add_argument("--importance_threshold", type=float, default=0.01,
                       help="特征重要性阈值")
    parser.add_argument("--top_n", type=int, default=None,
                       help="选择前N个重要特征 (可选)")
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
    parser.add_argument("--train_model", action="store_true",
                       help="是否训练新模型获取特征重要性")
    parser.add_argument("--evaluate_subsets", action="store_true",
                       help="是否评估不同特征子集")
    
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
        top_n_features=args.top_n,
        output_dir=args.output_dir,
        plot_results=True
    )
    
    # 如果需要训练模型
    if args.train_model or args.model_path is None:
        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"训练集形状: {X_train.shape}, 验证集形状: {X_val.shape}")
        
        # 训练模型获取特征重要性
        selector.train_model(X_train, y_train, X_val, y_val)
    
    # 获取特征重要性
    importance_df = selector.get_feature_importance()
    
    if importance_df is not None:
        logger.info(f"Top-10 重要特征:\n{importance_df.head(10)[['feature', 'importance_normalized']]}")
    
    # 选择特征
    selected_features = selector.select_features(X)
    
    logger.info(f"选中了{len(selected_features)}个特征:\n{selected_features[:10] + ['...'] if len(selected_features) > 10 else selected_features}")
    
    # 评估不同特征子集
    if args.evaluate_subsets:
        selector.evaluate_feature_subsets(X, y)
    
    logger.info("基于特征重要性的特征选择完成")

if __name__ == "__main__":
    main() 