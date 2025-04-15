"""
特征重要性分析器
用于评估特征重要性并筛选最重要的特征
支持多种特征重要性评估方法
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression, 
    SelectFromModel, RFE, RFECV
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime

from src.utils.logger import get_logger
from src.models.model_factory import ModelFactory

logger = get_logger(__name__)

class FeatureImportanceAnalyzer:
    """
    特征重要性分析器
    用于评估特征重要性并筛选最重要的特征
    支持多种特征重要性评估方法
    """
    
    def __init__(
        self,
        output_dir: str = "outputs/feature_importance",
        importance_methods: List[str] = ["model", "mutual_info", "permutation"],
        model_type: str = "xgboost",
        n_estimators: int = 100,
        n_features_to_select: int = 20,
        correlation_threshold: float = 0.7,
        random_state: int = 42,
        n_jobs: int = -1,
        plot_results: bool = True
    ):
        """
        初始化特征重要性分析器
        
        参数:
            output_dir: 输出目录
            importance_methods: 重要性评估方法列表，支持 'model'、'mutual_info'、'f_regression'、'permutation'
            model_type: 模型类型，用于模型特征重要性评估，支持 'xgboost'、'lightgbm'、'random_forest'
            n_estimators: 树模型的估计器数量
            n_features_to_select: 要选择的特征数量
            correlation_threshold: 特征相关性阈值，超过此阈值的高相关特征将被合并考虑
            random_state: 随机种子
            n_jobs: 并行任务数
            plot_results: 是否绘制结果图表
        """
        self.output_dir = output_dir
        self.importance_methods = importance_methods
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.n_features_to_select = n_features_to_select
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.plot_results = plot_results
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化重要性评分和选择的特征
        self.importance_scores = {}
        self.selected_features = {}
        
        # 检查重要性方法
        for method in importance_methods:
            if method not in ["model", "mutual_info", "f_regression", "permutation", "rfe", "rfecv"]:
                logger.warning(f"不支持的特征重要性方法: {method}")
                
    def create_model(self) -> Any:
        """
        根据指定的模型类型创建模型
        
        返回:
            创建的模型实例
        """
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            logger.warning(f"不支持的模型类型: {self.model_type}，使用 RandomForestRegressor")
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            
    def calculate_feature_correlation(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        计算特征间的相关性矩阵
        
        参数:
            X: 特征数据
            
        返回:
            相关性矩阵
        """
        return X.corr(method='pearson')
    
    def get_correlated_features(self, X: pd.DataFrame) -> List[Set[str]]:
        """
        获取高度相关的特征组
        
        参数:
            X: 特征数据
            
        返回:
            高度相关的特征组列表
        """
        corr_matrix = self.calculate_feature_correlation(X)
        
        # 查找高度相关的特征
        correlated_features = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > self.correlation_threshold:
                    feature_i = corr_matrix.columns[i]
                    feature_j = corr_matrix.columns[j]
                    
                    # 尝试将相关特征添加到已有的组中
                    added = False
                    for group in correlated_features:
                        if feature_i in group or feature_j in group:
                            group.add(feature_i)
                            group.add(feature_j)
                            added = True
                            break
                    
                    # 如果没有添加到现有组，则创建新组
                    if not added:
                        correlated_features.append({feature_i, feature_j})
        
        # 合并包含相同特征的组
        merged_groups = []
        for group in correlated_features:
            merged = False
            for i, existing_group in enumerate(merged_groups):
                if len(group.intersection(existing_group)) > 0:
                    merged_groups[i] = existing_group.union(group)
                    merged = True
                    break
            if not merged:
                merged_groups.append(group)
                
        return merged_groups
    
    def calculate_model_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        使用模型的内置特征重要性计算特征重要性
        
        参数:
            X: 特征数据
            y: 目标变量
            
        返回:
            特征重要性评分
        """
        # 创建并训练模型
        model = self.create_model()
        model.fit(X, y)
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            logger.warning(f"模型 {self.model_type} 没有 feature_importances_ 属性，无法计算模型特征重要性")
            return pd.Series(0, index=X.columns)
            
        return pd.Series(importances, index=X.columns)
    
    def calculate_mutual_info_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        使用互信息计算特征重要性
        
        参数:
            X: 特征数据
            y: 目标变量
            
        返回:
            特征重要性评分
        """
        selector = SelectKBest(mutual_info_regression, k='all')
        selector.fit(X, y)
        
        return pd.Series(selector.scores_, index=X.columns)
    
    def calculate_f_regression_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        使用F值回归计算特征重要性
        
        参数:
            X: 特征数据
            y: 目标变量
            
        返回:
            特征重要性评分
        """
        selector = SelectKBest(f_regression, k='all')
        selector.fit(X, y)
        
        return pd.Series(selector.scores_, index=X.columns)
    
    def calculate_permutation_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        使用排列重要性计算特征重要性
        
        参数:
            X: 特征数据
            y: 目标变量
            
        返回:
            特征重要性评分
        """
        # 创建并训练模型
        model = self.create_model()
        model.fit(X, y)
        
        # 计算排列重要性
        result = permutation_importance(
            model, X, y, 
            n_repeats=10, 
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        return pd.Series(result.importances_mean, index=X.columns)
    
    def perform_rfe(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.Series, List[str]]:
        """
        使用递归特征消除计算特征重要性
        
        参数:
            X: 特征数据
            y: 目标变量
            
        返回:
            特征重要性评分和选择的特征列表
        """
        # 创建模型
        model = self.create_model()
        
        # 创建RFE选择器
        n_features = min(self.n_features_to_select, X.shape[1])
        rfe = RFE(
            estimator=model,
            n_features_to_select=n_features,
            step=1,
            verbose=0
        )
        
        # 训练RFE选择器
        rfe.fit(X, y)
        
        # 获取支持特征的掩码和排名
        support = rfe.support_
        ranking = rfe.ranking_
        
        # 计算重要性分数（根据排名的倒数，排名越低越重要）
        importance = np.zeros(X.shape[1])
        for i, rank in enumerate(ranking):
            importance[i] = 1.0 / rank if rank > 0 else 0
            
        # 获取选择的特征
        selected_features = list(X.columns[support])
        
        return pd.Series(importance, index=X.columns), selected_features
    
    def perform_rfecv(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.Series, List[str]]:
        """
        使用带交叉验证的递归特征消除计算特征重要性
        
        参数:
            X: 特征数据
            y: 目标变量
            
        返回:
            特征重要性评分和选择的特征列表
        """
        # 创建模型
        model = self.create_model()
        
        # 创建RFECV选择器
        rfecv = RFECV(
            estimator=model,
            step=1,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # 训练RFECV选择器
        rfecv.fit(X, y)
        
        # 获取支持特征的掩码和排名
        support = rfecv.support_
        ranking = rfecv.ranking_
        
        # 计算重要性分数（根据排名的倒数，排名越低越重要）
        importance = np.zeros(X.shape[1])
        for i, rank in enumerate(ranking):
            importance[i] = 1.0 / rank if rank > 0 else 0
            
        # 获取选择的特征
        selected_features = list(X.columns[support])
        
        # 记录最佳特征数量
        logger.info(f"RFECV 确定的最佳特征数量: {rfecv.n_features_}")
        
        return pd.Series(importance, index=X.columns), selected_features
    
    def analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, pd.Series]:
        """
        分析特征重要性
        
        参数:
            X: 特征数据
            y: 目标变量
            
        返回:
            不同方法计算的特征重要性评分字典
        """
        logger.info(f"开始分析特征重要性，数据形状: X={X.shape}, y={y.shape}")
        
        # 存储各种方法的特征重要性得分
        importance_scores = {}
        
        # 计算不同方法的特征重要性
        for method in self.importance_methods:
            logger.info(f"使用 {method} 方法计算特征重要性")
            
            if method == "model":
                scores = self.calculate_model_importance(X, y)
                importance_scores[method] = scores
                self.selected_features[method] = list(scores.nlargest(self.n_features_to_select).index)
                
            elif method == "mutual_info":
                scores = self.calculate_mutual_info_importance(X, y)
                importance_scores[method] = scores
                self.selected_features[method] = list(scores.nlargest(self.n_features_to_select).index)
                
            elif method == "f_regression":
                scores = self.calculate_f_regression_importance(X, y)
                importance_scores[method] = scores
                self.selected_features[method] = list(scores.nlargest(self.n_features_to_select).index)
                
            elif method == "permutation":
                scores = self.calculate_permutation_importance(X, y)
                importance_scores[method] = scores
                self.selected_features[method] = list(scores.nlargest(self.n_features_to_select).index)
                
            elif method == "rfe":
                scores, selected = self.perform_rfe(X, y)
                importance_scores[method] = scores
                self.selected_features[method] = selected
                
            elif method == "rfecv":
                scores, selected = self.perform_rfecv(X, y)
                importance_scores[method] = scores
                self.selected_features[method] = selected
                
        # 保存重要性评分
        self.importance_scores = importance_scores
        
        # 计算不同方法的一致性
        self.calculate_consistency()
        
        # 保存结果
        self.save_results()
        
        # 绘制结果
        if self.plot_results:
            self.plot_importance_scores()
            
        return importance_scores
    
    def calculate_consistency(self) -> Dict[str, float]:
        """
        计算不同特征重要性方法之间的一致性
        
        返回:
            方法对之间的一致性评分字典
        """
        if len(self.selected_features) <= 1:
            return {}
            
        consistency = {}
        methods = list(self.selected_features.keys())
        
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                method1 = methods[i]
                method2 = methods[j]
                
                # 计算两个方法选择的特征的交集占比
                features1 = set(self.selected_features[method1])
                features2 = set(self.selected_features[method2])
                
                intersection = features1.intersection(features2)
                union = features1.union(features2)
                
                # Jaccard相似度
                jaccard = len(intersection) / len(union) if len(union) > 0 else 0
                
                # 同时记录交集特征数量
                consistency[f"{method1}_vs_{method2}"] = {
                    "jaccard": jaccard,
                    "intersection_count": len(intersection),
                    "intersection_features": list(intersection)
                }
                
                logger.info(f"方法 {method1} 和 {method2} 的一致性: {jaccard:.4f} (交集: {len(intersection)})")
                
        return consistency
    
    def get_consensus_features(self, min_count: int = 2) -> List[str]:
        """
        获取在多种方法中都被选为重要的特征
        
        参数:
            min_count: 至少在多少种方法中被选为重要的阈值
            
        返回:
            共识特征列表
        """
        if not self.selected_features:
            return []
            
        # 计算每个特征被选为重要的次数
        feature_count = {}
        for method, features in self.selected_features.items():
            for feature in features:
                if feature in feature_count:
                    feature_count[feature] += 1
                else:
                    feature_count[feature] = 1
                    
        # 获取出现次数超过阈值的特征
        consensus_features = [
            feature for feature, count in feature_count.items()
            if count >= min_count
        ]
        
        # 按计数排序
        consensus_features.sort(key=lambda x: feature_count[x], reverse=True)
        
        logger.info(f"找到 {len(consensus_features)} 个共识特征 (min_count={min_count})")
        return consensus_features
    
    def plot_importance_scores(self):
        """
        绘制特征重要性评分图表
        """
        if not self.importance_scores:
            logger.warning("没有可用的特征重要性评分，无法绘图")
            return
            
        # 创建结果目录
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 对于每种方法，绘制一个特征重要性条形图
        for method, scores in self.importance_scores.items():
            # 对特征重要性进行排序
            sorted_scores = scores.sort_values(ascending=False)
            
            # 只展示前N个特征（太多特征会导致图表混乱）
            top_n = min(30, len(sorted_scores))
            top_scores = sorted_scores.iloc[:top_n]
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x=top_scores.values, y=top_scores.index)
            plt.title(f'特征重要性 ({method})', fontsize=16)
            plt.xlabel('重要性评分', fontsize=12)
            plt.ylabel('特征名称', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"feature_importance_{method}_{timestamp}.png"))
            plt.close()
            
        # 绘制不同方法选择的特征的比较热力图
        if len(self.selected_features) > 1:
            # 获取所有方法选择的所有特征的并集
            all_features = set()
            for features in self.selected_features.values():
                all_features.update(features)
                
            # 创建比较矩阵
            methods = list(self.selected_features.keys())
            comparison_data = []
            
            for feature in all_features:
                row = [1 if feature in self.selected_features[method] else 0 for method in methods]
                comparison_data.append(row)
                
            comparison_df = pd.DataFrame(comparison_data, index=list(all_features), columns=methods)
            
            # 绘制热力图
            plt.figure(figsize=(12, max(8, len(all_features) * 0.3)))
            sns.heatmap(comparison_df, cmap='Blues', cbar=False, linewidths=0.5)
            plt.title('不同方法选择的特征比较', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"feature_selection_comparison_{timestamp}.png"))
            plt.close()
            
        # 如果有共识特征，绘制共识特征的条形图
        consensus_features = self.get_consensus_features(min_count=2)
        if consensus_features:
            # 计算每个共识特征的平均重要性
            avg_importance = {}
            for feature in consensus_features:
                scores = [
                    self.importance_scores[method][feature]
                    for method in self.importance_scores
                    if feature in self.importance_scores[method].index
                ]
                avg_importance[feature] = np.mean(scores)
                
            # 对平均重要性进行排序
            sorted_importance = {k: v for k, v in sorted(avg_importance.items(), key=lambda item: item[1], reverse=True)}
            
            # 绘制共识特征条形图
            plt.figure(figsize=(12, 8))
            sns.barplot(x=list(sorted_importance.values()), y=list(sorted_importance.keys()))
            plt.title('共识特征平均重要性', fontsize=16)
            plt.xlabel('平均重要性评分', fontsize=12)
            plt.ylabel('特征名称', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"consensus_features_{timestamp}.png"))
            plt.close()
    
    def save_results(self):
        """
        保存特征重要性分析结果
        """
        if not self.importance_scores:
            logger.warning("没有可用的特征重要性评分，无法保存结果")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存特征重要性评分
        for method, scores in self.importance_scores.items():
            # 保存为CSV
            scores_df = pd.DataFrame({'feature': scores.index, 'importance': scores.values})
            scores_df = scores_df.sort_values('importance', ascending=False)
            scores_path = os.path.join(self.output_dir, f"importance_scores_{method}_{timestamp}.csv")
            scores_df.to_csv(scores_path, index=False)
            
            logger.info(f"已保存 {method} 方法的特征重要性评分: {scores_path}")
            
        # 保存选择的特征
        selected_features = {}
        for method, features in self.selected_features.items():
            selected_features[method] = features
            
        # 保存为JSON
        selected_path = os.path.join(self.output_dir, f"selected_features_{timestamp}.json")
        with open(selected_path, 'w') as f:
            json.dump(selected_features, f, indent=2)
            
        logger.info(f"已保存选择的特征: {selected_path}")
        
        # 保存共识特征
        consensus_features = {
            'min_count_2': self.get_consensus_features(min_count=2),
            'min_count_3': self.get_consensus_features(min_count=3) if len(self.selected_features) >= 3 else []
        }
        
        # 保存为JSON
        consensus_path = os.path.join(self.output_dir, f"consensus_features_{timestamp}.json")
        with open(consensus_path, 'w') as f:
            json.dump(consensus_features, f, indent=2)
            
        logger.info(f"已保存共识特征: {consensus_path}")
        
        # 保存所有结果的摘要
        summary = {
            'model_type': self.model_type,
            'importance_methods': self.importance_methods,
            'n_features_to_select': self.n_features_to_select,
            'correlation_threshold': self.correlation_threshold,
            'selected_features': selected_features,
            'consensus_features': consensus_features,
            'timestamp': timestamp
        }
        
        # 保存为JSON
        summary_path = os.path.join(self.output_dir, f"importance_analysis_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"已保存分析摘要: {summary_path}")
    
    def select_features(self, X: pd.DataFrame, method: str = 'consensus', min_count: int = 2) -> List[str]:
        """
        选择特征
        
        参数:
            X: 特征数据
            method: 特征选择方法，可以是 importance_methods 中的一种，也可以是 'consensus'
            min_count: 当 method='consensus' 时，至少在多少种方法中被选为重要的阈值
            
        返回:
            选择的特征列表
        """
        if not self.importance_scores:
            logger.warning("尚未分析特征重要性，无法选择特征")
            return []
            
        if method == 'consensus':
            return self.get_consensus_features(min_count=min_count)
        elif method in self.selected_features:
            return self.selected_features[method]
        else:
            logger.warning(f"不支持的特征选择方法: {method}")
            return []
            
def parse_args():
    """
    解析命令行参数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="分析特征重要性并选择最重要的特征")
    
    parser.add_argument("--output_dir", type=str, default="outputs/feature_importance",
                      help="输出目录")
    parser.add_argument("--importance_methods", type=str, nargs='+', 
                      default=["model", "mutual_info", "permutation"],
                      help="特征重要性评估方法")
    parser.add_argument("--model_type", type=str, default="xgboost",
                      help="模型类型 (xgboost, lightgbm, random_forest)")
    parser.add_argument("--n_estimators", type=int, default=100,
                      help="树模型的估计器数量")
    parser.add_argument("--n_features", type=int, default=20,
                      help="要选择的特征数量")
    parser.add_argument("--correlation_threshold", type=float, default=0.7,
                      help="特征相关性阈值")
    parser.add_argument("--random_state", type=int, default=42,
                      help="随机种子")
    parser.add_argument("--n_jobs", type=int, default=-1,
                      help="并行任务数")
    parser.add_argument("--no_plot", action="store_true",
                      help="不绘制结果图表")
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
    parser.add_argument("--features", type=str, nargs='+',
                      help="使用的特征列表，如果不指定则使用所有可用特征")
    parser.add_argument("--selection_method", type=str, default="consensus",
                      help="最终特征选择方法 (consensus, model, mutual_info, permutation, rfe, rfecv)")
    parser.add_argument("--min_count", type=int, default=2,
                      help="使用consensus方法时，至少在多少种方法中被选为重要的阈值")
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 设置日志
    logger.info("开始分析特征重要性")
    
    # 创建特征重要性分析器
    analyzer = FeatureImportanceAnalyzer(
        output_dir=args.output_dir,
        importance_methods=args.importance_methods,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        n_features_to_select=args.n_features,
        correlation_threshold=args.correlation_threshold,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        plot_results=not args.no_plot
    )
    
    # 加载数据
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
    
    if X_train is None or y_train is None:
        logger.error("加载数据失败")
        return
    
    # 如果指定了特定特征，则仅使用这些特征
    if args.features:
        # 检查指定的特征是否存在于数据中
        missing_features = [f for f in args.features if f not in X_train.columns]
        if missing_features:
            logger.warning(f"以下特征不存在于数据中: {missing_features}")
            args.features = [f for f in args.features if f in X_train.columns]
            
        if not args.features:
            logger.error("没有可用的特征")
            return
            
        X_train = X_train[args.features]
        if X_test is not None:
            X_test = X_test[args.features]
    
    logger.info(f"加载数据完成，训练集形状: X={X_train.shape}, y={y_train.shape}")
    if X_test is not None:
        logger.info(f"测试集形状: X={X_test.shape}, y={y_test.shape}")
    
    # 分析特征重要性
    analyzer.analyze_feature_importance(X_train, y_train)
    
    # 选择特征
    selected_features = analyzer.select_features(
        X_train, 
        method=args.selection_method,
        min_count=args.min_count
    )
    
    if selected_features:
        logger.info(f"选择了 {len(selected_features)} 个特征: {selected_features}")
        
        # 保存选择的特征
        selection_path = os.path.join(args.output_dir, f"final_selected_features_{args.selection_method}.json")
        with open(selection_path, 'w') as f:
            json.dump(selected_features, f, indent=2)
            
        logger.info(f"已保存最终选择的特征: {selection_path}")
    else:
        logger.warning("没有选择任何特征")
    
    logger.info("特征重要性分析完成")

if __name__ == "__main__":
    main() 