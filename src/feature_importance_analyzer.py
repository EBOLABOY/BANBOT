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
from src.models.model_training import prepare_train_test_data

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
    parser.add_argument("--features", type=str, nargs='+',
                      help="使用的特征列表，如果不指定则使用所有可用特征")
    parser.add_argument("--selection_method", type=str, default="consensus",
                      help="最终特征选择方法 (consensus, model, mutual_info, permutation, rfe, rfecv)")
    parser.add_argument("--min_count", type=int, default=2,
                      help="使用consensus方法时，至少在多少种方法中被选为重要的阈值")
    parser.add_argument("--feature_file", type=str, default=None, help="Path to the feature file (e.g., features_1m.parquet)")
    parser.add_argument("--target_file", type=str, default=None, help="Path to the target file (e.g., targets_1m.parquet)")
    parser.add_argument("--target_column", type=str, default=None, help="Name of the target column in the target file")

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
    
    # Determine file paths
    if args.feature_file is None:
        # Construct path based on convention if not provided
        # Use args.symbols and args.timeframe if needed for path, but ensure they are parsed correctly
        if args.symbols and args.timeframe:
             # Example convention: data/processed/BTCUSDT/features_1m.parquet
             # Adjust this logic based on your actual file structure if different
             # For simplicity, let's assume a flat structure in data_dir for now
             args.feature_file = os.path.join(args.data_dir, f"features_{args.timeframe}.parquet")
        else:
             logger.error("Feature file path not specified and cannot be inferred without symbol and timeframe.")
             return # Or raise error

    if args.target_file is None:
        if args.symbols and args.timeframe:
             args.target_file = os.path.join(args.data_dir, f"targets_{args.timeframe}.parquet")
        else:
             logger.error("Target file path not specified and cannot be inferred without symbol and timeframe.")
             return # Or raise error


    logger.info(f"加载特征文件: {args.feature_file}")
    logger.info(f"加载目标文件: {args.target_file}")

    try:
        X = pd.read_parquet(args.feature_file)
        y_df = pd.read_parquet(args.target_file)

        # Ensure index is DatetimeIndex for alignment and splitting
        if not isinstance(X.index, pd.DatetimeIndex):
            try:
                X.index = pd.to_datetime(X.index)
            except Exception as e:
                logger.error(f"无法将特征文件索引转换为 DatetimeIndex: {e}")
                return
        if not isinstance(y_df.index, pd.DatetimeIndex):
             try:
                 y_df.index = pd.to_datetime(y_df.index)
             except Exception as e:
                 logger.error(f"无法将目标文件索引转换为 DatetimeIndex: {e}")
                 return

        # Infer or use specified target column
        if args.target_column:
            target_col = args.target_column
        elif len(y_df.columns) == 1:
            target_col = y_df.columns[0]
        else:
            # Try to find a column starting with 'target_'
            target_cols = [col for col in y_df.columns if col.startswith('target_')]
            if not target_cols:
                 # Try finding columns based on prediction horizon from args if available
                 # This part might need adjustment based on actual target naming convention
                 # For now, raise error if ambiguous
                 raise ValueError(f"在目标文件 {args.target_file} 中找不到目标列，请使用 --target_column 指定")

            target_col = target_cols[0] # Assume first target column found
            logger.info(f"自动检测到目标列: {target_col}")

        if target_col not in y_df.columns:
             raise ValueError(f"目标列 '{target_col}' 在文件 {args.target_file} 中不存在")

        y = y_df[target_col]

        # Align data - crucial step
        X, y = X.align(y, join='inner', axis=0)

        # Handle potential NaNs introduced by alignment or previous steps
        initial_len = len(X)
        # Check for infinite values as well
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)

        valid_mask = ~y.isna() & ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        dropped_count = initial_len - len(X)
        if dropped_count > 0:
            logger.warning(f"因对齐或 NaN 值移除了 {dropped_count} 条记录")


        if X.empty or y.empty:
             raise ValueError("对齐或清理 NaN 后数据为空")

    except FileNotFoundError as e:
        logger.error(f"找不到数据文件: {e}")
        return
    except Exception as e:
        logger.error(f"加载或处理数据时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Prepare train/test data using the imported function
    # Note: Test set might not be strictly needed for importance analysis itself
    # Validation set is useful for permutation importance / RFE with early stopping
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_train_test_data(
        X, y,
        test_size=0.2,       # Default test size, consider making configurable
        validation_size=0.1, # Default validation size, consider making configurable
        time_series_split=True # Default to time series split
    )

    if X_train is None or y_train is None:
        logger.error("数据分割失败，训练集为空")
        return

    # --- Added: Select only numeric columns for model training ---
    numeric_cols_train = X_train.select_dtypes(include=np.number).columns.tolist()
    original_cols_train = X_train.columns.tolist()
    non_numeric_cols_train = [col for col in original_cols_train if col not in numeric_cols_train]

    if non_numeric_cols_train:
        logger.warning(f"从训练特征集中移除以下非数值列: {non_numeric_cols_train}")
        X_train = X_train[numeric_cols_train]

    if X_val is not None:
        numeric_cols_val = X_val.select_dtypes(include=np.number).columns.tolist()
        original_cols_val = X_val.columns.tolist()
        non_numeric_cols_val = [col for col in original_cols_val if col not in numeric_cols_val]
        if non_numeric_cols_val:
             logger.warning(f"从验证特征集中移除以下非数值列: {non_numeric_cols_val}")
             X_val = X_val[numeric_cols_val]
             # Re-align val features in case user specified a subset via --features
             if args.features:
                 # Use the already filtered args.features list
                 val_available_cols = [f for f in args.features if f in X_val.columns]
                 X_val = X_val[val_available_cols]

    if X_train.empty:
         logger.error("在移除非数值列后，训练特征集为空。")
         return
    # --- End of Added Section ---

    # 如果指定了特定特征，则仅使用这些特征
    if args.features:
        # Check if specified features exist in the data
        available_cols = X_train.columns.tolist()
        missing_features = [f for f in args.features if f not in available_cols]
        if missing_features:
            logger.warning(f"以下特征不存在于数据中: {missing_features}")
            # Filter the list of features to use only available ones
            args.features = [f for f in args.features if f in available_cols]

        if not args.features:
            logger.error("指定的特征列表为空或所有特征均不可用")
            return

        logger.info(f"仅使用指定的 {len(args.features)} 个特征进行分析")
        X_train = X_train[args.features]
        # Apply feature selection to validation set as well if it exists
        if X_val is not None:
             # Ensure validation set columns exist before indexing
             val_available_cols = [f for f in args.features if f in X_val.columns]
             if len(val_available_cols) != len(args.features):
                  missing_in_val = set(args.features) - set(val_available_cols)
                  logger.warning(f"以下指定特征在验证集中缺失: {list(missing_in_val)}")
             if not val_available_cols:
                  logger.error("验证集中没有可用的指定特征")
                  X_val = None # Or handle differently
             else:
                  X_val = X_val[val_available_cols]


    logger.info(f"数据加载和分割完成，训练集形状: X={X_train.shape}, y={y_train.shape}")
    if X_val is not None:
        logger.info(f"验证集形状: X={X_val.shape}, y={y_val.shape}")
    if X_test is not None:
        logger.info(f"测试集形状: X={X_test.shape}, y={y_test.shape}")


    # 分析特征重要性
    # Pass X_train, y_train. Validation data might be used internally by some methods.
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
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        selection_path = os.path.join(args.output_dir, f"final_selected_features_{args.selection_method}.json")
        try:
            with open(selection_path, 'w') as f:
                json.dump(selected_features, f, indent=2)
            logger.info(f"已保存最终选择的特征: {selection_path}")
        except Exception as e:
            logger.error(f"保存选择的特征时出错: {e}")
    else:
        logger.warning("没有选择任何特征")

    logger.info("特征重要性分析完成")

if __name__ == "__main__":
    main() 