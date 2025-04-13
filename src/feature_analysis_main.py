"""
特征分析模块 - 用于分析特征重要性、相关性等
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
from datetime import datetime

# 导入项目内部模块
from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config
from src.features.feature_engineering import FeatureEngineer

# 设置日志记录器
logger = get_logger(__name__)

class FeatureAnalyzer:
    """
    特征分析类，提供各种特征分析方法
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        初始化特征分析器
        
        参数:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.results_dir = self.config.get("paths", {}).get("results_dir", "results")
        self.feature_dir = self.config.get("paths", {}).get("feature_dir", "data/processed/features")
        
        # 确保目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 特征工程器 - 用于加载数据
        self.feature_engineer = FeatureEngineer(config_path)
        
        logger.info("特征分析器已初始化")
    
    def load_data(self, symbol, timeframe, start_date=None, end_date=None):
        """
        加载特征数据
        
        参数:
            symbol: 交易对名称
            timeframe: 时间周期
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            特征数据和目标变量
        """
        # 构建正确的文件路径 - 查找feature目录下的symbol子目录
        features_path = os.path.join(self.feature_dir, symbol, f"features_{timeframe}.csv")
        targets_path = os.path.join(self.feature_dir, symbol, f"targets_{timeframe}.csv")
        
        # 检查文件是否存在
        if not os.path.exists(features_path) or not os.path.exists(targets_path):
            # 尝试从processed目录加载数据作为备用方案
            data_path = os.path.join("data/processed", f"processed_{symbol}_{timeframe}_20220101.csv")
            
            if not os.path.exists(data_path):
                logger.error(f"特征文件不存在: {features_path} 或 {targets_path}")
                logger.error(f"处理后的数据文件也不存在: {data_path}")
                return None, None
            
            # 从处理后的数据加载
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            
            # 分离特征和目标变量
            # 假设所有'price_change_'开头的列都是目标变量
            target_cols = [col for col in df.columns if col.startswith('price_change_')]
            if not target_cols:
                # 如果没有目标变量列，为后续分析创建一些基本的目标变量
                df['price_change_1h'] = df['close'].pct_change(periods=1)
                df['price_change_4h'] = df['close'].pct_change(periods=4)
                df['price_change_1d'] = df['close'].pct_change(periods=24)
                target_cols = ['price_change_1h', 'price_change_4h', 'price_change_1d']
                
            # 分离特征和目标
            targets_df = df[target_cols]
            
            # 特征数据 - 排除OHLCV基本列和目标列
            basic_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            exclude_cols = basic_cols + target_cols
            
            # 确保有特征列被识别 - 如果所有列都在exclude_cols中，则使用所有数值列作为特征
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            if not feature_cols:
                logger.warning("没有识别到特征列，将使用所有数值列（除了目标列）作为特征")
                feature_cols = [col for col in df.select_dtypes(include=['number']).columns 
                              if col not in target_cols and col not in ['timestamp']]
                
                # 如果仍然没有特征列，使用OHLCV列作为特征
                if not feature_cols:
                    logger.warning("没有识别到数值特征列，将使用OHLCV基本列作为特征")
                    feature_cols = [col for col in basic_cols if col != 'timestamp' and col in df.columns]
            
            features_df = df[feature_cols]
        else:
            # 从特征目录加载
            features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
            targets_df = pd.read_csv(targets_path, index_col=0, parse_dates=True)
        
        # 筛选日期范围
        if start_date:
            features_df = features_df[features_df.index >= pd.to_datetime(start_date)]
            targets_df = targets_df[targets_df.index >= pd.to_datetime(start_date)]
        
        if end_date:
            features_df = features_df[features_df.index <= pd.to_datetime(end_date)]
            targets_df = targets_df[targets_df.index <= pd.to_datetime(end_date)]
        
        logger.info(f"已加载 {symbol} 的 {timeframe} 数据，共 {len(features_df)} 条记录，{len(features_df.columns)} 个特征和 {len(targets_df.columns)} 个目标变量")
        return features_df, targets_df
    
    def calculate_feature_importance(self, features_df, targets_df, target_col='target_pct_60', 
                                    method='random_forest', n_estimators=100, top_n=30):
        """
        计算特征重要性
        
        参数:
            features_df: 特征数据
            targets_df: 目标变量
            target_col: 目标变量列名
            method: 特征重要性计算方法，可选 'random_forest', 'mutual_info'
            n_estimators: 随机森林的树数量
            top_n: 返回最重要的前N个特征
            
        返回:
            特征重要性DataFrame，如果计算失败则返回None
        """
        if features_df is None or targets_df is None:
            logger.error("特征数据或目标变量为空，无法计算特征重要性")
            return None
        
        # 确保目标变量在目标数据中
        if target_col not in targets_df.columns:
            logger.error(f"目标变量 {target_col} 不在目标数据中")
            # 列出可用的目标变量
            available_targets = targets_df.columns.tolist()
            if available_targets:
                logger.info(f"可用的目标变量有: {', '.join(available_targets[:10])}" + 
                           (f" 等共{len(available_targets)}个" if len(available_targets) > 10 else ""))
            return None
        
        try:
            # 准备数据
            X = features_df.copy()
            y = targets_df[target_col].copy()
            
            # 删除缺失值
            valid_indices = ~np.isnan(y)
            X = X[valid_indices]
            y = y[valid_indices]
            
            if len(y) == 0:
                logger.error(f"目标变量 {target_col} 全为NaN值，无法计算特征重要性")
                return None
                
            # 列名记录
            feature_names = X.columns.tolist()
            
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            importance_scores = {}
            
            if method == 'random_forest':
                # 使用随机森林计算特征重要性
                logger.info(f"使用随机森林方法计算特征重要性，目标变量: {target_col}")
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                model.fit(X_scaled, y)
                
                # 获取特征重要性
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
                
            elif method == 'mutual_info':
                # 使用互信息计算特征重要性
                logger.info(f"使用互信息方法计算特征重要性，目标变量: {target_col}")
                importances = mutual_info_regression(X_scaled, y)
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
            
            else:
                logger.error(f"不支持的特征重要性计算方法: {method}")
                return None
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # 仅保留前N个特征
            if top_n and len(importance_df) > top_n:
                importance_df = importance_df.head(top_n)
            
            logger.info(f"已计算 {method} 方法下的特征重要性，最重要的特征是 {importance_df['feature'].iloc[0]}")
            return importance_df
            
        except Exception as e:
            logger.exception(f"计算特征重要性时发生错误: {str(e)}")
            return None
    
    def calculate_feature_correlation(self, features_df, method='pearson', threshold=0.7):
        """
        计算特征相关性
        
        参数:
            features_df: 特征数据
            method: 相关性计算方法，可选 'pearson', 'spearman', 'kendall'
            threshold: 高相关性阈值
            
        返回:
            相关性矩阵和高相关性特征对
        """
        if features_df is None or features_df.empty:
            logger.error("特征数据为空，无法计算相关性")
            return None, None
        
        # 计算相关性矩阵
        corr_matrix = features_df.corr(method=method)
        
        # 找出高度相关的特征对
        high_corr_pairs = []
        
        # 获取上三角矩阵的索引
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        
        # 获取高相关性特征对
        high_corr = corr_matrix.where(upper_tri)
        high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], high_corr.iloc[i, j])
                          for i, j in zip(*np.where(np.abs(high_corr) >= threshold))]
        
        # 将高相关性特征对转换为DataFrame
        high_corr_df = pd.DataFrame(high_corr_pairs, columns=['feature1', 'feature2', 'correlation'])
        high_corr_df = high_corr_df.sort_values('correlation', ascending=False)
        
        logger.info(f"已计算特征相关性，发现 {len(high_corr_df)} 对高相关性特征对 (阈值 {threshold})")
        return corr_matrix, high_corr_df
    
    def analyze_feature_distributions(self, features_df, top_n=20):
        """
        分析特征分布
        
        参数:
            features_df: 特征数据
            top_n: 分析前N个特征
            
        返回:
            特征统计信息
        """
        if features_df is None or features_df.empty:
            logger.error("特征数据为空，无法分析分布")
            return None
        
        # 计算特征的基本统计量
        stats_df = features_df.describe().T
        
        # 添加其他统计量
        stats_df['skew'] = features_df.skew()
        stats_df['kurtosis'] = features_df.kurtosis()
        stats_df['missing'] = features_df.isnull().sum()
        stats_df['missing_percent'] = (features_df.isnull().sum() / len(features_df)) * 100
        
        # 按缺失值百分比排序
        stats_df = stats_df.sort_values('missing_percent', ascending=False)
        
        logger.info(f"已分析特征分布，平均缺失值比例为 {stats_df['missing_percent'].mean():.2f}%")
        return stats_df
    
    def select_best_features(self, importance_df, corr_matrix, max_features=50, corr_threshold=0.7):
        """
        选择最佳特征集
        
        参数:
            importance_df: 特征重要性DataFrame
            corr_matrix: 相关性矩阵
            max_features: 最大特征数量
            corr_threshold: 相关性阈值
            
        返回:
            最佳特征列表
        """
        if importance_df is None or corr_matrix is None:
            logger.error("特征重要性或相关性矩阵为空，无法选择最佳特征")
            return None
        
        # 按重要性排序的特征列表
        sorted_features = importance_df['feature'].tolist()
        
        # 选择的特征集
        selected_features = []
        
        # 贪婪特征选择
        for feature in sorted_features:
            # 如果已经选择了足够多的特征，停止
            if len(selected_features) >= max_features:
                break
                
            # 检查是否与已选特征高度相关
            if not selected_features:
                # 第一个特征直接选择
                selected_features.append(feature)
            else:
                # 检查与已选特征的相关性
                is_correlated = False
                for selected in selected_features:
                    if abs(corr_matrix.loc[feature, selected]) > corr_threshold:
                        is_correlated = True
                        break
                
                # 如果不与任何已选特征高度相关，则选择该特征
                if not is_correlated:
                    selected_features.append(feature)
        
        logger.info(f"已选择 {len(selected_features)} 个最佳特征")
        return selected_features
    
    def visualize_feature_importance(self, importance_df, output_path=None, top_n=20):
        """
        可视化特征重要性
        
        参数:
            importance_df: 特征重要性DataFrame
            output_path: 输出路径
            top_n: 展示前N个特征
        """
        if importance_df is None or importance_df.empty:
            logger.error("特征重要性数据为空，无法可视化")
            return
        
        # 限制展示的特征数量
        if top_n and len(importance_df) > top_n:
            df_plot = importance_df.head(top_n)
        else:
            df_plot = importance_df.copy()
        
        # 设置图表样式和大小
        plt.figure(figsize=(12, 8))
        sns.set(style="whitegrid")
        
        # 绘制条形图
        plot = sns.barplot(x='importance', y='feature', data=df_plot)
        
        # 添加标题和标签
        plt.title('特征重要性分析', fontsize=16)
        plt.xlabel('重要性', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        
        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性图表已保存至 {output_path}")
        else:
            # 保存到默认位置
            save_path = os.path.join(self.results_dir, "feature_importance.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性图表已保存至 {save_path}")
        
        plt.close()
    
    def visualize_correlation_matrix(self, corr_matrix, output_path=None, top_n=30):
        """
        可视化相关性矩阵
        
        参数:
            corr_matrix: 相关性矩阵
            output_path: 输出路径
            top_n: 展示前N个特征
        """
        if corr_matrix is None or corr_matrix.empty:
            logger.error("相关性矩阵为空，无法可视化")
            return
        
        # 限制展示的特征数量
        if top_n and len(corr_matrix) > top_n:
            # 获取对角元素方差最大的top_n个特征
            vars_diag = np.diag(corr_matrix)
            top_indices = np.argsort(vars_diag)[-top_n:]
            corr_plot = corr_matrix.iloc[top_indices, top_indices]
        else:
            corr_plot = corr_matrix.copy()
        
        # 设置图表大小
        plt.figure(figsize=(15, 12))
        
        # 绘制热图
        mask = np.triu(np.ones_like(corr_plot, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        sns.heatmap(corr_plot, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
        
        # 添加标题
        plt.title('特征相关性矩阵', fontsize=16)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"相关性矩阵图表已保存至 {output_path}")
        else:
            # 保存到默认位置
            save_path = os.path.join(self.results_dir, "correlation_matrix.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"相关性矩阵图表已保存至 {save_path}")
        
        plt.close()
    
    def run_analysis(self, symbol, timeframe, target_col='target_pct_60', 
                    start_date=None, end_date=None, importance_method='random_forest', 
                    corr_method='pearson', top_n=30, corr_threshold=0.7, max_features=50):
        """
        运行特征分析
        
        参数:
            symbol: 交易对
            timeframe: 时间周期
            target_col: 目标变量
            start_date: 开始日期
            end_date: 结束日期
            importance_method: 特征重要性计算方法
            corr_method: 相关性计算方法
            top_n: 展示前N个特征
            corr_threshold: 相关性阈值
            max_features: 最大特征数量
        """
        logger.info(f"开始分析 {symbol} 的 {timeframe} 数据，目标变量: {target_col}")
        
        # 加载特征和目标数据
        features_df, targets_df = self.load_data(symbol, timeframe, start_date, end_date)
        if features_df is None or targets_df is None:
            logger.error(f"无法加载 {symbol} 的 {timeframe} 数据")
            return
        
        logger.info(f"已加载 {symbol} 的 {timeframe} 数据，共 {len(features_df)} 条记录，"
                   f"{features_df.shape[1]} 个特征和 {targets_df.shape[1]} 个目标变量")
        
        # 如果指定的目标变量不存在，尝试找一个替代的目标变量
        if target_col not in targets_df.columns:
            logger.warning(f"指定的目标变量 {target_col} 不存在，尝试查找合适的替代变量")
            
            # 优先考虑target_pct开头的变量
            target_pct_cols = [col for col in targets_df.columns if 'target_pct_' in col]
            if target_pct_cols:
                target_col = target_pct_cols[0]
                logger.info(f"已选择 {target_col} 作为替代目标变量")
            # 其次考虑target_direction开头的变量    
            elif any(col.startswith('target_direction_') for col in targets_df.columns):
                target_col = next(col for col in targets_df.columns if col.startswith('target_direction_'))
                logger.info(f"已选择 {target_col} 作为替代目标变量")
            # 然后考虑price_change开头的变量
            elif any(col.startswith('price_change_') for col in targets_df.columns):
                target_col = next(col for col in targets_df.columns if col.startswith('price_change_'))
                logger.info(f"已选择 {target_col} 作为替代目标变量")
            # 最后，使用任何可用的目标变量
            elif len(targets_df.columns) > 0:
                target_col = targets_df.columns[0]
                logger.info(f"已选择 {target_col} 作为替代目标变量")
            else:
                logger.error("无法找到合适的目标变量，分析终止")
                return
        
        # 数据统计分析
        stats_df = self.analyze_feature_distributions(features_df, top_n=top_n)
        
        # 特征重要性分析
        importance_df = self.calculate_feature_importance(
            features_df, targets_df, target_col, method=importance_method, top_n=top_n)
        
        # 如果特征重要性计算失败，可能是目标变量不存在，尝试其他可用的目标变量
        if importance_df is None and targets_df is not None and not targets_df.empty:
            logger.warning(f"使用目标变量 {target_col} 计算特征重要性失败，尝试使用其他可用目标变量")
            # 查找所有可用的目标变量
            available_targets = targets_df.columns.tolist()
            for alt_target in available_targets:
                logger.info(f"尝试使用替代目标变量: {alt_target}")
                importance_df = self.calculate_feature_importance(
                    features_df, targets_df, alt_target, method=importance_method, top_n=top_n)
                if importance_df is not None:
                    target_col = alt_target
                    logger.info(f"成功使用替代目标变量 {target_col} 计算特征重要性")
                    break
        
        # 特征相关性分析
        corr_matrix, high_corr_df = self.calculate_feature_correlation(
            features_df, method=corr_method, threshold=corr_threshold)
        
        # 目标相关性分析
        target_corr_df = None
        if importance_df is not None:
            # 获取最重要的特征
            top_features = importance_df['feature'].head(10).tolist()
            
            # 计算这些特征与目标的相关性
            target_corr_df = pd.DataFrame()
            if top_features:
                # 计算相关性
                corr_data = []  # 存储相关性数据的列表
                for feature in top_features:
                    if feature in features_df.columns:
                        corr = features_df[feature].corr(targets_df[target_col])
                        corr_data.append({
                            'feature': feature,
                            'correlation': corr
                        })
                
                # 创建DataFrame
                if corr_data:
                    target_corr_df = pd.DataFrame(corr_data)
                
                # 按相关性排序
                if not target_corr_df.empty:
                    target_corr_df = target_corr_df.sort_values('correlation', ascending=False)
        
        # 创建一个结果目录
        result_dir = os.path.join(self.results_dir, f"feature_analysis_{symbol}_{timeframe}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 可视化
        if importance_df is not None:
            # 保存特征重要性数据到CSV
            importance_csv = os.path.join(result_dir, f"feature_importance_{symbol}_{timeframe}.csv")
            importance_df.to_csv(importance_csv)
            logger.info(f"特征重要性数据已保存至 {importance_csv}")
            
            # 可视化特征重要性
            importance_path = os.path.join(result_dir, f"feature_importance_{symbol}_{timeframe}.png")
            self.visualize_feature_importance(importance_df, importance_path, top_n=top_n)
        else:
            logger.warning("特征重要性数据为空，跳过特征重要性图表生成")
        
        if corr_matrix is not None:
            # 可视化相关性矩阵
            corr_path = os.path.join(result_dir, f"correlation_matrix_{symbol}_{timeframe}.png")
            self.visualize_correlation_matrix(corr_matrix, corr_path, top_n=top_n)
        else:
            logger.warning("相关性矩阵为空，跳过相关性图表生成")
        
        # 保存统计数据到CSV
        if stats_df is not None:
            stats_csv = os.path.join(result_dir, f"feature_stats_{symbol}_{timeframe}.csv")
            stats_df.to_csv(stats_csv)
            logger.info(f"特征统计数据已保存至 {stats_csv}")
        
        # 保存高相关性对到CSV
        if high_corr_df is not None and not high_corr_df.empty:
            high_corr_csv = os.path.join(result_dir, f"high_correlation_pairs_{symbol}_{timeframe}.csv")
            high_corr_df.to_csv(high_corr_csv)
            logger.info(f"高相关性对数据已保存至 {high_corr_csv}")
        
        logger.info(f"{symbol} {timeframe} 的特征分析已完成")
        return result_dir

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='特征分析工具')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='交易对名称')
    
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='时间周期')
    
    parser.add_argument('--target', type=str, default='target_pct_60',
                        help='目标变量列名')
    
    parser.add_argument('--start-date', type=str, default=None,
                        help='开始日期 (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                        help='结束日期 (YYYY-MM-DD)')
    
    parser.add_argument('--importance-method', type=str, default='random_forest',
                        choices=['random_forest', 'mutual_info'],
                        help='特征重要性计算方法')
    
    parser.add_argument('--corr-method', type=str, default='pearson',
                        choices=['pearson', 'spearman', 'kendall'],
                        help='相关性计算方法')
    
    parser.add_argument('--top-n', type=int, default=30,
                        help='展示前N个特征')
    
    parser.add_argument('--corr-threshold', type=float, default=0.7,
                        help='相关性阈值')
    
    parser.add_argument('--max-features', type=int, default=50,
                        help='最大特征数量')
    
    parser.add_argument('--symbols', type=str, default=None,
                        help='多个交易对名称，以逗号分隔')
    
    parser.add_argument('--timeframes', type=str, default=None,
                        help='多个时间周期，以逗号分隔')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    setup_logging()
    
    # 打印启动信息
    logger.info("开始特征分析流程")
    
    try:
        # 初始化特征分析器
        analyzer = FeatureAnalyzer(args.config)
        
        # 处理多个交易对
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(',')]
        else:
            symbols = [args.symbol]
        
        # 处理多个时间周期
        if args.timeframes:
            timeframes = [t.strip() for t in args.timeframes.split(',')]
        else:
            timeframes = [args.timeframe]
        
        # 运行每个交易对和时间周期的分析
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"分析 {symbol} 的 {timeframe} 数据")
                
                try:
                    # 运行分析
                    result_dir = analyzer.run_analysis(
                        symbol=symbol,
                        timeframe=timeframe,
                        target_col=args.target,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        importance_method=args.importance_method,
                        corr_method=args.corr_method,
                        top_n=args.top_n,
                        corr_threshold=args.corr_threshold,
                        max_features=args.max_features
                    )
                except Exception as e:
                    logger.error(f"分析 {symbol} {timeframe} 时出错: {str(e)}")
                    logger.debug("详细错误信息:", exc_info=True)
                    continue
        
        logger.info("特征分析流程完成")
        return 0
    
    except Exception as e:
        logger.error(f"特征分析过程中发生错误: {str(e)}")
        logger.debug("详细错误信息:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 