"""
特征工程管道模块
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
import logging
import os
from tqdm import tqdm
import torch

from src.utils.logger import get_logger
from src.utils.config import load_config
# 从 pytorch_technical_indicators 导入 GPU 版本的指标计算类
from src.features.pytorch_technical_indicators import PyTorchCompatibleTechnicalIndicators
from src.features.microstructure_features import MicrostructureFeatures
from src.features.torch_utils import get_device

logger = get_logger(__name__)

class FeatureEngineer:
    """
    特征工程管道类，用于计算和处理特征
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        初始化特征工程管道
        
        参数:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.features_config = self.config.get("features", {})
        
        # 特征组
        self.feature_groups = self.features_config.get("groups", [
            "price_based", "volume_based", "volatility", 
            "trend", "momentum", "market_microstructure"
        ])
        
        # 窗口大小
        self.window_sizes = self.features_config.get("windows", {
            "short": [5, 10, 20],
            "medium": [50, 100, 200],
            "long": [500, 1000]
        })
        
        # 特征处理设置
        self.preprocessing = self.features_config.get("preprocessing", {
            "scaling": "standard",  # 标准化方法: standard, minmax, robust
            "fill_na": "forward",   # 缺失值填充方法: forward, mean, zero
            "outlier_method": "iqr" # 异常值处理方法: iqr, zscore, none
        })
        
        # 特征选择设置
        self.feature_selection = self.features_config.get("feature_selection", {
            "method": "mutual_info",    # 特征选择方法: mutual_info, f_regression, pca
            "n_features": 50,           # 选择的特征数量
            "threshold": None           # 特征重要性阈值
        })
        
        # 数据路径
        self.processed_data_path = "data/processed"
        self.features_path = "data/processed/features"
        os.makedirs(self.features_path, exist_ok=True)
        
        # 初始化特征计算器 - 使用 PyTorch 版本
        self.tech_indicators = PyTorchCompatibleTechnicalIndicators()
        self.microstructure = MicrostructureFeatures()
        
        # 缩放器
        self.scalers = {}
        
        # GPU设备
        self.device = get_device()
        self.use_gpu = self.device.type == 'cuda'
        
        logger.info(f"特征工程管道已初始化，使用设备: {self.device}")
        if self.use_gpu:
            logger.info("已启用GPU加速特征计算")
        else:
            logger.warning("未检测到可用GPU，将使用CPU计算，但使用PyTorch兼容接口")
    
    def compute_features(self, df, feature_groups=None):
        """
        计算特征
        
        参数:
            df: DataFrame对象，包含原始数据
            feature_groups: 要计算的特征组列表
            
        返回:
            包含计算特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法计算空数据的特征")
            return df
        
        # 使用默认特征组（如果未指定）
        if feature_groups is None:
            feature_groups = self.feature_groups
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        # 备份原始DataFrame，确保在任何情况下都能返回有效结果
        original_df = df.copy()
        
        # 转换窗口大小为列表
        try:
            window_sizes_list = []
            for size_group in self.window_sizes.values():
                window_sizes_list.extend(size_group)
            window_sizes_list = sorted(list(set(window_sizes_list)))
        except Exception as e:
            logger.error(f"处理窗口大小时出错: {str(e)}")
            window_sizes_list = [5, 10, 20, 50, 100, 200]  # 使用默认值
        
        try:
            # 按特征组计算特征 - 每次计算前检查result_df状态
            if result_df is not None and not result_df.empty and "price_based" in feature_groups:
                try:
                    temp_df = self.tech_indicators.calculate_price_features(result_df)
                    if temp_df is not None and not temp_df.empty:
                        result_df = temp_df
                    else:
                        logger.warning("价格特征计算返回空结果，跳过并使用当前DataFrame")
                except Exception as e:
                    logger.error(f"计算价格特征时出错: {str(e)}")
                    # 继续使用当前result_df，不更新
            
            if result_df is not None and not result_df.empty and "volume_based" in feature_groups:
                try:
                    temp_df = self.tech_indicators.calculate_volume_features(result_df)
                    if temp_df is not None and not temp_df.empty:
                        result_df = temp_df
                    else:
                        logger.warning("交易量特征计算返回空结果，跳过并使用当前DataFrame")
                except Exception as e:
                    logger.error(f"计算交易量特征时出错: {str(e)}")
                    # 继续使用当前result_df，不更新
            
            if result_df is not None and not result_df.empty and "volatility" in feature_groups:
                try:
                    temp_df = self.tech_indicators.calculate_volatility_features(result_df)
                    if temp_df is not None and not temp_df.empty:
                        result_df = temp_df
                    else:
                        logger.warning("波动性特征计算返回空结果，跳过并使用当前DataFrame")
                except Exception as e:
                    logger.error(f"计算波动性特征时出错: {str(e)}")
                    # 继续使用当前result_df，不更新
            
            if result_df is not None and not result_df.empty and "trend" in feature_groups:
                try:
                    temp_df = self.tech_indicators.calculate_trend_features(result_df)
                    if temp_df is not None and not temp_df.empty:
                        result_df = temp_df
                    else:
                        logger.warning("趋势特征计算返回空结果，跳过并使用当前DataFrame")
                except Exception as e:
                    logger.error(f"计算趋势特征时出错: {str(e)}")
                    # 继续使用当前result_df，不更新
            
            if result_df is not None and not result_df.empty and "momentum" in feature_groups:
                try:
                    temp_df = self.tech_indicators.calculate_momentum_features(result_df)
                    if temp_df is not None and not temp_df.empty:
                        result_df = temp_df
                    else:
                        logger.warning("动量特征计算返回空结果，跳过并使用当前DataFrame")
                except Exception as e:
                    logger.error(f"计算动量特征时出错: {str(e)}")
                    # 继续使用当前result_df，不更新
            
            if result_df is not None and not result_df.empty and "market_microstructure" in feature_groups:
                try:
                    # 计算市场微观结构特征（如果有相关数据）
                    has_bid_ask = all(col in result_df.columns for col in ['bid', 'ask'])
                    
                    if has_bid_ask:
                        try:
                            temp_df = self.microstructure.calculate_bid_ask_features(result_df)
                            if temp_df is not None and not temp_df.empty:
                                result_df = temp_df
                        except Exception as e:
                            logger.error(f"计算买卖盘特征时出错: {str(e)}")
                        
                        try:
                            temp_df = self.microstructure.calculate_liquidity_features(result_df, window_sizes_list)
                            if temp_df is not None and not temp_df.empty:
                                result_df = temp_df
                        except Exception as e:
                            logger.error(f"计算流动性特征时出错: {str(e)}")
                    
                    # 计算其他微观结构特征（基于OHLCV数据）
                    try:
                        temp_df = self.microstructure.calculate_order_flow_features(result_df, window_sizes_list)
                        if temp_df is not None and not temp_df.empty:
                            result_df = temp_df
                    except Exception as e:
                        logger.error(f"计算订单流特征时出错: {str(e)}")
                    
                    try:
                        temp_df = self.microstructure.calculate_volatility_clustering(result_df, window_sizes_list)
                        if temp_df is not None and not temp_df.empty:
                            result_df = temp_df
                    except Exception as e:
                        logger.error(f"计算波动率聚类特征时出错: {str(e)}")
                    
                    try:
                        temp_df = self.microstructure.calculate_price_impact(result_df, window_sizes_list)
                        if temp_df is not None and not temp_df.empty:
                            result_df = temp_df
                    except Exception as e:
                        logger.error(f"计算价格影响特征时出错: {str(e)}")
                except Exception as e:
                    logger.error(f"计算微观结构特征时出错: {str(e)}")
                    # 继续使用当前result_df，不更新
            
            # 检查result_df是否有效，如果无效则恢复使用原始DataFrame
            if result_df is None or result_df.empty:
                logger.warning("特征计算后得到空DataFrame，恢复使用原始数据")
                result_df = original_df.copy()
            
            # 计算技术指标（如果未在上述特征组中处理）
            if result_df is not None and not result_df.empty:
                try:
                    # 将技术指标名称映射到新的结构
                    indicators_mapping = {
                        'MACD': 'momentum',
                        'RSI': 'momentum',
                        'STOCH': 'momentum',
                        'BBANDS': 'volatility',
                        'ATR': 'volatility',
                        'ADX': 'trend',
                        'CCI': 'trend'
                    }
                    
                    # 获取需要计算的指标组
                    needed_groups = set()
                    for ind in ['MACD', 'RSI', 'STOCH', 'BBANDS', 'ATR', 'ADX', 'CCI']:
                        if indicators_mapping.get(ind) not in feature_groups:
                            needed_groups.add(indicators_mapping.get(ind))
                    
                    # 如果有需要单独计算的指标组
                    if needed_groups:
                        try:
                            temp_df = self.tech_indicators.calculate_indicators(
                                result_df, 
                                indicators=list(needed_groups),
                                window_sizes=None
                            )
                            if temp_df is not None and not temp_df.empty:
                                result_df = temp_df
                        except Exception as e:
                            logger.error(f"计算额外技术指标时出错: {str(e)}")
                            # 即使额外指标计算失败，也保留当前的结果_df
                except Exception as e:
                    logger.error(f"处理技术指标映射时出错: {str(e)}")
            
            # 最终检查结果
            if result_df is None or result_df.empty:
                logger.warning("所有特征计算失败，返回原始数据")
                result_df = original_df.copy()
            
            logger.info(f"已计算 {len(result_df.columns) - len(df.columns)} 个特征")
            return result_df
            
        except Exception as e:
            logger.error(f"特征计算过程中出错: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            # 返回原始DataFrame的副本，确保不会返回None
            return original_df.copy()
    
    def compute_features_batch(self, df, feature_groups=None):
        """
        计算批次数据的特征 (优化版，可利用GPU)
        
        参数:
            df: DataFrame对象，包含一批数据
            feature_groups: 要计算的特征组列表
            
        返回:
            包含计算特征的DataFrame
        """
        return self.compute_features(df, feature_groups)
    
    def preprocess_features(self, df, scaling=None, fill_na=None, handle_outliers=None):
        """
        预处理特征
        
        参数:
            df: DataFrame对象，包含特征
            scaling: 缩放方法，可选 'standard', 'minmax', 'robust', None
            fill_na: 缺失值填充方法，可选 'forward', 'mean', 'zero', None
            handle_outliers: 异常值处理方法，可选 'iqr', 'zscore', None
            
        返回:
            预处理后的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法预处理空数据")
            return df
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 使用配置中的默认值（如果未指定）
        if scaling is None:
            scaling = self.preprocessing.get("scaling", None)
        
        if fill_na is None:
            fill_na = self.preprocessing.get("fill_na", None)
        
        if handle_outliers is None:
            handle_outliers = self.preprocessing.get("outlier_method", None)
        
        # 分离特征和目标变量（假设'timestamp'是索引）
        feature_cols = [col for col in result_df.columns if col not in ['timestamp']]
        
        # 处理缺失值
        if fill_na == 'forward':
            result_df[feature_cols] = result_df[feature_cols].ffill()
            # 对于开头的NaN，使用后向填充
            result_df[feature_cols] = result_df[feature_cols].bfill()
        elif fill_na == 'mean':
            for col in feature_cols:
                if result_df[col].dtype in [np.float64, np.int64]:
                    mean_val = result_df[col].mean()
                    result_df[col] = result_df[col].fillna(mean_val)
        elif fill_na == 'zero':
            result_df[feature_cols] = result_df[feature_cols].fillna(0)
        
        # 处理异常值
        if handle_outliers == 'iqr':
            for col in feature_cols:
                if result_df[col].dtype in [np.float64, np.int64]:
                    Q1 = result_df[col].quantile(0.25)
                    Q3 = result_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    result_df[col] = result_df[col].clip(lower=lower_bound, upper=upper_bound)
        elif handle_outliers == 'zscore':
            for col in feature_cols:
                if result_df[col].dtype in [np.float64, np.int64]:
                    mean = result_df[col].mean()
                    std = result_df[col].std()
                    if std != 0:  # 避免除以零
                        z_scores = (result_df[col] - mean) / std
                        result_df[col] = result_df[col].mask(abs(z_scores) > 3, mean)
        
        # 特征缩放前确保没有无穷值和NaN
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        # 对于数值型特征中的NaN，用0填充
        result_df[numeric_cols] = result_df[numeric_cols].fillna(0)
        
        # 特征缩放
        if scaling == 'standard':
            scaler = StandardScaler()
            # 仅对数值列进行缩放
            numeric_cols = result_df[feature_cols].select_dtypes(include=[np.number]).columns
            result_df[numeric_cols] = scaler.fit_transform(result_df[numeric_cols])
            self.scalers['standard'] = scaler
        elif scaling == 'minmax':
            scaler = MinMaxScaler()
            numeric_cols = result_df[feature_cols].select_dtypes(include=[np.number]).columns
            result_df[numeric_cols] = scaler.fit_transform(result_df[numeric_cols])
            self.scalers['minmax'] = scaler
        elif scaling == 'robust':
            scaler = RobustScaler()
            numeric_cols = result_df[feature_cols].select_dtypes(include=[np.number]).columns
            result_df[numeric_cols] = scaler.fit_transform(result_df[numeric_cols])
            self.scalers['robust'] = scaler
        
        logger.info("特征预处理完成")
        return result_df
    
    def select_features(self, X, y=None, method=None, n_features=None, threshold=None):
        """
        特征选择
        
        参数:
            X: 特征矩阵
            y: 目标变量（如果使用监督特征选择）
            method: 特征选择方法，可选 'mutual_info', 'f_regression', 'pca', None
            n_features: 选择的特征数量
            threshold: 特征重要性阈值
            
        返回:
            选择后的特征矩阵和特征名称
        """
        if X is None or X.empty:
            logger.warning("无法对空数据进行特征选择")
            return X, []
        
        # 使用配置中的默认值（如果未指定）
        if method is None:
            method = self.feature_selection.get("method", None)
        
        if n_features is None:
            n_features = self.feature_selection.get("n_features", 50)
        
        if threshold is None:
            threshold = self.feature_selection.get("threshold", None)
        
        # 只选择数值列
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        # 如果数值特征少于所需特征数，调整n_features
        if len(numeric_cols) < n_features:
            n_features = len(numeric_cols)
            logger.warning(f"可用特征数量({len(numeric_cols)})少于所需数量({n_features})，已调整为{n_features}")
        
        # 处理缺失值
        if y is not None:
            # 检查目标变量中的NaN
            nan_count = np.isnan(y).sum()
            if nan_count > 0:
                logger.warning(f"目标变量中包含 {nan_count} 个NaN值，将被处理")
                
                # 创建包含特征和目标的DataFrame，以便同时处理NaN
                combined_df = pd.DataFrame(X_numeric)
                combined_df['target'] = y
                
                # 移除包含NaN的行
                combined_df_clean = combined_df.dropna()
                logger.info(f"移除含NaN的行后，剩余 {len(combined_df_clean)} 行数据，原始数据有 {len(combined_df)} 行")
                
                if len(combined_df_clean) == 0:
                    logger.error("处理NaN值后无剩余数据，无法进行特征选择")
                    return X, []
                
                # 分离特征和目标变量
                X_numeric = combined_df_clean.drop('target', axis=1)
                y = combined_df_clean['target']
        
        # 特征选择
        selected_features = []
        importance_scores = {}
        
        if method == 'mutual_info' and y is not None:
            # 使用互信息选择特征
            selector = SelectKBest(mutual_info_regression, k=n_features)
            X_selected = selector.fit_transform(X_numeric, y)
            
            # 获取选择的特征名称
            mask = selector.get_support()
            selected_features = X_numeric.columns[mask].tolist()
            
            # 记录特征重要性
            scores = selector.scores_
            for i, feature in enumerate(X_numeric.columns):
                importance_scores[feature] = scores[i]
        
        elif method == 'f_regression' and y is not None:
            # 使用F检验选择特征
            selector = SelectKBest(f_regression, k=n_features)
            X_selected = selector.fit_transform(X_numeric, y)
            
            # 获取选择的特征名称
            mask = selector.get_support()
            selected_features = X_numeric.columns[mask].tolist()
            
            # 记录特征重要性
            scores = selector.scores_
            for i, feature in enumerate(X_numeric.columns):
                importance_scores[feature] = scores[i]
        
        elif method == 'pca':
            # 使用PCA降维
            pca = PCA(n_components=n_features)
            X_selected = pca.fit_transform(X_numeric)
            
            # PCA没有选择原始特征，而是创建新的组件
            selected_features = [f'PC{i+1}' for i in range(n_features)]
            
            # 记录特征重要性（使用解释方差比例）
            explained_variance = pca.explained_variance_ratio_
            for i, feature in enumerate(selected_features):
                importance_scores[feature] = explained_variance[i]
                
            # 返回PCA转换后的数据作为DataFrame
            X_selected_df = pd.DataFrame(X_selected, index=X_numeric.index, columns=selected_features)
            return X_selected_df, selected_features
        
        else:
            # 不进行特征选择，保留所有数值特征
            X_selected = X_numeric.values
            selected_features = numeric_cols.tolist()
            
            # 默认的特征重要性为1
            for feature in selected_features:
                importance_scores[feature] = 1.0
        
        # 如果指定了阈值，并且使用了基于特征重要性的方法
        if threshold is not None and method in ['mutual_info', 'f_regression']:
            # 根据阈值筛选特征
            selected_features = [feature for feature, score in importance_scores.items() 
                               if score >= threshold]
            
            # 更新选择的特征矩阵
            X_selected = X_numeric[selected_features].values
        
        # 返回选择后的特征矩阵和特征名称
        X_selected_df = pd.DataFrame(X_selected, index=X_numeric.index, columns=selected_features)
        
        logger.info(f"特征选择完成，选择了 {len(selected_features)} 个特征")
        return X_selected_df, selected_features
    
    def create_lagged_features(self, df, lag_periods=None, columns=None):
        """
        创建滞后特征
        
        参数:
            df: DataFrame对象
            lag_periods: 滞后周期列表
            columns: 要创建滞后特征的列名列表
            
        返回:
            包含滞后特征的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法处理空数据")
            return df
        
        # 默认滞后周期
        if lag_periods is None:
            lag_periods = [1, 2, 3, 5, 10]
        
        # 默认使用所有数值列
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 创建滞后特征
        for col in columns:
            for lag in lag_periods:
                result_df[f'{col}_lag_{lag}'] = result_df[col].shift(lag)
        
        logger.info(f"已创建 {len(columns) * len(lag_periods)} 个滞后特征")
        return result_df
    
    def create_target_variables(self, df, horizons=None, target_type='price_change_pct'):
        """
        创建目标变量
        
        参数:
            df: DataFrame对象
            horizons: 预测时间范围列表（分钟）
            target_type: 目标变量类型，可选 'price_change_pct', 'direction', 'volatility'
            
        返回:
            包含目标变量的DataFrame
        """
        if df is None or df.empty:
            logger.warning("无法处理空数据")
            return df
        
        # 默认预测范围
        if horizons is None:
            horizons_config = self.config.get("models", {}).get("prediction_horizons", {})
            horizons = []
            for period_list in horizons_config.values():
                horizons.extend(period_list)
        
        # 创建副本，避免修改原始数据
        result_df = df.copy()
        
        # 检查是否有close列
        if 'close' not in result_df.columns:
            logger.warning("数据缺少close列，无法创建价格相关的目标变量")
            return result_df
        
        # 创建不同类型的目标变量
        for horizon in horizons:
            if target_type == 'price_change_pct':
                # 价格百分比变化
                result_df[f'target_pct_{horizon}'] = result_df['close'].pct_change(periods=horizon) * 100
            
            elif target_type == 'direction':
                # 价格方向（1=上涨，0=下跌）
                price_change = result_df['close'].pct_change(periods=horizon)
                result_df[f'target_direction_{horizon}'] = (price_change > 0).astype(int)
            
            elif target_type == 'volatility':
                # 未来波动率（收盘价的标准差）
                for i in range(len(result_df)):
                    if i + horizon < len(result_df):
                        future_prices = result_df['close'].iloc[i:i+horizon]
                        future_volatility = future_prices.pct_change().std() * np.sqrt(252)
                        result_df.loc[result_df.index[i], f'target_volatility_{horizon}'] = future_volatility
                    else:
                        result_df.loc[result_df.index[i], f'target_volatility_{horizon}'] = np.nan
        
        logger.info(f"已创建 {len(horizons)} 个目标变量")
        return result_df
    
    def load_data(self, filepath):
        """
        加载数据文件 - 优化版本
        
        参数:
            filepath: 文件路径
            
        返回:
            加载的DataFrame
        """
        try:
            # 读取CSV文件，使用更高效的引擎和方式
            # 1. 使用c引擎而非python引擎
            # 2. 只加载必要的列
            # 3. 使用适当的数据类型
            # 4. 通过chunksize处理大文件
            
            # 先尝试检查文件的列名
            import csv
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
            
            # 确定必要的列
            essential_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            usecols = [col for col in headers if col in essential_cols]
            
            # 设置dtypes加速加载
            dtypes = {
                'open': 'float32',
                'high': 'float32',
                'low': 'float32',
                'close': 'float32',
                'volume': 'float32'
            }
            
            # 如果文件非常大，使用分块读取
            file_size = os.path.getsize(filepath)
            chunk_size = 500000 if file_size > 100*1024*1024 else None  # 100MB以上使用分块
            
            if chunk_size:
                logger.info(f"文件 {filepath} 较大 ({file_size/1024/1024:.1f}MB)，使用分块读取")
                chunks = []
                for chunk in pd.read_csv(filepath, usecols=usecols, dtype=dtypes, chunksize=chunk_size, engine='c'):
                    # 处理timestamp
                    if "timestamp" in chunk.columns:
                        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
                    chunks.append(chunk)
                df = pd.concat(chunks)
            else:
                df = pd.read_csv(filepath, usecols=usecols, dtype=dtypes, engine='c')
                # 处理timestamp
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # 设置索引
            if "timestamp" in df.columns:
                df.set_index("timestamp", inplace=True)
            
            logger.debug(f"已加载数据文件 {filepath}，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"加载数据文件 {filepath} 时出错: {e}")
            return pd.DataFrame()
    
    def process_data_for_symbol(self, symbol, timeframe, start_date=None, end_date=None, data_dir="data/processed/merged", batch_size=None):
        """
        处理单个交易对的数据
        
        参数:
            symbol: 交易对名称
            timeframe: 时间框架
            start_date: 起始日期
            end_date: 截止日期
            data_dir: 数据目录
            batch_size: 批处理大小，用于处理大数据集
            
        返回:
            (特征DataFrame, 目标变量DataFrame)元组
        """
        # 构建文件路径
        if start_date and end_date:
            filename = f"{symbol}_{timeframe}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
        else:
            filename = f"{symbol}_{timeframe}.csv"
            
        filepath = os.path.join(data_dir, filename)
        
        # 检查文件是否存在
        if not os.path.exists(filepath):
            logger.warning(f"找不到文件: {filepath}")
            return None, None
        
        try:
            # 文件大小检查（如果超过100MB，使用分块读取）
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            use_chunking = file_size_mb > 100 or batch_size is not None
            
            # 加载数据
            df = None
            
            if use_chunking:
                logger.info(f"文件 {filepath} 较大 ({file_size_mb:.1f}MB)，使用分块读取")
                
                # 首先读取文件头来获取列名
                header_df = pd.read_csv(filepath, nrows=0)
                column_names = header_df.columns.tolist()
                
                # 确保时间戳列存在
                if 'timestamp' not in column_names:
                    logger.warning(f"文件 {filepath} 中没有timestamp列")
                    return None, None
                
                # 确定读取数据的批次大小
                if batch_size is None:
                    batch_size = 100000  # 默认批次大小
                
                # 计算文件的行数 (需要一次遍历)
                with open(filepath, 'r') as f:
                    total_rows = sum(1 for _ in f) - 1  # 减去标题行
                
                logger.info(f"数据大小为 {total_rows} 行，启用批处理 (每批 {batch_size} 行)")
                
                try:
                    # 直接读取整个文件，然后分批处理
                    logger.info("读取整个文件，然后分批处理...")
                    df = pd.read_csv(filepath)
                except Exception as load_e:
                    logger.error(f"读取文件 {filepath} 时出错: {str(load_e)}")
                    return None, None
            else:
                # 对于较小的文件，直接读取
                try:
                    df = pd.read_csv(filepath)
                except Exception as load_e:
                    logger.error(f"读取文件 {filepath} 时出错: {str(load_e)}")
                    return None, None
            
            # 确保数据加载成功
            if df is None or df.empty:
                logger.warning(f"文件 {filepath} 数据为空或加载失败")
                return None, None
                
            # 设置时间戳为索引
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            
            # 初始化变量
            features_df = None
            targets_df = None
            
            # 整体异常处理
            try:
                # 分批处理特征计算
                logger.info(f"开始计算 {symbol} {timeframe} 特征...")
                try:
                    features_df = self.compute_features(df)
                    if features_df is None or features_df.empty:
                        logger.warning(f"计算 {symbol} {timeframe} 特征失败，返回空DataFrame")
                        return None, None
                except Exception as feature_e:
                    logger.error(f"计算 {symbol} {timeframe} 特征时出错: {str(feature_e)}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    return None, None
                
                # 创建目标变量
                logger.info(f"开始为 {symbol} {timeframe} 创建目标变量...")
                try:
                    targets_df = self.create_target_variables(features_df)
                    if targets_df is None or targets_df.empty:
                        logger.warning(f"为 {symbol} {timeframe} 创建目标变量失败，返回空DataFrame")
                        return None, None
                except Exception as target_e:
                    logger.error(f"为 {symbol} {timeframe} 创建目标变量时出错: {str(target_e)}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    return None, None
                
                # 预处理特征
                logger.info(f"开始为 {symbol} {timeframe} 预处理特征...")
                try:
                    features_df = self.preprocess_features(features_df)
                    if features_df is None or features_df.empty:
                        logger.warning(f"为 {symbol} {timeframe} 预处理特征失败，返回空DataFrame")
                        return None, None
                except Exception as preprocess_e:
                    logger.error(f"为 {symbol} {timeframe} 预处理特征时出错: {str(preprocess_e)}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    return None, None
                
            except Exception as general_e:
                logger.error(f"处理 {filepath} 总体流程出错: {str(general_e)}")
                import traceback
                logger.debug(traceback.format_exc())
                return None, None
            
            # 检查处理是否成功
            if features_df is None or targets_df is None:
                logger.warning(f"处理 {filepath} 失败，特征或目标变量为空")
                return None, None
            
            # 将时间戳设为列而非索引，方便后续处理
            try:
                if isinstance(features_df.index, pd.DatetimeIndex) or features_df.index.name == 'timestamp':
                    features_df.reset_index(inplace=True)
                
                if isinstance(targets_df.index, pd.DatetimeIndex) or targets_df.index.name == 'timestamp':
                    targets_df.reset_index(inplace=True)
            except Exception as reset_e:
                logger.error(f"重置索引时出错: {str(reset_e)}")
                # 尝试恢复处理
                try:
                    features_df = features_df.reset_index()
                    targets_df = targets_df.reset_index()
                except:
                    pass
            
            # 创建输出目录
            try:
                output_dir = os.path.join(self.features_path, symbol)
                os.makedirs(output_dir, exist_ok=True)
                
                # 保存特征和目标变量
                features_file = os.path.join(output_dir, f"features_{timeframe}.csv")
                targets_file = os.path.join(output_dir, f"targets_{timeframe}.csv")
                
                features_df.to_csv(features_file, index=False)
                targets_df.to_csv(targets_file, index=False)
                
                logger.info(f"已处理 {symbol} {timeframe} 数据并保存至 {output_dir}")
            except Exception as save_e:
                logger.error(f"保存结果到 {output_dir} 时出错: {str(save_e)}")
                # 即使保存失败，也返回结果
            
            return features_df, targets_df
            
        except Exception as e:
            logger.error(f"处理 {filepath} 时出错: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, None
    
    def process_all_data(self, symbols=None, timeframes=None, start_date=None, end_date=None, data_dir="data/processed/merged", batch_size=None):
        """
        处理所有交易对的数据
        
        参数:
            symbols: 交易对列表，如果为None则处理所有交易对
            timeframes: 时间框架列表，如果为None则处理所有时间框架
            start_date: 起始日期
            end_date: 截止日期
            data_dir: 数据目录
            batch_size: 批处理大小，用于处理大数据集
            
        返回:
            处理后的数据字典 {symbol: {timeframe: (features_df, targets_df)}}
        """
        # 如果未指定交易对和时间框架，从数据目录自动检测
        if symbols is None or timeframes is None:
            available_files = os.listdir(data_dir)
            data_files = [f for f in available_files if f.endswith('.csv')]
            
            if not data_files:
                logger.warning(f"在目录 {data_dir} 中未找到任何CSV文件")
                return {}
            
            # 从文件名中提取交易对和时间框架
            if symbols is None:
                detected_symbols = set()
                for filename in data_files:
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        detected_symbols.add(parts[0])
                symbols = list(detected_symbols)
            
            if timeframes is None:
                detected_timeframes = set()
                for filename in data_files:
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        detected_timeframes.add(parts[1])
                timeframes = list(detected_timeframes)
            
            logger.info(f"检测到的交易对: {symbols}")
            logger.info(f"检测到的时间框架: {timeframes}")
        
        results = {}
        
        # 处理所有交易对和时间框架的组合
        for symbol in tqdm(symbols, desc="处理交易对"):
            results[symbol] = {}
            
            for timeframe in timeframes:
                logger.info(f"处理 {symbol} {timeframe} 数据...")
                
                features_df, targets_df = self.process_data_for_symbol(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    data_dir=data_dir,
                    batch_size=batch_size
                )
                
                if features_df is not None and targets_df is not None:
                    results[symbol][timeframe] = (features_df, targets_df)
                    logger.info(f"成功处理 {symbol} {timeframe} 数据: {len(features_df)} 行")
                else:
                    logger.warning(f"未能处理 {symbol} {timeframe} 数据")
        
        return results 