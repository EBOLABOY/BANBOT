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
# 导入原始的 TechnicalIndicators 类
from src.features.technical_indicators import TechnicalIndicators
from src.features.microstructure_features import MicrostructureFeatures

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
        
        # 初始化特征计算器 - 使用原始实现
        self.tech_indicators = TechnicalIndicators()
        self.microstructure = MicrostructureFeatures()
        
        # 缩放器
        self.scalers = {}
        
        # 设备配置 - 固定为CPU
        self.device = torch.device("cpu")
        
        logger.info(f"特征工程管道已初始化")
        logger.info(f"使用的技术指标计算器: {type(self.tech_indicators).__name__}")
    
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
            
            # --- 添加组合/衍生信号特征 ---
            logger.info("开始计算组合/衍生信号特征...")
            try:
                # 1. SMA 交叉信号 (5上穿20)
                if 'sma_5' in result_df.columns and 'sma_20' in result_df.columns:
                    # 当前周期 sma_5 > sma_20
                    current_cross = result_df['sma_5'] > result_df['sma_20']
                    # 上一周期 sma_5 <= sma_20
                    previous_state = result_df['sma_5'].shift(1) <= result_df['sma_20'].shift(1)
                    # 计算原始信号，可能包含NaN
                    raw_signal = (current_cross & previous_state)
                    # 填充NaN为False，然后转为整数 (0 或 1)
                    result_df['signal_sma_cross_5_20'] = raw_signal.fillna(False).astype(int)
                else:
                    logger.warning("缺少 sma_5 或 sma_20 列，无法计算SMA交叉信号")

                # 2. MACD Histogram 正负信号
                if 'macd_diff' in result_df.columns:
                    # 计算原始信号，可能包含NaN
                    raw_sign = np.sign(result_df['macd_diff'])
                    # 填充NaN为0，然后转为整数 (-1, 0, 1)
                    result_df['signal_macd_hist_sign'] = raw_sign.fillna(0).astype(int)
                else:
                    logger.warning("缺少 macd_diff 列，无法计算MACD Histogram信号")

                # 3. RSI 超买/超卖信号 (使用 rsi_14) - 保持原逻辑，它应能处理NaN
                rsi_col = 'rsi_14' # 假设使用14周期RSI
                if rsi_col in result_df.columns:
                    overbought_threshold = 70
                    oversold_threshold = 30
                    result_df['signal_rsi_ob_os'] = 0 # 默认为0 (中性)
                    # 确保只对非NaN的RSI值应用条件
                    valid_rsi = result_df[rsi_col].notna()
                    result_df.loc[valid_rsi & (result_df[rsi_col] > overbought_threshold), 'signal_rsi_ob_os'] = 1 # 超买
                    result_df.loc[valid_rsi & (result_df[rsi_col] < oversold_threshold), 'signal_rsi_ob_os'] = -1 # 超卖
                else:
                    logger.warning(f"缺少 {rsi_col} 列，无法计算RSI超买/超卖信号")
                logger.info("已计算组合/衍生信号特征")

            except Exception as signal_e:
                logger.error(f"计算组合/衍生信号特征时出错: {str(signal_e)}")
                # 即使信号计算出错，也保留当前的 result_df
            
            # 最终检查结果
            if result_df is None or result_df.empty:
                logger.warning("所有特征计算失败，返回原始数据")
                result_df = original_df.copy()
            
            calculated_cols = len(result_df.columns) - len(df.columns)
            logger.info(f"已计算 {calculated_cols} 个特征")
            
            # 添加 .copy() 来解决性能警告
            result_df = result_df.copy()
            
            return result_df
            
        except Exception as e:
            logger.error(f"特征计算过程中出错: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            # 返回原始DataFrame的副本，确保不会返回None
            return original_df.copy()
    
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
            (特征DataFrame, 目标变量DataFrame)元组 - 仅在处理成功时返回，否则返回 (None, None)
        """
        # 构建文件路径 - 优先尝试Parquet，然后是压缩CSV，最后是普通CSV
        possible_filenames = [
            f"{symbol}_{timeframe}.parquet",
            f"{symbol}_{timeframe}.csv.gz",
            f"{symbol}_{timeframe}.csv"
        ]
        if start_date and end_date:
            date_str = f"_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
            possible_filenames = [
                f"{symbol}_{timeframe}{date_str}.parquet",
                f"{symbol}_{timeframe}{date_str}.csv.gz",
                f"{symbol}_{timeframe}{date_str}.csv"
            ] + possible_filenames # 也检查没有日期的文件名

        filepath = None
        for fname in possible_filenames:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                filepath = fpath
                logger.info(f"找到数据文件: {filepath}")
                break
        
        # 检查文件是否存在
        if filepath is None:
            logger.warning(f"在 {data_dir} 中找不到 {symbol} {timeframe} 的数据文件 (尝试了 {possible_filenames})")
            return None, None
        
        try:
            all_features_chunks = []
            all_targets_chunks = []
            processed_rows = 0

            # 根据是否提供 batch_size 决定处理方式
            if batch_size and batch_size > 0 and filepath.endswith(('.csv', '.csv.gz')): # 分块仅对CSV有效
                logger.info(f"使用批处理模式，批次大小: {batch_size}")
                try:
                    # 使用 chunksize 读取
                    reader = pd.read_csv(filepath, chunksize=batch_size)
                    # 估算总块数 (如果需要进度条)，这可能需要再次读取或预先计算行数
                    # total_chunks = sum(1 for _ in pd.read_csv(filepath, chunksize=batch_size)) # 避免重复读取，可以省略或用其他方式估算

                    # for i, chunk_df in enumerate(tqdm(reader, desc=f"处理 {symbol} {timeframe} 块")):
                    for i, chunk_df in enumerate(reader):
                        logger.debug(f"处理块 {i+1}...")
                        # 设置时间戳索引
                        if 'timestamp' in chunk_df.columns:
                            try:
                                chunk_df['timestamp'] = pd.to_datetime(chunk_df['timestamp'])
                                chunk_df.set_index('timestamp', inplace=True)
                            except Exception as ts_err:
                                logger.warning(f"块 {i+1} 时间戳处理失败: {ts_err}, 跳过此块")
                                continue
                        else:
                            logger.warning(f"块 {i+1} 中缺少 'timestamp' 列, 跳过此块")
                            continue
                        
                        # --- 块内处理 --- 
                        features_chunk = None
                        targets_chunk = None
                        try:
                            features_chunk = self.compute_features(chunk_df)
                            if features_chunk is not None and not features_chunk.empty:
                                targets_chunk = self.create_target_variables(features_chunk)
                                if targets_chunk is not None and not targets_chunk.empty:
                                     features_chunk = self.preprocess_features(features_chunk)
                                     if features_chunk is None or features_chunk.empty: # 检查预处理是否成功
                                         logger.warning(f"块 {i+1} 预处理失败")
                                         features_chunk = None # 预处理失败则标记失败
                                         targets_chunk = None
                                else:
                                    logger.warning(f"块 {i+1} 创建目标变量失败")
                                    features_chunk = None # 标记失败
                            else:
                                logger.warning(f"块 {i+1} 计算特征失败")
                                features_chunk = None # 标记失败 (确保在所有失败路径都标记)
                        except Exception as chunk_proc_e:
                            logger.error(f"处理块 {i+1} 时出错: {chunk_proc_e}")
                            features_chunk = None # 标记失败
                        # ---------------

                        if features_chunk is not None and targets_chunk is not None and not features_chunk.empty:
                            processed_rows += len(chunk_df)
                            all_features_chunks.append(features_chunk.reset_index())
                            all_targets_chunks.append(targets_chunk.reset_index())
                    if not all_features_chunks or not all_targets_chunks:
                        logger.error(f"处理 {symbol} {timeframe} 时所有块都处理失败")
                        return None, None
                    
                    # 合并所有处理过的块
                    logger.info("合并所有处理过的块...")
                    features_df = pd.concat(all_features_chunks, ignore_index=True)
                    targets_df = pd.concat(all_targets_chunks, ignore_index=True)
                    logger.info(f"合并完成，总共处理 {processed_rows} 行")

                except Exception as batch_e:
                    logger.error(f"批处理 {filepath} 时出错: {str(batch_e)}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    return None, None
            else:
                # 不分批处理 (或输入是 Parquet)
                if filepath.endswith('.parquet'):
                    logger.info(f"加载 Parquet 文件 {filepath} 进行处理...")
                else:
                    logger.info(f"加载整个 CSV 文件 {filepath} 进行处理...")
                df = self.load_data(filepath)
                if df is None or df.empty:
                    logger.warning(f"文件 {filepath} 数据为空或加载失败")
                    return None, None
                
                # --- 整体处理 ---
                features_df = None
                targets_df = None
                try:
                    logger.info(f"开始计算 {symbol} {timeframe} 特征 (整体处理)...")
                    features_df = self.compute_features(df)
                    if features_df is not None and not features_df.empty:
                        logger.info(f"开始为 {symbol} {timeframe} 创建目标变量 (整体处理)...")
                        targets_df = self.create_target_variables(features_df)
                        if targets_df is not None and not targets_df.empty:
                            logger.info(f"开始为 {symbol} {timeframe} 预处理特征 (整体处理)...")
                            features_df = self.preprocess_features(features_df)
                            if features_df is None or features_df.empty:
                                logger.warning(f"为 {symbol} {timeframe} 预处理特征失败")
                                targets_df = None # 如果预处理失败，目标也无效了
                        else:
                            logger.warning(f"为 {symbol} {timeframe} 创建目标变量失败")
                            features_df = None # 如果目标失败，特征也没用了
                    else:
                        logger.warning(f"计算 {symbol} {timeframe} 特征失败")
                        features_df = None # 确保失败时 features_df 为 None
                except Exception as proc_e:
                    logger.error(f"整体处理 {symbol} {timeframe} 时出错: {proc_e}")
                    features_df = None
                    targets_df = None
                # --------------

                if features_df is None or targets_df is None:
                    logger.error(f"整体处理 {symbol} {timeframe} 失败")
                    return None, None

            # --- 保存为 Parquet ---
            output_dir = os.path.join(self.features_path, symbol)
            os.makedirs(output_dir, exist_ok=True)
            
            features_file_parquet = os.path.join(output_dir, f"features_{timeframe}.parquet")
            targets_file_parquet = os.path.join(output_dir, f"targets_{timeframe}.parquet")
            
            try:
                # 尝试优化数据类型
                features_df = features_df.infer_objects() # 推断最佳类型
                for col in features_df.select_dtypes(include=['float64']).columns:
                    features_df[col] = features_df[col].astype('float32')
                for col in features_df.select_dtypes(include=['int64', 'int32', 'int16', 'int8']).columns: # 检查所有整数类型
                    if pd.api.types.is_numeric_dtype(features_df[col]): 
                        min_val = features_df[col].min()
                        max_val = features_df[col].max()
                        if pd.notna(min_val) and pd.notna(max_val): # 确保有有效值
                            # 检查是否可以降级为更小的整数类型
                            if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                                features_df[col] = features_df[col].astype('int8')
                            elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                                features_df[col] = features_df[col].astype('int16')
                            elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                                features_df[col] = features_df[col].astype('int32')
                        # else: 保持原有的int64或其他较大整数类型
                        # else: # 如果列中存在NaN（即使理论上不应该在整数列），无法直接转换
                            pass # 明确添加 pass 来处理这个 else 分支

                targets_df = targets_df.infer_objects()
                for col in targets_df.select_dtypes(include=['float64']).columns:
                    targets_df[col] = targets_df[col].astype('float32')

                # 保存为 Parquet (使用 gzip 压缩)
                features_df.to_parquet(features_file_parquet, index=False, compression='gzip')
                targets_df.to_parquet(targets_file_parquet, index=False, compression='gzip')
                logger.info(f"已处理 {symbol} {timeframe} 数据并保存为 Parquet 至 {output_dir}")
            except ImportError:
                logger.error("需要安装 'pyarrow' 库才能保存为 Parquet 格式。请运行: pip install pyarrow")
                # 回退到保存为 CSV (如果需要)
                features_file_csv = os.path.join(output_dir, f"features_{timeframe}.csv.gz")
                targets_file_csv = os.path.join(output_dir, f"targets_{timeframe}.csv.gz")
                try:
                    features_df.to_csv(features_file_csv, index=False, compression='gzip')
                    targets_df.to_csv(targets_file_csv, index=False, compression='gzip')
                    logger.info(f"已回退保存为压缩 CSV (.gz) 至 {output_dir}")
                except Exception as csv_save_e:
                    logger.error(f"回退保存为 CSV 时也出错: {str(csv_save_e)}")
                    # 保存失败，但处理可能已完成
                    return features_df, targets_df
            except Exception as save_e:
                logger.error(f"保存结果为 Parquet 时出错: {str(save_e)}")
                # 保存失败，但处理可能已完成
                return features_df, targets_df # 或者返回 None, None 表示保存失败？ \
            
            # 如果保存成功，返回处理好的DataFrame (虽然 process_all_data 不再使用它们)
            return features_df, targets_df
            
        except Exception as e:
            logger.error(f"处理 {filepath} 时发生意外错误: {str(e)}")
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
            处理后的数据字典 {symbol: {timeframe: (features_df, targets_df)}} -- 注意：实际不再返回数据，直接保存文件
        """
        # 如果未指定交易对和时间框架，从数据目录自动检测
        if symbols is None or timeframes is None:
            try:
                available_files = os.listdir(data_dir)
                data_files = [f for f in available_files if f.endswith('.csv') or f.endswith('.parquet') or f.endswith('.csv.gz')]
                
                if not data_files:
                    logger.warning(f"在目录 {data_dir} 中未找到任何数据文件")
                    return {} # 返回空字典而不是None
                
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
                            # 假设时间框架是第二部分，移除文件扩展名
                            tf_part = os.path.splitext(parts[1])[0]
                            if tf_part.endswith('.csv'): # 处理.csv.gz的情况
                                tf_part = os.path.splitext(tf_part)[0]
                            detected_timeframes.add(tf_part)
                    timeframes = list(detected_timeframes)
                
                logger.info(f"检测到的交易对: {symbols}")
                logger.info(f"检测到的时间框架: {timeframes}")
            except FileNotFoundError:
                 logger.error(f"数据目录 {data_dir} 不存在")
                 return {}
            except Exception as detect_e:
                 logger.error(f"检测交易对和时间框架时出错: {detect_e}")
                 return {}
        
        # 记录处理结果，但不存储数据
        processed_count = 0
        failed_items = []
        
        # 处理所有交易对和时间框架的组合
        for symbol in tqdm(symbols, desc="处理交易对"):
            symbol_processed = False
            for timeframe in timeframes:
                logger.info(f"处理 {symbol} {timeframe} 数据...")
                
                # 调用处理单个文件的方法，它现在负责保存
                features_df, targets_df = self.process_data_for_symbol(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    data_dir=data_dir,
                    batch_size=batch_size
                )
                
                if features_df is not None and targets_df is not None:
                    logger.info(f"成功处理 {symbol} {timeframe} 数据")
                    symbol_processed = True
                else:
                    logger.warning(f"未能处理 {symbol} {timeframe} 数据")
                    failed_items.append(f"{symbol}_{timeframe}")
            
            if symbol_processed:
                processed_count += 1

        if failed_items:
            logger.warning(f"以下项目处理失败: {', '.join(failed_items)}")
        
        # 返回已处理交易对的数量，而不是整个数据集
        return processed_count 