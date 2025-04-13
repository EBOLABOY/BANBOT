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

from src.utils.logger import get_logger
from src.utils.config import load_config
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
        
        # 初始化特征计算器
        self.tech_indicators = TechnicalIndicators()
        self.microstructure = MicrostructureFeatures()
        
        # 缩放器
        self.scalers = {}
        
        logger.info("特征工程管道已初始化")
    
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
        
        # 转换窗口大小为列表
        window_sizes_list = []
        for size_group in self.window_sizes.values():
            window_sizes_list.extend(size_group)
        window_sizes_list = sorted(list(set(window_sizes_list)))
        
        # 按特征组计算特征
        if "price_based" in feature_groups:
            result_df = self.tech_indicators.calculate_price_features(result_df)
        
        if "volume_based" in feature_groups:
            result_df = self.tech_indicators.calculate_volume_features(result_df)
        
        if "volatility" in feature_groups:
            result_df = self.tech_indicators.calculate_volatility_features(result_df)
        
        if "trend" in feature_groups:
            result_df = self.tech_indicators.calculate_trend_features(result_df)
        
        if "momentum" in feature_groups:
            result_df = self.tech_indicators.calculate_momentum_features(result_df)
        
        if "market_microstructure" in feature_groups:
            # 计算市场微观结构特征（如果有相关数据）
            has_bid_ask = all(col in result_df.columns for col in ['bid', 'ask'])
            
            if has_bid_ask:
                result_df = self.microstructure.calculate_bid_ask_features(result_df)
                result_df = self.microstructure.calculate_liquidity_features(result_df, window_sizes_list)
            
            # 计算其他微观结构特征（基于OHLCV数据）
            result_df = self.microstructure.calculate_order_flow_features(result_df, window_sizes_list)
            result_df = self.microstructure.calculate_volatility_clustering(result_df, window_sizes_list)
            result_df = self.microstructure.calculate_price_impact(result_df, window_sizes_list)
        
        # 计算技术指标（如果未在上述特征组中处理）
        result_df = self.tech_indicators.calculate_indicators(
            result_df, 
            indicators=['MACD', 'RSI', 'STOCH', 'BBANDS', 'ATR', 'ADX', 'CCI'],
            window_sizes=window_sizes_list
        )
        
        logger.info(f"已计算 {len(result_df.columns) - len(df.columns)} 个特征")
        return result_df
    
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
            X_selected_df = pd.DataFrame(X_selected, index=X.index, columns=selected_features)
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
            X_selected = X[selected_features].values
        
        # 返回选择后的特征矩阵和特征名称
        X_selected_df = pd.DataFrame(X_selected, index=X.index, columns=selected_features)
        
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
    
    def process_data_for_symbol(self, symbol, timeframe, start_date=None, end_date=None, data_dir="data/processed/merged"):
        """
        为单个交易对处理数据
        
        参数:
            symbol: 交易对符号
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            data_dir: 数据目录
            
        返回:
            处理后的特征DataFrame和目标变量
        """
        # 构建文件名模式
        pattern = f"{symbol}_{timeframe}"
        if start_date:
            pattern += f"_{start_date.replace('-', '')}"
        if end_date:
            pattern += f"_{end_date.replace('-', '')}"
        
        # 查找匹配的文件
        files = [f for f in os.listdir(data_dir) if pattern in f]
        
        if not files:
            logger.warning(f"找不到匹配的数据文件: {pattern}")
            return None, None
        
        # 加载数据
        filepath = os.path.join(data_dir, files[0])
        df = pd.read_csv(filepath)
        
        # 设置时间索引
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        
        # 筛选日期范围
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]
        
        # 计算特征
        features_df = self.compute_features(df)
        
        # 创建目标变量
        targets_df = self.create_target_variables(features_df)
        
        # 处理特征
        processed_features = self.preprocess_features(features_df)
        
        logger.info(f"已处理 {symbol} 的 {timeframe} 数据，共 {len(processed_features)} 条记录和 {len(processed_features.columns)} 个特征")
        
        return processed_features, targets_df
    
    def process_all_data(self, symbols=None, timeframes=None, start_date=None, end_date=None, data_dir="data/processed/merged"):
        """
        处理所有交易对的数据
        
        参数:
            symbols: 交易对列表
            timeframes: 时间框架列表
            start_date: 开始日期
            end_date: 结束日期
            data_dir: 数据目录
            
        返回:
            处理后的特征和目标变量字典
        """
        # 如果未指定交易对和时间框架，读取所有可用的文件
        if symbols is None or timeframes is None:
            available_files = os.listdir(data_dir)
            symbol_timeframe_pairs = []
            
            for filename in available_files:
                parts = filename.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = parts[1]
                    symbol_timeframe_pairs.append((symbol, timeframe))
            
            # 去重
            symbol_timeframe_pairs = list(set(symbol_timeframe_pairs))
            
            if not symbols:
                symbols = list(set([pair[0] for pair in symbol_timeframe_pairs]))
            
            if not timeframes:
                timeframes = list(set([pair[1] for pair in symbol_timeframe_pairs]))
        
        # 处理结果存储
        processed_data = {}
        
        # 处理每个交易对和时间框架组合
        for symbol in tqdm(symbols, desc="处理交易对"):
            processed_data[symbol] = {}
            
            for timeframe in timeframes:
                try:
                    features_df, targets_df = self.process_data_for_symbol(
                        symbol, timeframe, start_date, end_date, data_dir
                    )
                    
                    if features_df is not None and targets_df is not None:
                        processed_data[symbol][timeframe] = {
                            'features': features_df,
                            'targets': targets_df
                        }
                        
                        # 保存处理后的数据
                        output_dir = os.path.join(self.features_path, symbol)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 确定文件名后缀
                        date_suffix = ""
                        if start_date:
                            date_suffix += f"_{start_date.replace('-', '')}"
                        if end_date:
                            date_suffix += f"_{end_date.replace('-', '')}"
                        
                        # 保存特征和目标变量
                        features_filepath = os.path.join(output_dir, f"features_{timeframe}{date_suffix}.csv")
                        targets_filepath = os.path.join(output_dir, f"targets_{timeframe}{date_suffix}.csv")
                        
                        features_df.to_csv(features_filepath)
                        targets_df.to_csv(targets_filepath)
                        
                        logger.info(f"已保存 {symbol} 的 {timeframe} 特征和目标变量")
                
                except Exception as e:
                    logger.error(f"处理 {symbol} 的 {timeframe} 数据时出错: {e}")
        
        return processed_data
        
    def load_data(self, filepath):
        """
        加载数据文件
        
        参数:
            filepath: 文件路径
            
        返回:
            加载的DataFrame
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(filepath)
            
            # 尝试将timestamp列转换为日期时间格式并设为索引
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            
            logger.debug(f"已加载数据文件 {filepath}，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"加载数据文件 {filepath} 时出错: {e}")
            return pd.DataFrame() 