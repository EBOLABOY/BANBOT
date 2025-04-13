"""
时间序列模型 - 包括ARIMA、Prophet和其他时间序列特化模型
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from prophet import Prophet
import json
from joblib import dump, load
from tqdm import tqdm

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ARIMAModel(BaseModel):
    """
    ARIMA模型类，支持自动参数选择
    """
    
    def __init__(self, 
                 name: str = "arima",
                 model_params: Dict = None,
                 prediction_horizon: int = 1,
                 target_type: str = "price_change_pct",
                 auto_order: bool = True,
                 model_dir: str = "models/saved_models"):
        """
        初始化ARIMA模型
        
        参数:
            name (str): 模型名称
            model_params (Dict): 模型参数字典
            prediction_horizon (int): 预测周期（步数）
            target_type (str): 目标变量类型
            auto_order (bool): 是否自动选择ARIMA参数
            model_dir (str): 模型保存目录
        """
        super().__init__(name, model_params, prediction_horizon, target_type, model_dir)
        
        self.auto_order = auto_order
        
        # 设置默认参数
        default_params = {
            "order": (1, 1, 1),  # (p, d, q)
            "seasonal_order": (0, 0, 0, 0),  # (P, D, Q, s)
            "trend": "c",
            "enforce_stationarity": False,
            "enforce_invertibility": False
        }
        
        # 更新参数
        if model_params:
            default_params.update(model_params)
        
        self.model_params = default_params
        self.data_frequency = None
        self.history_data = None
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], 
              validation_data: Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]] = None) -> Dict:
        """
        训练ARIMA模型
        
        参数:
            X: 训练特征数据（对于ARIMA模型，只使用时间索引）
            y: 训练目标数据
            validation_data: 可选的验证数据 (X_val, y_val)
            
        返回:
            Dict: 包含训练指标的字典
        """
        # 确保y是pandas Series，并且有日期索引
        if isinstance(y, np.ndarray):
            if isinstance(X, pd.DataFrame):
                # 使用X的索引
                y = pd.Series(y, index=X.index)
            else:
                raise ValueError("对于ARIMA模型，如果y是numpy数组，X必须是带有日期索引的DataFrame")
        
        # 保存原始目标变量数据
        self.history_data = y.copy()
        
        # 推断数据频率
        if pd.infer_freq(y.index) is not None:
            self.data_frequency = pd.infer_freq(y.index)
        else:
            # 尝试计算平均时间间隔
            time_diff = np.mean((y.index[1:] - y.index[:-1]).total_seconds())
            if time_diff < 60*60:  # 小于1小时
                self.data_frequency = f"{int(time_diff/60)}min"
            elif time_diff < 60*60*24:  # 小于1天
                self.data_frequency = f"{int(time_diff/3600)}H"
            else:
                self.data_frequency = f"{int(time_diff/86400)}D"
            
            logger.warning(f"无法推断数据频率，使用计算的平均时间间隔: {self.data_frequency}")
        
        logger.info(f"数据频率: {self.data_frequency}")
        
        # 如果启用自动参数选择
        if self.auto_order:
            logger.info("使用auto_arima自动选择ARIMA参数")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # 使用auto_arima自动选择最佳参数
                auto_model = pm.auto_arima(
                    y,
                    start_p=0, start_q=0,
                    max_p=5, max_q=5, max_d=2,
                    seasonal=False,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                
                # 获取最佳参数
                best_order = auto_model.order
                best_seasonal_order = auto_model.seasonal_order
                
                logger.info(f"自动选择的ARIMA参数 - order: {best_order}, seasonal_order: {best_seasonal_order}")
                
                # 更新模型参数
                self.model_params["order"] = best_order
                self.model_params["seasonal_order"] = best_seasonal_order
        
        # 提取参数
        order = self.model_params.pop("order")
        seasonal_order = self.model_params.pop("seasonal_order")
        
        # 训练模型
        logger.info(f"开始训练ARIMA模型 (order={order}, seasonal_order={seasonal_order})...")
        
        try:
            # 使用SARIMAX训练模型
            self.model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                **self.model_params
            )
            
            self.results = self.model.fit(disp=False)
            self.trained = True
            
            # 恢复参数（为了保存）
            self.model_params["order"] = order
            self.model_params["seasonal_order"] = seasonal_order
            
            # 计算训练指标
            train_pred = self.results.predict()
            train_metrics = {
                "aic": self.results.aic,
                "bic": self.results.bic
            }
            
            # 计算预测误差
            train_metrics["rmse"] = np.sqrt(np.mean((y - train_pred)**2))
            train_metrics["mae"] = np.mean(np.abs(y - train_pred))
            
            # 如果提供了验证数据，则计算验证指标
            if validation_data is not None:
                _, y_val = validation_data
                
                if isinstance(y_val, np.ndarray):
                    if isinstance(X, pd.DataFrame) and len(validation_data[0]) > 0:
                        y_val = pd.Series(y_val, index=validation_data[0].index)
                    else:
                        # 创建未来日期索引
                        future_index = [y.index[-1] + pd.Timedelta(self.data_frequency) * (i+1) for i in range(len(y_val))]
                        y_val = pd.Series(y_val, index=future_index)
                
                # 预测验证集
                val_pred = self.predict_from_history(steps=len(y_val))
                
                # 计算验证指标
                val_metrics = {
                    "val_rmse": np.sqrt(np.mean((y_val - val_pred)**2)),
                    "val_mae": np.mean(np.abs(y_val - val_pred))
                }
                
                # 更新训练指标
                train_metrics.update(val_metrics)
            
            # 更新元数据
            self.metadata["metrics"].update(train_metrics)
            self.metadata["arima_order"] = order
            self.metadata["arima_seasonal_order"] = seasonal_order
            
            logger.info(f"ARIMA模型训练完成，AIC: {train_metrics['aic']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
            
            return train_metrics
            
        except Exception as e:
            logger.error(f"ARIMA模型训练失败: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用模型进行预测（对于新的时间点）
        
        参数:
            X: 预测特征数据（主要用于获取索引）
            
        返回:
            np.ndarray: 预测结果
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        # 这种方法不适用于带有滞后和外部变量的模型
        # 主要用于简单预测，更复杂的情况请使用forecast方法
        if isinstance(X, pd.DataFrame):
            future_steps = len(X)
            
            # 获取预测开始日期（数据最后一天之后的下一个时间点）
            if self.history_data is not None and len(self.history_data) > 0:
                start_date = self.history_data.index[-1] + pd.Timedelta(self.data_frequency)
            else:
                start_date = X.index[0]
            
            # 创建未来日期索引
            future_index = pd.date_range(start=start_date, periods=future_steps, freq=self.data_frequency)
            
            # 进行预测
            forecast = self.results.get_forecast(steps=future_steps)
            predictions = forecast.predicted_mean
            
            # 如果预测日期与X日期不匹配，尝试重新索引
            if len(future_index) == len(X) and not future_index.equals(X.index):
                predictions = pd.Series(predictions, index=future_index).reindex(X.index)
                
            return predictions.values
        else:
            # 如果X是numpy数组，只使用其长度
            future_steps = len(X)
            forecast = self.results.get_forecast(steps=future_steps)
            return forecast.predicted_mean.values
    
    def predict_from_history(self, steps: int = 1) -> pd.Series:
        """
        从历史数据开始预测未来的步数
        
        参数:
            steps (int): 要预测的未来步数
            
        返回:
            pd.Series: 预测结果
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        # 进行预测
        forecast = self.results.get_forecast(steps=steps)
        predictions = forecast.predicted_mean
        
        return predictions
    
    def get_confidence_intervals(self, X: Union[pd.DataFrame, np.ndarray], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取预测的置信区间
        
        参数:
            X: 预测特征数据（主要用于获取索引）
            alpha (float): 显著性水平
            
        返回:
            Tuple[np.ndarray, np.ndarray]: (下界, 上界)
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        if isinstance(X, pd.DataFrame):
            future_steps = len(X)
        else:
            future_steps = len(X)
        
        # 获取预测及置信区间
        forecast = self.results.get_forecast(steps=future_steps)
        ci = forecast.conf_int(alpha=alpha)
        
        # 返回下界和上界
        return ci.iloc[:, 0].values, ci.iloc[:, 1].values

    def update(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        使用新数据更新模型
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 新的特征数据
            y (Union[pd.Series, np.ndarray]): 新的目标数据
        """
        logger.info(f"更新{self.name}模型...")
        
        # 确保y是一维数组或序列
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        elif isinstance(y, np.ndarray) and y.ndim > 1:
            y = y.ravel()
        
        # 将y转换为pandas Series以保留索引信息
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # 将新数据追加到训练数据
        if hasattr(self, 'train_data') and self.train_data is not None:
            if isinstance(self.train_data, pd.Series):
                # 确保索引没有重叠
                y = y[~y.index.isin(self.train_data.index)]
                self.train_data = pd.concat([self.train_data, y])
            else:
                self.train_data = np.append(self.train_data, y)
        else:
            self.train_data = y
        
        # 重新训练模型
        self.train(None, self.train_data)
        
        logger.info(f"{self.name}模型更新完成")
    
    def save(self, filepath: str) -> None:
        """
        保存模型到文件
        
        参数:
            filepath (str): 文件路径
        """
        logger.info(f"保存{self.name}模型到{filepath}...")
        
        if self.results is None:
            raise ValueError("模型尚未训练，无法保存")
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型及其参数
        model_data = {
            "model_params": self.model_params,
            "target_type": self.target_type,
            "name": self.name,
            "results": self.results.to_json(),
            "trained": self.trained,
            "data_frequency": self.data_frequency,
            "history_data": self.history_data.to_dict() if hasattr(self, 'history_data') else None,
            "metadata": self.metadata
        }
        
        # 保存模型数据
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=4)
        
        logger.info(f"{self.name}模型已保存")
    
    def load(self, filepath: str) -> None:
        """
        从文件加载模型
        
        参数:
            filepath (str): 文件路径
        """
        logger.info(f"从{filepath}加载{self.name}模型...")
        
        # 加载模型数据
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # 加载模型参数
        self.model_params = model_data["model_params"]
        self.target_type = model_data["target_type"]
        self.name = model_data["name"]
        
        # 加载训练数据
        if model_data["history_data"] is not None:
            if isinstance(model_data["history_data"], dict):
                self.history_data = pd.Series(model_data["history_data"])
            else:
                self.history_data = pd.Series(model_data["history_data"])
        else:
            self.history_data = None
        
        # 加载训练结果
        if model_data["results"] is not None:
            self.results = sm.tsa.statespace.sarimax.SARIMAXResults.from_json(model_data["results"])
        else:
            self.results = None
        
        # 加载训练标志
        self.trained = model_data["trained"]
        
        # 加载数据频率
        self.data_frequency = model_data["data_frequency"]
        
        # 加载元数据
        self.metadata = model_data["metadata"]
        
        logger.info(f"{self.name}模型已加载")


class ProphetModel(BaseModel):
    """
    Prophet模型类，用于时间序列预测
    """
    
    def __init__(self, 
                 name: str = "prophet",
                 model_params: Dict = None,
                 prediction_horizon: int = 1,
                 target_type: str = "price_change_pct",
                 model_dir: str = "models/saved_models"):
        """
        初始化Prophet模型
        
        参数:
            name (str): 模型名称
            model_params (Dict): 模型参数字典
            prediction_horizon (int): 预测周期（步数）
            target_type (str): 目标变量类型
            model_dir (str): 模型保存目录
        """
        super().__init__(name, model_params, prediction_horizon, target_type, model_dir)
        
        # 设置默认参数
        default_params = {
            "growth": "linear",
            "changepoints": None,
            "n_changepoints": 25,
            "changepoint_prior_scale": 0.05,
            "seasonality_mode": "additive",
            "yearly_seasonality": "auto",
            "weekly_seasonality": "auto",
            "daily_seasonality": "auto"
        }
        
        # 更新参数
        if model_params:
            default_params.update(model_params)
        
        self.model_params = default_params
        self.data_frequency = None
        self.history_df = None
        self.feature_columns = None
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], 
              validation_data: Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]] = None) -> Dict:
        """
        训练Prophet模型
        
        参数:
            X: 训练特征数据（可选的外部回归变量）
            y: 训练目标数据
            validation_data: 可选的验证数据 (X_val, y_val)
            
        返回:
            Dict: 包含训练指标的字典
        """
        # 确保y是pandas Series，并且有日期索引
        if isinstance(y, np.ndarray):
            if isinstance(X, pd.DataFrame):
                # 使用X的索引
                y = pd.Series(y, index=X.index)
            else:
                raise ValueError("对于Prophet模型，如果y是numpy数组，X必须是带有日期索引的DataFrame")
        
        # 推断数据频率
        if pd.infer_freq(y.index) is not None:
            self.data_frequency = pd.infer_freq(y.index)
        else:
            # 尝试计算平均时间间隔
            time_diff = np.mean((y.index[1:] - y.index[:-1]).total_seconds())
            if time_diff < 60*60:  # 小于1小时
                self.data_frequency = f"{int(time_diff/60)}min"
            elif time_diff < 60*60*24:  # 小于1天
                self.data_frequency = f"{int(time_diff/3600)}H"
            else:
                self.data_frequency = f"{int(time_diff/86400)}D"
            
            logger.warning(f"无法推断数据频率，使用计算的平均时间间隔: {self.data_frequency}")
        
        logger.info(f"数据频率: {self.data_frequency}")
        
        # 准备Prophet训练数据
        df = pd.DataFrame({
            'ds': y.index,
            'y': y.values
        })
        
        # 保存训练数据
        self.history_df = df.copy()
        
        # 如果X是DataFrame且有额外特征（不只是日期索引），添加为外部回归变量
        if isinstance(X, pd.DataFrame) and X.shape[1] > 0:
            self.feature_columns = X.columns.tolist()
            for col in X.columns:
                df[col] = X[col].values
        
        # 创建和训练模型
        logger.info("开始训练Prophet模型...")
        
        try:
            # 从模型参数中提取额外的回归变量
            regressors = self.model_params.pop("extra_regressors", [])
            
            # 创建模型
            self.model = Prophet(**self.model_params)
            
            # 添加外部回归变量
            if self.feature_columns:
                for col in self.feature_columns:
                    self.model.add_regressor(col)
            
            # 添加指定的额外回归变量
            for regressor in regressors:
                if regressor in df.columns and regressor != 'ds' and regressor != 'y':
                    if regressor not in (self.feature_columns or []):
                        self.model.add_regressor(regressor)
            
            # 训练模型
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(df)
            
            self.trained = True
            
            # 计算训练指标
            train_pred = self.model.predict(df)['yhat'].values
            train_metrics = {
                "rmse": np.sqrt(np.mean((df['y'].values - train_pred)**2)),
                "mae": np.mean(np.abs(df['y'].values - train_pred))
            }
            
            # 如果提供了验证数据，则计算验证指标
            if validation_data is not None:
                X_val, y_val = validation_data
                
                if isinstance(y_val, np.ndarray):
                    if isinstance(X_val, pd.DataFrame):
                        y_val = pd.Series(y_val, index=X_val.index)
                    else:
                        # 创建未来日期索引
                        future_index = [y.index[-1] + pd.Timedelta(self.data_frequency) * (i+1) for i in range(len(y_val))]
                        y_val = pd.Series(y_val, index=future_index)
                
                # 准备验证数据
                val_df = pd.DataFrame({
                    'ds': y_val.index,
                    'y': y_val.values
                })
                
                # 如果有外部回归变量，添加到验证数据
                if isinstance(X_val, pd.DataFrame) and X_val.shape[1] > 0:
                    for col in X_val.columns:
                        if col in self.feature_columns:
                            val_df[col] = X_val[col].values
                
                # 预测验证集
                val_pred = self.model.predict(val_df)['yhat'].values
                
                # 计算验证指标
                val_metrics = {
                    "val_rmse": np.sqrt(np.mean((val_df['y'].values - val_pred)**2)),
                    "val_mae": np.mean(np.abs(val_df['y'].values - val_pred))
                }
                
                # 更新训练指标
                train_metrics.update(val_metrics)
            
            # 更新元数据
            self.metadata["metrics"].update(train_metrics)
            
            logger.info(f"Prophet模型训练完成，RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}")
            
            return train_metrics
            
        except Exception as e:
            logger.error(f"Prophet模型训练失败: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        使用模型进行预测
        
        参数:
            X: 预测特征数据
            
        返回:
            np.ndarray: 预测结果
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        if isinstance(X, pd.DataFrame):
            # 准备预测数据
            future_df = pd.DataFrame({
                'ds': X.index
            })
            
            # 如果有外部回归变量，添加到预测数据
            if self.feature_columns:
                for col in self.feature_columns:
                    if col in X.columns:
                        future_df[col] = X[col].values
            
            # 进行预测
            forecast = self.model.predict(future_df)
            return forecast['yhat'].values
        else:
            # 如果X是numpy数组，创建未来日期序列
            if self.history_df is not None:
                last_date = self.history_df['ds'].iloc[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(self.data_frequency),
                    periods=len(X),
                    freq=self.data_frequency
                )
                
                future_df = pd.DataFrame({
                    'ds': future_dates
                })
                
                # 进行预测
                forecast = self.model.predict(future_df)
                return forecast['yhat'].values
            else:
                raise ValueError("无法预测，没有历史数据来创建日期序列")
    
    def predict_with_intervals(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        进行预测并返回置信区间
        
        参数:
            X: 预测特征数据
            
        返回:
            pd.DataFrame: 包含预测值和置信区间的DataFrame
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        if isinstance(X, pd.DataFrame):
            # 准备预测数据
            future_df = pd.DataFrame({
                'ds': X.index
            })
            
            # 如果有外部回归变量，添加到预测数据
            if self.feature_columns:
                for col in self.feature_columns:
                    if col in X.columns:
                        future_df[col] = X[col].values
        else:
            # 如果X是numpy数组，创建未来日期序列
            if self.history_df is not None:
                last_date = self.history_df['ds'].iloc[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(self.data_frequency),
                    periods=len(X),
                    freq=self.data_frequency
                )
                
                future_df = pd.DataFrame({
                    'ds': future_dates
                })
            else:
                raise ValueError("无法预测，没有历史数据来创建日期序列")
        
        # 进行预测
        forecast = self.model.predict(future_df)
        
        # 返回预测结果，包括预测值和置信区间
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def update(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """
        使用新数据更新模型
        
        参数:
            X (Union[pd.DataFrame, np.ndarray]): 新的特征数据
            y (Union[pd.Series, np.ndarray]): 新的目标数据
        """
        logger.info(f"更新{self.name}模型...")
        
        # 准备新数据
        new_df = pd.DataFrame({
            'ds': X.index,
            'y': y.values
        })
        
        # 将新数据追加到训练数据
        if hasattr(self, 'history_df') and self.history_df is not None:
            # 确保没有重复的日期
            new_df = new_df[~new_df['ds'].isin(self.history_df['ds'])]
            df = pd.concat([self.history_df, new_df])
        else:
            df = new_df
        
        # 重新训练模型
        self.history_df = df
        self.train(X, y)
        
        logger.info(f"{self.name}模型更新完成")
    
    def save(self, filepath: str) -> None:
        """
        保存模型到文件
        
        参数:
            filepath (str): 文件路径
        """
        logger.info(f"保存{self.name}模型到{filepath}...")
        
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型
        model_dir = os.path.dirname(filepath)
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        
        # 保存Prophet模型
        model_json_path = os.path.join(model_dir, f"{model_name}_prophet.json")
        with open(model_json_path, 'w') as f:
            json.dump(self.model.to_json(), f)
        
        # 保存训练数据
        train_data_path = os.path.join(model_dir, f"{model_name}_train_data.csv")
        if hasattr(self, 'history_df') and self.history_df is not None:
            self.history_df.to_csv(train_data_path, index=False)
        
        # 保存模型元数据
        model_data = {
            "model_params": self.model_params,
            "target_type": self.target_type,
            "name": self.name,
            "model_path": model_json_path,
            "train_data_path": train_data_path,
            "data_frequency": self.data_frequency,
            "history_df": self.history_df.to_dict() if hasattr(self, 'history_df') else None,
            "metadata": self.metadata
        }
        
        # 保存元数据
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=4)
        
        logger.info(f"{self.name}模型已保存")
    
    def load(self, filepath: str) -> None:
        """
        从文件加载模型
        
        参数:
            filepath (str): 文件路径
        """
        logger.info(f"从{filepath}加载{self.name}模型...")
        
        # 加载模型元数据
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # 加载模型参数
        self.model_params = model_data["model_params"]
        self.target_type = model_data["target_type"]
        self.name = model_data["name"]
        
        # 加载Prophet模型
        model_path = model_data["model_path"]
        with open(model_path, 'r') as f:
            model_json = json.load(f)
        
        self.model = Prophet.from_json(model_json)
        
        # 加载训练数据
        train_data_path = model_data["train_data_path"]
        if os.path.exists(train_data_path):
            self.history_df = pd.read_csv(train_data_path)
            self.history_df['ds'] = pd.to_datetime(self.history_df['ds'])
        else:
            self.history_df = None
        
        # 加载数据频率
        self.data_frequency = model_data["data_frequency"]
        
        # 加载元数据
        self.metadata = model_data["metadata"]
        
        logger.info(f"{self.name}模型已加载")


# 工厂函数
def create_time_series_model(model_type: str, target_type: str = "price_change_pct", model_params: Dict = None, name: str = None) -> BaseModel:
    """
    创建时间序列模型
    
    参数:
        model_type (str): 模型类型，如"arima"或"prophet"
        target_type (str): 目标类型，如"price_change_pct"或"direction"
        model_params (Dict): 模型参数
        name (str): 模型名称
        
    返回:
        BaseModel: 时间序列模型实例
    """
    model_type = model_type.lower()
    
    if model_type == "arima":
        name = name or "ARIMA"
        return ARIMAModel(target_type=target_type, model_params=model_params, name=name)
    elif model_type == "prophet":
        name = name or "Prophet"
        return ProphetModel(target_type=target_type, model_params=model_params, name=name)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}") 