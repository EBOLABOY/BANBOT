"""
深度学习模型 - 包括RNN、LSTM和其他神经网络模型
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Any
from datetime import datetime
import pickle
import json
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, BatchNormalization
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l1, l2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DeepLearningModel(BaseModel):
    """
    深度学习模型基类，提供通用的深度学习模型功能
    """
    
    def __init__(self, 
                 name: str = "deep_learning",
                 model_params: Dict = None,
                 prediction_horizon: int = 1,
                 target_type: str = "price_change_pct",
                 input_sequence_length: int = 10,
                 model_dir: str = "models/saved_models"):
        """
        初始化深度学习模型
        
        参数:
            name (str): 模型名称
            model_params (Dict): 模型参数字典
            prediction_horizon (int): 预测周期（步数）
            target_type (str): 目标变量类型
            input_sequence_length (int): 输入序列长度（时间步）
            model_dir (str): 模型保存目录
        """
        if not TF_AVAILABLE:
            raise ImportError("未找到TensorFlow库，请安装tensorflow: pip install tensorflow")
            
        super().__init__(name, model_params, prediction_horizon, target_type, model_dir)
        
        self.input_sequence_length = input_sequence_length
        
        # 设置默认参数
        default_params = {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse" if target_type in ["price_change_pct", "volatility"] else "binary_crossentropy",
            "metrics": ["mae"] if target_type in ["price_change_pct", "volatility"] else ["accuracy"],
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
            "validation_split": 0.1
        }
        
        # 更新参数
        if model_params:
            default_params.update(model_params)
        
        self.model_params = default_params
        self.keras_model = None
        self.history = None
        self.scaler = None
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        构建神经网络模型（子类必须实现）
        
        参数:
            input_shape (Tuple[int, int]): 输入形状 (时间步, 特征数)
            
        返回:
            tf.keras.Model: 构建的Keras模型
        """
        raise NotImplementedError("子类必须实现_build_model方法")
    
    def _prepare_sequences(self, X: Union[pd.DataFrame, np.ndarray], 
                         y: Optional[Union[pd.Series, np.ndarray]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        将输入数据转换为序列格式
        
        参数:
            X: 输入特征数据
            y: 目标数据（可选）
            
        返回:
            Tuple[np.ndarray, Optional[np.ndarray]]: (X序列, y)
        """
        # 转换为numpy数组
        if isinstance(X, pd.DataFrame):
            X_data = X.values
        else:
            X_data = X
        
        # 创建序列
        X_seq = []
        y_seq = []
        
        for i in range(len(X_data) - self.input_sequence_length):
            X_seq.append(X_data[i:i+self.input_sequence_length])
            
            if y is not None:
                if isinstance(y, pd.Series):
                    y_seq.append(y.iloc[i+self.input_sequence_length])
                else:
                    y_seq.append(y[i+self.input_sequence_length])
        
        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        else:
            return X_seq, None
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray], 
              validation_data: Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]] = None) -> Dict:
        """
        训练深度学习模型
        
        参数:
            X: 训练特征数据
            y: 训练目标数据
            validation_data: 可选的验证数据 (X_val, y_val)
            
        返回:
            Dict: 包含训练指标的字典
        """
        # 保存特征名称（如果可用）
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # 转换为序列格式
        X_seq, y_seq = self._prepare_sequences(X, y)
        
        # 如果提供了验证数据，也转换为序列格式
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
            val_data = (X_val_seq, y_val_seq)
        else:
            val_data = None
        
        # 获取输入形状
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        
        # 构建模型
        self.keras_model = self._build_model(input_shape)
        
        # 编译模型
        optimizer = self.model_params.get("optimizer", "adam")
        if optimizer == "adam":
            optimizer = Adam(learning_rate=self.model_params.get("learning_rate", 0.001))
        
        self.keras_model.compile(
            optimizer=optimizer,
            loss=self.model_params.get("loss", "mse"),
            metrics=self.model_params.get("metrics", ["mae"])
        )
        
        # 准备训练回调
        callbacks = []
        
        # 早停
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.model_params.get("early_stopping_patience", 10),
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # 学习率降低
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.model_params.get("reduce_lr_patience", 5),
            min_lr=1e-6
        )
        callbacks.append(reduce_lr)
        
        # 模型检查点
        checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.name}_best.h5")
        
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        callbacks.append(checkpoint)
        
        # 训练模型
        logger.info(f"开始训练{self.__class__.__name__}...")
        
        # 设置验证数据
        if val_data is not None:
            validation_split = 0
            validation_data = val_data
        else:
            validation_split = self.model_params.get("validation_split", 0.1)
            validation_data = None
        
        # 训练
        self.history = self.keras_model.fit(
            X_seq, y_seq,
            epochs=self.model_params.get("epochs", 100),
            batch_size=self.model_params.get("batch_size", 32),
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.trained = True
        
        # 计算训练指标
        train_loss = self.history.history['loss'][-1]
        train_metrics = {"train_loss": train_loss}
        
        # 添加其他训练指标
        metrics = self.model_params.get("metrics", [])
        for metric in metrics:
            if metric in self.history.history:
                train_metrics[f"train_{metric}"] = self.history.history[metric][-1]
        
        # 添加验证指标
        if "val_loss" in self.history.history:
            train_metrics["val_loss"] = self.history.history['val_loss'][-1]
            
            for metric in metrics:
                val_metric = f"val_{metric}"
                if val_metric in self.history.history:
                    train_metrics[val_metric] = self.history.history[val_metric][-1]
        
        # 更新元数据
        self.metadata["metrics"].update(train_metrics)
        
        logger.info(f"{self.__class__.__name__}训练完成，训练损失: {train_loss:.4f}")
        
        return train_metrics
    
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
        
        # 转换为序列格式
        X_seq, _ = self._prepare_sequences(X)
        
        # 使用模型预测
        predictions = self.keras_model.predict(X_seq)
        
        # 对于分类问题，转换为类别
        if self.target_type == "direction" and len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # 多分类情况
            predictions = np.argmax(predictions, axis=1)
        elif self.target_type == "direction":
            # 二分类情况
            predictions = (predictions > 0.5).astype(int)
        
        # 补充序列长度导致的缺失预测值
        pad_length = self.input_sequence_length
        padded_predictions = np.zeros(len(X))
        padded_predictions[:pad_length] = np.nan
        padded_predictions[pad_length:] = predictions.flatten()
        
        return padded_predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预测概率（分类模型使用）
        
        参数:
            X: 预测特征数据
            
        返回:
            np.ndarray: 预测概率
        """
        if not self.trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        if self.target_type != "direction":
            raise NotImplementedError("概率预测仅适用于分类模型")
        
        # 转换为序列格式
        X_seq, _ = self._prepare_sequences(X)
        
        # 使用模型预测
        probas = self.keras_model.predict(X_seq)
        
        # 补充序列长度导致的缺失预测值
        pad_length = self.input_sequence_length
        padded_probas = np.zeros((len(X), probas.shape[1] if len(probas.shape) > 1 else 1))
        padded_probas[:pad_length] = np.nan
        padded_probas[pad_length:] = probas
        
        return padded_probas
    
    def save(self, filepath: str = None, include_model: bool = True) -> str:
        """
        保存模型到指定路径
        
        参数:
            filepath (str): 保存路径
            include_model (bool): 是否包含模型
            
        返回:
            str: 保存的文件路径
        """
        if not self.trained and include_model:
            logger.warning("模型尚未训练，无法保存模型对象")
            include_model = False
        
        # 如果未指定文件路径，则自动生成
        if filepath is None:
            # 创建目录（如果不存在）
            os.makedirs(self.model_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            horizon = self.prediction_horizon
            target = self.target_type.split("_")[0]
            filename = f"{self.name}_{target}_h{horizon}_{timestamp}.pkl"
            
            filepath = os.path.join(self.model_dir, filename)
        
        # 更新元数据中的保存时间
        self.metadata["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存对象
        if include_model:
            # 保存Keras模型
            keras_path = filepath.replace(".pkl", "_keras.h5")
            self.keras_model.save(keras_path)
            
            # 保存其他对象
            with open(filepath, "wb") as f:
                pickle.dump({
                    "feature_names": self.feature_names,
                    "metadata": self.metadata,
                    "model_params": self.model_params,
                    "input_sequence_length": self.input_sequence_length,
                    "trained": self.trained,
                    "class_name": self.__class__.__name__,
                    "keras_model_path": keras_path
                }, f)
            
            logger.info(f"模型已保存至 {filepath}")
        else:
            # 仅保存元数据
            metadata_path = filepath.replace(".pkl", "_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            
            logger.info(f"模型元数据已保存至 {metadata_path}")
            return metadata_path
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'DeepLearningModel':
        """
        从文件加载模型
        
        参数:
            filepath (str): 模型文件路径
            
        返回:
            DeepLearningModel: 加载的模型实例
        """
        if not TF_AVAILABLE:
            raise ImportError("未找到TensorFlow库，请安装tensorflow: pip install tensorflow")
        
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            
            # 获取模型类
            class_name = data.get("class_name", cls.__name__)
            
            # 导入模型模块
            import importlib
            
            # 查找模型类
            if class_name != cls.__name__:
                try:
                    module = importlib.import_module("models.deep_learning_models")
                    model_class = getattr(module, class_name)
                except (ImportError, AttributeError):
                    logger.warning(f"找不到类 {class_name}，使用基类加载")
                    model_class = cls
            else:
                model_class = cls
            
            # 创建模型实例
            metadata = data.get("metadata", {})
            model_instance = model_class(
                name=filepath.split("/")[-1].split("_")[0],
                model_params=data.get("model_params", {}),
                prediction_horizon=metadata.get("prediction_horizon", 1),
                target_type=metadata.get("target_type", "price_change_pct"),
                input_sequence_length=data.get("input_sequence_length", 10)
            )
            
            # 加载Keras模型
            keras_path = data.get("keras_model_path")
            if keras_path and os.path.exists(keras_path):
                model_instance.keras_model = load_model(keras_path)
            else:
                logger.warning(f"找不到Keras模型文件: {keras_path}")
            
            # 恢复模型状态
            model_instance.feature_names = data.get("feature_names")
            model_instance.metadata = metadata
            model_instance.trained = data.get("trained", False)
            
            logger.info(f"已从 {filepath} 加载模型")
            return model_instance
            
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 8), save_path: str = None) -> None:
        """
        绘制训练历史
        
        参数:
            figsize: 图表大小
            save_path: 保存路径（可选）
        """
        if not self.trained or self.history is None:
            logger.warning("模型尚未训练，无法绘制训练历史")
            return
        
        plt.figure(figsize=figsize)
        
        # 绘制损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.history.history['loss'], label='训练损失')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.ylabel('损失')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        # 绘制指标曲线
        metrics = self.model_params.get("metrics", [])
        if metrics:
            plt.subplot(2, 1, 2)
            for metric in metrics:
                if metric in self.history.history:
                    plt.plot(self.history.history[metric], label=f'训练 {metric}')
                val_metric = f'val_{metric}'
                if val_metric in self.history.history:
                    plt.plot(self.history.history[val_metric], label=f'验证 {metric}')
            plt.title('模型评估指标')
            plt.ylabel('指标值')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练历史图表已保存至 {save_path}")
        else:
            plt.show()


class SimpleRNNModel(DeepLearningModel):
    """
    简单RNN模型
    """
    
    def __init__(self, 
                 name: str = "simple_rnn",
                 model_params: Dict = None,
                 prediction_horizon: int = 1,
                 target_type: str = "price_change_pct",
                 input_sequence_length: int = 10,
                 model_dir: str = "models/saved_models"):
        """
        初始化简单RNN模型
        
        参数:
            name (str): 模型名称
            model_params (Dict): 模型参数字典
            prediction_horizon (int): 预测周期（步数）
            target_type (str): 目标变量类型
            input_sequence_length (int): 输入序列长度（时间步）
            model_dir (str): 模型保存目录
        """
        super().__init__(name, model_params, prediction_horizon, target_type, input_sequence_length, model_dir)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        构建SimpleRNN模型
        
        参数:
            input_shape (Tuple[int, int]): 输入形状 (时间步, 特征数)
            
        返回:
            tf.keras.Model: 构建的Keras模型
        """
        # 获取模型参数
        units = self.model_params.get("rnn_units", [64, 32])
        dropout_rate = self.model_params.get("dropout_rate", 0.2)
        l2_reg = self.model_params.get("l2_reg", 0.001)
        
        # 创建Sequential模型
        model = Sequential()
        
        # 添加RNN层
        if len(units) > 1:
            # 多层RNN，除最后一层外，返回序列
            for i, unit in enumerate(units[:-1]):
                model.add(SimpleRNN(
                    unit,
                    return_sequences=True,
                    input_shape=input_shape,
                    kernel_regularizer=l2(l2_reg)
                ))
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))
            
            # 最后一层RNN，不返回序列
            model.add(SimpleRNN(
                units[-1],
                return_sequences=False,
                kernel_regularizer=l2(l2_reg)
            ))
        else:
            # 单层RNN
            model.add(SimpleRNN(
                units[0],
                input_shape=input_shape,
                kernel_regularizer=l2(l2_reg)
            ))
        
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # 添加输出层
        if self.target_type == "direction":
            # 分类问题
            model.add(Dense(1, activation='sigmoid'))
        else:
            # 回归问题
            model.add(Dense(1, activation='linear'))
        
        return model


class LSTMModel(DeepLearningModel):
    """
    LSTM模型
    """
    
    def __init__(self, 
                 name: str = "lstm",
                 model_params: Dict = None,
                 prediction_horizon: int = 1,
                 target_type: str = "price_change_pct",
                 input_sequence_length: int = 10,
                 model_dir: str = "models/saved_models"):
        """
        初始化LSTM模型
        
        参数:
            name (str): 模型名称
            model_params (Dict): 模型参数字典
            prediction_horizon (int): 预测周期（步数）
            target_type (str): 目标变量类型
            input_sequence_length (int): 输入序列长度（时间步）
            model_dir (str): 模型保存目录
        """
        super().__init__(name, model_params, prediction_horizon, target_type, input_sequence_length, model_dir)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        构建LSTM模型
        
        参数:
            input_shape (Tuple[int, int]): 输入形状 (时间步, 特征数)
            
        返回:
            tf.keras.Model: 构建的Keras模型
        """
        # 获取模型参数
        units = self.model_params.get("lstm_units", [128, 64])
        dropout_rate = self.model_params.get("dropout_rate", 0.2)
        recurrent_dropout = self.model_params.get("recurrent_dropout", 0.0)
        l2_reg = self.model_params.get("l2_reg", 0.001)
        
        # 创建Sequential模型
        model = Sequential()
        
        # 添加LSTM层
        if len(units) > 1:
            # 多层LSTM，除最后一层外，返回序列
            for i, unit in enumerate(units[:-1]):
                model.add(LSTM(
                    unit,
                    return_sequences=True,
                    input_shape=input_shape,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l2(l2_reg)
                ))
                model.add(BatchNormalization())
            
            # 最后一层LSTM，不返回序列
            model.add(LSTM(
                units[-1],
                return_sequences=False,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l2(l2_reg)
            ))
        else:
            # 单层LSTM
            model.add(LSTM(
                units[0],
                input_shape=input_shape,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l2(l2_reg)
            ))
        
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # 添加全连接层
        dense_units = self.model_params.get("dense_units", [32])
        for unit in dense_units:
            model.add(Dense(unit, activation='relu', kernel_regularizer=l2(l2_reg)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # 添加输出层
        if self.target_type == "direction":
            # 分类问题
            model.add(Dense(1, activation='sigmoid'))
        else:
            # 回归问题
            model.add(Dense(1, activation='linear'))
        
        return model


class GRUModel(DeepLearningModel):
    """
    GRU模型
    """
    
    def __init__(self, 
                 name: str = "gru",
                 model_params: Dict = None,
                 prediction_horizon: int = 1,
                 target_type: str = "price_change_pct",
                 input_sequence_length: int = 10,
                 model_dir: str = "models/saved_models"):
        """
        初始化GRU模型
        
        参数:
            name (str): 模型名称
            model_params (Dict): 模型参数字典
            prediction_horizon (int): 预测周期（步数）
            target_type (str): 目标变量类型
            input_sequence_length (int): 输入序列长度（时间步）
            model_dir (str): 模型保存目录
        """
        super().__init__(name, model_params, prediction_horizon, target_type, input_sequence_length, model_dir)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        构建GRU模型
        
        参数:
            input_shape (Tuple[int, int]): 输入形状 (时间步, 特征数)
            
        返回:
            tf.keras.Model: 构建的Keras模型
        """
        # 获取模型参数
        units = self.model_params.get("gru_units", [128, 64])
        dropout_rate = self.model_params.get("dropout_rate", 0.2)
        recurrent_dropout = self.model_params.get("recurrent_dropout", 0.0)
        l2_reg = self.model_params.get("l2_reg", 0.001)
        
        # 创建Sequential模型
        model = Sequential()
        
        # 添加GRU层
        if len(units) > 1:
            # 多层GRU，除最后一层外，返回序列
            for i, unit in enumerate(units[:-1]):
                model.add(GRU(
                    unit,
                    return_sequences=True,
                    input_shape=input_shape,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l2(l2_reg)
                ))
                model.add(BatchNormalization())
            
            # 最后一层GRU，不返回序列
            model.add(GRU(
                units[-1],
                return_sequences=False,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l2(l2_reg)
            ))
        else:
            # 单层GRU
            model.add(GRU(
                units[0],
                input_shape=input_shape,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l2(l2_reg)
            ))
        
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # 添加全连接层
        dense_units = self.model_params.get("dense_units", [32])
        for unit in dense_units:
            model.add(Dense(unit, activation='relu', kernel_regularizer=l2(l2_reg)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # 添加输出层
        if self.target_type == "direction":
            # 分类问题
            model.add(Dense(1, activation='sigmoid'))
        else:
            # 回归问题
            model.add(Dense(1, activation='linear'))
        
        return model


class CNNLSTMModel(DeepLearningModel):
    """
    CNN-LSTM混合模型，先使用CNN提取特征，再使用LSTM处理时序
    """
    
    def __init__(self, 
                 name: str = "cnn_lstm",
                 model_params: Dict = None,
                 prediction_horizon: int = 1,
                 target_type: str = "price_change_pct",
                 input_sequence_length: int = 10,
                 model_dir: str = "models/saved_models"):
        """
        初始化CNN-LSTM模型
        
        参数:
            name (str): 模型名称
            model_params (Dict): 模型参数字典
            prediction_horizon (int): 预测周期（步数）
            target_type (str): 目标变量类型
            input_sequence_length (int): 输入序列长度（时间步）
            model_dir (str): 模型保存目录
        """
        super().__init__(name, model_params, prediction_horizon, target_type, input_sequence_length, model_dir)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        构建CNN-LSTM混合模型
        
        参数:
            input_shape (Tuple[int, int]): 输入形状 (时间步, 特征数)
            
        返回:
            tf.keras.Model: 构建的Keras模型
        """
        # 获取模型参数
        filters = self.model_params.get("cnn_filters", [64, 128])
        kernel_size = self.model_params.get("kernel_size", 3)
        lstm_units = self.model_params.get("lstm_units", [64])
        dropout_rate = self.model_params.get("dropout_rate", 0.2)
        l2_reg = self.model_params.get("l2_reg", 0.001)
        
        # 创建Sequential模型
        model = Sequential()
        
        # 添加卷积层
        model.add(Conv1D(
            filters=filters[0],
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            input_shape=input_shape,
            kernel_regularizer=l2(l2_reg)
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        
        # 添加额外的卷积层
        for filter_size in filters[1:]:
            model.add(Conv1D(
                filters=filter_size,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                kernel_regularizer=l2(l2_reg)
            ))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
        
        # 添加LSTM层
        if len(lstm_units) > 1:
            # 多层LSTM，除最后一层外，返回序列
            for i, unit in enumerate(lstm_units[:-1]):
                model.add(LSTM(
                    unit,
                    return_sequences=True,
                    dropout=dropout_rate,
                    kernel_regularizer=l2(l2_reg)
                ))
                model.add(BatchNormalization())
            
            # 最后一层LSTM，不返回序列
            model.add(LSTM(
                lstm_units[-1],
                return_sequences=False,
                dropout=dropout_rate,
                kernel_regularizer=l2(l2_reg)
            ))
        else:
            # 单层LSTM
            model.add(LSTM(
                lstm_units[0],
                dropout=dropout_rate,
                kernel_regularizer=l2(l2_reg)
            ))
        
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # 添加全连接层
        dense_units = self.model_params.get("dense_units", [32])
        for unit in dense_units:
            model.add(Dense(unit, activation='relu', kernel_regularizer=l2(l2_reg)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # 添加输出层
        if self.target_type == "direction":
            # 分类问题
            model.add(Dense(1, activation='sigmoid'))
        else:
            # 回归问题
            model.add(Dense(1, activation='linear'))
        
        return model 


def create_deep_learning_model(
    model_type: str,
    target_type: str,
    model_params: Dict = None,
    prediction_horizon: int = 1,
    input_sequence_length: int = 10,
    name: Optional[str] = None,
    model_dir: str = "models/saved_models"
) -> DeepLearningModel:
    """
    创建深度学习模型实例

    参数:
        model_type (str): 模型类型，如 "lstm", "gru", "simple_rnn", "cnn_lstm"
        target_type (str): 目标变量类型
        model_params (Dict): 模型特定参数
        prediction_horizon (int): 预测周期
        input_sequence_length (int): 输入序列长度
        name (str, optional): 模型名称
        model_dir (str): 模型保存目录

    返回:
        DeepLearningModel: 深度学习模型实例
    """
    logger.info(f"Creating deep learning model: {model_type}")
    model_type = model_type.lower()

    if model_type == "lstm":
        return LSTMModel(
            name=name or "lstm_model",
            model_params=model_params,
            prediction_horizon=prediction_horizon,
            target_type=target_type,
            input_sequence_length=input_sequence_length,
            model_dir=model_dir
        )
    elif model_type == "gru":
        return GRUModel(
            name=name or "gru_model",
            model_params=model_params,
            prediction_horizon=prediction_horizon,
            target_type=target_type,
            input_sequence_length=input_sequence_length,
            model_dir=model_dir
        )
    elif model_type == "simple_rnn":
        return SimpleRNNModel(
            name=name or "simple_rnn_model",
            model_params=model_params,
            prediction_horizon=prediction_horizon,
            target_type=target_type,
            input_sequence_length=input_sequence_length,
            model_dir=model_dir
        )
    elif model_type == "cnn_lstm":
        return CNNLSTMModel(
            name=name or "cnn_lstm_model",
            model_params=model_params,
            prediction_horizon=prediction_horizon,
            target_type=target_type,
            input_sequence_length=input_sequence_length,
            model_dir=model_dir
        )
    # 在这里可以添加其他深度学习模型的创建逻辑
    else:
        raise ValueError(f"Unsupported deep learning model type: {model_type}") 