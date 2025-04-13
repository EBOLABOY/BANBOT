import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 设置随机种子以确保结果可重复
np.random.seed(42)
tf.random.set_seed(42)

# 创建输出目录
os.makedirs('data/models/deep_learning', exist_ok=True)

class DeepLearningModels:
    def __init__(self, sequence_length=30, prediction_horizon=1, batch_size=64, epochs=100):
        """
        初始化深度学习模型参数
        
        参数:
        - sequence_length: 时间序列长度（用多少天的数据来预测）
        - prediction_horizon: 预测范围（预测未来几天）
        - batch_size: 批次大小
        - epochs: 训练轮数
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.epochs = epochs
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def load_data(self, filepath):
        """加载并预处理数据"""
        print("加载数据...")
        try:
            self.df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            print(f"数据已加载: {self.df.shape[0]} 行, {self.df.shape[1]} 列")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {filepath}")
            return False
            
        # 获取特征列表（排除目标变量）
        self.features = self.df.columns.tolist()
        if 'target' in self.features:
            self.features.remove('target')
        
        # 保存原始日期索引以便后续可视化
        self.dates = self.df.index
        
        # 对于回归任务，我们使用收盘价变化率作为目标变量
        self.df['price_change'] = self.df['close'].pct_change()
        
        # 移除前N行，因为它们包含NaN（由于计算移动平均，变化率等）
        self.df = self.df.dropna()
        
        return True
        
    def prepare_sequences(self, is_classification=False):
        """
        将时间序列数据转换为监督学习格式的序列
        
        参数:
        - is_classification: 是否为分类任务（True）或回归任务（False）
        """
        print(f"准备{self.sequence_length}天序列数据，预测未来{self.prediction_horizon}天...")
        
        # 选择要使用的特征
        selected_features = [
            'close', 'volume', 'stoch_k', 'stoch_d', 'rsi_14', 'macd_hist',
            'williams_r', 'cci_20', 'adx', 'buy_sell_pressure', 'volatility_20',
            'funding_rate', 'fear_greed_simple'
        ]
        
        # 确保所有选定特征都存在
        available_features = [f for f in selected_features if f in self.df.columns]
        print(f"使用 {len(available_features)} 个特征进行深度学习模型训练")
        
        X_data = []
        y_data = []
        
        # 创建序列
        for i in range(len(self.df) - self.sequence_length - self.prediction_horizon + 1):
            # 特征序列 (t, t+1, ..., t+sequence_length-1)
            X_sequence = self.df[available_features].iloc[i:i+self.sequence_length].values
            
            # 目标变量 (t+sequence_length, ..., t+sequence_length+prediction_horizon-1)
            if is_classification:
                # 分类任务: 预测价格方向 (1=上涨, 0=下跌)
                future_price = self.df['close'].iloc[i+self.sequence_length+self.prediction_horizon-1]
                current_price = self.df['close'].iloc[i+self.sequence_length-1]
                y_label = 1 if future_price > current_price else 0
            else:
                # 回归任务: 预测价格变化百分比
                y_label = self.df['price_change'].iloc[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon].values
            
            X_data.append(X_sequence)
            y_data.append(y_label)
        
        # 转换为numpy数组
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # 标准化特征
        n_samples, n_timesteps, n_features = X_data.shape
        X_reshaped = X_data.reshape(n_samples * n_timesteps, n_features)
        X_reshaped = self.scaler_X.fit_transform(X_reshaped)
        X_scaled = X_reshaped.reshape(n_samples, n_timesteps, n_features)
        
        # 对于回归任务，标准化目标变量
        if not is_classification and len(y_data.shape) > 1 and y_data.shape[1] > 1:
            y_reshaped = y_data.reshape(-1, 1)
            y_scaled = self.scaler_y.fit_transform(y_reshaped)
            y_scaled = y_scaled.reshape(y_data.shape)
        elif not is_classification:
            y_scaled = self.scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()
        else:
            y_scaled = y_data
        
        print(f"数据准备完成: X形状={X_scaled.shape}, y形状={y_scaled.shape}")
        
        # 分割为训练集和测试集 (80% 训练, 20% 测试)
        train_size = int(len(X_scaled) * 0.8)
        self.X_train, self.X_test = X_scaled[:train_size], X_scaled[train_size:]
        self.y_train, self.y_test = y_scaled[:train_size], y_scaled[train_size:]
        
        # 保存测试集对应的日期索引，用于后续可视化
        sequence_end_indices = np.arange(self.sequence_length, len(self.df) - self.prediction_horizon + 1)
        self.test_dates = self.dates[sequence_end_indices[train_size:]]
        
        print(f"训练集: {self.X_train.shape[0]} 样本, 测试集: {self.X_test.shape[0]} 样本")
        
        # 保存数据特征数目，用于构建模型
        self.n_features = n_features
        
    def build_lstm_model(self, is_classification=False):
        """构建LSTM模型"""
        print("构建LSTM模型...")
        
        model = Sequential()
        
        # 第一层LSTM，返回序列以输入到下一层LSTM
        model.add(LSTM(100, activation='tanh', return_sequences=True, 
                       input_shape=(self.sequence_length, self.n_features)))
        model.add(Dropout(0.2))
        
        # 第二层LSTM
        model.add(LSTM(80, activation='tanh', return_sequences=False))
        model.add(Dropout(0.2))
        
        # 输出层
        if is_classification:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), 
                         metrics=['accuracy'])
        else:
            model.add(Dense(self.prediction_horizon))
            model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), 
                         metrics=['mae'])
        
        print(model.summary())
        self.model = model
        return model
        
    def build_gru_model(self, is_classification=False):
        """构建GRU模型"""
        print("构建GRU模型...")
        
        model = Sequential()
        
        # 第一层GRU，返回序列以输入到下一层GRU
        model.add(GRU(100, activation='tanh', return_sequences=True, 
                      input_shape=(self.sequence_length, self.n_features)))
        model.add(Dropout(0.2))
        
        # 第二层GRU
        model.add(GRU(80, activation='tanh', return_sequences=False))
        model.add(Dropout(0.2))
        
        # 输出层
        if is_classification:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), 
                         metrics=['accuracy'])
        else:
            model.add(Dense(self.prediction_horizon))
            model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), 
                         metrics=['mae'])
        
        print(model.summary())
        self.model = model
        return model
    
    def build_transformer_model(self, is_classification=False):
        """构建Transformer模型"""
        print("构建Transformer模型...")
        
        # 定义输入层
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # 自注意力机制
        x = inputs
        
        # Transformer编码器块
        for _ in range(2):  # 可以根据需要堆叠多个编码器块
            # 多头注意力
            attention_output = MultiHeadAttention(
                num_heads=8, key_dim=16
            )(x, x)
            
            # 残差连接和层归一化
            x1 = LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # 前馈神经网络
            ff = Dense(256, activation='relu')(x1)
            ff = Dropout(0.2)(ff)
            ff = Dense(self.n_features)(ff)
            
            # 残差连接和层归一化
            x = LayerNormalization(epsilon=1e-6)(x1 + ff)
        
        # 全局池化
        x = tf.reduce_mean(x, axis=1)
        
        # 全连接层
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # 输出层
        if is_classification:
            outputs = Dense(1, activation='sigmoid')(x)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), 
                         metrics=['accuracy'])
        else:
            outputs = Dense(self.prediction_horizon)(x)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), 
                         metrics=['mae'])
        
        print(model.summary())
        self.model = model
        return model
        
    def train_model(self, model_name="lstm", is_classification=False):
        """
        训练指定类型的模型
        
        参数:
        - model_name: 模型类型 ('lstm', 'gru', 或 'transformer')
        - is_classification: 是否为分类任务
        """
        print(f"\n开始训练 {model_name.upper()} 模型...")
        
        # 选择模型类型
        if model_name.lower() == "lstm":
            model = self.build_lstm_model(is_classification)
        elif model_name.lower() == "gru":
            model = self.build_gru_model(is_classification)
        elif model_name.lower() == "transformer":
            model = self.build_transformer_model(is_classification)
        else:
            print(f"错误: 未知的模型类型 '{model_name}'")
            return
        
        # 设置回调函数
        task_type = "分类" if is_classification else "回归"
        model_path = f'data/models/deep_learning/{model_name}_{task_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5'
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # 训练模型
        history = model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存训练历史
        history_path = f'data/models/deep_learning/{model_name}_{task_type}_history.png'
        self.plot_training_history(history, history_path)
        
        # 评估模型
        self.evaluate_model(is_classification)
        
        # 保存模型信息
        model_info = {
            'model_name': model_name,
            'is_classification': is_classification,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'model_path': model_path,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y if not is_classification else None,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        joblib.dump(model_info, f'data/models/deep_learning/{model_name}_{task_type}_info.joblib')
        print(f"模型保存至: {model_path}")
        
        return model
        
    def evaluate_model(self, is_classification=False):
        """评估模型性能"""
        print("\n模型评估:")
        
        # 预测测试集
        y_pred = self.model.predict(self.X_test)
        
        if is_classification:
            # 二分类任务
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            accuracy = accuracy_score(self.y_test, y_pred_classes)
            
            print(f"测试集准确率: {accuracy:.4f}")
            
            # 可视化预测结果
            plt.figure(figsize=(12, 6))
            plt.scatter(range(len(self.y_test)), self.y_test, color='blue', label='实际值')
            plt.scatter(range(len(y_pred_classes)), y_pred_classes, color='red', label='预测值')
            plt.title('比特币价格方向预测 (1=上涨, 0=下跌)')
            plt.xlabel('时间步')
            plt.ylabel('价格方向')
            plt.legend()
            plt.grid(True)
            plt.savefig('data/models/deep_learning/classification_prediction.png')
            
        else:
            # 回归任务
            # 反标准化预测结果
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred_rescaled = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
                y_test_rescaled = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).reshape(self.y_test.shape)
            else:
                y_pred_rescaled = self.scaler_y.inverse_transform(y_pred).flatten()
                y_test_rescaled = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
            
            # 计算评估指标
            mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
            
            print(f"均方误差 (MSE): {mse:.6f}")
            print(f"均方根误差 (RMSE): {rmse:.6f}")
            print(f"平均绝对误差 (MAE): {mae:.6f}")
            
            # 可视化预测结果
            plt.figure(figsize=(12, 6))
            
            # 如果预测范围为1，直接绘制
            if self.prediction_horizon == 1:
                plt.plot(self.test_dates[-100:], y_test_rescaled[-100:], label='实际值', color='blue')
                plt.plot(self.test_dates[-100:], y_pred_rescaled[-100:], label='预测值', color='red')
            else:
                # 对于多步预测，只展示第一步
                plt.plot(self.test_dates[-100:], y_test_rescaled[-100:, 0], label='实际值', color='blue')
                plt.plot(self.test_dates[-100:], y_pred_rescaled[-100:, 0], label='预测值', color='red')
            
            plt.title('比特币价格变化率预测')
            plt.xlabel('日期')
            plt.ylabel('价格变化率')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('data/models/deep_learning/regression_prediction.png')
        
    def plot_training_history(self, history, filename):
        """绘制训练历史"""
        plt.figure(figsize=(12, 5))
        
        # 绘制损失
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('训练轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        # 绘制指标（准确率或平均绝对误差）
        plt.subplot(1, 2, 2)
        if 'accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='训练准确率')
            plt.plot(history.history['val_accuracy'], label='验证准确率')
            plt.title('模型准确率')
            plt.ylabel('准确率')
        else:
            plt.plot(history.history['mae'], label='训练MAE')
            plt.plot(history.history['val_mae'], label='验证MAE')
            plt.title('平均绝对误差')
            plt.ylabel('MAE')
            
        plt.xlabel('训练轮次')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    def predict_future(self, days=7, is_classification=False):
        """预测未来几天的价格走势"""
        print(f"\n预测未来{days}天价格走势...")
        
        # 获取最新的序列数据
        latest_sequence = self.df[:-self.prediction_horizon].iloc[-self.sequence_length:].copy()
        
        # 选择特征
        selected_features = [
            'close', 'volume', 'stoch_k', 'stoch_d', 'rsi_14', 'macd_hist',
            'williams_r', 'cci_20', 'adx', 'buy_sell_pressure', 'volatility_20',
            'funding_rate', 'fear_greed_simple'
        ]
        
        available_features = [f for f in selected_features if f in latest_sequence.columns]
        X_pred = latest_sequence[available_features].values
        
        # 标准化
        X_pred_scaled = self.scaler_X.transform(X_pred)
        X_pred_scaled = X_pred_scaled.reshape(1, self.sequence_length, len(available_features))
        
        # 逐步预测
        results = []
        current_sequence = X_pred_scaled
        last_price = self.df['close'].iloc[-1]
        
        for i in range(days):
            # 预测下一个值
            y_pred = self.model.predict(current_sequence)
            
            if is_classification:
                # 预测价格方向
                direction = "上涨" if y_pred[0][0] > 0.5 else "下跌"
                confidence = y_pred[0][0] if y_pred[0][0] > 0.5 else 1 - y_pred[0][0]
                
                results.append({
                    'day': i + 1,
                    'date': (self.df.index[-1] + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    'direction': direction,
                    'probability': float(y_pred[0][0]),
                    'confidence': float(confidence)
                })
                
            else:
                # 预测价格变化率
                change_pred = self.scaler_y.inverse_transform(y_pred)[0][0]
                price_pred = last_price * (1 + change_pred)
                last_price = price_pred
                
                results.append({
                    'day': i + 1,
                    'date': (self.df.index[-1] + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    'price_change': float(change_pred),
                    'predicted_price': float(price_pred)
                })
                
            # 更新序列（移除最早的一天，添加新预测的一天）
            # 这里简化处理，实际应用中可能需要更复杂的特征生成
            if not is_classification and i < days - 1:
                # 这里添加一个简化的更新策略，只更新收盘价
                new_row = current_sequence[0, -1:].copy()
                # 更新收盘价特征（假设是第一个特征）
                new_row[0, 0] = y_pred[0][0]
                
                # 移除最早的一天并添加新预测的一天
                current_sequence = np.concatenate([current_sequence[:, 1:, :], new_row], axis=1)
        
        # 将结果保存为CSV
        result_df = pd.DataFrame(results)
        result_path = 'data/models/deep_learning/future_predictions.csv'
        result_df.to_csv(result_path, index=False)
        print(f"未来预测结果已保存至: {result_path}")
        
        # 可视化未来预测
        plt.figure(figsize=(10, 6))
        
        if is_classification:
            # 绘制方向置信度
            plt.bar(result_df['date'], result_df['probability'], color=['green' if p > 0.5 else 'red' for p in result_df['probability']])
            plt.axhline(y=0.5, color='black', linestyle='--')
            plt.title('未来价格方向预测 (>0.5表示上涨)')
            plt.ylabel('上涨概率')
        else:
            # 绘制预测价格
            plt.plot(result_df['date'], result_df['predicted_price'], 'b-o')
            # 添加最后一个已知价格点
            plt.plot([self.df.index[-1].strftime('%Y-%m-%d')], [self.df['close'].iloc[-1]], 'ro', markersize=8)
            plt.title('未来价格预测')
            plt.ylabel('价格 (USD)')
        
        plt.xlabel('日期')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('data/models/deep_learning/future_prediction_chart.png')
        
        return results

def train_multiple_models():
    """训练多种深度学习模型"""
    data_file = 'data/features/btc_full_features.csv'
    
    # 参数配置
    configs = [
        # 分类模型 - 预测价格方向
        {"model": "lstm", "sequence_length": 30, "prediction_horizon": 1, "is_classification": True},
        {"model": "gru", "sequence_length": 30, "prediction_horizon": 1, "is_classification": True},
        {"model": "transformer", "sequence_length": 30, "prediction_horizon": 1, "is_classification": True},
        
        # 回归模型 - 预测价格变化率
        {"model": "lstm", "sequence_length": 30, "prediction_horizon": 1, "is_classification": False},
        {"model": "gru", "sequence_length": 30, "prediction_horizon": 1, "is_classification": False},
        
        # 多步预测
        {"model": "lstm", "sequence_length": 30, "prediction_horizon": 3, "is_classification": False}
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*80}")
        model_type = config['model'].upper()
        task_type = "分类" if config['is_classification'] else "回归"
        horizon = config['prediction_horizon']
        print(f"训练 {model_type} 模型 ({task_type}, 预测范围: {horizon}天)")
        print(f"{'='*80}")
        
        try:
            # 创建模型实例
            dl_model = DeepLearningModels(
                sequence_length=config['sequence_length'],
                prediction_horizon=config['prediction_horizon'],
                batch_size=64,
                epochs=50  # 减少轮次以加快训练
            )
            
            # 加载数据
            if not dl_model.load_data(data_file):
                continue
                
            # 准备序列数据
            dl_model.prepare_sequences(is_classification=config['is_classification'])
            
            # 训练模型
            model = dl_model.train_model(
                model_name=config['model'],
                is_classification=config['is_classification']
            )
            
            # 预测未来
            if model is not None:
                future_predictions = dl_model.predict_future(
                    days=5,
                    is_classification=config['is_classification']
                )
                
                # 记录结果
                model_result = {
                    "model_type": config['model'],
                    "task_type": "分类" if config['is_classification'] else "回归",
                    "prediction_horizon": config['prediction_horizon'],
                    "status": "成功",
                    "predictions": future_predictions
                }
                results.append(model_result)
            
        except Exception as e:
            print(f"错误: 训练 {model_type} 模型时出错: {str(e)}")
            results.append({
                "model_type": config['model'],
                "task_type": "分类" if config['is_classification'] else "回归",
                "prediction_horizon": config['prediction_horizon'],
                "status": f"失败: {str(e)}",
                "predictions": None
            })
    
    # 保存训练结果摘要
    summary_df = pd.DataFrame([
        {
            "模型类型": r["model_type"],
            "任务类型": r["task_type"],
            "预测范围": r["prediction_horizon"],
            "状态": r["status"]
        } for r in results
    ])
    
    summary_df.to_csv('data/models/deep_learning/training_summary.csv', index=False)
    print(f"\n模型训练摘要已保存至: data/models/deep_learning/training_summary.csv")
    
    return results

def compare_with_ensemble_model():
    """比较深度学习模型与集成模型的性能"""
    # 待实现 - 这需要加载已训练的模型并进行比较
    pass

if __name__ == "__main__":
    print("比特币价格预测 - 深度学习模型")
    print("=" * 50)
    
    # 检查GPU是否可用
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"找到 {len(gpus)} 个GPU: {gpus}")
        # 设置内存增长模式
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("未检测到GPU，将使用CPU进行训练")
    
    # 训练多个模型
    train_multiple_models() 