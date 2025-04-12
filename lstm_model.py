import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import time
from torch.amp import autocast, GradScaler
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import RobustScaler
import joblib
import warnings
import torch.cuda.profiler as profiler
import threading
import psutil
import gc
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message="Given trait value dtype.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置 Matplotlib
# 移除强制指定中文字体的行，允许在 Ubuntu 上使用默认字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置中文显示
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加系统监控函数
def get_gpu_utilization():
    """获取GPU使用率"""
    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return f"{util.gpu}%"
        except:
            return "N/A"
    return "N/A"

def get_gpu_memory():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = info.used / 1e9
            total_gb = info.total / 1e9
            return f"{used_gb:.2f}GB / {total_gb:.2f}GB"
        except:
            return "N/A"
    return "N/A"

def get_cpu_utilization():
    """获取CPU使用率"""
    return f"{psutil.cpu_percent()}"

def get_memory_utilization():
    """获取系统内存使用率"""
    return f"{psutil.virtual_memory().percent}"

# 检查计算环境和依赖项
def check_environment():
    """检查计算环境和依赖项，并显示硬件信息"""
    print("==== 计算环境检查 ====")
    
    # 检查PyTorch版本和CUDA支持
    if torch.cuda.is_available():
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {device_props.name}")
            print(f"  显存: {device_props.total_memory/1e9:.2f} GB")
            print(f"  计算能力: {device_props.major}.{device_props.minor}")
            print(f"  多处理器数量: {device_props.multi_processor_count}")
            
        # 测试张量运算性能
        start_time = time.time()
        test_size = 4000
        a = torch.randn(test_size, test_size, device='cuda')
        b = torch.randn(test_size, test_size, device='cuda')
        torch.cuda.synchronize()
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        matmul_time = time.time() - start
        print(f"矩阵乘法性能测试 ({test_size}x{test_size}): {matmul_time:.4f}秒")
        
        # 测试AMP性能
        with torch.cuda.amp.autocast():
            torch.cuda.synchronize()
            start = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            matmul_time_amp = time.time() - start
        print(f"AMP矩阵乘法性能测试: {matmul_time_amp:.4f}秒")
        print(f"AMP速度提升: {matmul_time/matmul_time_amp:.2f}x")
        
    else:
        print("警告: CUDA不可用，将使用CPU进行计算")
        print(f"CPU核心数: {os.cpu_count()}")
    
    # 检查内存
    mem = psutil.virtual_memory()
    print(f"系统内存: {mem.total/1e9:.2f} GB, 可用: {mem.available/1e9:.2f} GB")
    
    # 检查重要包
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'torch', 'sklearn', 
        'tensorboard', 'joblib'
    ]
    
    print("\n包版本检查:")
    for package in required_packages:
        try:
            if package == 'torch':
                version = torch.__version__
            elif package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                import importlib
                version = importlib.import_module(package).__version__
            print(f"  {package}: {version}")
        except (ImportError, AttributeError):
            print(f"  {package}: 未安装或无法检测版本")
    
    print("\n==== 环境检查完成 ====\n")

# 性能监控函数
def log_system_info(interval=30):
    """在后台线程中定期记录系统资源使用情况"""
    while True:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i} 使用率: {torch.cuda.utilization(i)}%")
                print(f"GPU {i} 显存占用: {torch.cuda.memory_allocated(i)/1e9:.2f}GB / {torch.cuda.get_device_properties(i).total_memory/1e9:.2f}GB")
        print(f"CPU 使用率: {psutil.cpu_percent()}%, 内存使用率: {psutil.virtual_memory().percent}%")
        time.sleep(interval)

# 性能分析装饰器
def profile_time(func):
    """装饰器，用于分析函数执行时间"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} 执行时间: {elapsed:.4f}秒")
        return result
    return wrapper

# 改进的LSTM模型 - 增加层数、隐藏单元和Dropout
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.3):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 启用cuDNN加速
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            
        # LSTM层 - 简化为单向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False  # 改为单向LSTM减少过拟合
        )
        
        # 简化网络结构 - 移除注意力机制
        
        # 全连接层 - 减少层数和神经元
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
        # 批归一化 - 保留一层
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name:
                # 检查参数的维度，对于BatchNorm等一维权重使用正态分布初始化
                if len(param.data.shape) >= 2:
                    nn.init.kaiming_normal_(param.data, mode='fan_out', nonlinearity='relu')
                else:
                    # 一维张量(如BatchNorm的weight)使用正态分布初始化
                    nn.init.normal_(param.data, mean=1.0, std=0.02)
        
    def forward(self, x):
        # x的形状: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        # LSTM层
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
        
        # 获取最后一个时间步的输出
        out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # FC层 - 简化版
        out = self.fc1(out)  # (batch_size, hidden_size//2)
        out = F.relu(out)
        out = self.bn1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)  # (batch_size, 1)
        
        return out

# 注意力机制LSTM模型
class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super(AttentionLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 启用cuDNN加速
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 使用双向LSTM
        )
        
        # 注意力矩阵
        self.attn = nn.Linear(hidden_size * 2, 1)  # *2因为是双向LSTM
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, 1)
        
        # 正则化和激活
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        
    def forward(self, x):
        # LSTM前向传播，获取所有时间步的输出
        # x: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]
        
        # 计算注意力权重
        attn_weights = self.attn(lstm_out)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # 在时间维度上做softmax
        
        # 加权求和
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch_size, hidden_size*2]
        
        # 全连接层
        out = self.dropout(context)
        out = self.fc(out)  # [batch_size, 1]
        
        return out

# 方向感知损失函数
class DirectionAwareLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(DirectionAwareLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # 标准MSE损失
        mse_loss = self.mse(pred, target)
        
        # 方向损失 - 预测方向与实际方向的一致性
        batch_size = pred.size(0)
        
        # 只有当batch_size大于1时才能计算方向损失
        if batch_size > 1:
            pred_diff = pred[1:] - pred[:-1]
            target_diff = target[1:] - target[:-1]
            
            # 计算符号一致的比例，然后转换为损失（越不一致损失越大）
            sign_match = torch.sign(pred_diff) * torch.sign(target_diff)
            direction_loss = torch.mean(1.0 - sign_match)
            
            # 组合损失：MSE权重alpha + 方向损失权重(1-alpha)
            return self.alpha * mse_loss + (1 - self.alpha) * direction_loss
        else:
            return mse_loss

# Focal Loss变体，专注于难以预测的样本
class FocalMSELoss(nn.Module):
    """
    结合MSE损失和方向预测损失的混合损失函数。
    gamma控制困难样本的权重，direction_weight控制方向损失的权重。
    """
    def __init__(self, gamma=2.0, alpha=0.5, direction_weight=0.3): # 将方向权重降低到0.3
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.direction_weight = direction_weight
        self.eps = 1e-6  # 添加平滑因子，防止数值不稳定
        
    def forward(self, pred, target):
        # 计算MSE
        mse = F.mse_loss(pred, target, reduction='none')
        
        # Focal权重 - 为难样本赋予更高权重
        pt = torch.exp(-mse)
        focal_weight = (1 - pt) ** self.gamma
        
        # 应用focal weight到MSE
        focal_loss = self.alpha * focal_weight * mse
        focal_loss = focal_loss.mean()
        
        # 计算方向预测损失
        # 添加平滑处理，避免极小值导致梯度不稳定
        if pred.dim() > 2:  # 处理3D输入 (batch, seq, feature)
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            target_diff = target[:, 1:, :] - target[:, :-1, :]
        else:  # 处理2D输入 (batch, feature)
            pred_diff = pred[1:, :] - pred[:-1, :]
            target_diff = target[1:, :] - target[:-1, :]
        
        # 平滑化差异值，防止极小值计算问题
        pred_diff_sign = torch.sign(pred_diff + self.eps)
        target_diff_sign = torch.sign(target_diff + self.eps)
        
        # 计算方向匹配损失 (1表示方向不匹配)
        direction_match = (pred_diff_sign * target_diff_sign < 0).float()
        
        # 对于非常小的变化（接近噪声），减少其在方向损失中的权重
        small_change_mask = (torch.abs(target_diff) < 0.001).float()
        weighted_direction_match = direction_match * (1.0 - 0.5 * small_change_mask)
        
        direction_loss = weighted_direction_match.mean()
        
        # 总损失
        total_loss = (1 - self.direction_weight) * focal_loss + self.direction_weight * direction_loss
        
        return total_loss

# 加载模型训练数据
def load_training_data():
    """加载并预处理训练数据，计算技术指标和基于币安API的衍生品特征"""
    # 显示数据加载进度
    print("开始加载训练数据...")
    
    # 开启多线程数据处理
    torch.set_num_threads(os.cpu_count())
    
    # 尝试加载预处理数据（如果存在）
    cache_file = 'data/processed_data_cache_v2.npz'
    scaler_file = 'data/scaler_v2.pkl' # 特征缩放器
    target_scaler_file = 'data/target_scaler_v2.pkl' # 目标缩放器文件 - !!! 更新为 RobustScaler !!!
    original_df_file = 'data/original_df_v2.csv'
    feature_info_file = 'data/model_ready/feature_info.json' # 修改为使用feature_info.json

    # 检查所有必需的缓存文件是否存在
    if os.path.exists(cache_file) and os.path.exists(scaler_file) and os.path.exists(target_scaler_file) and os.path.exists(original_df_file):
        print(f"找到预处理数据缓存，直接加载: {cache_file}, {scaler_file}, {target_scaler_file}")
        
        # 检查特征文件是否存在并加载
        use_cache = True
        custom_features = []
        try:
            if os.path.exists(feature_info_file):
                with open(feature_info_file, 'r') as f:
                    feature_info = json.load(f)
                    if 'features' in feature_info and isinstance(feature_info['features'], list):
                        custom_features = feature_info['features']
                        print(f"特征文件中包含 {len(custom_features)} 个特征")
                        
                        # 检查缓存的特征是否与当前指定的特征相同
                        cache_feature_file = 'data/model_ready/cached_features.json'
                        if os.path.exists(cache_feature_file):
                            with open(cache_feature_file, 'r') as cf:
                                cached_features = json.load(cf)
                                if 'features' in cached_features and cached_features['features'] != custom_features:
                                    print("特征列表已更改，将重新处理数据...")
                                    use_cache = False
                        else:
                            print("找不到缓存特征文件，将重新处理数据...")
                            use_cache = False
        except Exception as e:
            print(f"检查特征文件时出错: {e}")
            
        if use_cache:
            try:
                data = np.load(cache_file, allow_pickle=True)
                X_train = data['X_train']
                y_train = data['y_train']
                X_val = data['X_val']
                y_val = data['y_val']
                X_test = data['X_test']
                y_test = data['y_test']
                feature_scaler = joblib.load(scaler_file) # 加载特征缩放器
                target_scaler = joblib.load(target_scaler_file) # 加载目标缩放器 - 改为 RobustScaler
                print("成功加载预处理数据缓存!")
                original_df = pd.read_csv(original_df_file, index_col=0, parse_dates=['timestamp'])
                # !!! 确保返回6个值 !!!
                return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_scaler, target_scaler, original_df
            except Exception as e:
                print(f"加载缓存文件失败: {e}，将重新处理数据")
        else:
            print("由于特征列表变更，跳过缓存加载，将重新处理数据")
    else:
        print("缓存文件不完整或不存在，将重新处理数据...") # 提示缓存不完整
    
    # 读取数据文件并预处理
    print("从原始数据文件处理特征...")
    
    # 创建数据集文件夹（如果不存在）
    os.makedirs('data/model_ready', exist_ok=True)
    
    # 优先从feature_info.json中读取序列长度
    sequence_length = 30  # 默认值作为回退
    prediction_horizon = 1
    
    # 尝试从特征文件读取序列长度
    try:
        if os.path.exists(feature_info_file):
            with open(feature_info_file, 'r') as f:
                feature_info_data = json.load(f)
                if 'sequence_length' in feature_info_data and isinstance(feature_info_data['sequence_length'], int):
                    sequence_length = feature_info_data['sequence_length']
                    print(f"从特征文件读取序列长度: {sequence_length}")
                if 'prediction_horizon' in feature_info_data and isinstance(feature_info_data['prediction_horizon'], int):
                    prediction_horizon = feature_info_data['prediction_horizon']
                    print(f"从特征文件读取预测周期: {prediction_horizon}")
    except Exception as e:
        print(f"读取特征文件时出错: {e}，将使用默认序列长度: {sequence_length}")
        
    print(f"使用序列长度: {sequence_length}, 预测周期: {prediction_horizon}")
    
    train_ratio = 0.7
    val_ratio = 0.15
    
    # 加载原始数据 - 假设包含币安API数据列
    # **重要：确保你的CSV文件已包含以下或类似的列名**
    # 'funding_rate', 'open_interest', 'taker_buy_volume', 'taker_sell_volume'
    # 'spread' (可选), 'depth_imbalance' (可选)
    try:
        # 修正文件名，与 get_binance_data.py 的输出一致
        df = pd.read_csv('data/processed/btc_with_funding_rate_features.csv') 
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True) # 确保时间顺序
    except FileNotFoundError:
        # 修正错误信息中的文件名
        print("错误：无法找到包含币安特征的数据文件 'data/processed/btc_with_funding_rate_features.csv'")
        print("请确保已通过币安API获取数据并合并到此文件。")
        return None, None, None, None, None
    
    print("增强数据预处理和特征工程 (加入币安API特征)...")
    
    # 1. 基础时间序列特征 (保持不变)
    df['log_return'] = np.log(df['close']).diff()
    # 年化波动率 (假设小时数据，一年约 365*24 小时)
    df['volatility_20'] = df['log_return'].rolling(20).std() * np.sqrt(365*24)
    df['volatility_60'] = df['log_return'].rolling(60).std() * np.sqrt(365*24) # 增加一个更长周期
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ma20'] = df['volume'].rolling(20).mean() # 增加一个更长周期
    df['volume_ratio'] = df['volume'] / (df['volume_ma5'] + 1e-9) # 防除零

    # 2. 技术指标
    print("  计算技术指标 (ATR, RSI, CCI)...")
    df['atr_14'] = calculate_atr(df, period=14)
    df['rsi_14'] = calculate_rsi(df, period=14)
    df['cci_20'] = calculate_cci(df, period=20)
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['trend_signal'] = np.where(df['sma_20'] > df['sma_50'], 1, -1).astype(int)
    # 添加EMA
    print("  计算EMA...")
    df['ema_12'] = calculate_ema(df, period=12)
    df['ema_26'] = calculate_ema(df, period=26)
    # 添加MACD
    print("  计算MACD...")
    df['macd_line'], df['macd_signal'], df['macd_hist'] = calculate_macd(df)
    # 添加Bollinger Bands
    print("  计算Bollinger Bands...")
    _, _, _, df['bb_percent_b'], df['bb_bandwidth'] = calculate_bollinger_bands(df)
    # 添加Stochastic Oscillator
    print("  计算Stochastic Oscillator...")
    df['stoch_k'], df['stoch_d'] = calculate_stochastic_oscillator(df)
    # 添加Williams %R
    print("  计算Williams %R...")
    df['williams_r'] = calculate_williams_r(df)
    # 添加ADX
    print("  计算ADX...")
    _, _, df['adx'] = calculate_adx(df)

    # 3. 动量和情绪指标
    print("  计算动量和情绪指标...")
    df['price_momentum_5'] = df['close'].pct_change(5) # 5周期价格变化率
    df['price_momentum_20'] = df['close'].pct_change(20) # 20周期价格变化率
    # 简化的情绪指标 (成交量 * 波动率)
    df['sentiment_vol_x_volatility'] = (df['volume'] * df['volatility_20']).rolling(5).mean()
    # 简化版恐惧贪婪 (之前定义，如果calculate_fear_greed可用)
    if 'calculate_fear_greed' in globals():
        df['fear_greed_simple'] = calculate_fear_greed(df)
    else:
        # 如果函数不存在，提供一个备用或置空
        df['fear_greed_simple'] = 0 # 或者 np.nan

    # 4. 基于币安API数据的特征
    print("  处理基于币安API的特征...")
    binance_api_features = []
    
    if 'funding_rate' in df.columns:
        df['funding_rate_ma3'] = df['funding_rate'].rolling(3).mean()
        df['funding_rate_change'] = df['funding_rate'].diff()
        binance_api_features.extend(['funding_rate', 'funding_rate_ma3', 'funding_rate_change'])
        print("已加入资金费率特征.")
    
    if 'open_interest' in df.columns:
        df['oi_change'] = df['open_interest'].diff()
        df['oi_pct_change'] = df['open_interest'].pct_change()
        # 持仓量与价格的关系 (可能暗示杠杆变化)
        df['oi_price_ratio'] = df['open_interest'] / df['close'] 
        binance_api_features.extend(['open_interest', 'oi_change', 'oi_pct_change', 'oi_price_ratio'])
        print("已加入持仓量特征.")

    if 'taker_buy_volume' in df.columns and 'taker_sell_volume' in df.columns:
        df['taker_volume'] = df['taker_buy_volume'] + df['taker_sell_volume']
        df['taker_buy_sell_ratio'] = df['taker_buy_volume'] / (df['taker_sell_volume'] + 1e-6) # 防除零
        df['taker_buy_ratio'] = df['taker_buy_volume'] / (df['taker_volume'] + 1e-6)
        df['taker_volume_ma5'] = df['taker_volume'].rolling(5).mean()
        binance_api_features.extend(['taker_volume', 'taker_buy_sell_ratio', 'taker_buy_ratio', 'taker_volume_ma5'])
        print("已加入吃单量特征.")
        
    # 可选：微观结构特征 (如果能获取)
    if 'spread' in df.columns:
        df['spread_ma5'] = df['spread'].rolling(5).mean()
        binance_api_features.extend(['spread', 'spread_ma5'])
        print("已加入买卖价差特征.")
        
    if 'depth_imbalance' in df.columns:
        df['depth_imbalance_ma5'] = df['depth_imbalance'].rolling(5).mean()
        binance_api_features.extend(['depth_imbalance', 'depth_imbalance_ma5'])
        print("已加入订单簿不平衡度特征.")
        
    # 清理新特征引入的NaN
    # 注意：需要更仔细地处理，这里简单填充
    df = df.ffill().bfill() 

    # 过滤异常数据 (在所有特征计算后)
    initial_features = ['high', 'low', 'open', 'close', 'volume'] # 保留这些基础列以备用
    # base_features = [f for f in feature_info['features'] if f in df.columns] # 移除对 feature_info 的依赖
    generated_features = [
        # 基础时间序列特征
        'log_return', 'volatility_20', 'volatility_60', 
        'volume_ma5', 'volume_ma20', 'volume_ratio',
        # 技术指标
        'atr_14', 'rsi_14', 'cci_20', 'sma_20', 'sma_50', 'trend_signal',
        'ema_12', 'ema_26', 'macd_line', 'macd_signal', 'macd_hist',
        'bb_percent_b', 'bb_bandwidth', 'stoch_k', 'stoch_d',
        'williams_r', 'adx',
        # 动量和情绪指标
        'price_momentum_5', 'price_momentum_20',
        'sentiment_vol_x_volatility', 'fear_greed_simple'
    ] + binance_api_features

    # all_features_to_check = base_features + generated_features # 不再需要 base_features
    all_features_to_check = initial_features + generated_features # 检查初始列和生成列
    df = df.dropna(subset=[col for col in all_features_to_check if col in df.columns])

    # 异常值处理 (对波动较大的新特征也处理)
    cols_to_clip = [
        # 波动率和基础指标
        'volatility_20', 'volatility_60', 'volume_ratio', 
        'log_return', 'price_momentum_5', 'price_momentum_20',
        # 技术指标
        'atr_14', 'rsi_14', 'cci_20', 
        'macd_line', 'macd_signal', 'macd_hist',
        'bb_percent_b', 'bb_bandwidth', 
        'stoch_k', 'stoch_d', 'williams_r', 'adx',
        # 情绪指标
        'sentiment_vol_x_volatility',
        # 币安API特征(如果存在)
        'funding_rate', 'funding_rate_change', 'oi_change',
        'taker_volume', 'taker_buy_sell_ratio'
    ]
    for col in cols_to_clip:
        if col in df.columns:
            q_low = df[col].quantile(0.005)
            q_high = df[col].quantile(0.995)
            df[col] = df[col].clip(lower=q_low, upper=q_high)

    # 获取最终的特征列表和目标变量
    target_col = 'close'  # 直接定义目标列
    
    # 优先从feature_info.json中读取特征列表
    custom_features = []
    try:
        if os.path.exists(feature_info_file):
            print(f"从特征文件读取特征列表: {feature_info_file}")
            with open(feature_info_file, 'r') as f:
                feature_info = json.load(f)
                if 'features' in feature_info and isinstance(feature_info['features'], list):
                    custom_features = feature_info['features']
                    print(f"成功读取自定义特征列表，包含 {len(custom_features)} 个特征")
    except Exception as e:
        print(f"读取特征文件失败: {e}，将使用生成的特征")
    
    # 如果成功读取了自定义特征列表，则使用它，否则使用生成的特征
    if custom_features:
        # 过滤掉不存在于DataFrame中的特征
        available_features = [f for f in custom_features if f in df.columns]
        missing_features = [f for f in custom_features if f not in df.columns and f != target_col]
        
        if missing_features:
            print(f"警告: 以下特征在数据中不存在，将被忽略: {missing_features}")
            
        final_features = [f for f in available_features if f != target_col]
        print(f"使用自定义特征列表: {final_features}")
    else:
        # 使用原有的特征生成逻辑
        print("使用自动生成的特征列表")
    final_features = list(set([f for f in generated_features if f in df.columns])) # 只使用成功生成的特征
    
    # 确保至少有一些特征可用
    if len(final_features) == 0:
        print("错误: 没有可用特征! 将使用基本特征集")
        # 回退到基本特征集
        final_features = [col for col in ['open', 'high', 'low', 'volume'] if col in df.columns]

    # 如果 target_col 是特征之一，从 final_features 中移除
    if target_col in final_features:
        final_features.remove(target_col)

    print(f"最终使用的特征数量: {len(final_features)}")
    print(f"最终特征列表: {final_features}")

    if target_col not in df.columns:
        print(f"错误：目标列 '{target_col}' 不在DataFrame中！")
        return None, None, None, None, None
        
    print(f"最终使用的特征数量: {len(final_features)}")
    
    # 选择所需的特征并确保没有NaN (再次检查)
    data_to_scale = df[final_features].values
    if np.isnan(data_to_scale).any():
        print("警告：特征数据中仍存在NaN值，使用ffill/bfill填充。")
        data_to_scale = pd.DataFrame(data_to_scale).ffill().bfill().values

    target_data = df[target_col].values.reshape(-1, 1)
    
    # 数据标准化 - 使用鲁棒缩放器 (仅对特征进行标准化)
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(data_to_scale)
    
    # 增强数据标准化 - 使用鲁棒缩放器
    print("应用增强的鲁棒特征缩放...")
    feature_scaler = RobustScaler() # 特征缩放器
    
    # 对每个特征单独使用鲁棒缩放
    scaled_features_global = feature_scaler.fit_transform(data_to_scale)
    
    # 再对每列单独缩放以处理异常值
    for i in range(data_to_scale.shape[1]):
        # 捕获并处理潜在异常
        try:
            feature = data_to_scale[:, i].reshape(-1, 1)
            col_scaler = RobustScaler()
            scaled_col = col_scaler.fit_transform(feature).reshape(-1)
            if np.any(np.isinf(scaled_col)):
                print(f"警告: 特征 {i} 缩放后出现无穷值，使用全局缩放结果")
                scaled_features[:, i] = scaled_features_global[:, i]
            else:
                scaled_features[:, i] = scaled_col
        except Exception as e:
            print(f"警告: 特征 {i} 缩放失败: {e}，使用全局缩放结果")
            scaled_features[:, i] = scaled_features_global[:, i]
    
    scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=3.0, neginf=-3.0)
    print("增强特征缩放完成")

    # !!! 修改：使用 RobustScaler 缩放目标变量 Y !!!
    print("应用目标变量缩放 (RobustScaler)...")
    target_scaler = RobustScaler() # 目标缩放器 - 改为 RobustScaler
    scaled_target = target_scaler.fit_transform(target_data)
    print("目标变量缩放完成")
    
    # 创建序列数据
    X, y = [], []
    # 确保索引不越界
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:(i + sequence_length), :])
        # 使用缩放后的目标值
        y.append(scaled_target[i + sequence_length]) 
    
    X = np.array(X)
    y = np.array(y) # 形状已经是 (n_samples, 1)
    
    if len(X) == 0:
        print("错误：处理后没有生成任何数据序列，请检查数据长度和sequence_length设置。")
        return None, None, None, None, None
    
    # 分割训练集、验证集和测试集
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"数据形状: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"验证集: X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"测试集: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 保存更新后的特征信息和scaler/数据
    updated_feature_info = {
        'features': final_features,
        'target': target_col,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'binance_features_added': binance_api_features # 记录添加的币安特征
    }
    with open(feature_info_file, 'w') as f:
        json.dump(updated_feature_info, f, indent=4)
    
    # 保存当前使用的特征列表到缓存特征文件
    cache_feature_file = 'data/model_ready/cached_features.json'
    with open(cache_feature_file, 'w') as cf:
        json.dump({'features': final_features + [target_col]}, cf, indent=4)
    
    print("保存预处理数据缓存...")
    np.savez(cache_file, 
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val, 
             X_test=X_test, y_test=y_test)
    joblib.dump(scaler, scaler_file)
    joblib.dump(target_scaler, target_scaler_file) # 保存 RobustScaler
    df.to_csv(original_df_file)
    
    print("数据处理完成!")
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    # 清理内存
    del df, data_to_scale, scaled_features, target_data
    gc.collect()
    
    # 加载保存的原始DataFrame以供后续使用
    original_df_loaded = pd.read_csv(original_df_file, index_col=0, parse_dates=['timestamp'])
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_scaler, target_scaler, original_df_loaded

# 添加技术分析指标计算函数
def calculate_atr(df, period=14):
    """计算平均真实范围"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean() # 使用EMA计算ATR更常见
    
    return atr

def calculate_rsi(df, period=14):
    """计算相对强弱指标"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 使用EMA计算平均增益和损失
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9) # 防除零
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_cci(df, period=20):
    """计算商品通道指数"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    tp_ma = tp.rolling(period).mean()
    # 使用绝对值的平均值，而不是平均绝对偏差
    tp_md = (tp - tp_ma).abs().rolling(period).mean()
    
    cci = (tp - tp_ma) / (0.015 * tp_md + 1e-9) # 防除零
    
    return cci

def calculate_ema(df, period):
    """计算指数移动平均线"""
    return df['close'].ewm(span=period, adjust=False).mean()

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """计算MACD指标"""
    ema_fast = calculate_ema(df, fast_period)
    ema_slow = calculate_ema(df, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    return macd_line, signal_line, macd_hist

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """计算布林带指标"""
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    # %B 指标
    percent_b = (df['close'] - lower_band) / (upper_band - lower_band + 1e-9) # 防除零
    
    # 带宽
    bandwidth = (upper_band - lower_band) / (sma + 1e-9) # 防除零
    
    return upper_band, sma, lower_band, percent_b, bandwidth

def calculate_stochastic_oscillator(df, k_period=14, d_period=3):
    """计算随机震荡指标 (%K, %D)"""
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    
    percent_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9) # 防除零
    percent_d = percent_k.rolling(d_period).mean()
    
    return percent_k, percent_d

def calculate_williams_r(df, period=14):
    """计算威廉姆斯 %R 指标"""
    high_max = df['high'].rolling(period).max()
    low_min = df['low'].rolling(period).min()
    
    williams_r = -100 * (high_max - df['close']) / (high_max - low_min + 1e-9) # 防除零
    
    return williams_r

def calculate_adx(df, period=14):
    """计算平均趋向指数 (ADX)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 计算 +DM, -DM, TR
    move_up = high.diff()
    move_down = -low.diff()
    
    plus_dm = pd.Series(np.where((move_up > move_down) & (move_up > 0), move_up, 0.0))
    minus_dm = pd.Series(np.where((move_down > move_up) & (move_down > 0), move_down, 0.0))
    
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # 使用EMA平滑
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9))
    
    # 计算DX和ADX
    dx = 100 * ( (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) )
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return plus_di, minus_di, adx

def calculate_fear_greed(df):
    """简化版恐惧与贪婪指标"""
    # 使用价格变化率和波动率的组合
    # 确保 volatility 列已计算
    if 'volatility_20' not in df.columns: # 使用我们新计算的volatility_20
         df['volatility_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(365*24)
         df['volatility_20'] = df['volatility_20'].ffill().bfill() # 填充可能产生的NaN
         
    price_change = df['close'].pct_change(5) # 5周期变化
    # 使用波动率的滚动均值作为基准
    vol_rolling_mean = df['volatility_20'].rolling(30).mean()
    vol_ratio = df['volatility_20'] / (vol_rolling_mean + 1e-9)
    
    # 归一化到0-100范围，50是中性
    price_score = 50 + (price_change * 100).clip(-50, 50)
    vol_score = 50 - ((vol_ratio - 1) * 25).clip(-25, 25) # 波动率高于均值时分数降低（更贪婪? 或更恐慌? 取决于定义）
    
    # 组合指标
    fear_greed = (price_score * 0.7 + vol_score * 0.3)
    
    return fear_greed

# 计算评估指标
def calculate_metrics(y_true, y_pred):
    # 计算均方误差 (MSE)
    mse = np.mean((y_true - y_pred) ** 2)
    
    # 计算均方根误差 (RMSE)
    rmse = np.sqrt(mse)
    
    # 计算平均绝对误差 (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 计算平均绝对百分比误差 (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # 计算方向准确率 (DA) - 预测方向与实际方向一致的比例
    direction_true = np.diff(y_true.flatten())
    direction_pred = np.diff(y_pred.flatten())
    
    # 确保长度一致且大于0
    if len(direction_true) > 0 and len(direction_true) == len(direction_pred):
        # 只考虑非零变化的方向
        valid_indices = (direction_true != 0)
        if np.sum(valid_indices) > 0:
            direction_accuracy = np.mean(np.sign(direction_true[valid_indices]) == np.sign(direction_pred[valid_indices])) * 100
        else:
            direction_accuracy = 50.0 # 如果没有有效方向变化，默认为50%
    else:
        direction_accuracy = 0.0
    
    # 计算相关系数
    correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    
    # 计算上升趋势预测准确率 - !!! 修正计算方式 !!!
    actual_ups = (direction_true > 0)
    predicted_ups = (direction_pred > 0)
    if np.sum(actual_ups) > 0:
        up_acc = np.sum(actual_ups & predicted_ups) / np.sum(actual_ups) * 100
    else:
        up_acc = 0.0 # 或设置为 NaN 或其他标记值
    
    # 计算下降趋势预测准确率 - !!! 修正计算方式 !!!
    actual_downs = (direction_true < 0)
    predicted_downs = (direction_pred < 0)
    if np.sum(actual_downs) > 0:
        down_acc = np.sum(actual_downs & predicted_downs) / np.sum(actual_downs) * 100
    else:
        down_acc = 0.0 # 或设置为 NaN 或其他标记值
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'direction_accuracy': direction_accuracy,
        'correlation': correlation,
        'up_trend_accuracy': up_acc,
        'down_trend_accuracy': down_acc
    }

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
                epochs=100, patience=20, model_save_path='models', l2_weight=1e-5):
    """
    训练模型并保存最佳模型
    支持早停和学习率调度
    """
    # 初始化性能跟踪变量
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    val_dir_accs = []
    
    # 确保model_save_path目录存在
    os.makedirs(model_save_path, exist_ok=True)
    
    # 创建tensorboard日志记录器
    log_dir = os.path.join(model_save_path, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 创建梯度缩放器 (支持混合精度训练)
    scaler = GradScaler()
    
    start_time = time.time()
    
    # 主训练循环
    print(f"开始训练，总计训练 {epochs} 轮...")
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0
        batch_times = []
        samples_per_sec_list = []
        
        # 每个epoch前重置性能监控变量
        epoch_start = time.time()
        batch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 使用自动混合精度
            with autocast(device_type='cuda' if device=='cuda' else 'cpu', enabled=device=='cuda'):
                output = model(data)
                
                # 计算损失 (包括L2正则化)
                loss = criterion(output, target)
                
                # 添加L2正则化 (手动实现，更灵活)
                if l2_weight > 0:
                    l2_reg = 0.0
                    for param in model.parameters():
                        l2_reg += torch.norm(param, 2)
                    loss += l2_weight * l2_reg
            
            # 反向传播 (使用梯度缩放)
            scaler.scale(loss).backward()
            
            # 梯度裁剪，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新权重 (使用梯度缩放)
            scaler.step(optimizer)
            scaler.update()
            
            # 累加损失
            train_loss += loss.item() * data.size(0)
            
            # 性能监控
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            samples_per_second = data.size(0) / batch_time
            samples_per_sec_list.append(samples_per_second)
            
            # 重置批次开始时间
            batch_start = time.time()
            
        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 保存到tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                with autocast(device_type='cuda' if device=='cuda' else 'cpu', enabled=device=='cuda'):
                    output = model(data)
                    # 计算验证损失 (不包括L2正则化)
                    loss = criterion(output, target)
                
                val_loss += loss.item() * data.size(0)
                all_val_preds.append(output.cpu())
                all_val_targets.append(target.cpu())
        
        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # 保存到tensorboard
        writer.add_scalar('Loss/validation', val_loss, epoch)
        
        # 计算方向预测准确率
        all_val_preds = torch.cat(all_val_preds)
        all_val_targets = torch.cat(all_val_targets)
        
        # 计算验证集的方向准确率
        if len(all_val_preds) > 1:
            # 确定数据结构 (批次和序列)
            if len(all_val_preds.shape) > 2:  # 3D: [batch, seq, feature]
                batch_size, seq_len, _ = all_val_preds.shape
                
                # 如果有序列维度，计算每个序列内的方向变化
                if seq_len > 1:
                    # 计算每个样本每个序列位置的方向预测准确率
                    correct_dirs = 0
                    total_dirs = 0
                    
                    for i in range(batch_size):
                        # 提取序列
                        seq_pred = all_val_preds[i]
                        seq_target = all_val_targets[i]
                        
                        # 计算连续点之间的差值
                        pred_diff = seq_pred[1:] - seq_pred[:-1]
                        target_diff = seq_target[1:] - seq_target[:-1]
                        
                        # 计算非零变化的索引 (忽略平值)
                        valid_idx = (target_diff != 0)
                        
                        if valid_idx.sum() > 0:
                            # 方向是否匹配 (符号相同)
                            dir_match = (torch.sign(pred_diff) == torch.sign(target_diff))
                            correct_dirs += dir_match[valid_idx].sum().item()
                            total_dirs += valid_idx.sum().item()
                    
                    # 计算总体方向准确率
                    val_dir_acc = correct_dirs / total_dirs * 100 if total_dirs > 0 else 50.0
                else:
                    # 没有序列维度，视为批次数据
                    val_dir_acc = 50.0  # 默认值
            else:
                # 2D数据 [batch, feature]
                # 计算相邻样本间的差异
                pred_diff = all_val_preds[1:] - all_val_preds[:-1]
                target_diff = all_val_targets[1:] - all_val_targets[:-1]
                
                # 只考虑目标有明确变化方向的点
                valid_idx = (target_diff != 0).flatten()
                
                if valid_idx.sum() > 0:
                    # 找出方向预测正确的比例
                    dir_match = (torch.sign(pred_diff) == torch.sign(target_diff)).flatten()
                    val_dir_acc = dir_match[valid_idx].float().mean().item() * 100
                else:
                    val_dir_acc = 50.0
        else:
            val_dir_acc = 50.0  # 如果验证集太小，使用默认值
        
        val_dir_accs.append(val_dir_acc)
        
        # 保存到tensorboard
        writer.add_scalar('Metrics/direction_accuracy', val_dir_acc, epoch)
        
        # 计算本轮训练的平均性能指标
        avg_batch_time = np.mean(batch_times)
        avg_samples_per_sec = np.mean(samples_per_sec_list)
        
        # 更新学习率 - 使用验证损失作为指标！
        scheduler.step(val_loss)  # 这里修复 - 为ReduceLROnPlateau提供metrics参数
        
        # 显示进度信息，包括方向准确率
        print(f"轮次 [{epoch}/{epochs}], 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, 验证方向准确率: {val_dir_acc:.2f}%, "
              f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch % 1 == 0:  # 每轮显示性能指标
            # 获取GPU利用率和内存
            if device == 'cuda':
                gpu_util = torch.cuda.utilization(0)
                gpu_mem = torch.cuda.memory_allocated()/1e6
                gpu_total = torch.cuda.get_device_properties(0).total_memory/1e6
                
                print(f"GPU {gpu_util}%% {avg_samples_per_sec:.1f} samples/s, 批处理时间: {avg_batch_time:.4f}s,"
                      f"显存: {gpu_mem/1000:.2f}GB / {gpu_total/1000:.2f}GB", end="")
            
            # 计算更多性能指标
            samples = len(train_loader.dataset)
            elapsed = time.time() - epoch_start
            print(f"          轮次 {epoch} 性能指标: 总样本数: {samples}, 总时间: {elapsed:.2f}s")
            print(f"平均吞吐量: {samples/elapsed:.1f} samples/s, 平均批处理时间: {avg_batch_time:.4f}s")
        
        # 检查是否为最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dir_acc': val_dir_acc
            }, f'{model_save_path}/best_lstm_model.pth')
            
            print(f"验证损失改善至 {best_val_loss:.6f}. 模型已保存!")
        else:
            early_stop_counter += 1
            print(f"验证损失未改善。早停计数器: {early_stop_counter}/{patience}")
            
            if early_stop_counter >= patience:
                print(f"早停：{patience} 轮未见改善")
                break
    
    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"训练完成！总计耗时: {total_time/60:.2f} 分钟")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    # 关闭tensorboard writer
    writer.close()
    
    # 加载最佳模型
    best_model_data = torch.load(f'{model_save_path}/best_lstm_model.pth')
    model.load_state_dict(best_model_data['model_state_dict'])
    
    return model, train_losses, val_losses, val_dir_accs

# 测试模型
def test_model(model, X_test, y_test, target_scaler, original_df, test_start_idx, device='cpu'):
    """
    测试模型并返回预测结果（原始价格尺度）
    增强版本：添加裁剪逻辑，限制预测值范围，
    防止逆缩放后产生极端值
    """
    model.eval()
    test_timestamps = original_df.index[test_start_idx:test_start_idx+len(X_test)]
    
    with torch.no_grad():
        # 确保输入数据在正确的设备上
        if isinstance(X_test, np.ndarray):
            X_test = torch.FloatTensor(X_test).to(device)
        elif X_test.device.type != device:
            X_test = X_test.to(device)
            
        # 进行预测
        y_pred_scaled = model(X_test)
        
        # 确保形状正确
        if len(y_pred_scaled.shape) > 2:
            y_pred_scaled = y_pred_scaled.squeeze(-1)
        
        # 移回CPU用于后处理    
        y_pred_scaled = y_pred_scaled.cpu().numpy()
        
        # 计算安全的裁剪边界，基于训练数据分布
        y_test_np = y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test
        
        # 使用百分位数计算裁剪边界，略微扩大范围以允许一些合理预测
        q_low, q_high = np.percentile(y_test_np, [1, 99])
        margin = (q_high - q_low) * 0.2  # 20%的边界扩展
        
        clip_low = q_low - margin
        clip_high = q_high + margin
        
        # 检查并修正极端异常值
        if clip_low < -10 or clip_high > 10:
            # 这可能表明数据分布存在问题，使用更保守的边界
            clip_low = max(clip_low, -3)
            clip_high = min(clip_high, 10)
        
        # 检查是否有NaN值
        if np.isnan(y_pred_scaled).any():
            print("警告: 预测结果包含NaN值，将替换为0")
            y_pred_scaled = np.nan_to_num(y_pred_scaled, nan=0.0)
        
        print(f"对缩放后的预测值进行裁剪，范围: [{clip_low:.4f}, {clip_high:.4f}]")
        y_pred_scaled_clipped = np.clip(y_pred_scaled, clip_low, clip_high)
        
        # 检查裁剪是否改变了大量值
        clipped_pct = np.mean((y_pred_scaled != y_pred_scaled_clipped)) * 100
        if clipped_pct > 10:
            print(f"警告: {clipped_pct:.2f}%的预测值被裁剪，可能表明模型输出不稳定")
        
        # 计算逆缩放前后的统计量，帮助识别潜在问题
        print("缩放预测的统计指标:")
        print(f"  均值: {np.mean(y_pred_scaled_clipped):.4f}, 中位数: {np.median(y_pred_scaled_clipped):.4f}")
        print(f"  标准差: {np.std(y_pred_scaled_clipped):.4f}")
        print(f"  最小值: {np.min(y_pred_scaled_clipped):.4f}, 最大值: {np.max(y_pred_scaled_clipped):.4f}")
        
        # 执行逆缩放，同时捕获潜在的溢出警告
        print("执行逆缩放...")
        
        # 确保输入格式正确
        if len(y_pred_scaled_clipped.shape) == 1:
            y_pred_scaled_clipped = y_pred_scaled_clipped.reshape(-1, 1)
            
        # 使用try-except块处理可能的溢出
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                y_pred = target_scaler.inverse_transform(y_pred_scaled_clipped)
                
                if len(w) > 0:
                    print(f"警告：逆缩放过程中捕获到{len(w)}个警告。尝试替代方法...")
                    # 如果有警告，使用更安全的逆变换方法
                    # 首先获取缩放器的参数
                    if hasattr(target_scaler, 'scale_') and hasattr(target_scaler, 'min_'):
                        # MinMaxScaler
                        scale = target_scaler.scale_
                        min_val = target_scaler.min_
                        y_pred = y_pred_scaled_clipped / scale + min_val
                    elif hasattr(target_scaler, 'scale_') and hasattr(target_scaler, 'center_'):
                        # RobustScaler
                        scale = target_scaler.scale_
                        center = target_scaler.center_
                        y_pred = y_pred_scaled_clipped * scale + center
                    elif hasattr(target_scaler, 'mean_') and hasattr(target_scaler, 'scale_'):
                        # StandardScaler
                        scale = target_scaler.scale_
                        mean = target_scaler.mean_
                        y_pred = y_pred_scaled_clipped * scale + mean
                    else:
                        # 如果无法识别缩放器类型，使用原始结果
                        print("无法识别缩放器类型，使用原始结果")
                        pass
        except Exception as e:
            print(f"逆缩放时出错: {str(e)}，尝试手动逆缩放...")
            # 手动尝试逆缩放
            if hasattr(target_scaler, 'scale_'):
                if hasattr(target_scaler, 'min_'):
                    # MinMaxScaler
                    scale = target_scaler.scale_
                    min_val = target_scaler.min_
                    y_pred = y_pred_scaled_clipped / scale + min_val
                elif hasattr(target_scaler, 'center_'):
                    # RobustScaler
                    scale = target_scaler.scale_
                    center = target_scaler.center_
                    y_pred = y_pred_scaled_clipped * scale + center
                elif hasattr(target_scaler, 'mean_'):
                    # StandardScaler
                    scale = target_scaler.scale_
                    mean = target_scaler.mean_
                    y_pred = y_pred_scaled_clipped * scale + mean
                else:
                    print("无法手动逆缩放，使用原始缩放值")
                    y_pred = y_pred_scaled_clipped
            else:
                print("无法手动逆缩放，使用原始缩放值")
                y_pred = y_pred_scaled_clipped
        
        # 处理逆缩放后的异常值
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            print("警告: 逆缩放后发现NaN或Inf值，替换为合理值")
            # 计算非NaN和非Inf值的中位数作为替换值
            valid_mask = ~(np.isnan(y_pred) | np.isinf(y_pred))
            if np.any(valid_mask):
                replace_value = np.median(y_pred[valid_mask])
            else:
                # 如果所有值都是NaN或Inf，使用原始数据的中位数
                replace_value = np.median(original_df.iloc[test_start_idx:test_start_idx+len(X_test)].values)
            
            y_pred = np.nan_to_num(y_pred, nan=replace_value, posinf=replace_value*1.5, neginf=replace_value*0.5)
        
        # 确保输出是一维数组
        if y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
            
        # 对应地转换y_test为原始尺度
        if torch.is_tensor(y_test):
            y_test_np = y_test.cpu().numpy()
        else:
            y_test_np = y_test
            
        if len(y_test_np.shape) == 1:
            y_test_np = y_test_np.reshape(-1, 1)
            
        try:
            y_test_original = target_scaler.inverse_transform(y_test_np)
            if y_test_original.shape[1] == 1:
                y_test_original = y_test_original.flatten()
        except Exception as e:
            print(f"逆变换y_test时出错: {str(e)}")
            # 使用原始数据作为后备
            y_test_original = original_df.iloc[test_start_idx:test_start_idx+len(X_test)].values
            if len(y_test_original.shape) > 1 and y_test_original.shape[1] == 1:
                y_test_original = y_test_original.flatten()
                
    # 显示逆缩放完成消息
    print("逆缩放完成")
    
    # 返回预测值、原始y值和时间戳
    return y_pred, y_test_original, test_timestamps

# 改进的交易信号生成
def generate_trading_signals(y_test_original, y_pred, test_timestamps, original_df, threshold_pct=0.3, volatility_window=20):
    """
    根据原始价格尺度的预测生成交易信号
    
    参数:
    - y_test_original: 测试集的原始价格（目标变量）
    - y_pred: 测试集的预测价格
    - test_timestamps: 对应的时间戳
    - original_df: 原始数据
    - threshold_pct: 触发交易的最小价格变化百分比
    - volatility_window: 波动率计算窗口大小
    
    返回:
    - 包含交易信号的DataFrame
    """
    signals = []
    print(f"生成交易信号（基于原始价格尺度），基准阈值为价格的 {threshold_pct}%...")
    
    # 计算整个预测序列的波动性统计量
    price_changes_pct = np.abs(np.diff(y_test_original) / y_test_original[:-1]) * 100
    median_change_pct = np.median(price_changes_pct)
    mean_change_pct = np.mean(price_changes_pct)
    
    # 根据整体市场特性调整基准阈值
    adaptive_threshold_pct = min(max(threshold_pct, median_change_pct * 0.3), 1.0)
    print(f"自适应基准阈值: {adaptive_threshold_pct:.2f}%，基于中位数价格变化: {median_change_pct:.2f}%")
    
    # 创建用于计算滚动波动率的DataFame
    volatility_df = pd.DataFrame(index=test_timestamps)
    volatility_df['close'] = y_test_original
    
    # 计算滚动波动率(使用对数收益率)
    volatility_df['returns'] = np.log(volatility_df['close'] / volatility_df['close'].shift(1))
    volatility_df['volatility'] = volatility_df['returns'].rolling(window=volatility_window).std() * np.sqrt(24)  # 转换为日波动率
    volatility_df['volatility_pct'] = volatility_df['volatility'] * 100  # 转换为百分比
    volatility_df['volatility_abs'] = volatility_df['volatility'] * volatility_df['close']  # 绝对价格波动

    # 确保正确的起始索引
    start_idx = volatility_window
    
    # 计算预测价格变化的统计信息
    pred_changes = np.diff(y_pred)
    pred_changes_pct = np.abs(pred_changes / y_pred[:-1]) * 100
    pred_changes_mean = np.mean(pred_changes_pct)
    pred_changes_std = np.std(pred_changes_pct)
    
    # 相对强度指标 (RSI) 计算，用于平衡买卖信号
    close_series = pd.Series(y_test_original)
    delta = close_series.diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # 确保不除以零
    avg_loss = avg_loss.replace(0, 1e-10)  
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # 转换为numpy便于索引
    rsi_values = rsi.values
    
    # 价格区间分析
    price_range_pct = (np.max(y_test_original) - np.min(y_test_original)) / np.min(y_test_original) * 100
    range_factor = min(max(price_range_pct / 50, 0.5), 2.0)  # 价格变化范围因子
    
    for i in range(start_idx, len(y_test_original)):
        # 当前时间戳和相关价格数据
        timestamp = test_timestamps[i]
        current_price = y_test_original[i]
        predicted_price = y_pred[i]
        previous_price = y_test_original[i-1]
        
        # 计算价格变化百分比
        price_change = predicted_price - current_price
        price_change_pct = price_change / current_price * 100
        
        # 获取当前波动率
        current_volatility = volatility_df['volatility_pct'].iloc[i]
        current_volatility_abs = volatility_df['volatility_abs'].iloc[i]
        
        # 当前RSI值
        current_rsi = rsi_values[i] if i < len(rsi_values) else 50
        
        # 动态阈值 (基于当前波动性和价格范围调整)
        dynamic_threshold_pct = adaptive_threshold_pct * (0.5 + current_volatility / np.mean(volatility_df['volatility_pct'].iloc[start_idx:]) * 0.5) * range_factor
        
        # 信号仓位大小 (基于预测与当前波动率的比值)
        signal_strength = min(abs(price_change_pct) / (dynamic_threshold_pct * 2), 1.0)
        
        # 预测置信度 (使用统计分析确定异常值)
        if i > 0 and i < len(pred_changes_pct):
            z_score = (pred_changes_pct[i-1] - pred_changes_mean) / (pred_changes_std + 1e-6)
            confidence = max(min(1.0 - abs(z_score)/3, 1.0), 0.2)  # 将置信度限制在[0.2, 1.0]区间
        else:
            confidence = 0.5  # 默认中等置信度
        
        # 考虑RSI超买超卖，平衡信号分布
        rsi_modifier = 1.0
        if current_rsi > 70:  # 超买区域，增加卖出信号概率
            rsi_modifier = 1.2
        elif current_rsi < 30:  # 超卖区域，增加买入信号概率
            rsi_modifier = 0.8
        
        # 生成交易信号
        if price_change_pct > dynamic_threshold_pct * rsi_modifier:
            signal = "买入"
        elif price_change_pct < -dynamic_threshold_pct / rsi_modifier:
            signal = "卖出"
        else:
            signal = "持有"
        
        # 通过置信度调整信号，提高多样性
        # 对于低置信度的预测，增加生成"持有"信号的概率
        if confidence < 0.5 and np.random.random() < 0.7 * (1-confidence):
            signal = "持有"
            signal_strength = 0.0
            
        # 强制平衡交易信号
        # 如果卖出信号连续过多，降低卖出阈值
        if i > 10:
            recent_signals = [s['signal'] for s in signals[-10:]]
            buy_count = sum(1 for s in recent_signals if s == "买入")
            sell_count = sum(1 for s in recent_signals if s == "卖出")
            
            # 如果卖出信号远多于买入信号，增加买入概率
            if sell_count > buy_count * 3:
                if signal == "卖出" and np.random.random() < 0.4:
                    if abs(price_change_pct) < dynamic_threshold_pct * 1.5:
                        signal = "持有"
                    elif price_change_pct > 0 and price_change_pct > dynamic_threshold_pct * 0.7:
                        signal = "买入"
                        
            # 如果买入信号远多于卖出信号，增加卖出概率
            elif buy_count > sell_count * 3:
                if signal == "买入" and np.random.random() < 0.4:
                    if abs(price_change_pct) < dynamic_threshold_pct * 1.5:
                        signal = "持有"
                    elif price_change_pct < 0 and abs(price_change_pct) > dynamic_threshold_pct * 0.7:
                        signal = "卖出"
        
        # 记录信号
        signals.append({
            'timestamp': timestamp,
            'price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volatility_pct': current_volatility,
            'volatility_abs': current_volatility_abs,
            'dynamic_threshold_pct': dynamic_threshold_pct,
            'signal': signal,
            'confidence': confidence,
            'signal_strength': signal_strength,
            'rsi': current_rsi
        })
    
    # 创建DataFrame并保存
    signals_df = pd.DataFrame(signals)
    
    # 计算交易信号分布
    signal_counts = signals_df['signal'].value_counts()
    total_signals = len(signals_df)
    
    print("\n交易信号分布:")
    for signal_type, count in signal_counts.items():
        percentage = count / total_signals * 100
        print(f"{signal_type}: {count} ({percentage:.2f}%)")
    
    # 保存交易信号到CSV
    output_path = 'models/enhanced_trading_signals.csv'
    signals_df.to_csv(output_path, index=False)
    print(f"交易信号已保存到 {output_path}")
    
    # 可视化
    plt.figure(figsize=(14, 10))
    
    # 价格图
    plt.subplot(311)
    plt.plot(signals_df['timestamp'], signals_df['price'], label='实际价格', color='blue')
    plt.plot(signals_df['timestamp'], signals_df['predicted_price'], label='预测价格', color='orange', linestyle='--')
    
    # 标记买入和卖出点
    buy_signals = signals_df[signals_df['signal'] == '买入']
    sell_signals = signals_df[signals_df['signal'] == '卖出']
    
    plt.scatter(buy_signals['timestamp'], buy_signals['price'], color='green', marker='^', s=50, label='买入信号')
    plt.scatter(sell_signals['timestamp'], sell_signals['price'], color='red', marker='v', s=50, label='卖出信号')
    
    plt.title('价格预测和交易信号')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    
    # 价格变化百分比和动态阈值
    plt.subplot(312)
    plt.plot(signals_df['timestamp'], signals_df['price_change_pct'], label='价格变化%', color='purple')
    plt.plot(signals_df['timestamp'], signals_df['dynamic_threshold_pct'], label='动态阈值%', color='red', linestyle='-.')
    plt.plot(signals_df['timestamp'], -signals_df['dynamic_threshold_pct'], color='red', linestyle='-.')
    plt.ylabel('百分比(%)')
    plt.title('价格变化百分比与动态阈值')
    plt.legend()
    plt.grid(True)
    
    # RSI指标
    plt.subplot(313)
    plt.plot(signals_df['timestamp'], signals_df['rsi'], label='RSI', color='green')
    plt.axhline(y=70, color='r', linestyle='--', label='超买(70)')
    plt.axhline(y=30, color='g', linestyle='--', label='超卖(30)')
    plt.axhline(y=50, color='gray', linestyle=':')
    plt.title('相对强弱指标(RSI)')
    plt.ylabel('RSI值')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/trading_signals_visualization.png')
    print("交易信号可视化已保存到 models/trading_signals_visualization.png")
    
    return signals_df

# 回测交易策略
def backtest_strategy(signals_df, initial_capital=10000.0, commission_rate=0.001, slippage_pct=0.001, 
                     risk_control=True, position_sizing='volatility', max_drawdown_pct=0.2):
    print(f"开始回测交易策略，初始资金: ${initial_capital}...")
    
    # 初始化交易执行器和风险管理器
    risk_manager = RiskManagement(
        initial_capital=initial_capital,
        max_drawdown_pct=max_drawdown_pct,
        position_sizing=position_sizing
    )
    
    executor = TradingExecutor(
        initial_capital=initial_capital,
        risk_manager=risk_manager
    )
    
    # 设置交易费用
    executor.commission_rate = commission_rate
    executor.slippage_pct = slippage_pct
    
    # 遍历每个交易信号
    for i, row in signals_df.iterrows():
        timestamp = row['timestamp']
        price = row['current_price']
        signal = row['signal']
        # !!! 使用正确的列名 volatility_abs !!!
        volatility = row['volatility_abs'] 
        
        # 计算信号可信度 - 基于预测变化与阈值的比例
        pred_change_pct = abs(row['predicted_change_pct'])
        # !!! 使用正确的列名 dynamic_threshold_pct !!!
        dynamic_threshold = row['dynamic_threshold_pct'] 
        confidence = min(1.0, pred_change_pct / (dynamic_threshold * 2)) if dynamic_threshold > 0 else 0.5
        
        # 执行交易
        success, message = executor.execute_signal(
            signal=signal,
            price=price,
            timestamp=timestamp,
            volatility=volatility,
            confidence=confidence
        )
        
        if success:
            print(f"[{timestamp}] {message}")
        
        # 检查风险控制是否触发
        if risk_control and risk_manager.risk_triggered:
            print(f"风险控制触发，停止交易: {risk_manager.risk_message}")
            break
    
    # 计算绩效指标
    performance = executor.get_performance_metrics()
    
    # 保存交易历史
    executor.save_trade_history('models/detailed_backtest_trades.csv')
        
    # 输出回测结果 (修正缩进)
    print("\n回测结果摘要:")
    print(f"初始资金: ${initial_capital:.2f}")
    print(f"最终资金: ${performance['current_capital']:.2f}")
    print(f"总收益率: {performance['total_return']:.2%}")
    print(f"总交易次数: {performance['total_trades']}")
    print(f"获胜交易: {performance['win_count']}")
    print(f"亏损交易: {performance['loss_count']}")
    print(f"胜率: {performance['win_rate']:.2%}")
    print(f"平均盈利: ${performance['avg_profit']:.2f}")
    print(f"平均亏损: ${performance['avg_loss']:.2f}")
    print(f"盈亏比: {performance['profit_factor']:.2f}")
    print(f"夏普比率: {performance['sharpe_ratio']:.2f}")
    print(f"最大回撤: {performance['max_drawdown']:.2%}")
    
    # 创建资金曲线
    trades_df = pd.DataFrame(executor.trades)
    if len(trades_df) > 0:
        # 可视化回测结果
        plt.figure(figsize=(15, 12))
        
        # 价格和交易图
        plt.subplot(3, 1, 1)
        plt.plot(signals_df['timestamp'], signals_df['current_price'], label='BTC Price', color='blue')
        
        # 标记买入和卖出交易
        buy_trades = trades_df[trades_df['type'] == 'buy']
        sell_trades = trades_df[trades_df['type'] == 'sell']
        
        plt.scatter(buy_trades['timestamp'], buy_trades['price'], 
                    color='green', marker='^', s=100, label='Buy')
        plt.scatter(sell_trades['timestamp'], sell_trades['price'], 
                    color='red', marker='v', s=100, label='Sell')
        
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Bitcoin Price and Trade Points')
        plt.legend()
        plt.grid(True)
        
        # 创建资金曲线
        capital_history = [initial_capital]
        timestamps = [signals_df.iloc[0]['timestamp']]
        
        for i, trade in trades_df.iterrows():
            capital_history.append(trade['capital_after'])
            timestamps.append(trade['timestamp'])
        
            # 投资组合价值图
            plt.subplot(3, 1, 2)
            plt.plot(timestamps, capital_history, label='Portfolio Value', color='purple') # Changed label
            # 修正缩进级别
            plt.xlabel('Time')
            plt.ylabel('Value ($)')
            plt.title('Portfolio Value Over Time') # Changed title
            plt.legend() # Added legend
            plt.grid(True)
            
            # 买入持有策略对比
            plt.subplot(3, 1, 3)
            
            # 计算买入持有策略的价值
            first_price = signals_df.iloc[0]['current_price']
            buy_hold_btc = initial_capital / first_price
            buy_hold_values = [initial_capital]
            
            for price in signals_df['current_price']:
                buy_hold_values.append(buy_hold_btc * price)
            
            # 添加交易策略与买入持有策略对比
            plt.plot(timestamps, capital_history, label='Trading Strategy', color='blue')
            plt.plot([signals_df.iloc[0]['timestamp']] + list(signals_df['timestamp']), 
                    buy_hold_values, label='Buy and Hold', color='orange')
            
            # 计算相对收益率
            final_strategy_return = (capital_history[-1] - initial_capital) / initial_capital
            final_buyhold_return = (buy_hold_values[-1] - initial_capital) / initial_capital
            
            # 修正缩进级别
            plt.title(f'Trading Strategy vs Buy and Hold | Strategy: {final_strategy_return:.2%}, Buy & Hold: {final_buyhold_return:.2%}') # Changed title
            plt.xlabel('Time')
            plt.ylabel('Value ($)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/enhanced_backtest_results.png')
        plt.close()
        
        print(f"增强回测结果可视化已保存到 models/enhanced_backtest_results.png")
        
        # 导出性能指标
        performance_df = pd.DataFrame([performance])
        performance_df.to_csv('models/backtest_performance_metrics.csv', index=False)
        print(f"性能指标已保存到 models/backtest_performance_metrics.csv")
    else:
        print("回测期间没有执行交易")
    
    return trades_df if len(trades_df) > 0 else None, performance

# 时序数据增强类
class TimeSeriesAugmentation:
    """
    用于时序数据增强的类，支持多种增强方法：
    1. 添加高斯噪声
    2. 随机缩放
    3. 时间扭曲（Time Warping）
    4. 窗口滑动
    """
    def __init__(self, 
                 noise_level=0.008,  # 降低噪声
                 scale_range=(0.97, 1.03),  # 缩小缩放范围 
                 jitter_prob=0.2):  # 降低扰动概率
        self.noise_level = noise_level
        self.scale_range = scale_range
        self.jitter_prob = jitter_prob
        
    def add_noise(self, x):
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_level, x.shape)
        return x + noise
    
    def random_scaling(self, x):
        """随机缩放"""
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return x * scale_factor
    
    def window_slicing(self, x, y, slice_ratio=0.9):
        """窗口切片，随机选取一个子序列"""
        seq_len = x.shape[0]
        win_len = int(seq_len * slice_ratio)
        if win_len >= seq_len:
            return x, y
            
        start = np.random.randint(0, seq_len - win_len + 1)
        return x[start:start+win_len], y
    
    def time_warping(self, x, sigma=0.2):
        """时间扭曲，对时间轴进行非线性变换"""
        seq_len, feat_dim = x.shape
        
        # 创建时间扭曲函数
        time_warp = np.random.normal(loc=1.0, scale=sigma, size=seq_len)
        time_warp = np.cumsum(time_warp)
        time_warp = time_warp / time_warp[-1] * seq_len
        
        # 线性插值
        new_x = np.zeros_like(x)
        for i in range(feat_dim):
            new_x[:, i] = np.interp(np.arange(seq_len), time_warp, x[:, i])
        
        return new_x
        
    def __call__(self, x, y=None):
        """应用多种增强方法"""
        # 深拷贝输入数据，防止修改原始数据
        x_aug = x.copy()
        
        # 随机决定是否应用每种增强方法
        if np.random.rand() < self.jitter_prob:
            x_aug = self.add_noise(x_aug)
            
        if np.random.rand() < self.jitter_prob:
            x_aug = self.random_scaling(x_aug)
            
        if np.random.rand() < self.jitter_prob and x.shape[0] > 10:
            x_aug = self.time_warping(x_aug)
            
        return x_aug if y is None else (x_aug, y)

# 创建增强数据集
def create_augmented_dataset(X, y, num_augmentations=2):  # 从3减到2
    """创建数据增强后的数据集
    
    参数:
        X: 原始特征数据，形状 (samples, seq_len, features)
        y: 原始标签数据，形状 (samples, 1)
        num_augmentations: 每个样本要创建的增强样本数
    
    返回:
        增强后的X和y
    """
    print(f"开始创建增强数据集，每个样本创建 {num_augmentations} 个增强版本...")
    start_time = time.time()
    
    # 创建增强器
    augmenter = TimeSeriesAugmentation(
        noise_level=0.01,
        scale_range=(0.95, 1.05),
        jitter_prob=0.3
    )
    
    # 并行增强处理
    num_samples = X.shape[0]
    total_samples = num_samples * (num_augmentations + 1)  # 原始 + 增强
    
    # 预分配空间以提高效率
    X_aug = np.zeros((total_samples, X.shape[1], X.shape[2]), dtype=X.dtype)
    y_aug = np.zeros((total_samples, 1 if len(y.shape) == 1 else y.shape[1]), dtype=y.dtype)
    
    # 首先复制原始数据
    X_aug[:num_samples] = X
    y_aug[:num_samples] = y.reshape(-1, 1) if len(y.shape) == 1 else y
    
    # 使用多进程池并行处理数据增强
    def augment_batch(batch_idx):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        if end <= start:
            return
            
        for i in range(start, end):
            for j in range(num_augmentations):
                idx = num_samples + i * num_augmentations + j
                if idx < total_samples:
                    X_aug[idx], y_aug[idx] = augmenter(X[i:i+1], y[i:i+1].reshape(-1, 1) if len(y.shape) == 1 else y[i:i+1])
    
    # 批处理大小
    batch_size = 1000
    batch_count = (num_samples + batch_size - 1) // batch_size
    
    # 使用并行处理 - 改用 ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(augment_batch, i) for i in range(batch_count)]
        
        # 显示进度条
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            print(f"数据增强进度: {i+1}/{batch_count} 批次", end='\r')
    
    print(f"\n数据增强完成! 用时: {time.time() - start_time:.2f}秒")
    print(f"增强前: {X.shape} -> 增强后: {X_aug.shape}")
    
    # 打乱数据顺序
    indices = np.random.permutation(total_samples)
    X_aug = X_aug[indices]
    y_aug = y_aug[indices]
    
    # 清理内存
    gc.collect()
    
    return X_aug, y_aug

# 超参数搜索函数
def hyperparameter_tuning(X_train, y_train, X_val, y_val, input_size, device, n_trials=10):
    """使用简单的随机搜索进行超参数优化"""
    print("开始超参数搜索...")
    
    best_val_loss = float('inf')
    best_params = None
    results = []
    
    for trial in range(n_trials):
        # 随机选择超参数
        hidden_size = np.random.choice([128, 256, 512, 1024])
        num_layers = np.random.choice([2, 3, 4, 5])
        dropout = np.random.uniform(0.2, 0.5)
        learning_rate = 10 ** np.random.uniform(-5, -3)
        batch_size = np.random.choice([32, 64, 128, 256])
        alpha = np.random.uniform(0.5, 0.9)  # DirectionAwareLoss的权重参数
        
        print(f"\n试验 {trial+1}/{n_trials}:")
        print(f"参数: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout:.3f}, lr={learning_rate:.6f}, batch_size={batch_size}, alpha={alpha:.2f}")
        
        # 创建模型
        model = EnhancedLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        
        # 创建数据加载器
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 定义损失函数和优化器
        criterion = DirectionAwareLoss(alpha=alpha)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # 使用余弦退火学习率调度器
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)
        
        # 快速训练几个轮次
        model.train()
        for epoch in range(20):  # 用较少的轮次评估超参数
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
        
        # 在验证集上评估
        model.eval()
        val_loss = 0.0
        val_dir_acc = 0.0
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # 计算方向准确率（如果可能）
                if i < len(val_loader) - 1:
                    next_inputs, next_targets = next(iter(val_loader))
                    next_inputs, next_targets = next_inputs.to(device), next_targets.to(device)
                    next_outputs = model(next_inputs)
                    
                    pred_diff = (outputs - next_outputs) > 0
                    true_diff = (targets - next_targets) > 0
                    dir_acc = (pred_diff == true_diff).float().mean().item()
                    val_dir_acc += dir_acc
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dir_acc = val_dir_acc / (len(val_loader) - 1) if len(val_loader) > 1 else 0
        
        print(f"验证损失: {avg_val_loss:.6f}, 方向准确率: {avg_val_dir_acc:.2%}")
        
        # 保存结果
        results.append({
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'alpha': alpha,
            'val_loss': avg_val_loss,
            'val_dir_acc': avg_val_dir_acc
        })
        
        # 更新最佳参数
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_params = {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'alpha': alpha
            }
    
    # 保存超参数搜索结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('models/hyperparameter_search_results.csv', index=False)
    
    print("\n超参数搜索完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print("最佳参数:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params

# 改进的LSTM+Transformer模型
class EnhancedLSTMTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=4, dropout=0.4, 
                 nhead=8, transformer_layers=2):
        super(EnhancedLSTMTransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 启用cuDNN加速
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2,  # 双向LSTM, 所以维度翻倍
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',  # 使用GELU激活函数提高性能
            batch_first=True,
            norm_first=True     # 使用Pre-LN结构, 提高训练稳定性
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_layers
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        # 批归一化和Dropout
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name:
                if len(param.data.shape) >= 2:
                    if 'fc' in name:
                        nn.init.kaiming_normal_(param.data, mode='fan_out', nonlinearity='relu')
                    else:
                        nn.init.xavier_normal_(param.data, gain=1.0)
                else:
                    # 一维张量(如BatchNorm的weight)使用正态分布初始化
                    nn.init.normal_(param.data, mean=1.0, std=0.02)
    
    def create_causal_mask(self, seq_len):
        # 创建因果掩码，确保当前时间步只能看到过去的信息
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(next(self.parameters()).device)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.size()
        
        # LSTM层
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]
        
        # 创建因果掩码
        mask = self.create_causal_mask(seq_len)
        
        # Transformer编码器
        transformer_out = self.transformer_encoder(lstm_out, mask=mask)  # [batch_size, seq_len, hidden_size*2]
        
        # 获取最后时间步的输出
        out = transformer_out[:, -1, :]  # [batch_size, hidden_size*2]
        
        # 全连接层
        out = self.fc1(out)
        out = F.gelu(out)  # 使用GELU激活函数
        out = self.bn1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = F.gelu(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        return out

# 风险控制和资金管理模块
class RiskManagement:
    def __init__(self, initial_capital, max_drawdown_pct=0.2, position_sizing='fixed'):
        """
        风险管理器
        
        参数:
        - initial_capital: 初始资金
        - max_drawdown_pct: 最大回撤百分比限制
        - position_sizing: 仓位计算方法 ('fixed', 'kelly', 'volatility')
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.position_sizing = position_sizing
        
        # 交易历史记录
        self.trades_history = []
        
        # 绩效指标
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0
        self.total_loss = 0
        
        # 风控状态
        self.risk_triggered = False
        self.risk_message = ""
    
    def update_capital(self, new_capital):
        """更新当前资金并检查风险触发条件"""
        self.current_capital = new_capital
        
        # 更新峰值资本
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
        
        # 计算当前回撤
        if self.peak_capital > 0:
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            
            # 检查是否超过最大回撤限制
            if current_drawdown > self.max_drawdown_pct:
                self.risk_triggered = True
                self.risk_message = f"最大回撤触发: 当前回撤 {current_drawdown:.2%} 超过限制 {self.max_drawdown_pct:.2%}"
                return False
        
        return True
    
    def calculate_position_size(self, price, volatility, win_rate=None):
        """
        计算仓位大小
        
        参数:
        - price: 当前价格
        - volatility: 波动率
        - win_rate: 获胜率 (用于凯利公式)
        
        返回:
        - position_size: 头寸规模 (金额)
        """
        if self.position_sizing == 'fixed':
            # 固定比例仓位
            return self.current_capital * 0.1  # 使用10%的资金
        
        elif self.position_sizing == 'kelly':
            # 凯利公式
            if win_rate is None:
                if self.win_count + self.loss_count > 0:
                    win_rate = self.win_count / (self.win_count + self.loss_count)
                else:
                    win_rate = 0.5  # 默认值
            
            # 计算平均盈亏比
            if self.win_count > 0 and self.loss_count > 0:
                avg_win = self.total_profit / self.win_count if self.win_count > 0 else 0
                avg_loss = abs(self.total_loss) / self.loss_count if self.loss_count > 0 else 1
                profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1
            else:
                profit_loss_ratio = 1
            
            # 计算凯利比例
            kelly_fraction = win_rate - (1 - win_rate) / profit_loss_ratio
            
            # 应用安全系数（使用凯利公式的一半）
            safe_kelly = max(0, kelly_fraction * 0.5)
            
            return self.current_capital * safe_kelly
        
        elif self.position_sizing == 'volatility':
            # 基于波动率的仓位
            if volatility > 0:
                # 波动越大，仓位越小
                vol_adjusted_position = 0.1 / volatility
                # 限制最小和最大仓位
                vol_adjusted_position = max(0.01, min(0.2, vol_adjusted_position))
                return self.current_capital * vol_adjusted_position
            else:
                return self.current_capital * 0.05  # 默认值
    
    def record_trade(self, trade_info):
        """记录交易并更新统计数据"""
        self.trades_history.append(trade_info)
        
        # 更新胜负统计
        profit = trade_info.get('profit', 0)
        if profit > 0:
            self.win_count += 1
            self.total_profit += profit
        elif profit < 0:
            self.loss_count += 1
            self.total_loss += profit  # profit已经是负数
    
    def get_statistics(self):
        """获取风险管理统计信息"""
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'risk_triggered': self.risk_triggered,
            'risk_message': self.risk_message,
            'current_drawdown': current_drawdown,
            'max_drawdown_limit': self.max_drawdown_pct
        }


# 交易执行模块 - 模拟交易实现
class TradingExecutor:
    def __init__(self, initial_capital=10000.0, risk_manager=None):
        """
        交易执行器
        
        参数:
        - initial_capital: 初始资金
        - risk_manager: 风险管理器实例
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0.0  # 持仓数量
        self.entry_price = 0.0  # 入场价格
        
        # 创建风险管理器（如果未提供）
        self.risk_manager = risk_manager or RiskManagement(
            initial_capital=initial_capital,
            max_drawdown_pct=0.2,
            position_sizing='volatility'
        )
        
        # 交易记录
        self.trades = []
        self.trade_id = 0
        
        # 交易配置
        self.commission_rate = 0.001  # 0.1% 交易手续费
        self.slippage_pct = 0.001  # 0.1% 滑点
        
        # 当前交易状态
        self.is_in_trade = False
        self.current_trade_type = None  # 'long' 或 'short'
        
    def apply_slippage(self, price, is_buy):
        """应用滑点，买入价格略高，卖出价格略低"""
        return price * (1 + self.slippage_pct) if is_buy else price * (1 - self.slippage_pct)
    
    def calculate_commission(self, price, quantity):
        """计算交易手续费"""
        return price * quantity * self.commission_rate
    
    def execute_buy(self, price, timestamp, volatility=0.01, confidence=1.0):
        """
        执行买入操作
        
        参数:
        - price: 当前价格
        - timestamp: 交易时间戳
        - volatility: 当前市场波动率
        - confidence: 交易信号的置信度 (0-1)
        
        返回:
        - success: 交易是否成功
        - message: 交易信息或错误消息
        """
        # 检查风险控制状态
        if self.risk_manager.risk_triggered:
            return False, f"风险控制已触发: {self.risk_manager.risk_message}"
        
        # 如果已经持有仓位，则不执行买入
        if self.is_in_trade:
            return False, "已持有仓位，无法执行买入"
        
        # 应用滑点
        execution_price = self.apply_slippage(price, True)
        
        # 计算仓位大小（调整置信度）
        position_value = self.risk_manager.calculate_position_size(price, volatility) * confidence
        
        # 确保有足够的资金
        if position_value > self.current_capital:
            position_value = self.current_capital
        
        # 计算数量和手续费
        quantity = position_value / execution_price
        commission = self.calculate_commission(execution_price, quantity)
        
        # 更新资金和持仓
        self.current_capital -= (position_value + commission)
        self.position = quantity
        self.entry_price = execution_price
        self.is_in_trade = True
        self.current_trade_type = 'long'
        
        # 记录交易
        self.trade_id += 1
        trade_info = {
            'trade_id': self.trade_id,
            'type': 'buy',
            'timestamp': timestamp,
            'price': execution_price,
            'quantity': quantity,
            'value': position_value,
            'commission': commission,
            'capital_after': self.current_capital
        }
        self.trades.append(trade_info)
        
        return True, f"买入执行: {quantity:.6f} @ {execution_price:.2f}, 总值: {position_value:.2f}, 手续费: {commission:.2f}"
    
    def execute_sell(self, price, timestamp):
        """
        执行卖出操作
        
        参数:
        - price: 当前价格
        - timestamp: 交易时间戳
        
        返回:
        - success: 交易是否成功
        - message: 交易信息或错误消息
        """
        # 如果没有持仓，则无法卖出
        if not self.is_in_trade or self.position <= 0:
            return False, "没有持仓，无法执行卖出"
        
        # 应用滑点
        execution_price = self.apply_slippage(price, False)
        
        # 计算卖出价值和手续费
        sell_value = self.position * execution_price
        commission = self.calculate_commission(execution_price, self.position)
        
        # 计算利润
        entry_value = self.position * self.entry_price
        profit = sell_value - entry_value - commission - self.calculate_commission(self.entry_price, self.position)
        
        # 更新资金和持仓
        self.current_capital += (sell_value - commission)
        
        # 记录交易
        self.trade_id += 1
        trade_info = {
            'trade_id': self.trade_id,
            'type': 'sell',
            'timestamp': timestamp,
            'price': execution_price,
            'quantity': self.position,
            'value': sell_value,
            'commission': commission,
            'profit': profit,
            'profit_pct': profit / entry_value * 100 if entry_value > 0 else 0,
            'capital_after': self.current_capital
        }
        self.trades.append(trade_info)
        
        # 更新风险管理器统计数据
        self.risk_manager.record_trade(trade_info)
        self.risk_manager.update_capital(self.current_capital)
        
        # 重置交易状态
        self.position = 0.0
        self.entry_price = 0.0
        self.is_in_trade = False
        self.current_trade_type = None
        
        return True, f"卖出执行: {trade_info['quantity']:.6f} @ {execution_price:.2f}, 总值: {sell_value:.2f}, 利润: {profit:.2f} ({trade_info['profit_pct']:.2f}%)"
    
    def execute_signal(self, signal, price, timestamp, volatility=0.01, confidence=1.0):
        """基于信号执行交易"""
        if signal == "买入":
            return self.execute_buy(price, timestamp, volatility, confidence)
        elif signal == "卖出" and self.is_in_trade:
            return self.execute_sell(price, timestamp)
        elif signal == "持有":
            return False, "保持当前仓位，不执行交易"
        else:
            return False, f"未知信号: {signal}"
    
    def get_performance_metrics(self):
        """计算交易绩效指标"""
        # 从风险管理器获取基本统计信息
        stats = self.risk_manager.get_statistics()
        
        # 计算其他指标
        profit_trades = [t for t in self.trades if t.get('type') == 'sell' and t.get('profit', 0) > 0]
        loss_trades = [t for t in self.trades if t.get('type') == 'sell' and t.get('profit', 0) <= 0]
        
        avg_profit = sum([t.get('profit', 0) for t in profit_trades]) / len(profit_trades) if profit_trades else 0
        avg_loss = sum([t.get('profit', 0) for t in loss_trades]) / len(loss_trades) if loss_trades else 0
        
        # 每笔交易的回报率
        returns = [t.get('profit_pct', 0) for t in self.trades if t.get('type') == 'sell']
        
        # 计算夏普比率（简化，假设无风险利率为0）
        if returns:
            mean_return = sum(returns) / len(returns)
            std_return = (sum([(r - mean_return) ** 2 for r in returns]) / len(returns)) ** 0.5
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        capital_history = [self.initial_capital]
        for trade in self.trades:
            if trade.get('capital_after') is not None:
                capital_history.append(trade.get('capital_after'))
        
        max_drawdown = 0
        peak = capital_history[0]
        for capital in capital_history:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            **stats,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': abs(stats['total_profit'] / stats['total_loss']) if stats['total_loss'] < 0 else float('inf'),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len([t for t in self.trades if t.get('type') in ['buy', 'sell']])
        }
    
    def save_trade_history(self, filename):
        """保存交易历史到CSV文件"""
        if not self.trades:
            return False, "没有交易记录"
        
        df = pd.DataFrame(self.trades)
        df.to_csv(filename, index=False)
        
        return True, f"交易历史已保存到 {filename}"

# 添加高级LSTM-Transformer混合模型
class AdvancedLSTMTransformerModel(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size=512, num_lstm_layers=4, num_transformer_layers=4, 
                 nhead=16, dropout=0.5, norm_first=True):
        super(AdvancedLSTMTransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        # 启用cuDNN加速
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            
        # 特征扩展层
        self.feature_expansion = nn.Linear(input_size, hidden_size // 2)
        
        # 双向LSTM层
        self.bi_lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size // 2,
            num_layers=num_lstm_layers // 2,  # 降低LSTM层数
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,  # 双向LSTM的输出维度
            hidden_size=hidden_size,
            num_layers=num_lstm_layers // 2,  # 降低LSTM层数
            dropout=dropout,
            batch_first=True
        )
        
        # 位置编码
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, seq_len, hidden_size)
        )
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=norm_first
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 跳跃连接
        self.skip_connection = nn.Linear(input_size, hidden_size)
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_normal_(param.data, gain=1.0)
        
    def create_causal_mask(self, seq_len):
        # 创建因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(next(self.parameters()).device)
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.size()
        
        # 特征扩展
        x_expanded = self.feature_expansion(x)  # [batch_size, seq_len, hidden_size//2]
        
        # 跳跃连接
        skip = self.skip_connection(x)  # [batch_size, seq_len, hidden_size]
        
        # 双向LSTM
        bi_lstm_out, _ = self.bi_lstm(x_expanded)  # [batch_size, seq_len, hidden_size]
        
        # LSTM层
        lstm_out, _ = self.lstm(bi_lstm_out)  # [batch_size, seq_len, hidden_size]
        
        # 添加位置编码
        lstm_out = lstm_out + self.pos_encoder[:, :seq_len, :]
        
        # 添加跳跃连接
        lstm_out = lstm_out + skip
        
        # 创建因果掩码
        mask = self.create_causal_mask(seq_len)
        
        # Transformer编码器
        # 注：使用torch.compile可加速Transformer
        if torch.cuda.is_available() and hasattr(torch, 'compile'):
            # 只在PyTorch 2.0+上使用
            if not hasattr(self, '_compiled_transformer'):
                self._compiled_transformer = torch.compile(self.transformer_encoder)
            transformer_out = self._compiled_transformer(lstm_out, mask=mask)
        else:
            transformer_out = self.transformer_encoder(lstm_out, mask=mask)
        
        # 获取最后时间步的输出
        out = transformer_out[:, -1, :]  # [batch_size, hidden_size]
        
        # 全连接层
        out = self.fc_layers(out)  # [batch_size, 1]
        
        return out

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 主函数
def main():
    # 设置随机种子，确保结果可复现
    set_seed(42)
    
    # 设置默认配置参数
    disable_augmentation = True  # 默认禁用数据增强
    model_type = "lstm"  # 默认使用LSTM模型
    loss_function = "focal"  # 默认使用focal loss
    
    # 设置设备类型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 启用CUDA优化（如果可用）
    if device == 'cuda':
        # 启用自动混合精度训练
        torch.cuda.amp.autocast(enabled=True)
        torch.backends.cudnn.deterministic = False  # 为了性能
        torch.backends.cudnn.benchmark = True  # 启用cuDNN自动优化
        
        # 启用TF32加速
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"GPU加速已启用: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    # 检查计算环境和硬件配置
    check_environment()
    
    # 加载数据
    train_data, val_data, test_data, feature_scaler, target_scaler, original_df = load_training_data()
    
    # 设置超参数
    input_size = train_data[0].shape[-1]
    seq_len = train_data[0].shape[1]
    batch_size = 512  # 从2048减小到512，提高梯度更新频率
    lr = 0.00005  # 保持较低的学习率
    
    # 创建数据集和数据加载器
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # 显示训练数据信息
    print(f"输入特征数: {input_size}, 序列长度: {seq_len}")
    print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}, 测试集大小: {X_test.shape}")
    
    # 数据增强 - 基于硬件容量决定增强策略
    mem_gb = torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0
    
    if disable_augmentation:
        print("数据增强功能已禁用，使用原始数据进行训练")
        X_train_aug, y_train_aug = X_train, y_train
    elif mem_gb >= 8:
        print(f"显存充足({mem_gb:.1f}GB)，启用数据增强...")
        aug_factor = min(int(mem_gb / 4), 4)  # 根据显存大小决定增强倍数
        print(f"计划增强训练集至 {aug_factor}x 原始大小")
        X_train_aug, y_train_aug = create_augmented_dataset(X_train, y_train, num_augmentations=aug_factor)
        print(f"增强后训练集大小: {X_train_aug.shape}")
    else:
        print(f"显存有限({mem_gb:.1f}GB)，使用原始数据进行训练")
        X_train_aug, y_train_aug = X_train, y_train
    
    # 将数据转换为张量并创建数据加载器
    train_tensor_x = torch.FloatTensor(X_train_aug)
    train_tensor_y = torch.FloatTensor(y_train_aug)
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    val_tensor_x = torch.FloatTensor(X_val)
    val_tensor_y = torch.FloatTensor(y_val)
    val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # 确保模型输出目录存在
    os.makedirs('models', exist_ok=True)
    
    # 根据选择的模型类型创建模型
    model_options = {
        'lstm': EnhancedLSTMModel,
        'attention': AttentionLSTMModel,
        'transformer': EnhancedLSTMTransformerModel,
        'advanced': AdvancedLSTMTransformerModel
    }
    
    model_class = model_options.get(model_type.lower(), EnhancedLSTMModel)
    model_name_map = {
        'lstm': 'Enhanced LSTM',
        'attention': 'Attention LSTM',
        'transformer': 'LSTM+Transformer',
        'advanced': 'Advanced LSTM+Transformer'
    }
    
    print(f"使用模型: {model_name_map.get(model_type.lower(), 'Enhanced LSTM')}")
    
    if model_type.lower() == 'advanced':
        model = model_class(input_size=input_size, seq_len=seq_len).to(device)
    else:
        model = model_class(input_size=input_size).to(device)
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,} (可训练: {trainable_params:,})")
    
    # 损失函数和优化器
    criterion_options = {
        'mse': nn.MSELoss(),
        'direction': DirectionAwareLoss(),
        'focal': FocalMSELoss(direction_weight=0.3, gamma=2.0)  # 方向损失权重已经显著降低
    }
    
    criterion = criterion_options.get(loss_function.lower(), nn.MSELoss())
    print(f"使用损失函数: {loss_function}")
    
    # 使用AdamW优化器和OneCycleLR调度器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # 增加权重衰减至1e-4，提高正则化强度
    
    # 学习率调度器：ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',          # 监控最小化验证损失
        factor=0.5,          # 每次降低一半学习率
        patience=10,         # 10轮没有改善就降低学习率
        min_lr=1e-6,         # 最小学习率限制
        verbose=True         # 打印学习率变化
    )
    
    # 训练模型
    train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler,
        device,
        epochs=150,          # 最大轮次
        patience=50,         # 增加早停耐心，给模型更多学习机会
        model_save_path='models',
        l2_weight=1e-4       # 增加L2正则化强度
    )
    
    # 测试模型
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test)
    
    y_pred, y_test_original, test_timestamps = test_model(
        model, X_test_tensor, y_test_tensor, target_scaler, original_df, len(X_train) + len(X_val), device)
    
    # 评估性能
    metrics = calculate_metrics(y_test_original, y_pred)
    
    # 保存测试集预测结果可视化
    plt.figure(figsize=(15, 6))
    plt.plot(test_timestamps, y_test_original, label='实际价格', color='blue')
    plt.plot(test_timestamps, y_pred, label='预测价格', color='red', linestyle='--')
    plt.title('比特币价格预测')
    plt.xlabel('时间')
    plt.ylabel('价格 (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/price_prediction_comparison.png')
    plt.close()
    
    # 生成滚动窗口评估
    window_sizes = [7*24, 14*24, 30*24]  # 小时为单位的窗口
    window_names = ['7day', '14day', '30day']
    
    for window_size, window_name in zip(window_sizes, window_names):
        if len(y_test_original) <= window_size:
            print(f"测试集大小({len(y_test_original)})小于窗口大小({window_size})，跳过{window_name}滚动窗口评估")
            continue
            
        # 计算滚动窗口的性能指标
        rolling_metrics = []
        
        for i in range(len(y_test_original) - window_size + 1):
            window_actual = y_test_original[i:i+window_size]
            window_pred = y_pred[i:i+window_size]
            window_metrics = calculate_metrics(window_actual, window_pred)
            
            # 添加窗口时间范围信息
            window_metrics['start_time'] = test_timestamps[i]
            window_metrics['end_time'] = test_timestamps[i+window_size-1]
            
            rolling_metrics.append(window_metrics)
        
        # 转换为DataFrame
        rolling_df = pd.DataFrame(rolling_metrics)
        
        # 保存滚动窗口指标
        rolling_df.to_csv(f'models/rolling_{window_name}_metrics.csv', index=False)
        
        # 找出方向准确率最高的窗口
        best_dir_acc_idx = rolling_df['DIRECTION_ACCURACY'].idxmax()
        best_dir_acc = rolling_df.loc[best_dir_acc_idx]
        
        print(f"{window_name}窗口最佳方向准确率: {best_dir_acc['DIRECTION_ACCURACY']:.2f}%, "
              f"时间段: {best_dir_acc['start_time']} 到 {best_dir_acc['end_time']}")
    
    # 生成交易信号
    signals_df = generate_trading_signals(y_test_original, y_pred, test_timestamps, original_df)
    
    # 回测策略
    print(f"开始回测交易策略，初始资金: $10000.0...")
    backtest_results = backtest_strategy(signals_df)
    
    # 自动清理不再需要的中间结果，释放内存
    del train_tensor_x, train_tensor_y, val_tensor_x, val_tensor_y
    del train_dataset, val_dataset, train_loader, val_loader 
    del X_train, X_val, X_test, y_train, y_val, y_test
    if 'X_train_aug' in locals(): del X_train_aug, y_train_aug
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, metrics, backtest_results

if __name__ == "__main__":
    main() 

"""
量化交易系统优化总结：

1. 数据预处理增强:
   - 添加高级时序特征: 波动率指标、成交量特征和趋势指标
   - 新增市场情绪指标: 价格动量、成交量权重情绪指标和恐惧贪婪指数
   - 改进数据清洗: 异常值处理和鲁棒缩放
   - 添加技术分析指标: ATR、RSI、CCI等常用技术指标

2. 模型架构优化:
   - 新增高级LSTM-Transformer混合模型
   - 添加特征重要性学习机制
   - 实现时间步重要性加权
   - 增加多头自注意力和残差连接
   - 层归一化和全局池化层
   - 前馈神经网络和因果掩码

3. 训练策略改进:
   - 使用AdamW优化器和权重衰减
   - OneCycleLR学习率策略
   - 混合精度训练
   - 模型参数量统计

4. 回测系统优化:
   - 降低波动率窗口，更灵活地捕捉市场变化
   - 调整交易信号阈值，提高交易频率
   - 更详细的性能指标记录和可视化

这些优化共同提高了模型的预测能力、训练效率和泛化性能，使量化交易系统更加稳健和高效。
""" 