import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import ta  # 技术分析库
from binance.client import Client
from dotenv import load_dotenv
import datetime

# 加载 .env 文件
load_dotenv()

# 获取API密钥
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

# 初始化币安客户端
client = Client(api_key, api_secret)

# 读取已保存的K线数据
def load_kline_data(filename):
    print(f"正在加载K线数据: {filename}")
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 转换数值列为浮点数
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 过滤未来的数据（如果有）
    current_time = datetime.datetime.now()
    df = df[df['timestamp'] <= current_time]
    
    return df

# 创建技术指标特征
def add_technical_indicators(df):
    print("正在添加技术指标...")
    # 基本移动平均线
    df['SMA5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['SMA10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
    
    # 指数移动平均线
    df['EMA5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['EMA10'] = ta.trend.ema_indicator(df['close'], window=10)
    df['EMA20'] = ta.trend.ema_indicator(df['close'], window=20)
    
    # 相对强弱指数 (RSI)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # 布林带
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_lower'] = bollinger.bollinger_lband()
    
    # 计算波动性 (Volatility)
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    # 成交量变化
    df['volume_change'] = df['volume'].pct_change()
    
    # 计算价格震荡幅度 (模拟价差spread的历史状态)
    df['price_range'] = (df['high'] - df['low']) / df['close']
    
    # 计算之前N个周期的累积成交量（可以用来模拟交易活跃度）
    df['cum_volume_3h'] = df['volume'].rolling(window=3).sum()
    df['cum_volume_6h'] = df['volume'].rolling(window=6).sum()
    df['cum_volume_12h'] = df['volume'].rolling(window=12).sum()
    
    # 计算价格动量
    df['momentum_1h'] = df['close'].pct_change(periods=1)
    df['momentum_3h'] = df['close'].pct_change(periods=3)
    df['momentum_6h'] = df['close'].pct_change(periods=6)
    df['momentum_12h'] = df['close'].pct_change(periods=12)
    df['momentum_24h'] = df['close'].pct_change(periods=24)
    
    return df

# 获取当前的深度数据和成交数据 (这只能获取当前的数据，不是历史数据)
def get_current_orderbook_and_trades(symbol):
    print(f"获取当前的订单簿和成交数据: {symbol}")
    try:
        # 获取订单簿数据
        depth_data = client.get_order_book(symbol=symbol)
        bids = depth_data['bids']
        asks = depth_data['asks']
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        
        # 计算订单簿累积量（深度）
        bid_depth = sum([float(x[1]) for x in bids[:10]])  # 前10个买单的累积量
        ask_depth = sum([float(x[1]) for x in asks[:10]])  # 前10个卖单的累积量
        
        # 获取最近的成交数据
        trades = client.get_recent_trades(symbol=symbol)
        trade_prices = [float(trade['price']) for trade in trades]
        trade_volumes = [float(trade['qty']) for trade in trades]
        average_trade_price = sum(trade_prices) / len(trade_prices)
        total_trade_volume = sum(trade_volumes)
        
        # 计算买卖单比例（买卖压力）
        buy_sell_ratio = bid_depth / ask_depth if ask_depth > 0 else 1.0
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'buy_sell_ratio': buy_sell_ratio,
            'average_trade_price': average_trade_price,
            'total_trade_volume': total_trade_volume
        }
    except Exception as e:
        print(f"获取实时订单簿和成交数据时出错: {e}")
        return None

# 处理异常值
def handle_outliers(df):
    print("正在处理数据中的异常值...")
    # 数值类型的列
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # 对每一列进行处理
    for col in numeric_columns:
        # 替换无穷大值为NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # 对于每一列，计算可接受范围(使用四分位数范围IQR方法检测异常值)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 定义异常值的上下限
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # 将超出范围的值替换为边界值
        df.loc[df[col] > upper_bound, col] = upper_bound
        df.loc[df[col] < lower_bound, col] = lower_bound
    
    # 使用前一个有效值填充NaN
    df = df.fillna(method='ffill')
    # 使用后一个有效值填充NaN（如果前面的ffill仍有NaN）
    df = df.fillna(method='bfill')
    # 如果还有NaN，用0填充
    df = df.fillna(0)
    
    return df

# 对数据进行标准化处理
def normalize_data(df):
    print("正在对数据进行归一化...")
    # 选择需要归一化的列
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'close_time', 'ignore']]
    
    # 初始化归一化器
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 对特征进行归一化
    df_normalized = df.copy()
    df_normalized[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df_normalized, scaler

# 主处理函数
def process_data_with_features():
    # 1. 加载K线数据
    kline_filename = "data/btcusdt_1h_data_2021_to_now.csv"
    df = load_kline_data(kline_filename)
    
    # 2. 添加技术指标特征
    df = add_technical_indicators(df)
    
    # 3. 获取当前的深度数据和成交数据（仅用于实时交易，不用于历史回测）
    current_data = get_current_orderbook_and_trades("BTCUSDT")
    
    if current_data:
        print("当前实时市场数据:")
        for key, value in current_data.items():
            print(f"  {key}: {value}")
    
    # 4. 处理异常值
    df = handle_outliers(df)
    
    # 5. 删除缺失值（如果还有的话）
    df = df.dropna()
    print(f"处理后的数据行数: {len(df)}")
    
    # 6. 对数据进行归一化
    df_normalized, scaler = normalize_data(df)
    
    # 7. 保存处理后的数据
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv("data/processed/btc_with_features.csv", index=False)
    df_normalized.to_csv("data/processed/btc_normalized_with_features.csv", index=False)
    
    print("\n数据处理完成!")
    print(f"原始特征数据保存至: data/processed/btc_with_features.csv")
    print(f"归一化特征数据保存至: data/processed/btc_normalized_with_features.csv")
    
    # 8. 显示数据统计信息
    print("\n数据统计信息:")
    print(f"总行数: {len(df)}")
    print(f"特征数量: {len(df.columns)}")
    print(f"日期范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 9. 显示部分特征名称
    print("\n部分特征名称:")
    print(", ".join(df.columns.tolist()[:15]) + ", ...")
    
    return df, df_normalized, scaler

if __name__ == "__main__":
    df, df_normalized, scaler = process_data_with_features()
    
    # 显示原始数据和归一化后数据的前几行
    print("\n原始数据前5行(部分列):")
    print(df[['timestamp', 'open', 'close', 'volume', 'RSI', 'price_range']].head())
    
    print("\n归一化数据前5行(部分列):")
    print(df_normalized[['timestamp', 'open', 'close', 'volume', 'RSI', 'price_range']].head()) 