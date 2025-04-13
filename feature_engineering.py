import pandas as pd
import numpy as np
# 使用ta库代替talib
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import os
import matplotlib.pyplot as plt

# 读取数据
print("加载数据...")
df = pd.read_csv('data/processed/btc_with_funding_rate_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

print(f"数据范围: {df.index.min()} 至 {df.index.max()}")
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# 确保输出目录存在
os.makedirs('data/features', exist_ok=True)

# 第1步: 计算基础技术指标
print("计算基础技术指标...")
# 移动平均线
df['sma_7'] = ta.trend.sma_indicator(df['close'], window=7)
df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
df['sma_100'] = ta.trend.sma_indicator(df['close'], window=100)

# RSI
df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

# MACD
macd = ta.trend.MACD(df['close'])
df['macd_line'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_hist'] = macd.macd_diff()

# CCI
df['cci_20'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)

# Williams %R
df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)

# 随机指标
stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=5, smooth_window=3)
df['stoch_k'] = stoch.stoch()
df['stoch_d'] = stoch.stoch_signal()

# ADX
df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

# 波动率
df['volatility_20'] = df['close'].rolling(20).std()

# 收益率
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['return_1d'] = df['close'].pct_change()
df['return_5d'] = df['close'].pct_change(5)

# 价格动量
df['price_momentum_5'] = df['close'] - df['close'].shift(5)
df['price_momentum_10'] = df['close'] - df['close'].shift(10)

# 第2步: 基本指标转换与增强
print("基本指标转换与增强...")

# 交叉信号特征
df['macd_cross'] = 0
df.loc[(df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1)), 'macd_cross'] = 1  # 金叉
df.loc[(df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1)), 'macd_cross'] = -1  # 死叉

# RSI超买超卖区域信号
df['rsi_zone'] = 0
df.loc[df['rsi_14'] > 70, 'rsi_zone'] = 1  # 超买
df.loc[df['rsi_14'] < 30, 'rsi_zone'] = -1  # 超卖

# 价格与移动平均线交叉
df['price_sma20_cross'] = 0
df.loc[(df['close'] > df['sma_20']) & (df['close'].shift(1) <= df['sma_20'].shift(1)), 'price_sma20_cross'] = 1
df.loc[(df['close'] < df['sma_20']) & (df['close'].shift(1) >= df['sma_20'].shift(1)), 'price_sma20_cross'] = -1

# 变化率特征
df['rsi_change'] = df['rsi_14'] - df['rsi_14'].shift(1)
df['macd_hist_change'] = df['macd_hist'] - df['macd_hist'].shift(1)
df['cci_change'] = df['cci_20'] - df['cci_20'].shift(1)

# 多周期变化率
df['rsi_change_3d'] = df['rsi_14'] - df['rsi_14'].shift(3)
df['cci_change_5d'] = df['cci_20'] - df['cci_20'].shift(5)

# 指标加速度（二阶变化）
df['rsi_acceleration'] = df['rsi_change'] - df['rsi_change'].shift(1)

# 振荡指标与趋势指标组合
df['rsi_adx_product'] = df['rsi_14'] * df['adx']

# 价格与波动性关系 - 避免除以0和极端值
volatility_20_safe = df['volatility_20'].copy()
volatility_20_safe = volatility_20_safe.replace(0, np.nan)
df['price_volatility_ratio'] = df['close'] / volatility_20_safe
df['price_volatility_ratio'] = df['price_volatility_ratio'].replace([np.inf, -np.inf], np.nan)

# 随机指标与Williams %R组合
df['stoch_williams_divergence'] = df['stoch_k'] - (100 + df['williams_r'])  # 两者理论上互补

# 第3步: 高级特征构建
print("高级特征构建...")

# 技术指标综合评分
df['bull_score'] = 0
df.loc[df['rsi_14'] < 30, 'bull_score'] += 1
df.loc[df['macd_hist'] > 0, 'bull_score'] += 1
df.loc[df['cci_20'] < -100, 'bull_score'] += 1
df.loc[df['williams_r'] < -80, 'bull_score'] += 1
df.loc[df['stoch_k'] < 20, 'bull_score'] += 1

# 指标一致性得分
def get_direction(value):
    if pd.isna(value):
        return 0
    return 1 if value > 0 else (-1 if value < 0 else 0)

df['macd_direction'] = df['macd_hist'].apply(get_direction)
df['rsi_direction'] = df['rsi_change'].apply(get_direction)
df['momentum_direction'] = df['price_momentum_5'].apply(get_direction)

df['indicator_consensus'] = df['macd_direction'] + df['rsi_direction'] + df['momentum_direction']

# 时间序列特征
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

# 周期性编码
df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
df['day_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
df['day_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))

# 波动性和成交量特征 - 安全处理除法
volume_rolling_mean = df['volume'].rolling(5).mean()
volume_rolling_mean = volume_rolling_mean.replace(0, np.nan)
df['volume_weighted_price'] = df['close'] * df['volume'] / volume_rolling_mean
df['volume_weighted_price'] = df['volume_weighted_price'].replace([np.inf, -np.inf], np.nan)

# 异常成交量检测 - 安全处理标准差为0的情况
volume_rolling_std = df['volume'].rolling(20).std()
volume_rolling_std = volume_rolling_std.replace(0, np.nan)
df['volume_z_score'] = (df['volume'] - df['volume'].rolling(20).mean()) / volume_rolling_std
df['volume_z_score'] = df['volume_z_score'].replace([np.inf, -np.inf], np.nan)

# 波动性趋势特征 - 安全处理除法
volatility_shift = df['volatility_20'].shift(5)
volatility_shift = volatility_shift.replace(0, np.nan)
df['volatility_trend'] = df['volatility_20'] / volatility_shift
df['volatility_trend'] = df['volatility_trend'].replace([np.inf, -np.inf], np.nan)

# 第4步: 市场情绪特征
print("市场情绪特征...")

# 资金费率与价格变动的关系
df['funding_price_correlation'] = df['funding_rate'].rolling(24).corr(df['log_return'])

# 买卖压力平衡指标 - 安全处理除法
taker_sell_volume_safe = df['taker_sell_volume'].copy()
taker_sell_volume_safe = taker_sell_volume_safe.replace(0, np.nan)
df['buy_sell_pressure'] = df['taker_buy_volume'] / taker_sell_volume_safe
df['buy_sell_pressure'] = df['buy_sell_pressure'].replace([np.inf, -np.inf], np.nan)

# 简化的恐慌指数（基于RSI和波动率）- 安全处理除法
vol_rolling_max = df['volatility_20'].rolling(30).max()
vol_rolling_max = vol_rolling_max.replace(0, np.nan)
df['fear_greed_simple'] = (df['rsi_14'] - 50) / 50 - df['volatility_20'] / vol_rolling_max
df['fear_greed_simple'] = df['fear_greed_simple'].replace([np.inf, -np.inf], np.nan)
df['fear_greed_change'] = df['fear_greed_simple'] - df['fear_greed_simple'].shift(1)
df['fear_greed_ma_diff'] = df['fear_greed_simple'] - df['fear_greed_simple'].rolling(5).mean()

# 第5步: 技术指标背离特征
print("技术指标背离特征...")

# RSI与价格背离
df['price_higher_high'] = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
df['rsi_lower_high'] = (df['rsi_14'] < df['rsi_14'].shift(1)) & (df['rsi_14'].shift(1) > df['rsi_14'].shift(2))
df['bearish_divergence'] = df['price_higher_high'] & df['rsi_lower_high']

# MACD与价格背离
df['macd_hist_falling'] = df['macd_hist'] < df['macd_hist'].shift(1)
df['price_rising'] = df['close'] > df['close'].shift(1)
df['macd_price_divergence'] = df['macd_hist_falling'] & df['price_rising']

# 第6步: 拉格特征（时间滞后特征）
print("拉格特征（时间滞后特征）...")

# 为主要指标创建滞后特征
lag_features = ['rsi_14', 'macd_hist', 'cci_20', 'volume', 'volatility_20']
lag_periods = [1, 3, 5, 7, 14]

for feature in lag_features:
    for lag in lag_periods:
        df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

# 第7步: 统计和数学转换
print("统计和数学转换...")

# 对数转换
df['volume_log'] = np.log1p(df['volume'])

# 标准化/归一化（创建副本而不是直接修改原始特征）
features_to_scale = ['rsi_14', 'macd_hist', 'volume_log', 'volatility_20']

# 选择标准化方法
scaler = MinMaxScaler()
# 确保没有nan和inf
features_data = df[features_to_scale].fillna(0).replace([np.inf, -np.inf], 0)
scaled_features = scaler.fit_transform(features_data)

# 将标准化结果添加为新列
for i, feature in enumerate(features_to_scale):
    df[f'{feature}_scaled'] = scaled_features[:, i]

# 第8步: 特征选择与评估
print("特征选择与评估...")

# 准备特征矩阵和目标变量
# 使用所有生成的特征，排除原始价格数据和日期
X = df.drop(['open', 'high', 'low', 'close', 'volume', 'number_of_trades', 
              'taker_buy_volume', 'taker_sell_volume', 'funding_rate'], axis=1)
              
# 预测任务: 预测下一个时间点的收盘价变化方向
df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
y = df['target'].fillna(0)

# 填充缺失值并替换无穷大
X = X.fillna(0).replace([np.inf, -np.inf], 0)

# 使用随机森林评估特征重要性
print("评估特征重要性...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# 保存特征重要性
feature_importance.to_csv('data/features/feature_importance.csv', index=False)
print(f"前10个最重要特征:\n{feature_importance.head(10)}")

# 选择最重要的特征
selector = SelectFromModel(model, threshold='median')
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# 保存选择的特征
pd.DataFrame({'selected_features': selected_features}).to_csv('data/features/selected_features.csv', index=False)
print(f"选择了 {len(selected_features)} 个特征")

# 保存处理后的数据
print("保存处理后的数据...")
df.to_csv('data/features/btc_full_features.csv')

# 可视化一些关键特征
print("创建特征可视化...")

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df.index[-100:], df['rsi_14'].iloc[-100:])
plt.title('RSI (14)')
plt.axhline(y=70, color='r', linestyle='-')
plt.axhline(y=30, color='g', linestyle='-')

plt.subplot(2, 2, 2)
plt.plot(df.index[-100:], df['macd_hist'].iloc[-100:])
plt.title('MACD Histogram')

plt.subplot(2, 2, 3)
plt.plot(df.index[-100:], df['bull_score'].iloc[-100:])
plt.title('Bull Score')

plt.subplot(2, 2, 4)
plt.scatter(df.index[-100:], df['buy_sell_pressure'].iloc[-100:], c=df['close'].pct_change().iloc[-100:] > 0, cmap='coolwarm')
plt.title('Buy/Sell Pressure vs Price Change')

plt.tight_layout()
plt.savefig('data/features/key_features_visualization.png')

print("特征工程完成!") 