import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载已处理的数据
def load_processed_data(filename):
    print(f"正在加载已处理的数据: {filename}")
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# 计算特征之间的相关性
def analyze_correlations(df):
    print("正在计算特征之间的相关性...")
    
    # 选择所有数值型列，排除无意义的索引列
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    exclude_cols = ['Unnamed: 0']  # 明确排除索引列
    numeric_columns = [col for col in numeric_columns if col not in exclude_cols]
    
    # 计算相关性矩阵
    correlation_matrix = df[numeric_columns].corr()
    
    # 创建输出目录
    os.makedirs('analysis', exist_ok=True)
    
    # 保存相关性矩阵到CSV
    correlation_matrix.to_csv('analysis/feature_correlation_matrix.csv')
    
    # 绘制热力图
    plt.figure(figsize=(16, 14))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('特征相关性热力图', fontsize=16)
    plt.tight_layout()
    plt.savefig('analysis/correlation_heatmap.png', dpi=300)
    plt.close()
    
    print(f"相关性矩阵已保存至: analysis/feature_correlation_matrix.csv")
    print(f"相关性热力图已保存至: analysis/correlation_heatmap.png")
    
    return correlation_matrix

# 计算与目标变量的互信息
def calculate_mutual_info(df, target_col='close'):
    print(f"计算特征与目标变量 {target_col} 的互信息...")
    
    # 选择数值型特征，排除无意义的索引列
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    exclude_cols = ['Unnamed: 0']  # 明确排除索引列
    features = [col for col in numeric_columns if col not in exclude_cols and col != target_col]
    
    # 计算互信息
    mi_scores = mutual_info_regression(df[features], df[target_col])
    mi_scores = pd.Series(mi_scores, index=features)
    mi_scores = mi_scores.sort_values(ascending=False)
    
    # 保存互信息分数到CSV
    mi_scores.to_csv('analysis/mutual_info_scores.csv')
    
    # 绘制互信息条形图 (前20个特征)
    plt.figure(figsize=(12, 10))
    mi_scores.head(20).plot.barh()
    plt.title(f'前20个特征与{target_col}的互信息分数', fontsize=16)
    plt.xlabel('互信息分数', fontsize=14)
    plt.ylabel('特征', fontsize=14)
    plt.tight_layout()
    plt.savefig('analysis/mutual_info_scores.png', dpi=300)
    plt.close()
    
    print(f"互信息分数已保存至: analysis/mutual_info_scores.csv")
    print(f"互信息分数图已保存至: analysis/mutual_info_scores.png")
    
    return mi_scores

# 特征选择：根据相关性和互信息选择最相关的特征
def select_features(df, correlation_matrix, mi_scores, target_col='close', top_n=15, corr_threshold=0.8):
    print(f"正在选择与{target_col}最相关的特征...")
    
    # 基于互信息选择最重要的特征
    top_mi_features = mi_scores.head(top_n).index.tolist()
    
    # 从相关性矩阵找出高度相关的特征对
    high_corr_features = set()
    target_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)
    
    # 去除高度相关的冗余特征
    processed_features = []
    for feature in top_mi_features:
        # 如果这个特征已经被处理过了，跳过
        if feature in high_corr_features:
            continue
            
        processed_features.append(feature)
        
        # 找出与当前特征高度相关的其他特征
        correlated_features = correlation_matrix.index[
            (correlation_matrix[feature].abs() > corr_threshold) & 
            (correlation_matrix.index != feature)
        ].tolist()
        
        # 添加到高度相关集合中
        high_corr_features.update(correlated_features)
    
    # 最终选择的特征（确保target_col不在里面）
    selected_features = [f for f in processed_features if f != target_col]
    
    print(f"已选择 {len(selected_features)} 个特征用于模型训练:")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    
    return selected_features

# 准备用于训练的数据集
def prepare_training_data(df, selected_features, target_col='close', sequence_length=24, prediction_horizon=1):
    print(f"正在准备训练数据，序列长度为 {sequence_length}，预测周期为 {prediction_horizon}...")
    
    # 确保所有选择的特征都存在于数据集中
    features = [f for f in selected_features if f in df.columns]
    
    # 添加目标变量
    if target_col not in features:
        features.append(target_col)
        
    # 选择特征子集
    data = df[features].values
    
    # 创建序列数据
    X, y = [], []
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        X.append(data[i:(i + sequence_length), :])
        y.append(data[i + sequence_length + prediction_horizon - 1, df.columns.get_loc(target_col)])
    
    X = np.array(X)
    y = np.array(y)
    
    # 分割训练集、验证集和测试集
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 创建数据目录
    os.makedirs('data/model_ready', exist_ok=True)
    
    # 保存特征名称和目标变量
    feature_info = {
        'features': features,
        'target': target_col,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon
    }
    
    # 将特征信息保存为JSON格式
    import json
    with open('data/model_ready/feature_info.json', 'w') as f:
        json.dump(feature_info, f)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_info

def analyze_direction_correlation():
    """
    分析特征与未来价格方向的相关性
    """
    print("开始分析特征与价格方向的相关性...")
    
    # 加载原始数据
    try:
        df = pd.read_csv('data/original_df_v2.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)
    except FileNotFoundError:
        print("错误: 找不到原始数据文件。请先运行 lstm_model.py 生成处理后的数据文件。")
        return
    
    # 创建目标变量: 未来价格方向 (1=上涨, 0=下跌)
    # 1小时后的价格方向
    df['next_price'] = df['close'].shift(-1)
    df['price_direction'] = (df['next_price'] > df['close']).astype(int)
    
    # 3小时后的价格方向
    df['next_price_3h'] = df['close'].shift(-3)
    df['price_direction_3h'] = (df['next_price_3h'] > df['close']).astype(int)
    
    # 计算价格变化率
    df['price_change'] = df['close'].pct_change()
    df['price_change_next'] = df['price_change'].shift(-1)
    
    # 识别特征列
    # 排除非特征列和无意义的索引列
    exclude_cols = ['timestamp', 'next_price', 'price_direction', 'next_price_3h', 
                    'price_direction_3h', 'price_change_next', 'Unnamed: 0', 
                    'price_change', 'index']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 创建输出目录
    os.makedirs('analysis', exist_ok=True)
    
    # 删除存在NaN的行
    df_clean = df.dropna()
    print(f"分析数据形状: {df_clean.shape}")
    
    # 1. 相关性分析 - 与未来价格方向的相关性
    correlations = {}
    for col in feature_cols:
        correlations[col] = {
            'pearson_1h': df_clean[col].corr(df_clean['price_direction']),
            'pearson_3h': df_clean[col].corr(df_clean['price_direction_3h']),
            'pearson_change': df_clean[col].corr(df_clean['price_change_next'])
        }
    
    corr_df = pd.DataFrame(correlations).T
    corr_df = corr_df.sort_values(by='pearson_1h', ascending=False)
    
    # 保存相关性结果
    corr_df.to_csv('analysis/feature_direction_correlation.csv')
    
    # 2. 互信息分析 (非线性关系)
    X = df_clean[feature_cols]
    y_dir_1h = df_clean['price_direction']
    y_dir_3h = df_clean['price_direction_3h']
    y_change = df_clean['price_change_next']
    
    # 使用互信息计算非线性相关性
    mi_selector_1h = SelectKBest(mutual_info_regression, k='all')
    mi_selector_1h.fit(X, y_dir_1h)
    mi_scores_1h = mi_selector_1h.scores_
    
    mi_selector_3h = SelectKBest(mutual_info_regression, k='all')
    mi_selector_3h.fit(X, y_dir_3h)
    mi_scores_3h = mi_selector_3h.scores_
    
    mi_selector_change = SelectKBest(mutual_info_regression, k='all')
    mi_selector_change.fit(X, y_change)
    mi_scores_change = mi_selector_change.scores_
    
    # 创建互信息得分DataFrame
    mi_df = pd.DataFrame({
        'feature': feature_cols,
        'mi_score_1h': mi_scores_1h,
        'mi_score_3h': mi_scores_3h,
        'mi_score_change': mi_scores_change
    })
    mi_df = mi_df.sort_values(by='mi_score_1h', ascending=False)
    mi_df.to_csv('analysis/feature_mutual_info_scores.csv', index=False)
    
    # 3. 可视化 - 相关性热图 (Top 20 特征)
    plt.figure(figsize=(12, 10))
    top_features = corr_df.index[:20]
    
    direction_corr = df_clean[list(top_features) + ['price_direction', 'price_direction_3h']].corr()
    mask = np.zeros_like(direction_corr)
    mask[np.triu_indices_from(mask)] = True
    
    sns.heatmap(direction_corr, cmap='coolwarm', annot=False, mask=mask, 
                vmin=-0.3, vmax=0.3, center=0)
    plt.title('特征与未来价格方向的相关性热图 (Top 20)')
    plt.tight_layout()
    plt.savefig('analysis/direction_correlation_heatmap.png')
    
    # 4. 可视化 - 互信息得分条形图 (Top 20)
    plt.figure(figsize=(12, 8))
    top_mi_features = mi_df.head(20)
    sns.barplot(x='mi_score_1h', y='feature', data=top_mi_features)
    plt.title('特征与1小时价格方向的互信息分数 (Top 20)')
    plt.tight_layout()
    plt.savefig('analysis/direction_mutual_info_scores.png')
    
    print(f"分析完成。结果已保存到 analysis/ 目录")
    print("Top 5 线性相关特征:")
    print(corr_df.head(5))
    print("\nTop 5 非线性相关特征 (互信息):")
    print(mi_df.head(5))
    
    return corr_df, mi_df, feature_cols

# 主函数
def main():
    # 1. 加载已处理的数据
    df = load_processed_data('data/original_df_v2.csv')
    
    # 2. 计算相关性矩阵
    correlation_matrix = analyze_correlations(df)
    
    # 3. 计算互信息分数
    target_col = 'close'  # 我们的目标是预测收盘价
    mi_scores = calculate_mutual_info(df, target_col=target_col)
    
    # 4. 特征选择
    selected_features = select_features(df, correlation_matrix, mi_scores, target_col=target_col)
    
    # 5. 准备训练数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_info = prepare_training_data(
        df, selected_features, target_col=target_col, sequence_length=24, prediction_horizon=1
    )
    
    print("\n特征分析与选择完成！")
    print(f"已选择 {len(selected_features)} 个特征用于模型训练")
    print(f"特征信息已保存至: data/model_ready/feature_info.json")
    print("数据已准备好用于深度学习模型训练")

if __name__ == "__main__":
    main()
    analyze_direction_correlation() 