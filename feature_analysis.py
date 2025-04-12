import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
import os

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
    
    # 选择所有数值型列
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
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
    
    # 选择数值型特征
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    features = numeric_columns.drop(target_col) if target_col in numeric_columns else numeric_columns
    
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

# 主函数
def main():
    # 1. 加载已处理的数据
    df = load_processed_data('data/processed/btc_with_features.csv')
    
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