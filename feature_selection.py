import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 创建输出目录
os.makedirs('data/features/selection', exist_ok=True)

print("加载处理后的数据...")
# 读取包含所有特征的数据集
try:
    df = pd.read_csv('data/features/btc_full_features.csv', index_col='timestamp', parse_dates=True)
    print(f"数据已加载: {df.shape[0]} 行, {df.shape[1]} 列")
except FileNotFoundError:
    print("错误: 请先运行 feature_engineering.py 生成全特征数据集")
    exit(1)

# 目标变量（收盘价变化方向）
y = df['target'].fillna(0)

# 准备特征矩阵，排除原始数据和目标变量
excluded_columns = ['open', 'high', 'low', 'close', 'volume', 'number_of_trades', 
                    'taker_buy_volume', 'taker_sell_volume', 'funding_rate', 'target']
X = df.drop(excluded_columns, axis=1).fillna(0)

print(f"特征数量: {X.shape[1]}")

# 特征选择方法1: 特征重要性（随机森林）
print("\n方法1: 基于随机森林的特征重要性...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 特征重要性排序
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# 保存特征重要性
feature_importance.to_csv('data/features/selection/rf_feature_importance.csv', index=False)
print(f"前15个最重要特征:\n{feature_importance.head(15)}")

# 可视化前20个特征
plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('前20个最重要特征 (随机森林)')
plt.tight_layout()
plt.savefig('data/features/selection/top20_features_rf.png')

# 特征选择方法2: 递归特征消除 (RFE)
print("\n方法2: 递归特征消除 (RFE)...")
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), 
          n_features_to_select=30)
rfe.fit(X, y)

# 获取RFE选择的特征
rfe_selected_features = X.columns[rfe.support_]
rfe_feature_ranking = pd.DataFrame({
    'feature': X.columns,
    'ranking': rfe.ranking_
}).sort_values('ranking')

# 保存RFE结果
rfe_feature_ranking.to_csv('data/features/selection/rfe_feature_ranking.csv', index=False)
print(f"RFE选择的特征 (前15个):\n{rfe_feature_ranking.head(15)}")

# 特征选择方法3: 互信息 (Mutual Information)
print("\n方法3: 互信息分析...")
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': mi_scores
}).sort_values('importance', ascending=False)

# 保存互信息结果
mi_feature_importance.to_csv('data/features/selection/mutual_info_scores.csv', index=False)
print(f"基于互信息的重要特征 (前15个):\n{mi_feature_importance.head(15)}")

# 可视化互信息得分
plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=mi_feature_importance.head(20))
plt.title('前20个最重要特征 (互信息)')
plt.tight_layout()
plt.savefig('data/features/selection/top20_features_mi.png')

# 综合特征选择: 结合所有方法选择最终特征集
print("\n综合特征选择...")

# 获取每种方法的前30个特征
rf_top30 = set(feature_importance.head(30)['feature'])
rfe_top30 = set(rfe_feature_ranking.head(30)['feature'])
mi_top30 = set(mi_feature_importance.head(30)['feature'])

# 计算特征出现在不同方法中的次数
feature_votes = {}
for feature in X.columns:
    votes = 0
    if feature in rf_top30:
        votes += 1
    if feature in rfe_top30:
        votes += 1
    if feature in mi_top30:
        votes += 1
    feature_votes[feature] = votes

# 获取至少在两种方法中被选中的特征
consensus_features = [feature for feature, votes in feature_votes.items() if votes >= 2]
print(f"至少被两种方法选中的特征数量: {len(consensus_features)}")
print(f"共识特征 (前15个):\n{consensus_features[:15]}")

# 保存共识特征
pd.DataFrame({'feature': consensus_features}).to_csv('data/features/selection/consensus_features.csv', index=False)

# 特征相关性分析
print("\n特征相关性分析...")
selected_features_df = X[consensus_features]

# 计算相关性矩阵
corr_matrix = selected_features_df.corr()

# 保存相关性矩阵
corr_matrix.to_csv('data/features/selection/correlation_matrix.csv')

# 可视化相关性热图 (如果特征太多，只可视化前30个)
plt.figure(figsize=(16, 14))
features_to_plot = consensus_features[:30] if len(consensus_features) > 30 else consensus_features
sns.heatmap(corr_matrix.loc[features_to_plot, features_to_plot], annot=False, cmap='coolwarm')
plt.title('特征相关性热图')
plt.tight_layout()
plt.savefig('data/features/selection/correlation_heatmap.png')

# 时间序列交叉验证评估
print("\n使用时间序列交叉验证评估特征集...")

# 准备数据
X_selected = X[consensus_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

for train_idx, test_idx in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    cv_scores['accuracy'].append(accuracy_score(y_test, y_pred))
    cv_scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
    cv_scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
    cv_scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))

# 输出评估结果
print("\n时间序列交叉验证结果:")
for metric, scores in cv_scores.items():
    print(f"{metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# 保存评估结果
pd.DataFrame(cv_scores).to_csv('data/features/selection/cv_evaluation.csv', index=False)

# 最终特征重要性可视化
final_importances = pd.DataFrame({
    'feature': consensus_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='importance', y='feature', data=final_importances.head(20))
plt.title('最终选择的特征重要性 (前20个)')
plt.tight_layout()
plt.savefig('data/features/selection/final_feature_importance.png')

print("\n特征选择与评估完成！全部结果已保存到 data/features/selection/ 目录") 