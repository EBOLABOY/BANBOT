import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import os
import joblib
from datetime import datetime

# 创建输出目录
os.makedirs('data/models', exist_ok=True)

print("加载特征数据...")
# 读取包含所有特征的数据集
try:
    df = pd.read_csv('data/features/btc_full_features.csv', index_col='timestamp', parse_dates=True)
    print(f"数据已加载: {df.shape[0]} 行, {df.shape[1]} 列")
except FileNotFoundError:
    print("错误: 请先运行 feature_engineering.py 生成特征数据集")
    exit(1)

# 基于特征重要性选择顶级特征
top_features = [
    'stoch_k',                    # 随机指标K值
    'buy_sell_pressure',          # 买卖压力比率
    'funding_price_correlation',  # 资金费率与价格相关性
    'stoch_d',                    # 随机指标D值
    'volume_z_score',             # 成交量异常
    'rsi_acceleration',           # RSI加速度
    'rsi_change_3d',              # 3天RSI变化率
    'cci_20_lag_14',              # 14天前CCI
    'cci_change',                 # CCI变化率
    'volume_lag_14',              # 14天前成交量
    'macd_hist_lag_14',           # 14天前MACD直方图
    'williams_r',                 # 威廉指标
    'fear_greed_ma_diff',         # 恐惧贪婪指数与均值差
    'fear_greed_simple',          # 简化恐惧贪婪指数
    'volatility_trend'            # 波动性趋势
]

print(f"使用 {len(top_features)} 个顶级特征进行模型训练")

# 特征矩阵与目标变量准备
# 目标变量是下一时间点的价格变化方向(1表示上涨，0表示下跌)
X = df[top_features].fillna(0).replace([np.inf, -np.inf], 0)
y = df['target'].fillna(0).astype(int)

# 移除包含NaN的行
valid_idx = ~X.isnull().any(axis=1)
X = X[valid_idx]
y = y[valid_idx]

print(f"准备好的数据: {X.shape[0]} 行, {X.shape[1]} 列")
print(f"类别分布: 上涨={y.sum()}, 下跌={len(y)-y.sum()}, 上涨比例={y.mean():.2f}")

# 定义模型评估函数
def evaluate_model(model, X_test, y_test, model_name="模型"):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    
    # 计算各种评估指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # 打印评估结果
    print(f"{model_name} 评估结果:")
    print(f"准确率: {acc:.4f}")
    print(f"精确率: {prec:.4f}")
    print(f"召回率: {rec:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['下跌', '上涨'], 
                yticklabels=['下跌', '上涨'])
    plt.xlabel('预测')
    plt.ylabel('实际')
    plt.title(f'{model_name} 混淆矩阵')
    plt.savefig(f'data/models/{model_name}_confusion_matrix.png')
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'predictions': y_pred
    }

# 分割数据 - 使用时间序列交叉验证
print("\n使用时间序列交叉验证...")
tscv = TimeSeriesSplit(n_splits=5)

# 创建各种基础模型
models = {
    "随机森林": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42),
    "梯度提升": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "逻辑回归": LogisticRegression(random_state=42, max_iter=1000),
    "支持向量机": SVC(probability=True, random_state=42)
}

# 记录每个模型的交叉验证性能
cv_results = {model_name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} 
              for model_name in models.keys()}
cv_results['投票分类器'] = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

# 进行交叉验证
fold = 1
for train_idx, test_idx in tscv.split(X):
    print(f"\n正在评估折叠 {fold}/{tscv.n_splits}:")
    
    # 分割数据
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练和评估每个基本模型
    trained_models = {}
    for model_name, model in models.items():
        print(f"\n训练 {model_name}...")
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model
        
        # 评估模型
        results = evaluate_model(model, X_test_scaled, y_test, f"{model_name}_fold{fold}")
        
        # 记录结果
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            cv_results[model_name][metric].append(results[metric])
    
    # 创建投票分类器作为组合模型
    print("\n训练投票分类器（组合模型）...")
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'
    )
    voting_clf.fit(X_train_scaled, y_train)
    
    # 评估投票分类器
    results = evaluate_model(voting_clf, X_test_scaled, y_test, f"投票分类器_fold{fold}")
    
    # 记录结果
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        cv_results['投票分类器'][metric].append(results[metric])
    
    fold += 1

# 计算平均交叉验证结果
print("\n交叉验证平均结果:")
for model_name in list(models.keys()) + ['投票分类器']:
    print(f"\n{model_name}:")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        mean_val = np.mean(cv_results[model_name][metric])
        std_val = np.std(cv_results[model_name][metric])
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

# 可视化不同模型性能
plt.figure(figsize=(12, 8))
metrics = ['accuracy', 'precision', 'recall', 'f1']
model_names = list(models.keys()) + ['投票分类器']

for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    means = [np.mean(cv_results[model][metric]) for model in model_names]
    stds = [np.std(cv_results[model][metric]) for model in model_names]
    
    bars = plt.bar(model_names, means, yerr=stds, capsize=10)
    plt.title(f'平均{metric}得分')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # 在柱状图上方添加具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('data/models/model_comparison.png')

# 训练最终模型
print("\n训练最终模型（使用全部数据）...")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练各个基本模型
final_models = {}
for model_name, model in models.items():
    print(f"训练最终 {model_name}...")
    model.fit(X_scaled, y)
    final_models[model_name] = model

# 创建和训练最终的投票分类器
final_voting_clf = VotingClassifier(
    estimators=[(name, model) for name, model in final_models.items()],
    voting='soft'
)
final_voting_clf.fit(X_scaled, y)

# 保存模型
print("\n保存模型...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f'data/models/bitcoin_ensemble_model_{timestamp}.joblib'


model_package = {
    'voting_classifier': final_voting_clf,
    'base_models': final_models,
    'feature_names': top_features,
    'scaler': scaler,
    'cv_results': cv_results,
    'train_date': timestamp
}

joblib.dump(model_package, model_filename)
print(f"模型已保存至: {model_filename}")

# 特征重要性分析 (仅适用于随机森林和基于树的模型)
if 'RandomForestClassifier' in str(type(final_models["随机森林"])):
    print("\n分析特征重要性...")
    rf_model = final_models["随机森林"]
    
    # 获取并排序特征重要性
    feature_importance = pd.DataFrame({
        'feature': top_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 保存特征重要性
    feature_importance.to_csv('data/models/final_feature_importance.csv', index=False)
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('重要性')
    plt.title('最终模型特征重要性')
    plt.gca().invert_yaxis()  # 从上到下按重要性递减排序
    plt.tight_layout()
    plt.savefig('data/models/final_feature_importance.png')

print("\n组合模型训练完成!") 