import pandas as pd
import numpy as np
import joblib
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def load_latest_model():
    """加载最新训练的模型"""
    model_files = glob.glob('data/models/bitcoin_ensemble_model_*.joblib')
    if not model_files:
        print("错误: 没有找到训练好的模型，请先运行ensemble_model.py")
        return None
    
    # 按文件修改时间排序，获取最新模型
    latest_model_file = max(model_files, key=os.path.getmtime)
    print(f"加载模型: {latest_model_file}")
    
    # 加载模型包
    model_package = joblib.load(latest_model_file)
    return model_package

def prepare_features(data, feature_names):
    """准备模型输入特征"""
    # 确保所有必需的特征都存在
    missing_features = [f for f in feature_names if f not in data.columns]
    if missing_features:
        print(f"警告: 数据中缺少以下特征: {missing_features}")
        return None
    
    # 提取特征并处理缺失值
    X = data[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
    return X

def predict_price_direction(model_package, data):
    """使用模型预测价格方向"""
    # 提取模型组件
    voting_classifier = model_package['voting_classifier']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    
    # 准备特征
    X = prepare_features(data, feature_names)
    if X is None:
        return None
    
    # 标准化特征
    X_scaled = scaler.transform(X)
    
    # 使用组合模型进行预测
    predictions = voting_classifier.predict(X_scaled)
    probabilities = voting_classifier.predict_proba(X_scaled)
    
    # 组织预测结果
    results = pd.DataFrame({
        'date': data.index,
        'actual_price': data['close'],
        'predicted_direction': predictions,
        'prob_down': probabilities[:, 0],
        'prob_up': probabilities[:, 1]
    })
    
    return results

def evaluate_predictions(predictions, data, days_to_evaluate=30):
    """评估最近N天的预测结果"""
    # 获取最近N天的数据
    recent_data = predictions.tail(days_to_evaluate).copy()
    
    # 计算实际的价格变动方向
    actual_direction = (data['close'].shift(-1) > data['close']).astype(int)
    recent_data['actual_direction'] = actual_direction.tail(days_to_evaluate).values
    
    # 计算准确率
    correct = (recent_data['predicted_direction'] == recent_data['actual_direction']).sum()
    accuracy = correct / len(recent_data)
    
    # 计算盈亏
    recent_data['price_change_pct'] = data['close'].pct_change().shift(-1).tail(days_to_evaluate).values
    recent_data['trade_return'] = recent_data['price_change_pct'] * (recent_data['predicted_direction'] * 2 - 1)
    cumulative_return = (1 + recent_data['trade_return']).prod() - 1
    
    # 打印评估结果
    print(f"\n最近{days_to_evaluate}天预测评估:")
    print(f"准确率: {accuracy:.4f}")
    print(f"累积收益率: {cumulative_return:.4f} ({cumulative_return*100:.2f}%)")
    print(f"正确预测次数: {correct}/{len(recent_data)}")
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 价格与预测图
    plt.subplot(2, 1, 1)
    plt.plot(recent_data['date'], data['close'].tail(days_to_evaluate), 'b-', label='实际价格')
    
    # 标记预测正确和错误的点
    correct_points = recent_data[recent_data['predicted_direction'] == recent_data['actual_direction']]
    wrong_points = recent_data[recent_data['predicted_direction'] != recent_data['actual_direction']]
    
    plt.scatter(correct_points['date'], data['close'].loc[correct_points.index], 
                color='green', marker='^', s=100, label='预测正确')
    plt.scatter(wrong_points['date'], data['close'].loc[wrong_points.index], 
                color='red', marker='x', s=100, label='预测错误')
    
    plt.title(f'比特币价格预测结果 (准确率: {accuracy:.2f})')
    plt.ylabel('价格 (USD)')
    plt.legend()
    plt.grid(True)
    
    # 设置日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    
    # 预测概率图
    plt.subplot(2, 1, 2)
    plt.bar(recent_data['date'], recent_data['prob_up'], color='green', alpha=0.6, label='上涨概率')
    plt.bar(recent_data['date'], -recent_data['prob_down'], color='red', alpha=0.6, label='下跌概率')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('预测概率分布')
    plt.ylabel('概率')
    plt.legend()
    plt.grid(True)
    
    # 设置日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    
    plt.tight_layout()
    plt.savefig('data/models/prediction_evaluation.png')
    
    return recent_data

def predict_next_day(model_package, latest_data):
    """预测下一个交易日的价格方向"""
    # 准备特征
    X = prepare_features(latest_data.tail(1), model_package['feature_names'])
    if X is None:
        return None
    
    # 标准化特征
    X_scaled = model_package['scaler'].transform(X)
    
    # 使用组合模型预测
    voting_classifier = model_package['voting_classifier']
    prediction = voting_classifier.predict(X_scaled)[0]
    probabilities = voting_classifier.predict_proba(X_scaled)[0]
    
    # 获取预测结果
    direction = "上涨" if prediction == 1 else "下跌"
    confidence = max(probabilities)
    
    print("\n下一交易日预测:")
    print(f"日期: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"预测方向: {direction}")
    print(f"上涨概率: {probabilities[1]:.4f}")
    print(f"下跌概率: {probabilities[0]:.4f}")
    print(f"置信度: {confidence:.4f}")
    
    # 获取各个基础模型的预测
    base_predictions = {}
    for name, model in model_package['base_models'].items():
        base_pred = model.predict(X_scaled)[0]
        base_direction = "上涨" if base_pred == 1 else "下跌"
        base_predictions[name] = base_direction
    
    print("\n各模型预测:")
    for model_name, pred_direction in base_predictions.items():
        print(f"{model_name}: {pred_direction}")
    
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'direction': direction,
        'up_probability': probabilities[1],
        'down_probability': probabilities[0],
        'confidence': confidence,
        'base_model_predictions': base_predictions
    }

def main():
    """主函数，执行预测流程"""
    print("比特币价格方向预测")
    print("=" * 50)
    
    # 加载最新模型
    model_package = load_latest_model()
    if model_package is None:
        return
    
    # 加载最新数据
    print("\n加载数据...")
    try:
        data = pd.read_csv('data/features/btc_full_features.csv', index_col='timestamp', parse_dates=True)
        print(f"数据加载完成: {data.shape[0]} 行")
    except FileNotFoundError:
        print("错误: 找不到特征数据，请先运行feature_engineering.py")
        return
    
    # 进行历史预测
    print("\n进行历史预测评估...")
    predictions = predict_price_direction(model_package, data)
    if predictions is not None:
        # 保存预测结果
        predictions.to_csv('data/models/historical_predictions.csv')
        
        # 评估最近30天的预测
        recent_results = evaluate_predictions(predictions, data, days_to_evaluate=30)
        
        # 预测下一交易日
        next_day_prediction = predict_next_day(model_package, data)
        
        if next_day_prediction:
            # 保存预测结果
            pd.DataFrame([next_day_prediction]).to_csv('data/models/next_day_prediction.csv', index=False)
    
    print("\n预测完成!")

if __name__ == "__main__":
    main() 