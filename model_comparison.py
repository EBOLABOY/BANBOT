import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import glob
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 创建输出目录
os.makedirs('data/models/comparison', exist_ok=True)

def load_latest_ensemble_model():
    """加载最新的集成模型"""
    model_files = glob.glob('data/models/bitcoin_ensemble_model_*.joblib')
    if not model_files:
        print("警告: 找不到集成模型文件")
        return None
    
    # 按文件修改时间排序，获取最新模型
    latest_model_file = max(model_files, key=os.path.getmtime)
    print(f"加载集成模型: {latest_model_file}")
    
    # 加载模型包
    model_package = joblib.load(latest_model_file)
    return model_package

def load_deep_learning_models():
    """加载深度学习模型"""
    # 查找所有深度学习模型信息文件
    info_files = glob.glob('data/models/deep_learning/*_info.joblib')
    models = []
    
    for info_file in info_files:
        try:
            model_info = joblib.load(info_file)
            print(f"加载深度学习模型信息: {info_file}")
            
            # 检查模型文件是否存在
            if os.path.exists(model_info['model_path']):
                # 导入TensorFlow，需要时才加载
                import tensorflow as tf
                
                # 加载模型
                model = tf.keras.models.load_model(model_info['model_path'])
                model_info['model'] = model
                
                models.append(model_info)
            else:
                print(f"警告: 模型文件不存在: {model_info['model_path']}")
        except Exception as e:
            print(f"错误: 加载模型信息时出错: {str(e)}")
    
    print(f"共加载了 {len(models)} 个深度学习模型")
    return models

def load_test_data():
    """加载测试数据"""
    try:
        # 加载完整特征数据集
        data = pd.read_csv('data/features/btc_full_features.csv', index_col='timestamp', parse_dates=True)
        print(f"数据加载完成: {data.shape[0]} 行, {data.shape[1]} 列")
        
        # 使用最后30%的数据作为测试集
        test_size = int(len(data) * 0.3)
        test_data = data.iloc[-test_size:]
        
        print(f"测试集大小: {test_data.shape[0]} 行")
        return test_data
    except FileNotFoundError:
        print("错误: 找不到特征数据文件")
        return None

def evaluate_ensemble_model(model_package, test_data):
    """评估集成模型性能"""
    if model_package is None or test_data is None:
        return None
        
    print("\n评估集成模型...")
    
    # 提取模型组件
    voting_classifier = model_package['voting_classifier']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    
    # 准备特征和目标变量
    X_test = test_data[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
    y_test = test_data['target'].fillna(0).astype(int)
    
    # 标准化特征
    X_test_scaled = scaler.transform(X_test)
    
    # 使用模型预测
    y_pred = voting_classifier.predict(X_test_scaled)
    y_prob = voting_classifier.predict_proba(X_test_scaled)[:, 1]
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"集成模型性能:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['下跌', '上涨'],
                yticklabels=['下跌', '上涨'])
    plt.xlabel('预测')
    plt.ylabel('实际')
    plt.title('集成模型混淆矩阵')
    plt.savefig('data/models/comparison/ensemble_confusion_matrix.png')
    
    return {
        'model_type': '集成模型',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'probabilities': y_prob
    }

def evaluate_deep_learning_model(model_info, test_data):
    """评估深度学习模型性能"""
    if model_info is None or test_data is None or 'model' not in model_info:
        return None
    
    model_name = model_info['model_name']
    is_classification = model_info['is_classification']
    sequence_length = model_info['sequence_length']
    
    task_type = "分类" if is_classification else "回归"
    print(f"\n评估深度学习模型 ({model_name.upper()}, {task_type})...")
    
    try:
        # 选择要使用的特征
        selected_features = [
            'close', 'volume', 'stoch_k', 'stoch_d', 'rsi_14', 'macd_hist',
            'williams_r', 'cci_20', 'adx', 'buy_sell_pressure', 'volatility_20',
            'funding_rate', 'fear_greed_simple'
        ]
        
        # 确保所有选定特征都存在
        available_features = [f for f in selected_features if f in test_data.columns]
        
        # 准备序列数据
        X_sequences = []
        y_actual = []
        
        # 对于分类任务，目标是价格方向
        if is_classification:
            # 准备序列和目标
            for i in range(len(test_data) - sequence_length):
                X_seq = test_data[available_features].iloc[i:i+sequence_length].values
                X_sequences.append(X_seq)
                
                # 检查下一个时间点是否存在
                if i + sequence_length < len(test_data):
                    direction = test_data['target'].iloc[i+sequence_length]
                    y_actual.append(direction)
        
        # 对于回归任务，目标是价格变化率
        else:
            # 计算价格变化率
            test_data['price_change'] = test_data['close'].pct_change()
            test_data = test_data.dropna()
            
            # 准备序列和目标
            for i in range(len(test_data) - sequence_length):
                X_seq = test_data[available_features].iloc[i:i+sequence_length].values
                X_sequences.append(X_seq)
                
                # 检查下一个时间点是否存在
                if i + sequence_length < len(test_data):
                    change = test_data['price_change'].iloc[i+sequence_length]
                    y_actual.append(change)
        
        # 转换为numpy数组
        X_sequences = np.array(X_sequences)
        y_actual = np.array(y_actual)
        
        if len(X_sequences) == 0:
            print("错误: 无法生成序列数据")
            return None
        
        # 标准化
        scaler_X = model_info['scaler_X']
        n_samples, n_timesteps, n_features = X_sequences.shape
        X_reshaped = X_sequences.reshape(n_samples * n_timesteps, n_features)
        X_reshaped = scaler_X.transform(X_reshaped)
        X_scaled = X_reshaped.reshape(n_samples, n_timesteps, n_features)
        
        # 预测
        model = model_info['model']
        y_pred = model.predict(X_scaled)
        
        # 评估模型
        if is_classification:
            # 二分类评估
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            
            accuracy = accuracy_score(y_actual, y_pred_classes)
            precision = precision_score(y_actual, y_pred_classes, zero_division=0)
            recall = recall_score(y_actual, y_pred_classes, zero_division=0)
            f1 = f1_score(y_actual, y_pred_classes, zero_division=0)
            
            print(f"{model_name.upper()} 分类模型性能:")
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")
            
            # 计算混淆矩阵
            cm = confusion_matrix(y_actual, y_pred_classes)
            
            # 绘制混淆矩阵
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['下跌', '上涨'],
                        yticklabels=['下跌', '上涨'])
            plt.xlabel('预测')
            plt.ylabel('实际')
            plt.title(f'{model_name.upper()} 分类模型混淆矩阵')
            plt.savefig(f'data/models/comparison/{model_name}_classification_confusion_matrix.png')
            
            return {
                'model_type': f'{model_name.upper()} 分类',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred_classes,
                'probabilities': y_pred.flatten()
            }
            
        else:
            # 回归评估
            scaler_y = model_info['scaler_y']
            
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # 多步预测，取第一步
                y_pred_rescaled = scaler_y.inverse_transform(y_pred[:, 0].reshape(-1, 1)).flatten()
            else:
                y_pred_rescaled = scaler_y.inverse_transform(y_pred).flatten()
            
            mse = mean_squared_error(y_actual, y_pred_rescaled)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_actual, y_pred_rescaled)
            r2 = r2_score(y_actual, y_pred_rescaled)
            
            print(f"{model_name.upper()} 回归模型性能:")
            print(f"均方误差 (MSE): {mse:.6f}")
            print(f"均方根误差 (RMSE): {rmse:.6f}")
            print(f"平均绝对误差 (MAE): {mae:.6f}")
            print(f"R² 分数: {r2:.6f}")
            
            # 绘制预测 vs 实际
            plt.figure(figsize=(10, 6))
            plt.plot(y_actual, label='实际值', alpha=0.6)
            plt.plot(y_pred_rescaled, label='预测值', alpha=0.6)
            plt.title(f'{model_name.upper()} 回归模型预测 vs 实际')
            plt.xlabel('时间步')
            plt.ylabel('价格变化率')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'data/models/comparison/{model_name}_regression_predictions.png')
            
            # 绘制散点图
            plt.figure(figsize=(8, 8))
            plt.scatter(y_actual, y_pred_rescaled, alpha=0.5)
            plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'r--')
            plt.title(f'{model_name.upper()} 回归模型预测 vs 实际散点图')
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.grid(True)
            plt.savefig(f'data/models/comparison/{model_name}_regression_scatter.png')
            
            return {
                'model_type': f'{model_name.upper()} 回归',
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred_rescaled
            }
            
    except Exception as e:
        print(f"错误: 评估 {model_name} 模型时出错: {str(e)}")
        return None

def compare_models(ensemble_result, dl_results):
    """比较不同模型的性能"""
    print("\n模型性能比较:")
    
    # 分离分类和回归模型结果
    classification_results = [r for r in dl_results if 'accuracy' in r]
    regression_results = [r for r in dl_results if 'mse' in r]
    
    # 如果有集成模型结果，添加到分类结果中
    if ensemble_result is not None:
        classification_results.append(ensemble_result)
    
    # 比较分类模型
    if classification_results:
        print("\n分类模型比较:")
        
        # 创建比较数据框
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        clf_comparison = pd.DataFrame([
            [r['model_type']] + [r[m] for m in metrics]
            for r in classification_results
        ], columns=['模型'] + metrics)
        
        print(clf_comparison)
        
        # 保存比较结果
        clf_comparison.to_csv('data/models/comparison/classification_models_comparison.csv', index=False)
        
        # 绘制比较图
        plt.figure(figsize=(12, 8))
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            sns.barplot(x='模型', y=metric, data=clf_comparison)
            plt.title(f'{metric}比较')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/models/comparison/classification_models_comparison.png')
    
    # 比较回归模型
    if regression_results:
        print("\n回归模型比较:")
        
        # 创建比较数据框
        metrics = ['mse', 'rmse', 'mae', 'r2']
        reg_comparison = pd.DataFrame([
            [r['model_type']] + [r[m] for m in metrics]
            for r in regression_results
        ], columns=['模型'] + metrics)
        
        print(reg_comparison)
        
        # 保存比较结果
        reg_comparison.to_csv('data/models/comparison/regression_models_comparison.csv', index=False)
        
        # 绘制比较图
        plt.figure(figsize=(12, 8))
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            sns.barplot(x='模型', y=metric, data=reg_comparison)
            plt.title(f'{metric}比较')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/models/comparison/regression_models_comparison.png')

def compare_trading_performance(ensemble_result, dl_results, test_data):
    """比较不同模型的交易表现"""
    if test_data is None:
        return
    
    # 获取价格数据
    prices = test_data['close'].values
    
    # 初始化结果列表
    trading_results = []
    
    # 评估集成模型的交易性能
    if ensemble_result is not None and 'predictions' in ensemble_result:
        # 计算收益率
        returns = []
        capital = 1000.0  # 初始资金
        position = 0      # 0:空仓, 1:持仓
        
        for i in range(1, len(prices)):
            # 根据前一天的预测决定今天的仓位
            if i-1 < len(ensemble_result['predictions']):
                signal = ensemble_result['predictions'][i-1]
                
                # 计算收益
                if position == 1:  # 如果持仓
                    daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                else:
                    daily_return = 0
                
                # 更新资金
                capital *= (1 + daily_return)
                returns.append(capital)
                
                # 更新仓位
                position = signal
        
        # 计算最终收益率和夏普比率
        if returns:
            final_return = (returns[-1] / 1000.0) - 1
            daily_returns = [(returns[i] / returns[i-1]) - 1 for i in range(1, len(returns))]
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            
            trading_results.append({
                'model_type': ensemble_result['model_type'],
                'final_return': final_return,
                'sharpe_ratio': sharpe,
                'returns': returns
            })
    
    # 评估深度学习分类模型的交易性能
    for result in dl_results:
        if 'predictions' in result and 'accuracy' in result:
            # 计算收益率
            returns = []
            capital = 1000.0  # 初始资金
            position = 0      # 0:空仓, 1:持仓
            
            for i in range(1, len(prices)):
                # 确保我们有足够的预测值
                if i-1 < len(result['predictions']):
                    signal = result['predictions'][i-1]
                    
                    # 计算收益
                    if position == 1:  # 如果持仓
                        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                    else:
                        daily_return = 0
                    
                    # 更新资金
                    capital *= (1 + daily_return)
                    returns.append(capital)
                    
                    # 更新仓位
                    position = signal
            
            # 计算最终收益率和夏普比率
            if returns:
                final_return = (returns[-1] / 1000.0) - 1
                daily_returns = [(returns[i] / returns[i-1]) - 1 for i in range(1, len(returns))]
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
                
                trading_results.append({
                    'model_type': result['model_type'],
                    'final_return': final_return,
                    'sharpe_ratio': sharpe,
                    'returns': returns
                })
    
    # 绘制交易性能比较
    if trading_results:
        print("\n交易性能比较:")
        
        # 创建比较数据框
        perf_comparison = pd.DataFrame([
            {
                '模型': r['model_type'],
                '最终收益率': r['final_return'] * 100,  # 转为百分比
                '夏普比率': r['sharpe_ratio']
            }
            for r in trading_results
        ])
        
        print(perf_comparison)
        
        # 保存比较结果
        perf_comparison.to_csv('data/models/comparison/trading_performance_comparison.csv', index=False)
        
        # 绘制收益曲线
        plt.figure(figsize=(12, 6))
        for result in trading_results:
            plt.plot(result['returns'], label=result['model_type'])
        
        plt.title('模型交易性能比较')
        plt.xlabel('交易日')
        plt.ylabel('账户价值')
        plt.legend()
        plt.grid(True)
        plt.savefig('data/models/comparison/trading_performance_comparison.png')
        
        # 绘制收益率和夏普比率对比
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.barplot(x='模型', y='最终收益率', data=perf_comparison, ax=ax[0])
        ax[0].set_title('最终收益率比较 (%)')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)
        
        sns.barplot(x='模型', y='夏普比率', data=perf_comparison, ax=ax[1])
        ax[1].set_title('夏普比率比较')
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/models/comparison/trading_metrics_comparison.png')

def main():
    """主函数"""
    print("比特币价格预测模型比较")
    print("=" * 50)
    
    # 加载数据
    test_data = load_test_data()
    if test_data is None:
        print("错误: 无法加载测试数据")
        return
    
    # 加载集成模型
    ensemble_model = load_latest_ensemble_model()
    
    # 加载深度学习模型
    dl_models = load_deep_learning_models()
    
    # 评估模型
    ensemble_result = None
    dl_results = []
    
    # 评估集成模型
    if ensemble_model is not None:
        ensemble_result = evaluate_ensemble_model(ensemble_model, test_data)
    
    # 评估深度学习模型
    for model_info in dl_models:
        result = evaluate_deep_learning_model(model_info, test_data)
        if result is not None:
            dl_results.append(result)
    
    # 比较模型性能
    compare_models(ensemble_result, dl_results)
    
    # 比较交易性能
    compare_trading_performance(ensemble_result, dl_results, test_data)
    
    print("\n模型比较完成!")
    print(f"结果保存在 data/models/comparison/ 目录")

if __name__ == "__main__":
    main() 