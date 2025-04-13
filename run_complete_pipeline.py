import os
import time
import subprocess
import sys
from datetime import datetime

def run_command(command, description):
    """运行命令并显示进度"""
    print(f"\n{'='*80}")
    print(f"开始执行: {description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"\n❌ {description}失败，退出代码: {result.returncode}")
        return False
    
    print(f"\n✅ {description}完成！")
    print(f"用时: {end_time - start_time:.2f} 秒")
    print(f"{'='*80}")
    
    return True

def main():
    """运行完整的比特币价格预测流程"""
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 创建日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/pipeline_run_{timestamp}.log"
    
    print("\n🚀 比特币价格预测完整流程")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 日志文件: {log_file}")
    
    # 记录到日志文件
    with open(log_file, 'w') as f:
        f.write(f"比特币价格预测流程开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 步骤1: 特征工程
    print("\n🔧 第1步: 特征工程")
    feature_eng_success = run_command("python feature_engineering.py", "特征工程")
    if not feature_eng_success:
        print("特征工程失败，终止流程")
        return
    
    # 步骤2: 特征选择
    print("\n🔍 第2步: 特征选择与评估")
    feature_selection_success = run_command("python feature_selection.py", "特征选择与评估")
    if not feature_selection_success:
        print("特征选择失败，但继续流程")
    
    # 步骤3: 模型训练
    print("\n🧠 第3步: 组合模型训练")
    model_training_success = run_command("python ensemble_model.py", "组合模型训练")
    if not model_training_success:
        print("模型训练失败，终止流程")
        return
    
    # 步骤4: 预测
    print("\n🔮 第4步: 价格走势预测")
    prediction_success = run_command("python predict.py", "价格预测")
    if not prediction_success:
        print("预测失败，但流程已基本完成")
    
    # 计算总用时
    end_time = datetime.now()
    start_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    duration = end_time - start_time
    
    # 总结
    print("\n🎉 比特币价格预测流程完成！")
    print(f"⏱️ 总用时: {duration}")
    print(f"📊 结果位于 data/features/ 和 data/models/ 目录")
    
    # 记录到日志文件
    with open(log_file, 'a') as f:
        f.write(f"\n比特币价格预测流程完成: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总用时: {duration}\n")
        
        if os.path.exists("data/models/next_day_prediction.csv"):
            try:
                import pandas as pd
                next_day = pd.read_csv("data/models/next_day_prediction.csv")
                f.write("\n下一交易日预测:\n")
                f.write(f"方向: {next_day['direction'].values[0]}\n")
                f.write(f"上涨概率: {next_day['up_probability'].values[0]:.4f}\n")
                f.write(f"下跌概率: {next_day['down_probability'].values[0]:.4f}\n")
                f.write(f"置信度: {next_day['confidence'].values[0]:.4f}\n")
            except:
                f.write("无法读取下一交易日预测结果\n")

if __name__ == "__main__":
    main() 