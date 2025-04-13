import os
import time
import subprocess
import sys

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
        sys.exit(result.returncode)
    
    print(f"\n✅ {description}完成！")
    print(f"用时: {end_time - start_time:.2f} 秒")
    print(f"{'='*80}")
    
    return result.returncode

def main():
    """运行完整的特征工程和特征选择流程"""
    print("\n📊 比特币价格预测特征工程流程")
    print("👉 本脚本将依次执行特征工程和特征选择")
    
    # 确保输出目录存在
    os.makedirs('data/features', exist_ok=True)
    
    # 步骤1: 生成特征
    run_command("python feature_engineering.py", "特征工程")
    
    # 步骤2: 特征选择与评估
    run_command("python feature_selection.py", "特征选择与评估")
    
    print("\n🎉 整个特征工程流程已完成！")
    print("📁 结果文件位于 data/features/ 目录下")
    print("📈 可以开始使用生成的特征进行模型训练了")

if __name__ == "__main__":
    main() 