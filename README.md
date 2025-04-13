# 加密货币价格预测AI项目

基于机器学习和深度学习的加密货币价格预测系统。

## 项目目标

- 实现对BTC、ETH等加密货币的价格预测
- 支持多种时间窗口的预测（短期：1-24小时，中期：1-7天，长期：1-4周）
- 提供高准确度的价格方向和点位预测
- 开发可用于实际交易的策略

## 项目结构

```
├── data              # 数据目录
│   ├── raw           # 原始数据
│   ├── processed     # 处理后的数据
│   └── external      # 外部数据
├── logs              # 日志文件
├── models            # 模型相关
│   └── saved_models  # 保存的模型
├── notebooks         # Jupyter笔记本
├── src               # 源代码
│   ├── api           # API服务
│   ├── data          # 数据处理模块
│   ├── features      # 特征工程模块
│   ├── models        # 模型定义和训练
│   ├── utils         # 工具函数
│   └── visualization # 可视化模块
└── README.md         # 项目说明
```

## 安装与使用

### 环境配置

```bash
# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

## 评估指标

- 方向准确率
- RMSE (均方根误差)
- MAE (平均绝对误差)
- Sharpe比率
- 最大回撤 