# 比特币量化交易系统

这是一个基于深度学习（LSTM/Attention机制）的比特币量化交易系统，能够预测比特币价格走势并生成交易信号。

## 功能特点

- **数据获取**：通过币安API获取BTC/USDT的历史K线数据
- **特征工程**：提取并计算多种技术指标作为特征
- **深度学习模型**：
  - 双向LSTM网络
  - 注意力机制
  - 多层全连接层
- **交易信号生成**：基于波动率的动态阈值策略
- **策略回测**：对生成的交易信号进行历史回测

## 项目结构

```
├── data/                   # 数据存储目录
│   ├── processed/          # 预处理后的数据
│   └── model_ready/        # 模型训练准备数据
├── models/                 # 模型存储目录
├── analysis/               # 分析结果目录
├── get_binance_data.py     # 数据获取脚本
├── process_data_with_features.py # 数据处理脚本
├── feature_analysis.py     # 特征分析脚本
├── lstm_model.py           # LSTM深度学习模型
├── requirements.txt        # 项目依赖
├── setup_env.bat           # 环境设置脚本
└── README.md
```

## 安装与使用

### 环境要求

- Python 3.8+
- PyTorch
- 其他依赖库（见`requirements.txt`）

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/quant-trading-system.git
cd quant-trading-system
```

2. 设置环境：
```bash
# Windows
setup_env.bat

# Linux/Mac
python -m venv trading_env
source trading_env/bin/activate
pip install -r requirements.txt
```

3. 配置API密钥：
创建`.env`文件并添加以下内容：
```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

### 使用方法

1. 获取数据：
```bash
python get_binance_data.py
```

2. 处理数据并添加特征：
```bash
python process_data_with_features.py
```

3. 分析特征：
```bash
python feature_analysis.py
```

4. 训练模型：
```bash
python lstm_model.py
```

## 模型说明

系统中包含两种高级LSTM模型：

1. **EnhancedLSTMModel**：增强型LSTM模型，具有双向LSTM层和深度全连接层。
2. **AttentionLSTMModel**：注意力机制LSTM模型，能够自动关注重要的时间步。

默认使用的是AttentionLSTM模型，可以在`lstm_model.py`中修改`model_type`参数切换模型。

## 交易策略

系统使用基于价格预测和动态阈值的交易策略：

- **动态阈值**：根据市场波动性自动调整买入/卖出信号阈值
- **信号类型**：买入/卖出/持有
- **回测分析**：系统提供完整的回测功能，包括收益率、胜率和与买入持有策略的比较

## 注意事项

- 本项目仅供学习和研究使用，不构成投资建议
- 实盘交易需谨慎，建议先在模拟环境中测试
- 使用API密钥时需注意安全，不要泄露或上传到公开仓库 