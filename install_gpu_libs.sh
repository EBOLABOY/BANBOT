#!/bin/bash
# 安装GPU加速所需的库脚本

echo "安装RAPIDS库，用于GPU加速特征计算..."

# 检测CUDA版本
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | awk -F',' '{print $1}' | awk -F'V' '{print $2}')
echo "检测到CUDA版本: $CUDA_VERSION"

# 如果无法检测到CUDA版本，默认使用CUDA 11.8
if [ -z "$CUDA_VERSION" ]; then
    echo "无法检测到CUDA版本，默认使用CUDA 11.8"
    CUDA_VERSION="11.8"
fi

# 提取主版本号
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)

# 安装RAPIDS库
if [ "$CUDA_MAJOR" -ge "11" ]; then
    echo "使用pip安装RAPIDS库..."
    # 为CUDA 11.x安装适当的包
    pip install cudf-cu11 cuml-cu11 cupy-cuda11x numba
    
    echo "验证安装..."
    python -c "import cudf; print('cudf版本:', cudf.__version__); import cupy; print('cupy版本:', cupy.__version__); import numba; print('numba版本:', numba.__version__)"
else
    echo "当前CUDA版本 ($CUDA_VERSION) 可能不兼容RAPIDS库，推荐使用CUDA 11.x"
    echo "如果你确定要继续，请手动安装RAPIDS库: https://rapids.ai/start.html"
    exit 1
fi

echo "RAPIDS库安装完成!"
echo "现在你可以使用GPU加速进行特征计算了，使用以下命令:"
echo "python -m src.feature_engineering_main --symbol BTCUSDT --timeframe 1m --mode compute --use_gpu --batch_size 500000" 