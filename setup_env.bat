@echo off
echo ===== 创建深度学习交易系统环境 =====

:: 检查Python是否安装
python --version
if %ERRORLEVEL% neq 0 (
    echo Python未安装。请先安装Python 3.8或更高版本。
    exit /b 1
)

:: 创建虚拟环境
echo.
echo 创建虚拟环境 'trading_env'...
python -m venv trading_env
if %ERRORLEVEL% neq 0 (
    echo 创建虚拟环境失败。请确保已安装venv模块。
    exit /b 1
)

:: 激活虚拟环境
echo.
echo 激活虚拟环境...
call trading_env\Scripts\activate
if %ERRORLEVEL% neq 0 (
    echo 激活虚拟环境失败。
    exit /b 1
)

:: 安装依赖
echo.
echo 安装依赖库...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo 安装依赖失败。
    exit /b 1
)

echo.
echo ===== 环境设置完成 =====
echo.
echo 现在你可以使用以下命令激活环境：
echo call trading_env\Scripts\activate
echo.
echo 然后运行你的Python脚本
echo.
echo 按任意键继续...
pause > nul 