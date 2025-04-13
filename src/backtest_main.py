"""
回测主脚本 - 用于评估交易策略的表现
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from typing import Dict, List, Optional, Union

from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config
from src.models.model_training import load_data
from src.features.feature_engineering import FeatureEngineer

logger = get_logger(__name__)

class Backtest:
    """
    回测类，用于评估交易策略表现
    """
    
    def __init__(self, 
                 model_path: str,
                 price_data_path: str,
                 feature_data_path: str = None,
                 target_data_path: str = None,
                 initial_capital: float = 10000,
                 trading_fee: float = 0.001,
                 slippage: float = 0.0005,
                 output_dir: str = "results/backtest"):
        """
        初始化回测
        
        参数:
            model_path: 模型文件路径
            price_data_path: 价格数据文件路径
            feature_data_path: 特征数据文件路径
            target_data_path: 目标数据文件路径
            initial_capital: 初始资金
            trading_fee: 交易费用（占交易金额的比例）
            slippage: 滑点（占价格的比例）
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.price_data_path = price_data_path
        self.feature_data_path = feature_data_path
        self.target_data_path = target_data_path
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 加载数据
        self.price_data = self._load_price_data(price_data_path)
        if feature_data_path:
            self.feature_data = self._load_feature_data(feature_data_path)
        else:
            self.feature_data = None
            
        # 初始化回测结果
        self.results = {
            'timestamp': [],
            'price': [],
            'position': [],
            'cash': [],
            'holdings': [],
            'total_value': [],
            'returns': [],
            'prediction': []
        }
        
    def _load_model(self, model_path: str):
        """
        加载模型
        
        参数:
            model_path: 模型文件路径
            
        返回:
            加载的模型对象
        """
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"模型已从 {model_path} 加载")
            return model
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise
            
    def _load_price_data(self, price_data_path: str):
        """
        加载价格数据
        
        参数:
            price_data_path: 价格数据文件路径
            
        返回:
            价格数据DataFrame
        """
        try:
            df = pd.read_csv(price_data_path, index_col=0, parse_dates=True)
            logger.info(f"价格数据已从 {price_data_path} 加载，共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"加载价格数据时出错: {str(e)}")
            raise
            
    def _load_feature_data(self, feature_data_path: str):
        """
        加载特征数据
        
        参数:
            feature_data_path: 特征数据文件路径
            
        返回:
            特征数据DataFrame
        """
        try:
            df = pd.read_csv(feature_data_path, index_col=0, parse_dates=True)
            logger.info(f"特征数据已从 {feature_data_path} 加载，共 {len(df)} 条记录，{len(df.columns)} 个特征")
            return df
        except Exception as e:
            logger.error(f"加载特征数据时出错: {str(e)}")
            raise
            
    def run(self, features: List[str] = None, position_size: float = 0.2, threshold: float = 0.0):
        """
        运行回测
        
        参数:
            features: 要使用的特征列表，如果为None则使用所有特征
            position_size: 仓位大小（占总资金的比例）
            threshold: 交易阈值，预测值超过阈值才会交易
            
        返回:
            回测结果字典
        """
        if self.feature_data is None:
            logger.error("未加载特征数据，无法运行回测")
            return None
            
        # 如果指定了特征列表，则只使用这些特征
        if features:
            feature_data = self.feature_data[features]
        else:
            feature_data = self.feature_data
            
        # 确保价格数据和特征数据的索引匹配
        common_idx = feature_data.index.intersection(self.price_data.index)
        feature_data = feature_data.loc[common_idx]
        price_data = self.price_data.loc[common_idx]
        
        logger.info(f"回测数据准备完成，共 {len(common_idx)} 条记录")
        
        # 初始化回测状态
        cash = self.initial_capital
        position = 0
        
        # 记录初始状态
        timestamp = price_data.index[0]
        price = price_data.iloc[0]['close']
        holdings = position * price
        total_value = cash + holdings
        
        self.results['timestamp'].append(timestamp)
        self.results['price'].append(price)
        self.results['position'].append(position)
        self.results['cash'].append(cash)
        self.results['holdings'].append(holdings)
        self.results['total_value'].append(total_value)
        self.results['returns'].append(0)
        self.results['prediction'].append(0)
        
        # 遍历每个时间点
        for i in range(1, len(common_idx)):
            timestamp = price_data.index[i]
            price = price_data.iloc[i]['close']
            
            # 获取当前特征数据
            current_features = feature_data.iloc[i-1:i]
            
            # 进行预测
            prediction = self.model.predict(current_features)[0]
            
            # 记录预测值
            self.results['prediction'].append(prediction)
            
            # 根据预测值决定交易信号
            if prediction > threshold and position == 0:  # 买入信号
                # 计算买入数量
                buy_amount = cash * position_size
                buy_price = price * (1 + self.slippage)  # 考虑滑点
                fee = buy_amount * self.trading_fee
                shares = (buy_amount - fee) / buy_price
                
                # 更新状态
                position += shares
                cash -= (shares * buy_price + fee)
                
                logger.info(f"{timestamp}: 买入 {shares:.4f} 股，价格 {buy_price:.2f}，费用 {fee:.2f}，剩余现金 {cash:.2f}")
                
            elif prediction < -threshold and position > 0:  # 卖出信号
                # 计算卖出金额
                sell_price = price * (1 - self.slippage)  # 考虑滑点
                sell_amount = position * sell_price
                fee = sell_amount * self.trading_fee
                
                # 更新状态
                cash += (sell_amount - fee)
                position = 0
                
                logger.info(f"{timestamp}: 卖出全部持仓，价格 {sell_price:.2f}，费用 {fee:.2f}，现金 {cash:.2f}")
            
            # 计算当前持仓价值和总资产
            holdings = position * price
            total_value = cash + holdings
            
            # 计算收益率
            returns = (total_value / self.initial_capital - 1) * 100
            
            # 记录状态
            self.results['timestamp'].append(timestamp)
            self.results['price'].append(price)
            self.results['position'].append(position)
            self.results['cash'].append(cash)
            self.results['holdings'].append(holdings)
            self.results['total_value'].append(total_value)
            self.results['returns'].append(returns)
        
        # 计算回测评估指标
        metrics = self.calculate_metrics()
        
        # 转换结果为DataFrame
        results_df = pd.DataFrame(self.results)
        results_df.set_index('timestamp', inplace=True)
        
        # 保存回测结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(os.path.join(self.output_dir, f"backtest_results_{timestamp}.csv"))
        
        # 绘制并保存回测结果图表
        self.plot_results(results_df, timestamp)
        
        return {'results': results_df, 'metrics': metrics}
    
    def calculate_metrics(self):
        """
        计算回测评估指标
        
        返回:
            评估指标字典
        """
        # 转换回测结果为DataFrame
        results_df = pd.DataFrame(self.results)
        results_df.set_index('timestamp', inplace=True)
        
        # 计算日收益率
        daily_returns = results_df['total_value'].pct_change().dropna()
        
        # 计算年化收益率
        annual_return = daily_returns.mean() * 252 * 100
        
        # 计算年化波动率
        annual_volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # 计算夏普比率
        if annual_volatility == 0:
            sharpe_ratio = 0
        else:
            risk_free_rate = 0.01  # 假设无风险利率为1%
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # 计算最大回撤
        cum_returns = (1 + daily_returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns / rolling_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # 计算胜率
        trades = []
        current_position = 0
        entry_price = 0
        for i in range(1, len(results_df)):
            if results_df['position'].iloc[i] > 0 and current_position == 0:  # 开仓
                current_position = results_df['position'].iloc[i]
                entry_price = results_df['price'].iloc[i]
            elif results_df['position'].iloc[i] == 0 and current_position > 0:  # 平仓
                exit_price = results_df['price'].iloc[i]
                profit = exit_price - entry_price
                trades.append(profit)
                current_position = 0
        
        if trades:
            win_rate = sum(1 for x in trades if x > 0) / len(trades) * 100
            profit_loss_ratio = abs(sum(x for x in trades if x > 0) / sum(x for x in trades if x < 0)) if sum(x for x in trades if x < 0) != 0 else float('inf')
        else:
            win_rate = 0
            profit_loss_ratio = 0
        
        # 计算总收益率
        total_return = results_df['returns'].iloc[-1]
        
        metrics = {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_loss_ratio': float(profit_loss_ratio),
            'num_trades': len(trades)
        }
        
        logger.info("回测评估指标:")
        for key, value in metrics.items():
            logger.info(f"  - {key}: {value:.4f}")
        
        return metrics
    
    def plot_results(self, results_df: pd.DataFrame, timestamp: str):
        """
        绘制回测结果图表
        
        参数:
            results_df: 回测结果DataFrame
            timestamp: 时间戳，用于文件名
        """
        plt.figure(figsize=(12, 8))
        
        # 绘制价格和总资产
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(results_df.index, results_df['price'], 'b-', label='Price')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        ax2.plot(results_df.index, results_df['total_value'], 'r-', label='Portfolio Value')
        ax2.set_ylabel('Portfolio Value')
        ax2.legend(loc='upper right')
        
        # 绘制持仓情况
        ax3 = plt.subplot(2, 1, 2)
        ax3.plot(results_df.index, results_df['position'], 'g-', label='Position')
        ax3.set_ylabel('Position')
        ax3.set_xlabel('Date')
        ax3.legend(loc='upper left')
        
        ax4 = ax3.twinx()
        ax4.plot(results_df.index, results_df['returns'], 'm-', label='Returns (%)')
        ax4.set_ylabel('Returns (%)')
        ax4.legend(loc='upper right')
        
        plt.title('Backtest Results')
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(self.output_dir, f"backtest_chart_{timestamp}.png"), dpi=300)
        plt.close()
        
        logger.info(f"回测结果图表已保存至 {self.output_dir}/backtest_chart_{timestamp}.png")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="加密货币交易策略回测工具")
    
    # 主要参数
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    
    # 数据参数
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易对")
    parser.add_argument("--timeframe", type=str, default="1h", help="时间框架")
    parser.add_argument("--price_file", type=str, help="价格数据文件路径，如不指定则根据symbol和timeframe自动生成")
    parser.add_argument("--feature_file", type=str, help="特征文件路径，如不指定则根据symbol和timeframe自动生成")
    parser.add_argument("--target_file", type=str, help="目标文件路径，如不指定则根据symbol和timeframe自动生成")
    parser.add_argument("--features", type=str, help="要使用的特征列表，逗号分隔")
    
    # 回测参数
    parser.add_argument("--initial_capital", type=float, default=10000, help="初始资金")
    parser.add_argument("--trading_fee", type=float, default=0.001, help="交易费用（占比）")
    parser.add_argument("--slippage", type=float, default=0.0005, help="滑点（占比）")
    parser.add_argument("--position_size", type=float, default=0.2, help="仓位大小（总资金占比）")
    parser.add_argument("--threshold", type=float, default=0.0, help="交易阈值")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="results/backtest", help="输出目录")
    
    # 其他参数
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    default_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(config_path=args.config, default_level=default_level)
    
    logger.info("开始回测流程")
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 自动生成文件路径
        if args.price_file is None:
            args.price_file = os.path.join("data/raw", args.symbol + "_" + args.timeframe + "_" + "*.csv")
            # 获取最新的价格文件
            price_files = sorted([f for f in os.listdir("data/raw") if f.startswith(f"{args.symbol}_{args.timeframe}")])
            if price_files:
                args.price_file = os.path.join("data/raw", price_files[-1])
        
        if args.feature_file is None:
            # 自动生成特征文件路径
            feature_dir = os.path.join("data/processed/features", args.symbol)
            args.feature_file = os.path.join(feature_dir, f"features_{args.timeframe}.csv")
            
        if args.target_file is None:
            # 自动生成目标文件路径
            target_dir = os.path.join("data/processed/features", args.symbol)
            args.target_file = os.path.join(target_dir, f"targets_{args.timeframe}.csv")
        
        # 从配置文件中获取回测参数
        if args.initial_capital is None:
            args.initial_capital = config.get("evaluation", {}).get("backtest", {}).get("initial_capital", 10000)
            
        if args.trading_fee is None:
            args.trading_fee = config.get("evaluation", {}).get("backtest", {}).get("trading_fee", 0.001)
            
        if args.slippage is None:
            args.slippage = config.get("evaluation", {}).get("backtest", {}).get("slippage", 0.0005)
        
        # 创建回测对象
        backtest = Backtest(
            model_path=args.model,
            price_data_path=args.price_file,
            feature_data_path=args.feature_file,
            target_data_path=args.target_file,
            initial_capital=args.initial_capital,
            trading_fee=args.trading_fee,
            slippage=args.slippage,
            output_dir=args.output_dir
        )
        
        # 解析特征列表
        features = None
        if args.features:
            features = args.features.split(',')
            
        # 运行回测
        results = backtest.run(
            features=features,
            position_size=args.position_size,
            threshold=args.threshold
        )
        
        logger.info("回测流程完成")
        
        return 0
    except Exception as e:
        logger.error(f"回测过程中出错: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 