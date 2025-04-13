"""
回测主脚本 - 用于评估交易策略的表现
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
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
                 leverage: float = 3.0,  # 添加杠杆参数
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
            leverage: 杠杆倍数
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.price_data_path = price_data_path
        self.feature_data_path = feature_data_path
        self.target_data_path = target_data_path
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.leverage = leverage  # 保存杠杆倍数
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
            'position': [],  # 现在可能有正值(多头)或负值(空头)
            'position_type': [],  # 新增：记录头寸类型 ('long', 'short', 'none')
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
                model_data = pickle.load(f)
            logger.info(f"模型已从 {model_path} 加载")
            
            # 添加调试信息
            logger.info(f"模型类型: {type(model_data)}")
            
            # 处理模型是字典的情况
            if isinstance(model_data, dict):
                logger.info(f"模型是字典类型，键列表: {list(model_data.keys())}")
                
                # 检查字典中是否有model键
                if 'model' in model_data:
                    logger.info("使用'model'键中的模型对象")
                    return model_data['model']
                # 或者检查是否有trained_model键
                elif 'trained_model' in model_data:
                    logger.info("使用'trained_model'键中的模型对象")
                    return model_data['trained_model']
                # 如果有model_obj键
                elif 'model_obj' in model_data:
                    logger.info("使用'model_obj'键中的模型对象")
                    return model_data['model_obj']
                # 如果存在base_model
                elif 'base_model' in model_data:
                    logger.info("使用'base_model'键中的模型对象")
                    return model_data['base_model']
                else:
                    # 尝试查找可能的模型对象
                    for key, value in model_data.items():
                        # 检查value是否有predict方法
                        if hasattr(value, 'predict'):
                            logger.info(f"在键 '{key}' 中找到模型对象")
                            return value
                    
                    # 尝试查找xgb_model属性
                    for key, value in model_data.items():
                        if hasattr(value, 'xgb_model') and hasattr(value.xgb_model, 'predict'):
                            logger.info(f"在键 '{key}.xgb_model' 中找到模型对象")
                            return value.xgb_model
                    
                    # 如果找不到模型对象，输出字典的键以供诊断
                    logger.error(f"无法在模型字典中找到有效的模型对象。可用键: {list(model_data.keys())}")
                    
                    # 输出更多调试信息
                    for key, value in model_data.items():
                        logger.info(f"键 '{key}' 的值类型: {type(value)}")
                        
                    raise ValueError("加载的模型字典中没有有效的模型对象")
            
            # 如果模型本身就有predict方法
            if hasattr(model_data, 'predict'):
                logger.info("加载的对象直接具有predict方法")
                return model_data
                
            # 如果不是字典且没有predict方法，尝试查找xgb_model属性
            if hasattr(model_data, 'xgb_model'):
                logger.info("使用模型的xgb_model属性")
                return model_data.xgb_model
                
            logger.error(f"未知的模型类型: {type(model_data)}")
            raise ValueError(f"未能识别的模型类型: {type(model_data)}")
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
            threshold: 交易阈值，预测值绝对值超过阈值才会交易
            
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
        position = 0  # 0表示没有持仓，正数表示多头，负数表示空头
        position_type = "none"  # 持仓类型：多头、空头或无持仓
        entry_price = 0  # 记录开仓价格
        
        # 记录初始状态
        timestamp = price_data.index[0]
        price = price_data.iloc[0]['close']
        holdings = 0  # 初始持仓价值为0
        total_value = cash + holdings
        
        self.results['timestamp'].append(timestamp)
        self.results['price'].append(price)
        self.results['position'].append(position)
        self.results['position_type'].append(position_type)
        self.results['cash'].append(cash)
        self.results['holdings'].append(holdings)
        self.results['total_value'].append(total_value)
        self.results['returns'].append(0)
        self.results['prediction'].append(0)
        
        # 添加信号确认机制
        signal_count = 0  # 连续相同信号计数
        current_signal = 0  # 当前信号方向 (0: 无, 1: 多头, -1: 空头)
        min_holding_periods = 6  # 最小持仓时间（小时）
        holding_time = 0  # 当前持仓时间
        
        # 用于判断趋势的窗口
        trend_window = 5
        prev_predictions = [0] * trend_window
        
        # 遍历每个时间点
        for i in range(1, len(common_idx)):
            timestamp = price_data.index[i]
            price = price_data.iloc[i]['close']
            
            # 如果持有头寸，增加持仓时间
            if position != 0:
                holding_time += 1
            
            # 获取当前特征数据
            current_features = feature_data.iloc[i-1:i]
            
            # 进行预测，增强健壮性
            try:
                # 尝试直接预测
                prediction_result = self.model.predict(current_features)
                
                # 处理结果可能是数组的情况
                if isinstance(prediction_result, (list, np.ndarray)):
                    prediction = prediction_result[0]
                else:
                    prediction = prediction_result
                    
                logger.debug(f"预测值: {prediction}")
            except Exception as e:
                logger.error(f"预测失败: {str(e)}")
                # 使用一个保守的预测值
                prediction = 0
            
            # 记录预测值用于计算趋势
            prev_predictions.pop(0)
            prev_predictions.append(prediction)
            avg_prediction = sum(prev_predictions) / len(prev_predictions)
            
            # 记录预测值
            self.results['prediction'].append(prediction)
            
            # 提高交易阈值，减少噪声交易
            effective_threshold = max(threshold, 0.5)  # 基础阈值
            
            # 计算当前持仓的未实现盈亏
            if position != 0:
                # 多头持仓
                if position > 0:
                    unrealized_pnl = position * (price - entry_price) * self.leverage
                # 空头持仓
                else:
                    unrealized_pnl = abs(position) * (entry_price - price) * self.leverage
                
                # 计算浮动收益率，用于止损/止盈判断
                pnl_ratio = unrealized_pnl / (abs(position) * entry_price)
            else:
                unrealized_pnl = 0
                pnl_ratio = 0
            
            # 根据预测值和趋势确认决定交易信号
            # 多头信号：预测值 > 阈值，且平均预测值 > 0，且没有持仓或持有空头
            if prediction > effective_threshold and avg_prediction > 0 and (position <= 0):
                # 潜在多头信号
                if current_signal != 1:  # 新信号或与前一个信号方向不同
                    current_signal = 1
                    signal_count = 1
                else:  # 与前一个信号方向相同
                    signal_count += 1
                
                # 需要连续多个周期的相同信号才实际交易
                if signal_count >= 3:  # 至少需要3个周期的确认
                    # 如果有空头持仓，先平仓
                    if position < 0:
                        # 平空头持仓
                        close_price = price * (1 + self.slippage)  # 平空头时，价格需要上调(买入平空)
                        close_amount = abs(position) * close_price
                        fee = close_amount * self.trading_fee
                        
                        # 计算平仓后的收益
                        profit = abs(position) * (entry_price - close_price) * self.leverage
                        cash += profit - fee
                        
                        logger.info(f"{timestamp}: 平空头 {abs(position):.4f} 股，价格 {close_price:.2f}，费用 {fee:.2f}，"
                                   f"盈亏 {profit:.2f}，现金 {cash:.2f}")
                        
                        position = 0
                        position_type = "none"
                    
                    # 开多头仓位
                    order_amount = cash * position_size
                    contract_size = order_amount * self.leverage  # 杠杆放大仓位
                    buy_price = price * (1 + self.slippage)  # 考虑滑点
                    fee = contract_size * self.trading_fee
                    shares = contract_size / buy_price  # 计算合约数量
                    
                    # 更新状态
                    position = shares
                    position_type = "long"
                    entry_price = buy_price
                    cash -= fee  # 只扣除手续费，保证金从cash中冻结，由于已经计算了，无需再减
                    holding_time = 0  # 重置持仓时间
                    
                    logger.info(f"{timestamp}: 开多头 {shares:.4f} 股，价格 {buy_price:.2f}，费用 {fee:.2f}，"
                               f"杠杆 {self.leverage}倍，剩余现金 {cash:.2f}")
            
            # 空头信号：预测值 < -阈值，且平均预测值 < 0，且没有持仓或持有多头
            elif prediction < -effective_threshold and avg_prediction < 0 and (position >= 0):
                # 潜在空头信号
                if current_signal != -1:  # 新信号或与前一个信号方向不同
                    current_signal = -1
                    signal_count = 1
                else:  # 与前一个信号方向相同
                    signal_count += 1
                
                # 需要连续多个周期的相同信号才实际交易
                if signal_count >= 3:  # 至少需要3个周期的确认
                    # 如果有多头持仓，先平仓
                    if position > 0:
                        # 平多头持仓
                        close_price = price * (1 - self.slippage)  # 平多头时，价格需要下调(卖出平多)
                        close_amount = position * close_price
                        fee = close_amount * self.trading_fee
                        
                        # 计算平仓后的收益
                        profit = position * (close_price - entry_price) * self.leverage
                        cash += profit - fee
                        
                        logger.info(f"{timestamp}: 平多头 {position:.4f} 股，价格 {close_price:.2f}，费用 {fee:.2f}，"
                                   f"盈亏 {profit:.2f}，现金 {cash:.2f}")
                        
                        position = 0
                        position_type = "none"
                    
                    # 开空头仓位
                    order_amount = cash * position_size
                    contract_size = order_amount * self.leverage  # 杠杆放大仓位
                    sell_price = price * (1 - self.slippage)  # 考虑滑点
                    fee = contract_size * self.trading_fee
                    shares = contract_size / sell_price  # 计算合约数量
                    
                    # 更新状态
                    position = -shares  # 负数表示空头
                    position_type = "short"
                    entry_price = sell_price
                    cash -= fee  # 只扣除手续费
                    holding_time = 0  # 重置持仓时间
                    
                    logger.info(f"{timestamp}: 开空头 {shares:.4f} 股，价格 {sell_price:.2f}，费用 {fee:.2f}，"
                               f"杠杆 {self.leverage}倍，剩余现金 {cash:.2f}")
            
            # 平仓信号
            # 止损：多头下跌超过2%或空头上涨超过2%
            # 止盈：多头上涨超过5%或空头下跌超过5%
            # 持仓时间过长：超过48小时
            elif ((position > 0 and pnl_ratio < -0.02) or  # 多头止损
                 (position < 0 and pnl_ratio < -0.02) or  # 空头止损
                 (position != 0 and (pnl_ratio > 0.05)) or  # 止盈
                 (position != 0 and holding_time > 48)):  # 最大持仓时间
                
                if position > 0:  # 平多头
                    close_price = price * (1 - self.slippage)
                    close_amount = position * close_price
                    fee = close_amount * self.trading_fee
                    
                    # 计算平仓收益
                    profit = position * (close_price - entry_price) * self.leverage
                    cash += profit - fee
                    
                    # 记录平仓原因
                    if pnl_ratio < -0.02:
                        reason = "多头止损"
                    elif pnl_ratio > 0.05:
                        reason = "多头止盈"
                    elif holding_time > 48:
                        reason = "持仓时间过长"
                    else:
                        reason = "信号平仓"
                    
                    logger.info(f"{timestamp}: 平多头 {position:.4f} 股，价格 {close_price:.2f}，费用 {fee:.2f}，"
                               f"盈亏 {profit:.2f}，现金 {cash:.2f}，原因: {reason}")
                
                elif position < 0:  # 平空头
                    close_price = price * (1 + self.slippage)
                    close_amount = abs(position) * close_price
                    fee = close_amount * self.trading_fee
                    
                    # 计算平仓收益
                    profit = abs(position) * (entry_price - close_price) * self.leverage
                    cash += profit - fee
                    
                    # 记录平仓原因
                    if pnl_ratio < -0.02:
                        reason = "空头止损"
                    elif pnl_ratio > 0.05:
                        reason = "空头止盈"
                    elif holding_time > 48:
                        reason = "持仓时间过长"
                    else:
                        reason = "信号平仓"
                    
                    logger.info(f"{timestamp}: 平空头 {abs(position):.4f} 股，价格 {close_price:.2f}，费用 {fee:.2f}，"
                               f"盈亏 {profit:.2f}，现金 {cash:.2f}，原因: {reason}")
                
                # 重置状态
                position = 0
                position_type = "none"
                current_signal = 0
                signal_count = 0
            
            else:
                # 无新信号，保持当前状态
                pass
            
            # 计算当前持仓价值
            if position > 0:  # 多头
                holdings = position * price + (position * (price - entry_price) * (self.leverage - 1))
            elif position < 0:  # 空头
                holdings = abs(position) * entry_price - (abs(position) * (price - entry_price) * self.leverage)
            else:
                holdings = 0
            
            total_value = cash + holdings
            
            # 计算收益率
            returns = (total_value / self.initial_capital - 1) * 100
            
            # 记录状态
            self.results['timestamp'].append(timestamp)
            self.results['price'].append(price)
            self.results['position'].append(position)
            self.results['position_type'].append(position_type)
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
        plt.figure(figsize=(14, 10))
        
        # 绘制价格和总资产
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(results_df.index, results_df['price'], 'b-', label='Price')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        ax2.plot(results_df.index, results_df['total_value'], 'r-', label='Portfolio Value')
        ax2.set_ylabel('Portfolio Value')
        ax2.legend(loc='upper right')
        
        # 绘制持仓情况和预测值
        ax3 = plt.subplot(3, 1, 2)
        ax3.plot(results_df.index, results_df['position'], 'g-', label='Position')
        ax3.set_ylabel('Position')
        ax3.legend(loc='upper left')
        
        ax4 = ax3.twinx()
        ax4.plot(results_df.index, results_df['prediction'], 'm-', label='Prediction')
        ax4.set_ylabel('Prediction')
        ax4.legend(loc='upper right')
        
        # 绘制收益率
        ax5 = plt.subplot(3, 1, 3)
        ax5.plot(results_df.index, results_df['returns'], 'c-', label='Returns (%)')
        ax5.set_ylabel('Returns (%)')
        ax5.set_xlabel('Date')
        ax5.legend(loc='upper left')
        
        plt.title('Futures Trading Backtest Results')
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
    parser.add_argument("--leverage", type=float, default=3.0, help="杠杆倍数")
    
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
            leverage=args.leverage,
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