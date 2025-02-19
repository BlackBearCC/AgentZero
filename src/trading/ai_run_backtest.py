from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import torch
from pathlib import Path

from src.trading.feeds.crypto_feed import DataManager
from src.trading.models.datasets import MarketMicrostructureDataset
from src.trading.models.transformer import AlphaTransformer

class TransformerBacktester:
    """Transformer模型回测器"""
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.load_model(model_path)
        
    def load_model(self, path: str):
        """加载预训练模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model = AlphaTransformer(**checkpoint['model_config']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler_state']
        self.model.eval()
        print("模型加载成功")
        
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    initial_capital: float = 1_000_000,
                    transaction_cost: float = 0.0004,
                    stop_loss: float = 0.02, 
                    take_profit: float = 0.03,
                    position_limit: float = 0.5,
                    volatility_threshold: float = 0.02,
                    signal_threshold: float = 0.5,
                    smoothing_factor: float = 0.7,
                    prediction_horizon: int = 14,
                    verbose: bool = True
                    ) -> Dict:
        """运行回测"""
        print("开始回测...")
        
        # 准备数据
        dataset = MarketMicrostructureDataset(data)
        features = dataset.features
        
        # 初始化变量
        positions = []
        equity = [initial_capital]
        current_position = 0
        entry_price = 0
        trades = []
        signal_records = []
        
        # 遍历每个时间点
        window_size = 60
        for i in range(window_size, len(features) - prediction_horizon):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            
            # 获取预测
            window = features[i-window_size:i]
            x = torch.FloatTensor(window).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(x)
            
            future_returns = pred[0].cpu().numpy()
            
            # 信号生成优化
            short_term = np.median(future_returns[0:3])    # 短期信号
            medium_term = np.median(future_returns[3:8])   # 中期信号
            long_term = np.median(future_returns[8:14])    # 长期信号
            
            # 计算波动率
            volatility = np.percentile(future_returns, 75) - np.percentile(future_returns, 25)
            
            # 信号平滑处理
            if len(signal_records) > 0:
                last_medium_term = signal_records[-1].get('medium_term', medium_term)
                medium_term = smoothing_factor * last_medium_term + \
                             (1 - smoothing_factor) * medium_term
            
            # 开仓和调仓逻辑
            target_position = current_position  # 默认保持当前仓位
            
            if current_position == 0:
                # 趋势一致性检查
                trend_aligned = (np.sign(medium_term) == np.sign(long_term))
                # 信号强度检查
                signal_strong = abs(medium_term/volatility) > signal_threshold
                
                if trend_aligned and signal_strong:
                    target_position = np.sign(medium_term) * position_limit
            else:
                # 持仓调整
                if np.sign(short_term) != np.sign(current_position):
                    # 考虑平仓
                    target_position = 0
            
            # 记录信号
            signal_records.append({
                'time': current_time,
                'price': current_price,
                'short_term': short_term,
                'medium_term': medium_term,
                'long_term': long_term,
                'volatility': volatility,
                'current_position': current_position,
                'target_position': target_position,
                'trend_aligned': trend_aligned if current_position == 0 else None,
                'signal_strong': signal_strong if current_position == 0 else None
            })
            
            # 执行交易
            position_change = target_position - current_position
            if abs(position_change) > 0.1:  # 最小交易阈值
                # 计算交易成本
                transaction_fee = abs(position_change) * transaction_cost * equity[-1]
                
                # 执行交易
                trades.append({
                    'time': current_time,
                    'type': 'signal',
                    'price': current_price,
                    'size': position_change,
                    'cost': transaction_fee,
                    'pnl': 0  # 在交易时刻的收益为0
                })
                
                current_position = target_position
                if current_position != 0:
                    entry_price = current_price
            
            positions.append(current_position)
            
            # 更新权益
            price_change = data['close'].iloc[i] / data['close'].iloc[i-1] - 1
            equity_change = price_change * current_position * equity[-1]
            
            # 扣除交易成本
            if len(trades) > 0 and trades[-1]['time'] == current_time:
                equity_change -= trades[-1]['cost']
                
            equity.append(equity[-1] + equity_change)
            
            # 打印进度
            if i % 100 == 0:
                print(f"进度: {i}/{len(features)}, 当前权益: {equity[-1]:,.2f}")
        
        # 在回测循环结束后，创建返回数据结构
        dates = data.index[window_size:window_size+len(equity)]
        equity_series = pd.Series(equity, index=dates)
        position_series = pd.Series(positions, index=dates[:-1])  # positions比equity少一个
        
        # 转换交易记录为DataFrame
        if trades:  # 如果有交易
            trade_df = pd.DataFrame(trades)
            trade_df['time'] = pd.to_datetime(trade_df['time'])
        else:  # 如果没有交易
            trade_df = pd.DataFrame(columns=['time', 'type', 'price', 'size', 'pnl', 'cost'])
        
        # 转换信号记录为DataFrame
        signal_df = pd.DataFrame(signal_records)
        
        # 计算回测指标
        metrics = self._calculate_metrics(equity, positions, trades, dates)
        
        # # 绘制回测结果
        # self._plot_backtest_results(equity_series, position_series, trade_df)
        # self._plot_signal_analysis(signal_df)
        
        return {
            'equity': equity_series,
            'positions': position_series,
            'trades': trade_df,
            'signals': signal_df,
            'metrics': metrics
        }
    
    def _calculate_position(self, mu: float, sigma: float, limit: float = 1.0) -> float:
        """计算目标仓位"""
        if sigma < 1e-6:
            return 0
        kelly = mu / (sigma**2)
        return np.clip(kelly, -limit, limit)  # 限制仓位在[-limit, limit]之间
    
    def _calculate_metrics(self, 
                         equity: list, 
                         positions: list,
                         trades: list,
                         dates: pd.DatetimeIndex) -> Dict:
        """计算回测指标"""
        equity_series = pd.Series(equity, index=dates)
        position_series = pd.Series(positions, index=dates[:-1])
        returns = equity_series.pct_change().dropna()
        
        # 转换交易记录为DataFrame并添加更多信息
        trade_df = pd.DataFrame(trades)
        if not trade_df.empty:
            # 添加持仓时间
            trade_df['time'] = pd.to_datetime(trade_df['time'])
            trade_df['hold_time'] = None
            trade_df['exit_time'] = None
            trade_df['exit_price'] = None
            trade_df['return_pct'] = None
            
            # 计算每笔交易的详细信息
            current_position = 0
            entry_time = None
            entry_price = 0
            
            for i, row in trade_df.iterrows():
                if current_position == 0 and row['size'] != 0:  # 开仓
                    entry_time = row['time']
                    entry_price = row['price']
                    current_position = row['size']
                elif current_position != 0:  # 持仓中
                    if (current_position > 0 and row['size'] < 0) or \
                       (current_position < 0 and row['size'] > 0) or \
                       row['type'] == 'stop_loss':  # 平仓或止损
                        # 更新前一笔开仓交易的信息
                        mask = (trade_df['time'] == entry_time)
                        trade_df.loc[mask, 'exit_time'] = row['time']
                        trade_df.loc[mask, 'exit_price'] = row['price']
                        trade_df.loc[mask, 'hold_time'] = (row['time'] - entry_time).total_seconds() / 3600  # 转换为小时
                        trade_df.loc[mask, 'return_pct'] = \
                            (row['price']/entry_price - 1) * 100 * np.sign(current_position)
                        
                        current_position = row['size']  # 更新持仓
                        if current_position != 0:  # 如果是反手交易
                            entry_time = row['time']
                            entry_price = row['price']
            
            # 计算交易统计
            win_trades = trade_df[trade_df['pnl'] > 0]
            win_rate = len(win_trades) / len(trade_df) if len(trade_df) > 0 else 0
            avg_trade_return = trade_df['pnl'].mean()
            total_cost = trade_df['cost'].sum()
            avg_hold_time = trade_df['hold_time'].mean()  # 现在是以小时为单位
            
            # 打印详细交易记录
            print("\n====== 交易详情 ======")
            for i, trade in trade_df.iterrows():
                direction = "买入" if trade['size'] > 0 else "卖出"
                size = abs(trade['size'])
                hold_time_str = f"{trade['hold_time']:.1f}小时" if pd.notna(trade['hold_time']) else "持仓中"
                return_pct_str = f"{trade['return_pct']:.2f}%" if pd.notna(trade['return_pct']) else "持仓中"
                exit_time_str = trade['exit_time'] if pd.notna(trade['exit_time']) else "持仓中"
                exit_price_str = f"{trade['exit_price']:.2f}" if pd.notna(trade['exit_price']) else "-"
                
                print(f"""
交易 #{i+1}:
类型: {trade['type']}
方向: {direction}
规模: {size:.2f}
入场时间: {trade['time']}
入场价格: {trade['price']:.2f}
出场时间: {exit_time_str}
出场价格: {exit_price_str}
持仓时间: {hold_time_str}
收益率: {return_pct_str}
收益金额: {trade['pnl']:.2f}
交易成本: {trade['cost']:.2f}
                """)
            
            print(f"""
====== 交易统计 ======
总交易次数: {len(trade_df)}
盈利交易: {len(win_trades)}
亏损交易: {len(trade_df) - len(win_trades)}
胜率: {win_rate*100:.2f}%
平均收益: {avg_trade_return:.2f}
平均持仓时间: {avg_hold_time:.1f}小时
总交易成本: {total_cost:.2f}
最大单笔收益: {trade_df['pnl'].max():.2f}
最大单笔亏损: {trade_df['pnl'].min():.2f}
平均每笔成本: {total_cost/len(trade_df):.2f}
=====================
            """)
        
        # 计算其他回测指标
        total_return = (equity[-1] / equity[0] - 1) * 100
        annual_return = (1 + total_return/100) ** (252/len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol != 0 else 0
        max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100
        
        # 绘制回测图表
        # self._plot_backtest_results(equity_series, position_series, trade_df)
        
        return {
            'equity': equity_series,
            'positions': position_series,
            'trades': trade_df,
            'metrics': {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_vol': annual_vol,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_trade_return': avg_trade_return,
                'total_cost': total_cost,
                'trade_count': len(trade_df)
            }
        }

    def _plot_backtest_results(self, equity, positions, trades):
        """绘制回测结果图表"""
        plt.figure(figsize=(15, 12))
        
        # 绘制权益曲线
        plt.subplot(3, 1, 1)
        plt.plot(equity.index, equity.values, label='Equity')
        plt.title('Backtest Results')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        
        # 绘制仓位
        plt.subplot(3, 1, 2)
        plt.plot(positions.index, positions.values, label='Position')
        plt.ylabel('Position')
        plt.grid(True)
        plt.legend()
        
        # 绘制交易点
        if not trades.empty:
            # 买入点
            entries = trades[trades['size'] > 0]
            plt.scatter(entries['time'], entries['price'], 
                       marker='^', color='g', s=100, label='Buy')
            
            # 卖出点
            exits = trades[trades['size'] < 0]
            plt.scatter(exits['time'], exits['price'], 
                       marker='v', color='r', s=100, label='Sell')
            
            # 止损点
            stops = trades[trades['type'] == 'stop_loss']
            plt.scatter(stops['time'], stops['price'], 
                       marker='x', color='black', s=100, label='Stop Loss')
        
        plt.subplot(3, 1, 3)
        plt.plot(equity.pct_change().rolling(20).std() * np.sqrt(252), label='Rolling Volatility')
        plt.ylabel('Annualized Volatility')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def _plot_signal_analysis(self, signal_df):
        """绘制信号分析图表"""
        plt.figure(figsize=(15, 12))
        
        # 绘制价格和预期收益
        plt.subplot(3, 1, 1)
        plt.plot(signal_df['time'], signal_df['price'], label='Price')
        plt.title('Signal Analysis')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        
        # 绘制预期收益
        plt.subplot(3, 1, 2)
        plt.plot(signal_df['time'], signal_df['medium_term']*100, label='Medium Term Return (%)')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.ylabel('Medium Term Return (%)')
        plt.grid(True)
        plt.legend()
        
        # 绘制信号强度（预期收益/波动率）
        plt.subplot(3, 1, 3)
        signal_strength = signal_df['medium_term'] / signal_df['volatility']
        plt.plot(signal_df['time'], signal_strength, label='Signal Strength')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.ylabel('Signal Strength')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 加载回测数据
    data_mgr = DataManager()
    data = data_mgr.get_feed(
        symbol='BTC/USDT',
        timeframe='15m',
        start=datetime(2024, 6, 1),
        end=datetime(2025, 2, 1)
    )['indicator']
    
    # 运行回测
    backtester = TransformerBacktester('models/alpha_v1.pth')
    results = backtester.run_backtest(
        data=data,
        initial_capital=1_000_000,
        transaction_cost=0.0004,
        stop_loss=0.02,          # 2%止损
        take_profit=0.03,        # 3%止盈
        position_limit=0.5,      # 最大50%仓位
        volatility_threshold=0.02,# 2%波动率阈值
        signal_threshold=0.5,    # 信号阈值
        smoothing_factor=0.7,     # 信号平滑因子
        prediction_horizon=14    # 预测周期
    )