import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import sys

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.trading.strategies.grid_strategy import GridStrategy
from src.trading.feeds.crypto_feed import CCXTFeed
from src.utils.logger import Logger

class BacktestRunner:
    def __init__(self):
        self.logger = Logger("backtest")
        
    def load_data(self, 
                 symbol: str,
                 timeframe: str,
                 start: datetime,
                 end: datetime) -> pd.DataFrame:
        """从CCXT加载数据"""
        self.logger.info(f"开始获取数据: {symbol} {timeframe}")
        self.logger.info(f"时间范围: {start} - {end}")
        
        feed = CCXTFeed(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end
        )
        
        # 使用1分钟数据进行回测
        df = feed._df_1m.copy()
        
        # 确保数据在指定的时间范围内
        df = df[df.index >= pd.Timestamp(start)]
        df = df[df.index <= pd.Timestamp(end)]
        
        # 重置索引，保持时间戳列
        df['timestamp'] = df.index
        df = df.reset_index(drop=True)
        
        self.logger.info(f"数据范围: {df['timestamp'].min()} - {df['timestamp'].max()}")
        self.logger.info(f"数据点数: {len(df)}")
        
        return df
        
    def run(self,
            symbol: str,
            timeframe: str,
            start: datetime,
            end: datetime,
            strategy_params: Dict[str, Any] = None,
            plot: bool = True) -> Dict:
        """运行回测"""
        # 加载数据
        data = self.load_data(symbol, timeframe, start, end)
        
        if data.empty:
            self.logger.error("未获取到数据")
            return {}
            
        # 默认策略参数
        default_params = {
            'initial_capital': 10000,
            'commission_rate': 0.001,
            'grid_num': 20,
            'price_range': 0.1,
            'position_ratio': 0.01,
            'take_profit': 0.005,
            'symbol': symbol
        }
        
        # 更新策略参数
        if strategy_params:
            default_params.update(strategy_params)
            
        self.logger.info("策略参数:")
        for k, v in default_params.items():
            self.logger.info(f"  {k}: {v}")
            
        # 创建并运行策略
        self.logger.info("开始回测...")
        strategy = GridStrategy(data=data, **default_params)
        results = strategy.run()
        
        # 输出回测结果
        self.print_results(results)
        
        # 使用策略的plot_results方法
        if plot:
            strategy.plot_results(results, data)
            
        return results
    
    def print_results(self, results: Dict):
        """打印回测结果"""
        self.logger.info("====== 回测结果 ======")
        self.logger.info(f"初始资金: ${results['initial_capital']:,.2f}")
        self.logger.info(f"最终权益: ${results['final_equity']:,.2f}")
        self.logger.info(f"总收益率: {results['total_return']:.2%}")
        self.logger.info(f"总交易次数: {results['total_trades']}")
        self.logger.info(f"胜率: {results['win_rate']:.2%}")
        self.logger.info(f"平均收益率: {results['avg_return']:.2%}")
        self.logger.info(f"最大回撤: {results['max_drawdown']:.2%}")
        
    def plot_results(self, results: Dict, price_data: pd.DataFrame):
        """绘制回测结果"""
        plt.style.use('seaborn')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[2, 1, 1])
        
        # 绘制价格和权益曲线
        equity_curve = results['equity_curve']
        ax1.plot(price_data.index, price_data['close'], label='Price', color='gray', alpha=0.5)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(equity_curve['timestamp'], equity_curve['equity'], 
                     label='Portfolio Value', color='blue')
        ax1.set_title('Price and Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1_twin.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # 合并两个轴的图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 绘制交易分布
        if len(results['trades']) > 0:
            returns = results['trades']['return']
            ax2.hist(returns, bins=50, alpha=0.75)
            ax2.set_title('Trade Returns Distribution')
            ax2.set_xlabel('Return')
            ax2.set_ylabel('Frequency')
            ax2.grid(True)
            
        # 绘制回撤
        equity_series = equity_curve['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        
        ax3.fill_between(equity_curve['timestamp'], drawdown, 0, 
                        color='red', alpha=0.3)
        ax3.set_title('Drawdown')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown %')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    # 创建回测运行器
    runner = BacktestRunner()
    
    # 设置回测参数
    symbol = 'DOGE/USDT'
    timeframe = '1m'
    start = datetime(2025, 2, 1)
    end = datetime(2025, 2, 7)
    
    # 调整策略参数
    strategy_params = {
        'initial_capital': 10000,    # 初始资金
        'commission_rate': 0.0004,   # 币安合约手续费率
        'leverage': 10,              # 杠杆倍数
        'grid_num': 10,             # 单边网格数量
        'price_range': 0.1,        # 价格区间±2%
        'position_ratio': 0.8,      # 总资金使用比例80%
        'long_pos_limit': 0.4,      # 多头仓位上限40%
        'short_pos_limit': 0.4,     # 空头仓位上限40%
        'min_price_precision': 0.01, # 最小价格精度
        'min_qty_precision': 0.001   # 最小数量精度
    }
    
    # 运行回测
    results = runner.run(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        strategy_params=strategy_params,
        plot=True
    )

if __name__ == '__main__':
    main() 