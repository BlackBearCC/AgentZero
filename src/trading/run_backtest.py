import asyncio
import sys
from datetime import datetime
from pathlib import Path
import argparse
import backtrader as bt
from typing import Dict, Any

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# 使用绝对导入
from src.utils.logger import Logger
from src.trading.feeds.crypto_feed import DataManager
from src.trading.strategies.grid import AutoGridStrategy

# 创建回测专用logger
logger = Logger("backtest")
logger.info(f"添加 {project_root} 到 Python 路径")

def parse_date(date_str):
    """解析日期字符串为datetime对象"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"无效的日期格式: {date_str}. 请使用 YYYY-MM-DD 格式")

class BacktestRunner:
    """回测运行器"""
    
    def __init__(self):
        self.logger = logger  # 使用全局logger
        self.data_manager = DataManager()

    async def run(self,
                 symbol: str = 'DOGE/USDT',
                 timeframe: str = '15m',
                 start: datetime = None,
                 end: datetime = None,
                 initial_cash: float = 10000,
                 commission: float = 0.001,
                 strategy_params: Dict[str, Any] = None):
        """运行回测"""
        try:
            self.logger.info(f"初始化回测引擎 - {symbol}, {timeframe}")
            cerebro = bt.Cerebro()
            
            # 设置资金和手续费
            cerebro.broker.setcash(initial_cash)
            cerebro.broker.setcommission(commission=commission)
            self.logger.info(f"设置初始资金: ${initial_cash:.2f}, 手续费率: {commission:.4f}")
            
            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            
            # 获取数据并添加到回测引擎
            self.logger.info("开始加载历史数据...")
            data = self.data_manager.get_feed(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end
            )
            cerebro.adddata(data)
            
            # 添加策略
            self.logger.info("添加交易策略...")
            if strategy_params:
                cerebro.addstrategy(AutoGridStrategy, **strategy_params)
            else:
                cerebro.addstrategy(AutoGridStrategy)
            
            # 运行回测
            self.logger.info(f"开始回测 - {symbol}, {timeframe}")
            results = cerebro.run()
            strat = results[0]
            
            # 获取分析结果
            final_value = cerebro.broker.getvalue()
            trade_analysis = strat.analyzers.trades.get_analysis()
            
            # 计算交易统计
            total_trades = trade_analysis.get('total', {}).get('total', 0)
            won_trades = trade_analysis.get('won', {}).get('total', 0)
            lost_trades = trade_analysis.get('lost', {}).get('total', 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            # 计算盈亏统计
            avg_won = trade_analysis.get('won', {}).get('pnl', {}).get('average', 0)
            avg_lost = trade_analysis.get('lost', {}).get('pnl', {}).get('average', 0)
            profit_factor = abs(avg_won / avg_lost) if avg_lost != 0 else 0
            max_drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
            
            # 记录回测结果
            self.logger.info("\n=== 回测结果 ===")
            self.logger.info(f"初始资金: ${initial_cash:.2f}")
            self.logger.info(f"最终资金: ${final_value:.2f}")
            self.logger.info(f"总收益率: {((final_value/initial_cash - 1) * 100):.2f}%")
            self.logger.info(f"总交易次数: {total_trades}")
            self.logger.info(f"胜率: {win_rate:.2f}%")
            self.logger.info(f"盈亏比: {profit_factor:.2f}")
            self.logger.info(f"平均盈利: ${avg_won:.2f}")
            self.logger.info(f"平均亏损: ${abs(avg_lost):.2f}")
            self.logger.info(f"最大回撤: {max_drawdown:.2f}%")
            
            # 绘制图表
            cerebro.plot(style='candlestick', volume=False)
            
            return {
                'final_value': final_value,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"回测执行失败: {str(e)}")
            raise

async def main():
    """主函数"""
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='网格交易策略回测')
    parser.add_argument('--start', type=parse_date, default="2024-01-01",
                      help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=parse_date, default="2025-02-05",
                      help='回测结束日期 (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, default='DOGE/USDT',
                      help='交易对 (默认: DOGE/USDT)')
    parser.add_argument('--timeframe', type=str, default='15m',
                      help='K线周期 (默认: 15m)')
    parser.add_argument('--cash', type=float, default=10000,
                      help='初始资金 (默认: 10000)')
    
    args = parser.parse_args()
    
    # 检查日期范围
    if args.end <= args.start:
        logger.error("结束日期必须晚于开始日期")
        return
    
    # 策略参数配置
    strategy_params = {
        'grid_number': 50,         # 网格数量
        'position_size': 0.02,     # 每格仓位
        'atr_period': 14,          # ATR周期
        'vol_period': 20,          # 波动率周期
        'grid_min_spread': 0.002,  # 最小网格间距
        'grid_max_spread': 0.06,   # 最大网格间距
        'grid_expansion': 2.0      # 网格区间扩展系数
    }
    
    # 运行回测
    runner = BacktestRunner()
    await runner.run(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        initial_cash=args.cash,
        strategy_params=strategy_params
    )

if __name__ == '__main__':
    logger.info("启动回测脚本...")
    asyncio.run(main())
    logger.info("回测完成") 