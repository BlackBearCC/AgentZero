import backtrader as bt
from datetime import datetime
from typing import Dict, Any
import logging
from ..feeds.crypto_feed import DataManager
from ..strategies.grid import AutoGridStrategy

class BacktestRunner:
    """简单的回测运行器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_manager = DataManager()

    async def run(self,
                 symbol: str = 'BTC/USDT',
                 timeframe: str = '15m',
                 start: datetime = None,
                 end: datetime = None,
                 initial_cash: float = 100000,
                 commission: float = 0.001,
                 strategy_params: Dict[str, Any] = None):
        """运行回测"""
        try:
            # 创建回测引擎
            cerebro = bt.Cerebro()
            
            # 设置资金和手续费
            cerebro.broker.setcash(initial_cash)
            cerebro.broker.setcommission(commission=commission)
            
            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            
            # 获取数据并添加到回测引擎
            data = self.data_manager.get_feed(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end
            )
            cerebro.adddata(data)
            
            # 添加策略
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
            
            # 计算杠杆收益率
            leverage = strategy_params.get('leverage', 1)
            initial_margin = initial_cash / leverage
            returns = ((final_value - initial_cash) / initial_margin) * 100
            
            # 获取更详细的交易统计
            trade_analysis = strat.analyzers.trades.get_analysis()
            
            # 计算胜率和盈亏比
            total_trades = trade_analysis.get('total', {}).get('total', 0)
            won_trades = trade_analysis.get('won', {}).get('total', 0)
            lost_trades = trade_analysis.get('lost', {}).get('total', 0)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            # 计算平均盈亏
            avg_won = trade_analysis.get('won', {}).get('pnl', {}).get('average', 0)
            avg_lost = trade_analysis.get('lost', {}).get('pnl', {}).get('average', 0)
            profit_factor = abs(avg_won / avg_lost) if avg_lost != 0 else 0
            
            # 打印回测结果
            print(f'''
=== 回测结果 ===
初始资金: ${initial_cash:.2f}
初始保证金: ${initial_margin:.2f}
最终资金: ${final_value:.2f}
杠杆倍数: {leverage}x
总收益率(基于保证金): {returns:.2f}%
总交易次数: {total_trades}
胜率: {win_rate:.2f}%
盈亏比: {profit_factor:.2f}
平均盈利: ${avg_won:.2f}
平均亏损: ${abs(avg_lost):.2f}
最大回撤: {strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0):.2f}%

''')
            
            # 绘制图表
            cerebro.plot(style='candlestick', volume=False)
            
            return results
            
        except Exception as e:
            self.logger.error(f"回测执行失败: {str(e)}")
            raise 