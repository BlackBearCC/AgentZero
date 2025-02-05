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
            returns = (final_value / initial_cash - 1) * 100
            
            # 安全获取分析器结果
            analysis = strat.analyzers.sharpe.get_analysis()
            sharpe = analysis.get('sharperatio', 0.0) if analysis else 0.0
            
            dd_analysis = strat.analyzers.drawdown.get_analysis()
            max_dd = dd_analysis.get('max', {}).get('drawdown', 0.0) if dd_analysis else 0.0
            
            trade_analysis = strat.analyzers.trades.get_analysis()
            total_trades = trade_analysis.get('total', {}).get('total', 0) if trade_analysis else 0
            
            # 打印回测结果
            print(f'''
=== 回测结果 ===
起始资金: ${initial_cash:.2f}
最终资金: ${final_value:.2f}
总收益率: {returns:.2f}%
夏普比率: {sharpe if sharpe is not None else 0.0:.2f}
最大回撤: {max_dd if max_dd is not None else 0.0:.2f}%
总交易次数: {total_trades}
''')
            
            # 绘制图表
            # cerebro.plot(style='candlestick', volume=False)
            
            return results
            
        except Exception as e:
            self.logger.error(f"回测执行失败: {str(e)}")
            raise 