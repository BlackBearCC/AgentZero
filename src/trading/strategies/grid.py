import backtrader as bt
import numpy as np
from typing import List, Dict
from .base import BaseStrategy

class AutoGridStrategy(BaseStrategy):
    """自适应网格交易策略"""
    
    params = (
        ('risk_per_trade', 0.05),     # 提高到5%
        ('atr_period', 14),          # 波动率计算周期
        ('grid_expansion', 2.0),      # 扩大网格间距到2%
        ('max_grids', 8),            # 减少网格数量，提高每格仓位
        ('rebalance_days', 5),       # 网格再平衡周期
        ('max_drawdown', 15),        # 最大回撤止损线(%)
        ('min_profit_pct', 0.3),     # 最小利润率要求
    )

    def _init_indicators(self):
        """初始化技术指标"""
        # 核心指标
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.volatility = bt.indicators.EMA(self.data.close, period=20) / bt.indicators.EMA(self.data.close, period=50)
        self.sma = bt.indicators.SMA(self.data.close, period=20)
        
        # 动态网格参数
        self.price_center = None
        self.grid_levels: List[Dict] = []
        self.last_rebalance = 0
        
        # 风险控制
        self.trade_count = 0
        self.drawdown = bt.analyzers.DrawDown()
        
        # 日志初始化
        self.logger.info(f"策略初始化完成 - ATR周期: {self.p.atr_period}, "
                        f"网格扩展: {self.p.grid_expansion}, "
                        f"最大网格数: {self.p.max_grids}")

    def next(self):
        """策略主逻辑"""
        try:
            # 初始化价格中枢
            if self.price_center is None:
                self.price_center = self.data.close[0]
                self.adaptive_grid_adjustment()
                self.logger.info(f"初始价格中枢设置为: {self.price_center:.2f}")
                return

            # 定期网格再平衡
            if len(self.data) - self.last_rebalance >= self.p.rebalance_days:
                self.adaptive_grid_adjustment()
                self.last_rebalance = len(self.data)
                self.logger.debug("执行网格再平衡")

            # 执行网格交易
            self.execute_grid_orders()

            # 动态止损检查
            self._check_risk_control()
            
        except Exception as e:
            self.logger.error(f"策略执行错误: {str(e)}")

    def adaptive_grid_adjustment(self):
        """自适应网格生成算法"""
        try:
            # 动态调整公式优化（带均值回归因子）
            atr_ratio = self.atr[0] / self.data.close[0]
            grid_step = np.clip(
                0.8 * self.volatility[0] + 1.2 * atr_ratio - 
                0.2 * (self.data.close[0]/self.sma[0]-1),
                0.002,  # 最小间距0.2%
                0.015   # 最大间距1.5%
            )
            
            # 趋势自适应扩展系数
            trend_strength = abs(self.data.close[0] - self.sma[0]) / self.atr[0]
            expansion = np.clip(1.3 + 0.3 * trend_strength, 1.2, 2.0)
            
            # 计算网格边界
            upper = self.data.close[0] * (1 + expansion * grid_step)
            lower = self.data.close[0] * (1 - expansion * grid_step)
            
            # 生成斐波那契网格
            self.grid_levels = []
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            for level in fib_levels:
                price = lower + (upper - lower) * level
                weight = 1 + 0.5 * (1 - abs(0.5 - level))  # 中间层级权重更高
                
                self.grid_levels.append({
                    'price': price,
                    'side': 'buy' if price < self.data.close[0] else 'sell',
                    'size': self._calculate_position_size(price) * weight
                })
                    
            self.grid_levels.sort(key=lambda x: x['price'])
            self.logger.info(f"更新网格层数: {len(self.grid_levels)}, "
                           f"网格步长: {grid_step:.4f}")
            
        except Exception as e:
            self.logger.error(f"网格调整错误: {str(e)}")

    def execute_grid_orders(self):
        """执行网格交易逻辑"""
        try:
            current_price = self.data.close[0]
            position = self.getposition()
            
            for grid in self.grid_levels:
                price = grid['price']
                
                # 买入条件：价格触及买入网格且无持仓
                if (grid['side'] == 'buy' and 
                    not position.size and 
                    current_price <= price):
                    
                    size = grid['size']
                    self.buy(price=price, 
                            size=size, 
                            exectype=bt.Order.Limit)
                    
                    self.logger.info(f"创建买入订单 - 价格: {price:.2f}, "
                                   f"数量: {size:.4f}")
                        
                # 卖出条件：价格触及卖出网格且有持仓
                elif (grid['side'] == 'sell' and 
                      position.size > 0 and 
                      current_price >= price):
                    
                    self.sell(price=price, 
                            size=position.size, 
                            exectype=bt.Order.Limit)
                    
                    self.logger.info(f"创建卖出订单 - 价格: {price:.2f}, "
                                   f"数量: {position.size:.4f}")
                    
        except Exception as e:
            self.logger.error(f"订单执行错误: {str(e)}")

    def _calculate_position_size(self, price: float) -> float:
        """计算仓位大小"""
        return (self.broker.cash * self.p.risk_per_trade) / price

    def _check_risk_control(self):
        """风险控制检查"""
        try:
            # 动态止损阈值
            dynamic_stop = max(
                self.p.max_drawdown,
                0.5 * self.volatility[0] * 100  # 波动越大止损越宽松
            )
            
            # 分阶段止损
            current_drawdown = self.drawdown.get_analysis()['drawdown']
            if current_drawdown > dynamic_stop:
                # 先平仓50%
                position = self.getposition()
                if position.size > 0:
                    self.close(size=position.size * 0.5)
                    self.logger.warning(f"第一阶段止损: 平仓50%, 当前回撤: {current_drawdown:.2f}%")
            elif current_drawdown > 1.2 * dynamic_stop:
                self.close_all_positions()
                self.logger.warning(f"第二阶段止损: 全部平仓, 当前回撤: {current_drawdown:.2f}%")
                
            # 检查趋势（可选）
            if self.data.close[0] < self.sma[0] * 0.95:  # 价格显著低于均线
                self.logger.warning("价格显著低于均线，考虑调整策略参数")
                
        except Exception as e:
            self.logger.error(f"风险控制检查错误: {str(e)}")

    def close_all_positions(self):
        """强制平仓"""
        if self.getposition().size > 0:
            self.close()
            self.logger.info("执行全部平仓操作")

    def get_analysis(self) -> Dict:
        """获取策略分析结果"""
        analysis = super().get_analysis()
        analysis.update({
            'grid_count': len(self.grid_levels),
            'last_rebalance': self.last_rebalance,
            'price_center': self.price_center,
            'drawdown': self.drawdown.get_analysis()
        })
        return analysis

    def notify_trade(self, trade):
        """交易通知"""
        if trade.isclosed:
            self.logger.info(
                f"交易结束 - 毛利润: {trade.pnl:.2f}, "
                f"净利润: {trade.pnlcomm:.2f}"
            ) 