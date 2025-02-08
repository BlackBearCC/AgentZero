import backtrader as bt
import numpy as np
from typing import Dict
from .base import BaseStrategy
from src.utils.logger import Logger
from datetime import timedelta

class BinanceGridStrategy(BaseStrategy):
    """科学版双向合约网格策略"""
    
    params = (
        ('grids', 20),              # 单边网格数量
        ('position_ratio', 0.01),   # 每格风险比例
        ('price_range', 0.10),      # 价格波动区间(±10%)
        ('leverage', 10),           # 杠杆倍数
        ('take_profit', 0.005),     # 止盈间距(0.5%)
    )

    def __init__(self):
        super().__init__()
        self.logger = Logger("grid_strategy")
        
        # 核心数据结构
        self.grid_orders = {}       # {price: order}
        self.active_trades = {}     # {trade_id: trade_info}
        self.grid_levels = []       # 所有网格价格
        
        # 策略状态
        self.current_center = None  # 当前网格中心价
        self.upper_limit = None     # 区间上界
        self.lower_limit = None     # 区间下界
        
        # 风控参数
        self.max_drawdown = 0.20    # 最大回撤
        self.equity_high = self.broker.startingcash

        # 添加交易统计
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        self.total_loss = 0

    def notify_order(self, order):
        """订单处理改进版"""
        if order.status in [order.Submitted, order.Accepted]:
            # 订单提交和接受状态不输出日志，避免日志过多
            return
        
        if order.status == order.Completed:
            # 订单成交时输出详细信息
            order_type = "开仓" if order.ordtype in [bt.Order.Limit, bt.Order.Stop] else "平仓"
            direction = "买入" if order.isbuy() else "卖出"
            self.logger.info(f"{order_type}{direction} - 价格: {order.executed.price:.5f}, "
                            f"数量: {order.executed.size:.4f}, "
                            f"时间: {self.data.datetime.datetime()}")
            
            # 移除已完成订单
            if order.ref in self.grid_orders.values():
                price = [k for k,v in self.grid_orders.items() if v == order.ref][0]
                del self.grid_orders[price]
            
            # 处理开仓单
            if order.ordtype in [bt.Order.Limit, bt.Order.Stop]:
                self.process_entry_order(order)
            
            # 处理平仓单
            elif order.ordtype == bt.Order.Close:
                self.process_exit_order(order)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # 订单异常状态时输出警告信息
            self.logger.warning(f"订单异常 - 状态: {order.getstatusname()}, "
                              f"类型: {order.ordtypename()}, "
                              f"时间: {self.data.datetime.datetime()}")

    def process_entry_order(self, order):
        """处理开仓逻辑"""
        # 不做时间间隔检查，直接处理订单
        current_price = self.data.close[0]
        trade_side = 'long' if order.isbuy() else 'short'
        
        # 计算止盈价格
        if trade_side == 'long':
            tp_price = order.executed.price * (1 + self.p.take_profit)
        else:
            tp_price = order.executed.price * (1 - self.p.take_profit)
        
        # 立即挂出止盈单
        tp_order = self.sell(size=order.size, exectype=bt.Order.Limit, 
                            price=tp_price) if trade_side == 'long' else \
                   self.buy(size=order.size, exectype=bt.Order.Limit, 
                           price=tp_price)
        
        # 更新统计
        self.total_trades += 1
        
        # 记录交易信息
        self.active_trades[order.ref] = {
            'side': trade_side,
            'entry_price': order.executed.price,
            'size': order.size,
            'tp_order': tp_order.ref,
            'entry_time': self.data.datetime.datetime()
        }
        
        # 重新挂出网格单
        self.place_grid_order(order.executed.price)

    def place_grid_order(self, price):
        """在指定价格挂出双向订单"""
        # 多单（价格下方买入）
        buy_price = price * (1 - self.p.take_profit/2)
        if buy_price not in self.grid_orders:
            size = self.calculate_position_size(buy_price)
            order = self.buy(size=size, exectype=bt.Order.Limit, price=buy_price)
            self.grid_orders[buy_price] = order.ref
        
        # 空单（价格上方卖出）
        sell_price = price * (1 + self.p.take_profit/2)
        if sell_price not in self.grid_orders:
            size = self.calculate_position_size(sell_price)
            order = self.sell(size=size, exectype=bt.Order.Limit, price=sell_price)
            self.grid_orders[sell_price] = order.ref

    def calculate_position_size(self, price):
        """科学计算仓位"""
        risk_amount = self.broker.getvalue() * self.p.position_ratio
        leveraged_amount = risk_amount * self.p.leverage
        return leveraged_amount / price

    def initialize_grid(self):
        """科学初始化网格"""
        self.current_center = self.data.close[0]
        self.upper_limit = self.current_center * (1 + self.p.price_range)
        self.lower_limit = self.current_center * (1 - self.p.price_range)
        
        # 生成网格价格
        self.grid_levels = np.linspace(
            self.lower_limit, 
            self.upper_limit, 
            self.p.grids*2 + 1  # 包含中线
        )
        
        # 双向挂单
        for price in self.grid_levels:
            self.place_grid_order(price)

    def rebalance_grid(self):
        """网格再平衡逻辑"""
        current_price = self.data.close[0]
        
        # 价格偏离超过区间50%时重置
        if (current_price > self.current_center * (1 + self.p.price_range*0.5)) or \
           (current_price < self.current_center * (1 - self.p.price_range*0.5)):
            self.logger.warning("触发网格再平衡")
            self.initialize_grid()

    def next(self):
        """执行主逻辑"""
        # 首次运行初始化
        if not self.current_center:
            self.initialize_grid()
            return
            
        # 每4小时检查一次网格
        if self.data.datetime.time().hour % 4 == 0:
            self.rebalance_grid()
            self.check_risk()

    def check_risk(self):
        """风险控制"""
        current_equity = self.broker.getvalue()
        self.equity_high = max(self.equity_high, current_equity)
        drawdown = (self.equity_high - current_equity)/self.equity_high
        
        if drawdown > self.max_drawdown:
            self.logger.error(f"触发最大回撤限制 {drawdown:.2%}")
            self.close_all_positions()

    def close_all_positions(self):
        """紧急平仓"""
        for trade in list(self.active_trades.values()):
            self.cancel(trade['tp_order'])
            if trade['side'] == 'long':
                self.sell(size=trade['size'])
            else:
                self.buy(size=trade['size'])
        self.grid_orders.clear()

    def process_exit_order(self, order):
        """处理平仓订单逻辑"""
        # 找到对应的开仓订单
        entry_order_ref = [ref for ref, trade in self.active_trades.items() 
                          if trade['tp_order'] == order.ref]
        
        if entry_order_ref:
            trade_info = self.active_trades[entry_order_ref[0]]
            
            # 计算收益
            if trade_info['side'] == 'long':
                profit = (order.executed.price - trade_info['entry_price']) * trade_info['size']
            else:
                profit = (trade_info['entry_price'] - order.executed.price) * trade_info['size']
            
            # 记录交易结果
            self.logger.info(f"平仓成功 - 方向: {trade_info['side']}, "
                            f"收益: {profit:.2f}, "
                            f"持仓时间: {self.data.datetime.datetime() - trade_info['entry_time']}")
            
            # 清理交易记录
            del self.active_trades[entry_order_ref[0]]

    def stop(self):
        """策略终止"""
        self.logger.info("最终净值: %.2f" % self.broker.getvalue())