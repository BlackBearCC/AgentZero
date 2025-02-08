import backtrader as bt
import numpy as np
from typing import List, Dict
from .base import BaseStrategy
from src.utils.logger import Logger

class SimpleGridStrategy(BaseStrategy):
    """币安风格中性合约网格策略(修复版)"""
    
    params = (
        ('grid_number', 40),         # 网格线数量
        ('position_ratio', 0.01),    # 每格仓位比例(基于当前余额)
        ('price_range_ratio', 0.2),  # 价格区间比例(上下各10%)
        ('leverage', 10),            # 杠杆倍数
        ('take_profit_ratio', 0.05),# 止盈比例(0.5%)
        # ('stop_loss_ratio', 0.1),   # 止损比例(2%)
    )

    def __init__(self):
        super().__init__()
        self.logger = Logger("strategy")
        
        # 核心数据结构
        self.active_orders = []      
        self.grid_lines = []         
        self.position_size = 0       
        self.entry_prices = []       
        
        # 新增统计数据
        self.trade_stats = {
            'total_trades': 0,           # 总交易次数
            'tp_trades': 0,              # 止盈次数
            'total_profit': 0,           # 总盈利
            'positions': {}              # 持仓记录 {order_id: {'entry_time', 'entry_price', 'size', 'side'}}
        }
        
        # 策略状态
        self.upper_bound = None      # 当前区间上界
        self.lower_bound = None      # 当前区间下界
        self.base_price = None       # 网格基准价
        
        # 参数校验
        if self.p.leverage < 1 or self.p.leverage > 125:
            raise ValueError("杠杆倍数必须在1-125之间")

    def notify_order(self, order):
        """订单状态通知处理"""
        if order.status in [order.Completed]:
            # 记录成交信息
            self.logger.info(f"订单成交: {order.ordtypename()} {order.size:.4f}@{order.executed.price:.4f}")
            
            # 移除已完成订单
            if order in self.active_orders:
                self.active_orders.remove(order)
            
            # 记录交易统计
            self.trade_stats['total_trades'] += 1
            
            # 挂出止盈单
            if order.isbuy():
                # 做多后挂止盈卖单
                tp_price = order.executed.price * (1 + self.p.take_profit_ratio)
                tp_order = self.sell(size=abs(order.size), exectype=bt.Order.Limit, price=tp_price)
                self.active_orders.append(tp_order)
                # 记录开仓信息
                self.trade_stats['positions'][tp_order.ref] = {
                    'entry_time': self.data.datetime.datetime(),
                    'entry_price': order.executed.price,
                    'size': order.size,
                    'side': 'long'
                }
                self.logger.info(f"设置多单止盈: {order.size:.4f}@{tp_price:.4f}")
            else:
                # 做空后挂止盈买单 
                tp_price = order.executed.price * (1 - self.p.take_profit_ratio)
                tp_order = self.buy(size=abs(order.size), exectype=bt.Order.Limit, price=tp_price)
                self.active_orders.append(tp_order)
                # 记录开仓信息
                self.trade_stats['positions'][tp_order.ref] = {
                    'entry_time': self.data.datetime.datetime(),
                    'entry_price': order.executed.price,
                    'size': order.size,
                    'side': 'short'
                }
                self.logger.info(f"设置空单止盈: {order.size:.4f}@{tp_price:.4f}")
                
            # 如果是止盈单成交
            if order.ref in self.trade_stats['positions']:
                pos_info = self.trade_stats['positions'][order.ref]
                hold_time = self.data.datetime.datetime() - pos_info['entry_time']
                profit = 0
                
                if pos_info['side'] == 'long':
                    profit = (order.executed.price - pos_info['entry_price']) * abs(pos_info['size'])
                else:
                    profit = (pos_info['entry_price'] - order.executed.price) * abs(pos_info['size'])
                
                self.trade_stats['tp_trades'] += 1
                self.trade_stats['total_profit'] += profit
                
                self.logger.info(f"止盈成交统计: 方向={pos_info['side']} 持仓时间={hold_time} "
                               f"开仓价={pos_info['entry_price']:.4f} "
                               f"平仓价={order.executed.price:.4f} "
                               f"盈利={profit:.4f}")
                
                del self.trade_stats['positions'][order.ref]

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.logger.warning(f"订单异常: {order.status} {order.ordtypename()} {order.size:.4f}@{order.created.price:.4f}")

    def initialize_grid(self):
        """初始化网格系统"""
        self.cancel_all_orders()  # 清空历史订单
        
        self.base_price = self.data.close[0]
        price_range = self.base_price * self.p.price_range_ratio
        
        # 计算网格边界
        self.upper_bound = self.base_price + price_range
        self.lower_bound = self.base_price - price_range
        
        # 生成网格线
        self.grid_lines = np.linspace(
            self.lower_bound, 
            self.upper_bound, 
            self.p.grid_number + 1
        ).round(4)
        
        self.logger.info(f"当前价格: {self.base_price:.4f}")
        self.logger.info(f"网格区间: [{self.lower_bound:.4f}, {self.upper_bound:.4f}]")
        self.logger.info("网格价格明细:")
        
        # 双向挂单：每个网格线下方挂买单，上方挂卖单
        for i, price in enumerate(self.grid_lines):
            # 买单（做多）在网格线下方触发
            buy_price = price * (1 - 0.001)  # 略微低于网格线
            buy_size = self.calculate_position_size()
            order = self.buy(size=buy_size, exectype=bt.Order.Limit, price=buy_price)
            self.active_orders.append(order)
            
            # 卖单（做空）在网格线上方触发
            sell_price = price * (1 + 0.001)  # 略微高于网格线
            sell_size = self.calculate_position_size()
            order = self.sell(size=sell_size, exectype=bt.Order.Limit, price=sell_price)
            self.active_orders.append(order)
            
            # 打印网格信息
            self.logger.info(f"网格 {i+1:02d}: 价格={price:.4f} | "
                            f"买单={buy_price:.4f}(数量:{buy_size:.4f}) | "
                            f"卖单={sell_price:.4f}(数量:{sell_size:.4f})")

    def cancel_all_orders(self):
        """撤销所有未完成订单"""
        for order in self.active_orders:
            self.cancel(order)
        self.active_orders.clear()

    def rebalance_grid(self):
        """重新平衡网格系统"""
        current_price = self.data.close[0]
        
        # 当价格突破边界时重新初始化网格
        if current_price > self.upper_bound or current_price < self.lower_bound:
            self.logger.warning("价格突破区间，重新调整网格")
            self.initialize_grid()

    def calculate_position_size(self):
        """动态计算每格仓位大小"""
        equity = self.broker.getvalue()  
        # 考虑到要开多个网格，应该预留更多保证金
        risk_amount = equity * self.p.position_ratio * 0.8  # 增加0.8的安全系数
        return (risk_amount * self.p.leverage) / self.data.close[0]

    def next(self):
        """策略主逻辑"""
        # 首次运行初始化
        if self.upper_bound is None:
            self.initialize_grid()
            return
        
        # 检查是否需要重新平衡
        self.rebalance_grid()
        
        # 动态调整订单数量
        if len(self.active_orders) < self.p.grid_number * 0.8:
            self.initialize_grid()

    def get_balance_status(self):
        """获取账户风险数据"""
        equity = self.broker.getvalue()
        # 由于回测环境不支持保证金查询，仅返回净值信息
        return f"净值:{equity:.2f}"

    def stop(self):
        """策略结束时输出统计信息"""
        self.logger.info("策略统计信息:")
        self.logger.info(f"总交易次数: {self.trade_stats['total_trades']}")
        self.logger.info(f"止盈次数: {self.trade_stats['tp_trades']}")
        self.logger.info(f"总盈利: {self.trade_stats['total_profit']:.4f}")
        self.logger.info(f"策略运行结束 净值:{self.broker.getvalue():.2f}")