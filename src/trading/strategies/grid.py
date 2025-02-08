import backtrader as bt
import numpy as np
from typing import List, Dict
from .base import BaseStrategy
from src.utils.logger import Logger
from datetime import timedelta

class SimpleGridStrategy(BaseStrategy):
    """币安风格中性合约网格策略(修复版)"""
    
    params = (
        ('grid_number', 20),         # 网格线数量
        ('position_ratio', 0.01),    # 每格仓位比例(基于当前余额)
        ('price_range_ratio', 0.1),  # 价格区间比例(上下各10%)
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
            'total_hold_time': timedelta(),  # 总持仓时间
            'positions': {},             # 持仓记录
            'closed_trades': []          # 已平仓交易记录
        }
        
        # 策略状态
        self.upper_bound = None      # 当前区间上界
        self.lower_bound = None      # 当前区间下界
        self.base_price = None       # 网格基准价
        
        # 参数校验
        if self.p.leverage < 1 or self.p.leverage > 125:
            raise ValueError("杠杆倍数必须在1-125之间")

        # 使用execution数据源
        self.execution = self.datas[0]

    def notify_order(self, order):
        """订单状态通知处理"""
        current_time = self.execution.datetime.datetime()
        
        if order.status in [order.Completed]:
            self.logger.info(f"订单成交 - 时间: {current_time}, "
                            f"类型: {order.ordtypename()}, "
                            f"数量: {order.size:.4f}, "
                            f"价格: {order.executed.price:.4f}")
            
            # 移除已完成订单
            if order in self.active_orders:
                self.active_orders.remove(order)
            
            # 如果是网格单成交，挂出止盈单（下一个网格价格）
            if order.ref not in self.trade_stats['positions']:
                self.trade_stats['total_trades'] += 1
                grid_price = order.created.price
                grid_index = np.searchsorted(self.grid_lines, grid_price)
                
                if order.isbuy():
                    # 多单止盈 - 使用上一个网格价格
                    tp_price = self.grid_lines[grid_index + 1]
                    tp_order = self.sell(size=abs(order.size), exectype=bt.Order.Limit, price=tp_price)
                    self.active_orders.append(tp_order)
                    self.trade_stats['positions'][tp_order.ref] = {
                        'entry_time': current_time,
                        'entry_price': order.executed.price,
                        'size': order.size,
                        'side': 'long',
                        'grid_price': grid_price
                    }
                    self.logger.info(f"多单止盈挂单 - 时间: {current_time}, 价格: {tp_price:.4f}")
                    
                    # 在当前价格重新挂多单（如果该位置没有订单）
                    if grid_price not in self.grid_orders:
                        new_order = self.buy(size=self.calculate_position_size(), 
                                           exectype=bt.Order.Limit, 
                                           price=grid_price)
                        self.active_orders.append(new_order)
                        self.grid_orders[grid_price] = new_order
                else:
                    # 空单止盈 - 使用下一个网格价格
                    tp_price = self.grid_lines[grid_index - 1]
                    tp_order = self.buy(size=abs(order.size), exectype=bt.Order.Limit, price=tp_price)
                    self.active_orders.append(tp_order)
                    self.trade_stats['positions'][tp_order.ref] = {
                        'entry_time': current_time,
                        'entry_price': order.executed.price,
                        'size': order.size,
                        'side': 'short',
                        'grid_price': grid_price
                    }
                    self.logger.info(f"空单止盈挂单 - 时间: {current_time}, 价格: {tp_price:.4f}")
                    
                    # 在当前价格重新挂空单（如果该位置没有订单）
                    if grid_price not in self.grid_orders:
                        new_order = self.sell(size=self.calculate_position_size(), 
                                            exectype=bt.Order.Limit, 
                                            price=grid_price)
                        self.active_orders.append(new_order)
                        self.grid_orders[grid_price] = new_order
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.logger.warning(f"订单异常 - 时间: {current_time}, "
                              f"状态: {order.status}, "
                              f"类型: {order.ordtypename()}, "
                              f"数量: {order.size:.4f}, "
                              f"价格: {order.created.price:.4f}")
            if order in self.active_orders:
                self.active_orders.remove(order)

    def initialize_grid(self):
        """初始化网格系统"""
        self.cancel_all_orders()
        
        # 获取当前价格作为基准价
        self.base_price = self.execution.close[0]
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
        
        self.logger.info(f"初始化网格:")
        self.logger.info(f"基准价格: {self.base_price:.4f}")
        self.logger.info(f"价格区间: {self.lower_bound:.4f} - {self.upper_bound:.4f}")
        self.logger.info(f"网格数量: {len(self.grid_lines)}")
        self.logger.info(f"网格间距: {(self.upper_bound - self.lower_bound) / self.p.grid_number:.4f}")
        
        # 记录每个网格价格的订单状态
        self.grid_orders = {}  # 用于跟踪每个价格位置的订单
        
        # 在每个网格线位置挂单
        for i, price in enumerate(self.grid_lines[:-1]):  # 除了最后一个价格
            if price > self.base_price:
                # 价格线在当前价格上方，挂空单
                sell_size = self.calculate_position_size()
                order = self.sell(size=sell_size, exectype=bt.Order.Limit, price=price)
                self.active_orders.append(order)
                self.grid_orders[price] = order  # 记录该价格位置的订单
                self.logger.info(f"挂空单: 价格={price:.4f}, 止盈={self.grid_lines[i+1]:.4f}, 数量={sell_size:.4f}")
            elif price < self.base_price:
                # 价格线在当前价格下方，挂多单
                buy_size = self.calculate_position_size()
                order = self.buy(size=buy_size, exectype=bt.Order.Limit, price=price)
                self.active_orders.append(order)
                self.grid_orders[price] = order  # 记录该价格位置的订单
                self.logger.info(f"挂多单: 价格={price:.4f}, 止盈={self.grid_lines[i+1]:.4f}, 数量={buy_size:.4f}")

    def cancel_all_orders(self):
        """撤销所有未完成订单"""
        for order in self.active_orders:
            self.cancel(order)
        self.active_orders.clear()

    def rebalance_grid(self):
        """重新平衡网格系统"""
        current_price = self.execution.close[0]
        
        # 当价格突破边界时重新初始化网格
        if current_price > self.upper_bound or current_price < self.lower_bound:
            self.logger.warning("价格突破区间，重新调整网格")
            self.initialize_grid()

    def calculate_position_size(self):
        """计算每格仓位大小"""
        target_value = 200  # 每格固定价值200 USDT (已经考虑了杠杆)
        current_price = self.execution.close[0]
        return target_value / current_price  # 直接返回数量，因为target_value已经考虑了杠杆

    def next(self):
        """策略主逻辑 - 每分钟执行一次"""
        current_time = self.execution.datetime.datetime()
        current_price = self.execution.close[0]
        
        # 添加详细日志
        self.logger.debug(f"当前时间: {current_time}, 价格: {current_price:.4f}")
        
        # 首次运行初始化网格
        if self.upper_bound is None:
            self.base_price = self.execution.close[0]
            self.initialize_grid()
            return

    def get_balance_status(self):
        """获取账户风险数据"""
        equity = self.broker.getvalue()
        # 由于回测环境不支持保证金查询，仅返回净值信息
        return f"净值:{equity:.2f}"

    def stop(self):
        """策略结束时输出统计信息"""
        self.logger.info("\n====== 策略统计信息 ======")
        self.logger.info(f"总交易次数: {self.trade_stats['total_trades']}")
        self.logger.info(f"止盈次数: {self.trade_stats['tp_trades']}")
        self.logger.info(f"总盈利: {self.trade_stats['total_profit']:.4f}")
        
        if self.trade_stats['tp_trades'] > 0:
            avg_hold_time = self.trade_stats['total_hold_time'] / self.trade_stats['tp_trades']
            avg_profit = self.trade_stats['total_profit'] / self.trade_stats['tp_trades']
            self.logger.info(f"平均持仓时间: {avg_hold_time}")
            self.logger.info(f"平均每单盈利: {avg_profit:.4f}")
        
        self.logger.info("\n====== 交易明细 ======")
        for i, trade in enumerate(self.trade_stats['closed_trades'], 1):
            self.logger.info(
                f"交易 {i:04d}: "
                f"方向={trade['side']} "
                f"开仓时间={trade['entry_time']} "
                f"平仓时间={trade['exit_time']} "
                f"持仓时间={trade['hold_time']} "
                f"开仓价={trade['entry_price']:.4f} "
                f"平仓价={trade['exit_price']:.4f} "
                f"数量={trade['size']:.4f} "
                f"盈利={trade['profit']:.4f}"
            )
        
        self.logger.info(f"\n策略运行结束 净值:{self.broker.getvalue():.2f}")