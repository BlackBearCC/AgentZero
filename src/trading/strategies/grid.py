import backtrader as bt
import numpy as np
from typing import List, Dict
from .base import BaseStrategy
import logging

class AutoGridStrategy(BaseStrategy):
    """网格交易策略"""
    
    params = (
        ('grid_spacing', 0.02),      # 网格间距2%
        ('grid_number', 20),         # 网格数量
        ('position_size', 0.1),      # 每格仓位（占总资金的比例）
    )

    def __init__(self):
        super().__init__()
        self.grids = {}              # 网格状态字典 {price: {'has_position': False}}
        self.initial_price = None    # 初始价格
        self.last_price = None       # 上一次价格
        self.logger = logging.getLogger(self.__class__.__name__)

    def next(self):
        """策略主逻辑"""
        try:
            current_price = self.data.close[0]
            
            # 首次运行时初始化网格
            if self.initial_price is None:
                self.initial_price = current_price
                self.initialize_grids()
                self.last_price = current_price
                return
            
            # 检查是否触发网格
            self.check_grid_triggers(current_price)
            self.last_price = current_price
            
        except Exception as e:
            self.logger.error(f"策略执行错误: {str(e)}")

    def initialize_grids(self):
        """初始化网格"""
        try:
            # 清空旧网格
            self.grids = {}
            
            # 生成网格价格
            for i in range(-self.p.grid_number//2, self.p.grid_number//2 + 1):
                grid_price = self.initial_price * (1 + i * self.p.grid_spacing)
                grid_price = round(grid_price, 4)  # 四舍五入到4位小数
                
                self.grids[grid_price] = {
                    'has_position': False
                }
            
            # 在初始价格买入
            self.buy_at_grid(self.initial_price)
            
            self.logger.info(f"初始化网格 - 中心价格: {self.initial_price:.4f}, "
                           f"网格数量: {len(self.grids)}, "
                           f"间距: {self.p.grid_spacing:.4f}")
            
        except Exception as e:
            self.logger.error(f"网格初始化错误: {str(e)}")

    def check_grid_triggers(self, current_price):
        """检查是否触发网格"""
        try:
            # 获取排序后的网格价格
            grid_prices = sorted(self.grids.keys())
            
            # 找到当前价格所在的网格区间
            for i in range(len(grid_prices) - 1):
                lower_price = grid_prices[i]
                upper_price = grid_prices[i + 1]
                
                # 检查是否穿越网格
                if self.last_price <= lower_price < current_price:
                    # 上涨穿越网格，卖出
                    if self.grids[lower_price]['has_position']:
                        self.sell_at_grid(lower_price)
                elif current_price < lower_price <= self.last_price:
                    # 下跌穿越网格，买入
                    if not self.grids[lower_price]['has_position']:
                        self.buy_at_grid(lower_price)
                        
        except Exception as e:
            self.logger.error(f"网格触发检查错误: {str(e)}")

    def buy_at_grid(self, price):
        """在网格价格买入"""
        try:
            # 计算买入数量
            position_value = self.broker.getvalue() * self.p.position_size
            position_size = position_value / price
            
            # 创建买入订单
            self.buy(size=position_size, price=price, exectype=bt.Order.Limit)
            self.grids[price]['has_position'] = True
            
            self.logger.info(f"网格买入 - 价格: {price:.4f}, 数量: {position_size:.4f}")
            
        except Exception as e:
            self.logger.error(f"网格买入错误: {str(e)}")

    def sell_at_grid(self, price):
        """在网格价格卖出"""
        try:
            # 计算卖出数量
            position_value = self.broker.getvalue() * self.p.position_size
            position_size = position_value / price
            
            # 创建卖出订单
            self.sell(size=position_size, price=price, exectype=bt.Order.Limit)
            self.grids[price]['has_position'] = False
            
            self.logger.info(f"网格卖出 - 价格: {price:.4f}, 数量: {position_size:.4f}")
            
        except Exception as e:
            self.logger.error(f"网格卖出错误: {str(e)}")

    def notify_order(self, order):
        """订单状态更新通知"""
        if order.status in [bt.Order.Completed]:
            self.logger.info(
                f"订单完成 - 方向: {'买入' if order.isbuy() else '卖出'}, "
                f"价格: {order.executed.price:.4f}, "
                f"数量: {order.executed.size:.4f}, "
                f"价值: {order.executed.value:.2f}, "
                f"手续费: {order.executed.comm:.2f}"
            )

    def notify_trade(self, trade):
        """交易通知"""
        if trade.isclosed:
            self.logger.info(
                f"交易结束 - 毛利润: {trade.pnl:.2f}, "
                f"净利润: {trade.pnlcomm:.2f}"
            )