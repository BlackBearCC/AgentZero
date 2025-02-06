import backtrader as bt
import numpy as np
from typing import List, Dict
from .base import BaseStrategy
import logging

class AutoGridStrategy(BaseStrategy):
    """自适应网格交易策略
    
    这是一个基于ATR和波动率动态调整的网格交易策略。策略的主要特点：
    1. 结合ATR和波动率动态计算网格间距
    2. 根据市场状况自动调整网格区间范围
    3. 在价格偏离或波动率变化时自动重置网格
    4. 维护网格状态，记录每个网格点的持仓情况
    """
    
    params = (
        ('grid_number', 20),         # 网格数量
        ('position_size', 0.1),      # 每格仓位（占总资金的比例）
        ('atr_period', 14),          # ATR周期
        ('vol_period', 20),          # 波动率计算周期
        ('grid_min_spread', 0.005),  # 最小网格间距 0.5%
        ('grid_max_spread', 0.05),   # 最大网格间距 5%
        ('grid_expansion', 2.0),     # 网格区间扩展系数
    )

    def __init__(self):
        """初始化策略变量和技术指标"""
        super().__init__()
        # 核心数据结构
        self.grids = {}              # 存储网格信息
        self.initial_price = None    # 初始价格
        self.last_price = None       # 上一次价格
        self.current_spacing = None  # 当前网格间距
        
        # 技术指标
        self.atr = bt.indicators.ATR(
            self.data,
            period=self.p.atr_period
        )
        # 计算波动率
        self.volatility = bt.indicators.StdDev(
            self.data.close,
            period=self.p.vol_period
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)

    def adaptive_grid_adjustment(self):
        """自适应网格调整
        
        结合ATR和波动率动态计算网格间距和区间范围。
        
        Returns:
            tuple: (grid_spacing, upper_price, lower_price)
        """
        try:
            current_price = self.data.close[0]
            
            # 计算波动率比率
            vol_ratio = self.volatility[0] / current_price
            # 计算ATR比率
            atr_ratio = self.atr[0] / current_price
            
            # 动态计算网格间距
            grid_spacing = np.clip(
                vol_ratio * 1.5 + atr_ratio * 1.5,  # 综合波动指标
                self.p.grid_min_spread,           # 最小间距
                self.p.grid_max_spread            # 最大间距
            )
            
            # 计算网格区间范围
            expansion = self.p.grid_expansion
            upper_price = current_price * (1 + expansion * grid_spacing)
            lower_price = current_price * (1 - expansion * grid_spacing)
            
            self.logger.info(f"网格调整 - 波动率: {vol_ratio:.4f}, "
                           f"ATR比率: {atr_ratio:.4f}, "
                           f"间距: {grid_spacing:.4f}, "
                           f"区间: [{lower_price:.4f}, {upper_price:.4f}]")
            
            return grid_spacing, upper_price, lower_price
            
        except Exception as e:
            self.logger.error(f"网格调整错误: {str(e)}")
            return self.p.grid_min_spread, current_price * 1.05, current_price * 0.95

    def initialize_grids(self):
        """初始化网格系统"""
        try:
            # 清空现有网格
            self.grids = {}
            
            # 获取动态网格参数
            grid_spacing, upper_price, lower_price = self.adaptive_grid_adjustment()
            self.current_spacing = grid_spacing
            
            # 计算每个网格的价格
            price_range = upper_price - lower_price
            price_step = price_range / (self.p.grid_number - 1)
            
            # 生成网格价格点
            for i in range(self.p.grid_number):
                grid_price = lower_price + i * price_step
                grid_price = round(grid_price, 4)
                
                self.grids[grid_price] = {
                    'has_position': False
                }
            
            # 在初始价格买入
            self.buy_at_grid(self.initial_price)
            
            self.logger.info(f"初始化网格 - 中心价格: {self.initial_price:.4f}, "
                           f"网格数量: {len(self.grids)}, "
                           f"间距: {grid_spacing:.4f}")
            
        except Exception as e:
            self.logger.error(f"网格初始化错误: {str(e)}")

    def should_adjust_grids(self, current_price):
        """判断是否需要调整网格"""
        try:
            if self.current_spacing is None:
                return True
                
            # 获取新的网格参数
            new_spacing, _, _ = self.adaptive_grid_adjustment()
            
            # 计算价格偏离度
            price_deviation = abs(current_price - self.initial_price) / self.initial_price
            
            # 计算间距变化
            spacing_change = abs(new_spacing - self.current_spacing) / self.current_spacing
            
            # 判断是否需要调整
            return (price_deviation > 2 * self.current_spacing or  # 价格偏离过大
                    spacing_change > 0.2)                          # 间距变化显著
            
        except Exception as e:
            self.logger.error(f"网格调整检查错误: {str(e)}")
            return False

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
            
            # 检查是否需要重新调整网格
            if self.should_adjust_grids(current_price):
                self.logger.info("重新调整网格...")
                self.initial_price = current_price
                self.initialize_grids()
                return
            
            # 检查是否触发网格
            self.check_grid_triggers(current_price)
            self.last_price = current_price
            
        except Exception as e:
            self.logger.error(f"策略执行错误: {str(e)}")

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