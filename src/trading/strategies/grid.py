import backtrader as bt
import numpy as np
from typing import List, Dict
from .base import BaseStrategy
import logging

class AutoGridStrategy(BaseStrategy):
    """自适应网格交易策略
    
    这是一个基于ATR动态调整网格间距的网格交易策略。策略的主要特点：
    1. 使用ATR指标动态计算市场波动率
    2. 根据波动率自动调整网格间距（0.5%-5%）
    3. 在价格偏离或波动率变化时自动重置网格
    4. 维护网格状态，记录每个网格点的持仓情况
    """
    
    params = (
        ('grid_number', 20),         # 网格数量(上下各5个)
        ('position_size', 0.1),      # 每格仓位（占总资金的比例）
        ('atr_period', 14),          # ATR周期
        ('grid_min_spread', 0.002),  # 最小网格间距 0.2%
        ('grid_max_spread', 0.06),   # 最大网格间距 5%
        ('volatility_factor',2.0),  # 波动率系数
    )

    def __init__(self):
        """初始化策略变量和技术指标"""
        super().__init__()
        # 核心数据结构
        self.grids = {}              # 存储网格信息的字典，格式：{price: {'has_position': False}}
        self.initial_price = None    # 网格初始化时的价格
        self.last_price = None       # 上一个时间点的价格
        self.current_spacing = None  # 当前使用的网格间距
        
        # 技术指标
        self.atr = bt.indicators.ATR(
            self.data,               # 数据源
            period=self.p.atr_period # ATR周期
        )
        
        # 日志记录器
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_grid_spacing(self):
        """计算动态网格间距
        
        基于ATR指标计算市场波动率，并根据波动率调整网格间距。
        间距范围被限制在最小0.5%到最大5%之间。
        
        Returns:
            float: 计算得到的网格间距（以百分比表示）
        """
        try:
            # 计算当前的波动率（ATR相对于价格的比率）
            volatility = self.atr[0] / self.data.close[0]
            
            # 使用波动率和调整系数计算网格间距，并限制在指定范围内
            grid_spacing = np.clip(
                volatility * self.p.volatility_factor,  # 基础间距
                self.p.grid_min_spread,                 # 最小间距
                self.p.grid_max_spread                  # 最大间距
            )
            
            self.current_spacing = grid_spacing
            
            # 记录调整信息
            self.logger.info(f"网格间距调整 - ATR: {self.atr[0]:.4f}, "
                           f"波动率: {volatility:.4f}, "
                           f"网格间距: {grid_spacing:.4f}")
            
            return grid_spacing
            
        except Exception as e:
            self.logger.error(f"网格间距计算错误: {str(e)}")
            return self.p.grid_min_spread  # 错误时使用最小间距

    def initialize_grids(self):
        """初始化网格系统
        
        根据当前价格和计算得到的网格间距，生成一系列网格价格点。
        每个网格点都会记录其持仓状态。
        """
        try:
            # 清空现有网格
            self.grids = {}
            
            # 获取当前的网格间距
            grid_spacing = self.calculate_grid_spacing()
            
            # 生成网格价格点
            for i in range(-self.p.grid_number//2, self.p.grid_number//2 + 1):
                # 计算网格价格
                grid_price = self.initial_price * (1 + i * grid_spacing)
                grid_price = round(grid_price, 4)  # 保留4位小数
                
                # 初始化网格状态
                self.grids[grid_price] = {
                    'has_position': False  # 初始状态无持仓
                }
            
            # 在初始价格买入
            self.buy_at_grid(self.initial_price)
            
            # 记录初始化信息
            self.logger.info(f"初始化网格 - 中心价格: {self.initial_price:.4f}, "
                           f"网格数量: {len(self.grids)}, "
                           f"间距: {grid_spacing:.4f}")
            
        except Exception as e:
            self.logger.error(f"网格初始化错误: {str(e)}")

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

    def should_adjust_grids(self, current_price):
        """判断是否需要调整网格
        
        在以下情况下需要调整网格：
        1. 价格偏离初始价格超过两倍网格间距
        2. 波动率变化导致网格间距变化超过20%
        
        Args:
            current_price (float): 当前价格
            
        Returns:
            bool: 是否需要调整网格
        """
        try:
            # 首次运行时需要调整
            if self.current_spacing is None:
                return True
                
            # 计算价格偏离度
            price_deviation = abs(current_price - self.initial_price) / self.initial_price
            
            # 计算网格间距的变化
            new_spacing = self.calculate_grid_spacing()
            spacing_change = abs(new_spacing - self.current_spacing) / self.current_spacing
            
            # 判断是否需要调整
            return (price_deviation > 2 * self.current_spacing or  # 价格偏离过大
                    spacing_change > 0.2)                          # 间距变化显著
            
        except Exception as e:
            self.logger.error(f"网格调整检查错误: {str(e)}")
            return False

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