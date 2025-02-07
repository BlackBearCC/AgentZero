import backtrader as bt
import numpy as np
from typing import List, Dict
from .base import BaseStrategy
from src.utils.logger import Logger

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
        # 使用策略专用logger
        self.logger = Logger("strategy")
        
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
        
        # 添加时间检查
        self.start_time = None
        self.end_time = None

        # 验证策略参数
        if not 0 < self.p.position_size <= 1:
            raise ValueError(f"position_size 必须在 0-1 之间，当前值: {self.p.position_size}")
        
        if self.p.grid_number < 2:
            raise ValueError(f"grid_number 必须大于 1，当前值: {self.p.grid_number}")
        
        if self.p.grid_min_spread <= 0:
            raise ValueError(f"grid_min_spread 必须大于 0，当前值: {self.p.grid_min_spread}")

        self.initial_cash = self.broker.startingcash
        self.min_cash_required = self.initial_cash * self.p.position_size * (1 + 0.001)
        
        # 添加大小网格的状态
        self.outer_grids = {}  # 外层大网格
        self.inner_grids = {}  # 内层小网格
        self.is_ranging = False  # 是否处于横盘状态
        self.range_start_time = None  # 横盘开始时间
        self.last_trend_change = None  # 上次趋势改变时间
        
        self.logger.info(
            f"策略初始化 - "
            f"初始资金: ${self.initial_cash:.2f}, "
            f"每格资金: ${self.min_cash_required:.2f}"
        )

    def start(self):
        """策略启动时调用"""
        self.start_time = self.data.datetime.datetime(0)
        self.end_time = self.data.datetime.datetime(-1)
        self.logger.info(f"策略启动 - 回测区间: {self.start_time} 到 {self.end_time}")

    def adaptive_grid_adjustment(self):
        """自适应网格调整
        
        结合ATR和波动率动态计算网格间距和区间范围。
        
        Returns:
            tuple: (grid_spacing, upper_price, lower_price)
        """
        try:
            current_time = self.data.datetime.datetime(0)
            current_price = self.data.close[0]
            
            # 计算波动率比率
            vol_ratio = self.volatility[0] / current_price
            # 计算ATR比率
            atr_ratio = self.atr[0] / current_price
            
            # 动态计算网格间距
            grid_spacing = np.clip(
                vol_ratio * 1.5 + atr_ratio * 1.5,
                self.p.grid_min_spread,
                self.p.grid_max_spread
            )
            
            # 计算网格区间范围
            expansion = self.p.grid_expansion
            upper_price = current_price * (1 + expansion * grid_spacing)
            lower_price = current_price * (1 - expansion * grid_spacing)
            
            self.logger.info(f"网格调整 - 时间: {current_time}, "
                           f"波动率: {vol_ratio:.4f}, "
                           f"ATR比率: {atr_ratio:.4f}, "
                           f"间距: {grid_spacing:.4f}, "
                           f"区间: [{lower_price:.4f}, {upper_price:.4f}]")
            
            return grid_spacing, upper_price, lower_price
            
        except Exception as e:
            self.logger.error(f"网格调整错误: {str(e)}")
            return self.p.grid_min_spread, current_price * 1.05, current_price * 0.95

    def initialize_grids(self):
        """初始化或调整网格"""
        try:
            spacing, upper_limit, lower_limit = self.adaptive_grid_adjustment()
            current_price = self.data.close[0]
            
            if spacing <= 0:
                self.logger.error(f"间距无效: {spacing}")
                return
            
            # 如果是首次初始化
            if not self.grids:
                self.initial_price = current_price
                grid_number = 10 if self.is_ranging else self.p.grid_number
                price_range = 0.02 if self.is_ranging else 0.1
                
                # 计算初始网格
                price_step = self.initial_price * price_range / grid_number
                lower_price = self.initial_price * (1 - price_range/2)
                upper_price = self.initial_price * (1 + price_range/2)
                
                for i in range(grid_number):
                    grid_price = lower_price + i * price_step
                    grid_price = round(grid_price, 4)
                    
                    if grid_price <= 0:
                        continue
                        
                    self.grids[grid_price] = {
                        'has_position': False,
                        'is_outer': not self.is_ranging
                    }
            else:
                # 调整现有网格：保持现有网格，根据需要向上或向下延伸
                existing_prices = sorted(self.grids.keys())
                current_min_price = min(existing_prices)
                current_max_price = max(existing_prices)
                grid_step = spacing * current_price  # 使用新的间距
                
                # 向下延伸网格
                if current_price < current_min_price * 1.02:  # 接近或低于最低网格
                    new_price = current_min_price
                    while new_price > lower_limit:
                        new_price -= grid_step
                        new_price = round(new_price, 4)
                        if new_price not in self.grids:
                            self.grids[new_price] = {
                                'has_position': False,
                                'is_outer': not self.is_ranging
                            }
                            self.logger.info(f"向下添加新网格: {new_price:.4f}")
                
                # 向上延伸网格
                if current_price > current_max_price * 0.98:  # 接近或高于最高网格
                    new_price = current_max_price
                    while new_price < upper_limit:
                        new_price += grid_step
                        new_price = round(new_price, 4)
                        if new_price not in self.grids:
                            self.grids[new_price] = {
                                'has_position': False,
                                'is_outer': not self.is_ranging
                            }
                            self.logger.info(f"向上添加新网格: {new_price:.4f}")
                
                # 清理过远的网格（可选）
                prices_to_remove = []
                for price in self.grids:
                    if price < lower_limit * 0.5 or price > upper_limit * 1.5:
                        if not self.grids[price]['has_position']:  # 只清理没有持仓的网格
                            prices_to_remove.append(price)
                
                for price in prices_to_remove:
                    del self.grids[price]
                    self.logger.info(f"删除远端网格: {price:.4f}")

            self.current_spacing = spacing
            self.logger.info(
                f"网格调整完成 - "
                f"状态: {'横盘' if self.is_ranging else '趋势'}, "
                f"当前价格: {current_price:.4f}, "
                f"网格数量: {len(self.grids)}, "
                f"间距: {spacing:.4f}"
            )
            
        except Exception as e:
            self.logger.error(f"网格初始化错误: {str(e)}")

    def should_adjust_grids(self, current_price):
        """判断是否需要调整网格"""
        try:
            if self.current_spacing is None:
                return True
            
            # 获取新的网格参数
            new_spacing, _, _ = self.adaptive_grid_adjustment()
            
            # 检查是否进入横盘状态
            current_time = self.data.datetime.datetime(0)
            price_deviation = abs(current_price - self.initial_price) / self.initial_price
            
            if not self.is_ranging:
                # 判断是否进入横盘
                if price_deviation < 0.06:  # 价格波动小于2%
                    if self.range_start_time is None:
                        self.range_start_time = current_time
                    elif (current_time - self.range_start_time).total_seconds() > 3600:  # 1小时
                        self.is_ranging = True
                        self.logger.info(f"进入横盘状态 - 中心价格: {current_price:.4f}")
                        return True
                else:
                    self.range_start_time = None
            else:
                # 判断是否退出横盘
                if price_deviation > 0.06:  # 价格波动超过5%
                    self.is_ranging = False
                    self.logger.info(f"退出横盘状态 - 当前价格: {current_price:.4f}")
                    return True
            
            # 计算间距变化
            spacing_change = abs(new_spacing - self.current_spacing) / self.current_spacing
            
            # 根据状态使用不同的调整阈值
            if self.is_ranging:
                return spacing_change > 0.5  # 横盘时更敏感
            else:
                return (price_deviation > 0.05 or  # 价格偏离超过5%
                        spacing_change > 0.3)      # 间距变化超过30%
            
        except Exception as e:
            self.logger.error(f"网格调整检查错误: {str(e)}")
            return False

    def next(self):
        """策略主逻辑"""
        try:
            current_time = self.data.datetime.datetime(0)
            current_price = self.data.close[0]
            
            self.logger.debug(f"当前时间: {current_time}, 价格: {current_price:.4f}")
            
            # 首次运行时初始化网格
            if self.initial_price is None:
                self.initial_price = current_price
                self.initialize_grids()
                self.last_price = current_price
                return
            
            # 检查是否需要重新调整网格
            if self.should_adjust_grids(current_price):
                # self.logger.info("重新调整网格...")
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
            
            # 记录当前状态
            self.logger.debug(
                f"检查网格触发 - "
                f"当前价格: {current_price:.4f}, "
                f"上次价格: {self.last_price:.4f}, "
                f"现金: {self.broker.getcash():.2f}"
            )
            
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
            current_time = self.data.datetime.datetime(0)
            current_cash = self.broker.getcash()
            
            # 资金预检
            if current_cash < self.min_cash_required:
                self.logger.warning(
                    f"资金预检失败 - "
                    f"当前现金: ${current_cash:.2f}, "
                    f"最小要求: ${self.min_cash_required:.2f}, "
                    f"总资产: ${self.broker.getvalue():.2f}"
                )
                return
            
            # 使用初始资金计算仓位
            initial_cash = self.broker.startingcash
            position_value = initial_cash * self.p.position_size
            position_size = position_value / price
            
            # 详细记录买入前的状态
            self.logger.info(f"尝试网格买入 - 时间: {current_time}")
            self.logger.info(f"当前现金: ${current_cash:.2f}")
            self.logger.info(f"初始资金: ${initial_cash:.2f}")
            self.logger.info(f"目标价格: ${price:.4f}")
            
            # 检查是否有足够的资金
            commission_rate = 0.001
            required_cash = position_value * (1 + commission_rate)
            if required_cash > current_cash:
                self.logger.warning(
                    f"资金不足 - 需要: ${required_cash:.2f}, "
                    f"现有: ${current_cash:.2f}, "
                    f"目标仓位: ${position_value:.2f}"
                )
                return
            
            # 检查数量是否合理
            if position_size <= 0:
                self.logger.warning(f"计算的仓位大小无效: {position_size}")
                return
            
            # 创建买入订单
            self.buy(size=position_size, price=price, exectype=bt.Order.Limit)
            self.grids[price]['has_position'] = True
            
            self.logger.info(
                f"网格买入订单创建成功 - "
                f"价格: ${price:.4f}, "
                f"数量: {position_size:.4f}, "
                f"金额: ${position_value:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"网格买入错误: {str(e)}")

    def sell_at_grid(self, price):
        """在网格价格卖出"""
        try:
            current_time = self.data.datetime.datetime(0)
            initial_cash = self.broker.startingcash
            
            # 计算卖出数量
            position_value = initial_cash * self.p.position_size
            position_size = position_value / price
            
            # 创建卖出订单
            self.sell(size=position_size, price=price, exectype=bt.Order.Limit)
            self.grids[price]['has_position'] = False
            
            self.logger.info(
                f"网格卖出 - 时间: {current_time}, "
                f"价格: ${price:.4f}, "
                f"数量: {position_size:.4f}, "
                f"金额: ${position_value:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"网格卖出错误: {str(e)}")

    def notify_order(self, order):
        """订单状态更新通知"""
        if order.status in [bt.Order.Completed]:
            # 获取订单完成时间
            current_time = self.data.datetime.datetime(0)
            
            self.logger.info(
                f"订单完成 - 时间: {current_time}, "
                f"方向: {'买入' if order.isbuy() else '卖出'}, "
                f"价格: {order.executed.price:.4f}, "
                f"数量: {order.executed.size:.4f}, "
                f"价值: {order.executed.value:.2f}, "
                f"手续费: {order.executed.comm:.2f}, "
                f"剩余现金: {self.broker.getcash():.2f}"
            )

    def notify_trade(self, trade):
        """交易通知"""
        if trade.isclosed:
            # 获取交易结束时间
            current_time = self.data.datetime.datetime(0)
            
            self.logger.info(
                f"交易结束 - 时间: {current_time}, "
                f"毛利润: {trade.pnl:.2f}, "
                f"净利润: {trade.pnlcomm:.2f}, "
                f"剩余现金: {self.broker.getcash():.2f}, "
                f"总资产: {self.broker.getvalue():.2f}"
            )