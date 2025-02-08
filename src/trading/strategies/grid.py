import backtrader as bt
import numpy as np
from typing import List, Dict
from .base import BaseStrategy
from src.utils.logger import Logger

class AutoGridStrategy(BaseStrategy):
    """自适应网格交易策略
    
    实现类似币安的中性网格交易功能:
    1. 在价格区间内均匀设置网格
    2. 支持上移和下移网格
    3. 维护买卖单的状态
    """
    
    params = {
        # 基础参数
        'grid_number': 100,         # 网格数量
        'position_size': 0.01,      # 每格仓位比例
        'price_range': 0.2,        # 网格价格区间范围(上下各5%)
        
        # 移动参数
        'move_threshold': 0.08,    # 提高移动阈值到8%
    }
    
    def __init__(self):
        """初始化策略"""
        super().__init__()
        self.logger = Logger("strategy")  # 移除color参数
        
        # 核心数据结构
        self.grids = {}           # 存储网格信息
        self.initial_price = None # 初始价格
        self.upper_price = None   # 上边界价格
        self.lower_price = None   # 下边界价格
        
        # 验证参数
        if not 0 < self.p.position_size <= 1:
            raise ValueError(f"position_size 必须在 0-1 之间")
        if self.p.grid_number < 2:
            raise ValueError(f"grid_number 必须大于 1")
            
        # 计算所需资金
        self.initial_cash = self.broker.startingcash
        self.min_cash_required = self.initial_cash * self.p.position_size
    
    @property
    def current_time(self) -> str:
        """获取当前时间字符串"""
        return self.data.datetime.datetime().strftime('%Y-%m-%d %H:%M:%S')
    
    def initialize_grids(self):
        """初始化网格"""
        current_price = self.data.close[0]
        
        # 计算价格区间
        half_range = current_price * (self.p.price_range / 2)
        self.upper_price = current_price + half_range  
        self.lower_price = current_price - half_range
        
        # 计算网格间距
        grid_step = (self.upper_price - self.lower_price) / self.p.grid_number
        
        # 创建网格
        self.grids.clear()
        for i in range(self.p.grid_number + 1):
            price = round(self.lower_price + i * grid_step, 4)
            self.grids[price] = {
                'has_buy_order': False,
                'has_sell_order': False
            }
            
        self.logger.info(f"初始化网格 [{self.current_time}] - 价格区间: [{self.lower_price:.4f}, {self.upper_price:.4f}]", color='yellow')
    
    def should_move_grids(self, current_price):
        """判断是否需要移动网格"""
        # 提高移动门槛，只有价格真正接近边界时才移动
        if current_price >= self.upper_price * (1 - self.p.move_threshold/2):
            return 'up'
        elif current_price <= self.lower_price * (1 + self.p.move_threshold/2):
            return 'down'
        return None
    
    def move_grids(self, direction):
        """移动网格"""
        current_price = self.data.close[0]
        
        # 取消当前所有订单
        for price, grid in self.grids.items():
            grid['has_buy_order'] = False
            grid['has_sell_order'] = False
        
        # 重新计算价格区间，增加缓冲区
        half_range = current_price * (self.p.price_range / 2)
        if direction == 'up':
            self.upper_price = current_price + half_range * 1.1  # 增加10%缓冲
            self.lower_price = current_price - half_range * 0.9
        else:
            self.upper_price = current_price + half_range * 0.9
            self.lower_price = current_price - half_range * 1.1
            
        # 重新初始化网格
        self.initialize_grids()
        
        self.logger.info(f"移动网格 [{self.current_time}] - 方向: {direction}, 新区间: [{self.lower_price:.4f}, {self.upper_price:.4f}]", color='yellwo')
    
    def next(self):
        """策略主逻辑"""
        try:
            current_price = self.data.close[0]
            
            # 首次运行时初始化
            if self.initial_price is None:
                self.initial_price = current_price
                self.initialize_grids()
                return
                
            # 检查是否需要移动网格
            move_direction = self.should_move_grids(current_price)
            if move_direction:
                self.move_grids(move_direction)
                return
                
            # 检查是否触发网格
            for price in sorted(self.grids.keys()):
                grid = self.grids[price]
                
                # 价格下跌,触发买入
                if current_price <= price and not grid['has_buy_order']:
                    self.buy_at_grid(price)
                    grid['has_buy_order'] = True
                    
                # 价格上涨,触发卖出  
                elif current_price >= price and not grid['has_sell_order']:
                    self.sell_at_grid(price)
                    grid['has_sell_order'] = True
                    
        except Exception as e:
            self.logger.error(f"策略执行错误: {str(e)}")
            
    def buy_at_grid(self, price):
        """在网格价位买入"""
        try:
            position_value = self.initial_cash * self.p.position_size
            position_size = position_value / price
            
            self.buy(size=position_size, price=price, exectype=bt.Order.Limit)
            self.logger.info(f"网格买入 [{self.current_time}] - 价格: {price:.4f}, 数量: {position_size:.4f}",color='green')
            
        except Exception as e:
            self.logger.error(f"买入错误: {str(e)}")
            
    def sell_at_grid(self, price):
        """在网格价位卖出"""
        try:
            position_value = self.initial_cash * self.p.position_size
            position_size = position_value / price
            
            self.sell(size=position_size, price=price, exectype=bt.Order.Limit)
            self.logger.info(f"网格卖出 [{self.current_time}] - 价格: {price:.4f}, 数量: {position_size:.4f}",color='red')
            
        except Exception as e:
            self.logger.error(f"卖出错误: {str(e)}")
