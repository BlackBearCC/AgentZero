import backtrader as bt
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any

class BaseStrategy(bt.Strategy):
    """交易策略基类"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.trade_count = 0
        self._init_indicators()
        
    def _init_indicators(self):
        """初始化技术指标"""
        pass
        
    def next(self):
        """策略主逻辑"""
        pass
        
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Completed]:
            self.trade_count += 1
            # self.logger.info(
            #     f'{order.ordtypename()} 订单执行 @ {order.executed.price:.2f}'
            # )
            
    def get_analysis(self) -> Dict[str, Any]:
        """获取策略分析结果"""
        return {
            'trade_count': self.trade_count,
            'current_value': self.broker.getvalue(),
            'position': self.getposition().size
        } 