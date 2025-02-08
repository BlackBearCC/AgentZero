from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.utils.logger import Logger  # 添加导入

@dataclass
class Order:
    id: str
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'LIMIT' or 'MARKET'
    price: float
    quantity: float
    status: str = 'PENDING'  # PENDING, FILLED, CANCELLED
    filled_price: Optional[float] = None
    filled_time: Optional[datetime] = None
    # filled_quantity 和 quantity 相同，因为我们的回测系统只支持全部成交

@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    entry_time: datetime

class BacktestEngine:
    def __init__(self,
                 data: pd.DataFrame,
                 initial_capital: float = 10000,
                 commission_rate: float = 0.001,
                 leverage: float = 1.0,
                 **kwargs):  # 添加 **kwargs 来接收额外的参数
        """
        回测引擎初始化
        
        Args:
            data: DataFrame with OHLCV data
            initial_capital: 初始资金
            commission_rate: 手续费率
            leverage: 杠杆倍数
            **kwargs: 其他参数
        """
        self.data = data
        self.initial_capital = initial_capital
        self.current_capital = initial_capital  # 当前总资产
        self.available_capital = initial_capital  # 可用资金
        self.commission_rate = commission_rate
        self.leverage = leverage
        
        # 交易相关
        self.positions = {}  # 当前持仓
        self.pending_orders = []  # 待成交订单
        self.filled_orders = []  # 已成交订单
        self.trades = []  # 已完成交易
        self.equity_curve = []  # 权益曲线
        
        # 日志
        self.logger = Logger("backtest_engine")
        
    def place_limit_order(self, symbol: str, side: str, price: float, quantity: float) -> Order:
        """下限价单"""
        order = Order(
            id=f"ORDER_{len(self.pending_orders)}",
            timestamp=self.current_time,
            symbol=symbol,
            side=side,
            order_type='LIMIT',
            price=price,
            quantity=quantity
        )
        self.pending_orders.append(order)
        return order
        
    def _match_orders(self, bar: pd.Series):
        """订单撮合"""
        for order in self.pending_orders[:]:
            if order.status != 'PENDING':
                continue
            
            # 添加日志
            self.logger.debug(f"检查订单撮合 - 价格: {order.price}, 当前价: {bar['high']}-{bar['low']}, 方向: {order.side}")
            
            # 限价买单
            if order.side == 'BUY' and bar['low'] <= order.price:
                order.status = 'FILLED'
                order.filled_price = order.price
                order.filled_time = bar['timestamp']
                self.logger.info(f"买单成交 - 价格: {order.price}, 数量: {order.quantity}")
                
                # 处理成交订单
                if self._process_filled_order(order):
                    self.filled_orders.append(order)
                    self.pending_orders.remove(order)
                
            # 限价卖单
            elif order.side == 'SELL' and bar['high'] >= order.price:
                order.status = 'FILLED'
                order.filled_price = order.price
                order.filled_time = bar['timestamp']
                self.logger.info(f"卖单成交 - 价格: {order.price}, 数量: {order.quantity}")
                
                # 处理成交订单
                if self._process_filled_order(order):
                    self.filled_orders.append(order)
                    self.pending_orders.remove(order)
    
    def _process_filled_order(self, order: Order):
        """处理已成交订单"""
        # 考虑杠杆的手续费计算
        commission = order.quantity * order.filled_price * self.commission_rate
        
        # 计算所需保证金
        required_margin = (order.quantity * order.filled_price / self.leverage)
        
        if order.side == 'BUY':
            # 检查可用保证金是否足够
            if required_margin > self.available_capital:
                self.logger.warning(f"保证金不足 - 所需保证金: {required_margin:.2f}, 可用资金: {self.available_capital:.2f}")
                return False
            
            # 扣除保证金和手续费
            self.available_capital -= (required_margin + commission)
            self.current_capital -= commission
            
        elif order.side == 'SELL':
            # 开空仓
            if order.symbol not in self.positions:
                if required_margin > self.available_capital:
                    self.logger.warning(f"保证金不足 - 所需保证金: {required_margin:.2f}, 可用资金: {self.available_capital:.2f}")
                    return False
                # 扣除保证金和手续费
                self.available_capital -= (required_margin + commission)
                self.current_capital -= commission
                
            # 平多仓
            else:
                pos = self.positions[order.symbol]
                if pos.side == 'LONG':
                    if order.quantity > pos.quantity:
                        self.logger.warning(f"持仓不足 - 需要: {order.quantity}, 实际: {pos.quantity}")
                        return False
                    
                    # 计算收益（考虑杠杆）
                    profit = (order.filled_price - pos.entry_price) * order.quantity - commission
                    released_margin = (order.quantity * pos.entry_price / self.leverage)
                    
                    # 释放保证金，加上收益
                    self.available_capital += released_margin + profit
                    self.current_capital += profit
        
        return True
    
    def run(self) -> Dict:
        """运行回测"""
        for _, bar in self.data.iterrows():
            self.current_time = bar['timestamp']
            
            # 订单撮合
            self._match_orders(bar)
            
            # 处理成交订单的回调
            for order in self.filled_orders[:]:
                self.on_order_filled(order)
                self.filled_orders.remove(order)
            
            # 记录权益
            total_equity = self.current_capital
            for pos in self.positions.values():
                total_equity += pos.quantity * bar['close']
            self.equity_curve.append({
                'timestamp': bar['timestamp'],
                'equity': total_equity
            })
            
            # 策略逻辑在这里调用
            self.on_bar(bar)
            
        return self.get_results()
    
    def get_results(self) -> Dict:
        """获取回测结果统计"""
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)
        
        results = {
            'initial_capital': self.initial_capital,
            'final_equity': equity_df['equity'].iloc[-1],
            'total_return': (equity_df['equity'].iloc[-1] / self.initial_capital - 1),
            'total_trades': len(trades_df),
            'win_rate': len(trades_df[trades_df['profit'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_return': trades_df['return'].mean() if len(trades_df) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(equity_df['equity']),
            'equity_curve': equity_df,
            'trades': trades_df
        }
        return results
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """计算最大回撤"""
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        return abs(drawdowns.min())
    
    def on_bar(self, bar: pd.Series):
        """
        策略逻辑实现接口
        子类需要重写此方法
        """
        pass 

    def on_order_filled(self, order: Order):
        """
        订单成交回调接口
        子类需要重写此方法
        """
        pass 

    def update_capital(self):
        """更新资金状态"""
        # 计算当前持仓的市值
        total_position_value = sum(
            pos.quantity * self.data['close'].iloc[-1]
            for pos in self.positions.values()
        )
        
        # 计算持仓占用的保证金
        margin_used = total_position_value / self.leverage
        
        # 更新可用资金
        self.available_capital = self.current_capital - margin_used
        
        # 记录权益曲线
        self.equity_curve.append({
            'timestamp': self.data['timestamp'].iloc[-1],
            'equity': self.current_capital + total_position_value,
            'available': self.available_capital
        })