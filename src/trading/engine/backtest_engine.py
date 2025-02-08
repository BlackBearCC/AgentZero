from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.utils.logger import Logger  # 确保Logger类已正确实现

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

@dataclass
class Position:
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    quantity: float
    entry_time: datetime

class BacktestEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000,
        commission_rate: float = 0.001,
        leverage: float = 1.0,
        slippage_rate: float = 0.0,  # 新增滑点参数
        **kwargs
    ):
        self.data = data.sort_values('timestamp').reset_index(drop=True)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        self.commission_rate = commission_rate
        self.leverage = leverage
        self.slippage_rate = slippage_rate  # 滑点率（0表示无滑点）

        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        self.logger = Logger("backtest_engine")

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        price: Optional[float],
        quantity: float
    ) -> Optional[Order]:
        """下单（支持市价单和限价单）"""
        if order_type == 'MARKET':
            # 市价单价格为None，撮合时处理
            order_price = None
        elif order_type == 'LIMIT' and price is not None:
            order_price = price
        else:
            self.logger.error("无效的订单类型或价格")
            return None

        order = Order(
            id=f"ORDER_{len(self.pending_orders)}",
            timestamp=self.current_time,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=order_price,
            quantity=quantity
        )
        self.pending_orders.append(order)
        return order

    def _match_orders(self, bar: pd.Series):
        """订单撮合（支持市价单和限价单）"""
        high, low = bar['high'], bar['low']
        # 应用滑点：买单价上浮，卖单价下浮
        slippage = (high - low) * self.slippage_rate
        buy_price = high + slippage
        sell_price = low - slippage

        for order in self.pending_orders[:]:
            if order.status != 'PENDING':
                continue

            # 市价单处理
            if order.order_type == 'MARKET':
                if order.side == 'BUY':
                    fill_price = buy_price
                else:
                    fill_price = sell_price
                order.filled_price = fill_price
                order.status = 'FILLED'
                order.filled_time = bar['timestamp']
                self._process_filled_order(order)
                self.filled_orders.append(order)
                self.pending_orders.remove(order)
                continue

            # 限价单处理
            if order.side == 'BUY' and low <= order.price:
                order.filled_price = order.price
                order.status = 'FILLED'
                order.filled_time = bar['timestamp']
                if self._process_filled_order(order):
                    self.filled_orders.append(order)
                    self.pending_orders.remove(order)
            elif order.side == 'SELL' and high >= order.price:
                order.filled_price = order.price
                order.status = 'FILLED'
                order.filled_time = bar['timestamp']
                if self._process_filled_order(order):
                    self.filled_orders.append(order)
                    self.pending_orders.remove(order)

    def _process_filled_order(self, order: Order) -> bool:
        """处理已成交订单"""
        symbol = order.symbol
        filled_price = order.filled_price
        quantity = order.quantity
        commission = filled_price * quantity * self.commission_rate
        required_margin = (filled_price * quantity) / self.leverage

        # 扣除手续费（无论开平仓）
        self.current_capital -= commission
        self.available_capital -= commission

        if order.side == 'BUY':
            # 检查是否为平空单
            if symbol in self.positions and self.positions[symbol].side == 'SHORT':
                pos = self.positions[symbol]
                if quantity > pos.quantity:
                    self.logger.warning(f"平空单数量不足：需求 {quantity}，实际 {pos.quantity}")
                    return False

                # 计算利润和释放保证金
                profit = (pos.entry_price - filled_price) * quantity
                released_margin = (pos.entry_price * quantity) / self.leverage
                self.available_capital += released_margin + profit
                self.current_capital += profit  # 总资金更新

                # 记录交易
                self._record_trade(pos, order, profit)

                # 更新持仓
                if quantity == pos.quantity:
                    del self.positions[symbol]
                else:
                    pos.quantity -= quantity
            else:
                # 开多单
                if required_margin > self.available_capital:
                    self.logger.warning(f"保证金不足：需 {required_margin}，可用 {self.available_capital}")
                    return False

                self.available_capital -= required_margin
                if symbol in self.positions:
                    # 合并同方向持仓
                    pos = self.positions[symbol]
                    avg_price = (pos.entry_price * pos.quantity + filled_price * quantity) / (pos.quantity + quantity)
                    pos.entry_price = avg_price
                    pos.quantity += quantity
                else:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side='LONG',
                        entry_price=filled_price,
                        quantity=quantity,
                        entry_time=order.filled_time
                    )

        elif order.side == 'SELL':
            # 检查是否为平多单
            if symbol in self.positions and self.positions[symbol].side == 'LONG':
                pos = self.positions[symbol]
                if quantity > pos.quantity:
                    self.logger.warning(f"平多单数量不足：需求 {quantity}，实际 {pos.quantity}")
                    return False

                # 计算利润和释放保证金
                profit = (filled_price - pos.entry_price) * quantity
                released_margin = (pos.entry_price * quantity) / self.leverage
                self.available_capital += released_margin + profit
                self.current_capital += profit

                # 记录交易
                self._record_trade(pos, order, profit)

                # 更新持仓
                if quantity == pos.quantity:
                    del self.positions[symbol]
                else:
                    pos.quantity -= quantity
            else:
                # 开空单
                if required_margin > self.available_capital:
                    self.logger.warning(f"保证金不足：需 {required_margin}，可用 {self.available_capital}")
                    return False

                self.available_capital -= required_margin
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    avg_price = (pos.entry_price * pos.quantity + filled_price * quantity) / (pos.quantity + quantity)
                    pos.entry_price = avg_price
                    pos.quantity += quantity
                else:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side='SHORT',
                        entry_price=filled_price,
                        quantity=quantity,
                        entry_time=order.filled_time
                    )

        return True

    def _record_trade(self, position: Position, order: Order, profit: float):
        """记录交易结果"""
        self.trades.append({
            'symbol': position.symbol,
            'side': position.side,
            'entry_time': position.entry_time,
            'exit_time': order.filled_time,
            'entry_price': position.entry_price,
            'exit_price': order.filled_price,
            'quantity': order.quantity,
            'profit': profit,
            'return_pct': profit / (position.entry_price * order.quantity * self.leverage),
            'duration': (order.filled_time - position.entry_time).total_seconds() / 3600
        })

    def _calculate_equity(self, bar: pd.Series) -> float:
        """计算当前总权益（含未实现盈亏）"""
        equity = self.current_capital
        for pos in self.positions.values():
            if pos.side == 'LONG':
                unrealized = (bar['close'] - pos.entry_price) * pos.quantity
            else:
                unrealized = (pos.entry_price - bar['close']) * pos.quantity
            equity += unrealized
        return equity

    def run(self) -> Dict:
        """运行回测"""
        for _, bar in self.data.iterrows():
            self.current_time = bar['timestamp']

            # 先执行策略逻辑生成新订单
            self.on_bar(bar)

            # 撮合订单
            self._match_orders(bar)

            # 处理成交订单回调
            for order in self.filled_orders[:]:
                self.on_order_filled(order)
                self.filled_orders.remove(order)

            # 记录权益曲线
            equity = self._calculate_equity(bar)
            self.equity_curve.append({
                'timestamp': bar['timestamp'],
                'equity': equity,
                'price': bar['close']
            })

        return self.get_results()

    def get_results(self) -> Dict:
        """生成回测结果"""
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        # 计算统计指标
        total_return = equity_df['equity'].iloc[-1] / self.initial_capital - 1
        peak = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = drawdown.min()

        # 计算平均收益率
        avg_return = trades_df['return'].mean() if not trades_df.empty else 0

        results = {
            'initial_capital': self.initial_capital,
            'final_equity': equity_df['equity'].iloc[-1],
            'total_return': total_return,
            'avg_return': avg_return,  # 添加平均收益率
            'annualized_return': total_return / (len(equity_df) / 252),
            'max_drawdown': max_drawdown,
            'total_trades': len(trades_df),
            'win_rate': trades_df['profit'].gt(0).mean() if not trades_df.empty else 0,
            'profit_factor': trades_df[trades_df['profit'] > 0]['profit'].sum() / 
                            -trades_df[trades_df['profit'] < 0]['profit'].sum() if not trades_df.empty else 0,
            'equity_curve': equity_df,
            'trades': trades_df
        }
        return results

    def on_bar(self, bar: pd.Series):
        """策略逻辑接口（需子类实现）"""
        pass

    def on_order_filled(self, order: Order):
        """订单成交回调接口（需子类实现）"""
        pass