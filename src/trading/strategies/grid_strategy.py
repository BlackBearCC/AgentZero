from typing import Dict, List
import numpy as np
import pandas as pd
from src.trading.engine.backtest_engine import BacktestEngine
from src.trading.engine.position import Position
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

class GridStrategy(BacktestEngine):
    def __init__(self,
                 data: pd.DataFrame,
                 symbol: str,
                 grid_num: int = 20,
                 price_range: float = 0.1,
                 position_ratio: float = 0.01,
                 take_profit: float = 0.005,
                 **kwargs):
        """
        网格策略实现
        
        Args:
            data: DataFrame with OHLCV data
            symbol: 交易对名称
            grid_num: 单边网格数量，默认20个
            price_range: 价格区间范围（如±10%）
            position_ratio: 总资金使用比例
            take_profit: 网格间距（止盈点）
        """
        super().__init__(data, **kwargs)
        self.symbol = symbol
        self.grid_num = grid_num
        self.price_range = price_range
        self.position_ratio = position_ratio  # 修改为总资金使用比例
        self.take_profit = take_profit
        
        # 新增属性
        self.long_pos_limit = kwargs.get('long_pos_limit', 0.4)  # 多头仓位上限 40%
        self.short_pos_limit = kwargs.get('short_pos_limit', 0.4)  # 空头仓位上限 40%
        self.min_price_precision = kwargs.get('min_price_precision', 0.01)  # 最小价格精度
        self.min_qty_precision = kwargs.get('min_qty_precision', 0.001)  # 最小数量精度
        
        # 策略状态
        self.grid_orders: Dict[float, Dict[str, float]] = {}  # price -> {'order_id': id, 'side': side, 'quantity': size}
        self.center_price = None
        self.positions = {}    # 记录每个价格点的持仓
        
    def initialize_grid(self, center_price: float):
        """初始化网格"""
        self.center_price = center_price
        self.grid_orders.clear()
        self.positions.clear()
        
        # 计算网格间距
        grid_step = center_price * (self.price_range / self.grid_num)
        self.logger.info(f"网格间距: {grid_step:.8f}")
        
        # 修改资金分配逻辑
        total_margin = self.current_capital * self.position_ratio
        # 考虑仓位限制，将资金分配调整为不超过限制
        max_long_margin = self.current_capital * self.long_pos_limit
        max_short_margin = self.current_capital * self.short_pos_limit
        margin_per_grid = min(
            total_margin / (self.grid_num * 2),  # 原始计划的每格资金
            max_long_margin / self.grid_num,     # 考虑多头限制的每格资金
            max_short_margin / self.grid_num     # 考虑空头限制的每格资金
        )
        self.logger.info(f"每格资金量: {margin_per_grid:.2f}")
        
        # 计算每个网格的数量（考虑杠杆）
        position_size = round((margin_per_grid * self.leverage) / center_price, 8)
        
        # 在中心价格上下对称放置网格
        for i in range(self.grid_num):
            # 上方网格
            upper_price = round(center_price + (i + 1) * grid_step, 8)
            # 下方网格
            lower_price = round(center_price - (i + 1) * grid_step, 8)
            
            self.logger.info(f"创建网格 - 上方[{i}]: {upper_price}, 下方[{i}]: {lower_price}, 数量: {position_size}")
            
            # 上方放sell单，下方放buy单
            if position_size >= self.min_qty_precision:
                self._place_grid_order(upper_price, 'SELL', position_size)
                self._place_grid_order(lower_price, 'BUY', position_size)

    def _check_position_limits(self, side: str, new_size: float) -> bool:
        """检查是否超出仓位限制"""
        # 计算新订单的保证金占用（考虑杠杆）
        new_margin = (new_size * self.center_price) / self.leverage
        
        # 计算当前持仓的保证金占用
        current_long_margin = sum((p.quantity * self.center_price) / self.leverage 
                                for p in self.positions.values() 
                                if isinstance(p, Position) and p.side == 'LONG')
        current_short_margin = sum((p.quantity * self.center_price) / self.leverage 
                                 for p in self.positions.values() 
                                 if isinstance(p, Position) and p.side == 'SHORT')
        
        # 基于保证金计算仓位比例
        if side == 'BUY':
            new_long_ratio = (current_long_margin + new_margin) / self.initial_capital
            self.logger.info(f"多头仓位检查 - 当前: {current_long_margin:.2f}, 新增: {new_margin:.2f}, 比例: {new_long_ratio:.2%}")
            return new_long_ratio <= self.long_pos_limit
        else:
            new_short_ratio = (current_short_margin + new_margin) / self.initial_capital
            self.logger.info(f"空头仓位检查 - 当前: {current_short_margin:.2f}, 新增: {new_margin:.2f}, 比例: {new_short_ratio:.2%}")
            return new_short_ratio <= self.short_pos_limit

    def _place_grid_order(self, price: float, side: str, quantity: float):
        """在指定价格放置网格订单"""
        # 检查仓位限制
        if not self._check_position_limits(side, quantity):
            self.logger.warning(f"超出仓位限制 - {side} {quantity}")
            return
            
        order = self.place_limit_order(
            symbol=self.symbol,
            side=side,
            price=price,
            quantity=quantity
        )
        
        self.grid_orders[price] = {
            'order_id': order.id,
            'side': side,
            'quantity': quantity
        }
        self.logger.info(f"下单成功 - {side} {quantity}@{price}")

    def on_order_filled(self, order):
        """订单成交回调"""
        price = order.price
        if price not in self.grid_orders:
            self.logger.warning(f"找不到对应的网格订单 - 价格: {price}")
            return
        
        self.logger.info(f"订单成交 - {order.side} {order.quantity}@{price}")
        
        # 记录持仓
        if order.side == 'BUY':
            if self._is_closing_short(price):  # 如果是平空单
                self._close_short_position(price, order)
                self.logger.info(f"平空仓 - 价格: {price}")
            else:  # 开多单
                self.positions[price] = Position(
                    symbol=order.symbol,
                    side='LONG',
                    entry_price=price,
                    quantity=order.quantity,
                    entry_time=order.filled_time
                )
                self.logger.info(f"开多仓 - 价格: {price}")
        else:  # 'SELL'
            if self._is_closing_long(price):  # 如果是平多单
                self._close_long_position(price, order)
                self.logger.info(f"平多仓 - 价格: {price}")
            else:  # 开空单
                self.positions[price] = Position(
                    symbol=order.symbol,
                    side='SHORT',
                    entry_price=price,
                    quantity=order.quantity,
                    entry_time=order.filled_time
                )
                self.logger.info(f"开空仓 - 价格: {price}")
        
        # 移除已触发的订单
        del self.grid_orders[price]
        
        # 在对称的位置放置反向订单
        opposite_price = self.center_price * 2 - price
        opposite_side = 'SELL' if order.side == 'BUY' else 'BUY'
        self._place_grid_order(opposite_price, opposite_side, order.quantity)

    def _is_closing_long(self, price: float) -> bool:
        """检查是否是平多单"""
        return any(pos.side == 'LONG' and pos.entry_price < price 
                  for pos in self.positions.values())

    def _is_closing_short(self, price: float) -> bool:
        """检查是否是平空单"""
        return any(pos.side == 'SHORT' and pos.entry_price > price 
                  for pos in self.positions.values())

    def _close_long_position(self, price: float, order):
        """平多仓"""
        # 找到对应的多头持仓
        for pos_price, position in self.positions.items():
            if position.side == 'LONG' and position.entry_price < price:
                profit = (price - position.entry_price) * order.quantity
                self._process_trade(position, order, profit)
                del self.positions[pos_price]
                break

    def _close_short_position(self, price: float, order):
        """平空仓"""
        # 找到对应的空头持仓
        for pos_price, position in self.positions.items():
            if position.side == 'SHORT' and position.entry_price > price:
                profit = (position.entry_price - price) * order.quantity
                self._process_trade(position, order, profit)
                del self.positions[pos_price]
                break

    def _process_trade(self, position: Position, order, profit: float):
        """处理交易记录"""
        self.trades.append({
            'entry_time': position.entry_time,
            'exit_time': order.filled_time,
            'symbol': order.symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': order.price,
            'quantity': order.quantity,
            'profit': profit,
            'return': profit / (position.entry_price * order.quantity / self.leverage)
        })

    def calculate_position_size(self, price: float) -> float:
        """计算单个网格仓位大小"""
        # 考虑杠杆的可用保证金
        available_margin = self.current_capital
        
        # 单个网格的保证金使用比例
        max_margin_per_grid = available_margin * self.position_ratio / self.grid_num
        
        # 考虑杠杆的仓位大小
        position_size = (max_margin_per_grid * self.leverage) / price
        
        # 考虑手续费和滑点
        position_size = position_size * 0.98
        
        return round(position_size, 2)

    def on_bar(self, bar: pd.Series):
        """策略主逻辑"""
        # 只在第一次初始化网格
        if self.center_price is None:
            self.logger.info(f"初始化网格，中心价格: {bar['close']}")
            self.initialize_grid(bar['close'])
            return
        
        # 检查订单成交情况
        for order in self.pending_orders[:]:
            if order.status == 'FILLED':
                self.on_order_filled(order)

    def plot_results(self, results: Dict, price_data: pd.DataFrame):
        """绘制回测结果"""
        plt.style.use('seaborn')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[2, 1, 1])
        
        # 绘制K线图
        # 转换时间戳为matplotlib可用的格式
        price_data['date_num'] = mdates.date2num(price_data['timestamp'])
        ohlc = price_data[['date_num', 'open', 'high', 'low', 'close']].values
        
        # 绘制K线
        candlestick_ohlc(ax1, ohlc, width=0.0005, colorup='red', colordown='green', alpha=0.7)
        
        # 设置x轴为日期格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        
        # 绘制网格线
        if self.center_price is not None:
            grid_step = self.center_price * (self.price_range / self.grid_num)
            
            # 绘制中心价格线
            ax1.axhline(y=self.center_price, color='yellow', linestyle='--', alpha=0.5, label='Center Price')
            
            # 绘制上方网格（做空区域）
            for i in range(self.grid_num):
                grid_price = self.center_price + (i + 1) * grid_step
                ax1.axhline(y=grid_price, color='red', linestyle='--', alpha=0.2)
                
            # 绘制下方网格（做多区域）
            for i in range(self.grid_num):
                grid_price = self.center_price - (i + 1) * grid_step
                ax1.axhline(y=grid_price, color='green', linestyle='--', alpha=0.2)
        
        # 绘制交易点
        if len(results['trades']) > 0:
            trades_df = pd.DataFrame(results['trades'])
            
            # 将时间转换为与K线图相同的格式
            trades_df['entry_num'] = mdates.date2num(trades_df['entry_time'])
            trades_df['exit_num'] = mdates.date2num(trades_df['exit_time'])
            
            # 开多标记（绿色上三角）
            long_entries = trades_df[trades_df['side'] == 'LONG']
            ax1.scatter(long_entries['entry_num'], long_entries['entry_price'], 
                       marker='^', color='lime', s=100, label='Long Entry', alpha=1)
            
            # 平多标记（红色下三角）
            ax1.scatter(long_entries['exit_num'], long_entries['exit_price'], 
                       marker='v', color='magenta', s=100, label='Long Exit', alpha=1)
            
            # 开空标记（红色下三角）
            short_entries = trades_df[trades_df['side'] == 'SHORT']
            ax1.scatter(short_entries['entry_num'], short_entries['entry_price'], 
                       marker='v', color='magenta', s=100, label='Short Entry', alpha=1)
            
            # 平空标记（绿色上三角）
            ax1.scatter(short_entries['exit_num'], short_entries['exit_price'], 
                       marker='^', color='lime', s=100, label='Short Exit', alpha=1)
            
            # 用线连接开仓和平仓点
            for _, trade in trades_df.iterrows():
                ax1.plot([trade['entry_num'], trade['exit_num']], 
                        [trade['entry_price'], trade['exit_price']], 
                        color='gray', linestyle='--', alpha=0.3)
        
        # 添加多空区域标注
        y_min, y_max = ax1.get_ylim()
        ax1.fill_between(price_data['date_num'], 
                        self.center_price, y_max, 
                        color='red', alpha=0.1, label='Short Zone')
        ax1.fill_between(price_data['date_num'], 
                        y_min, self.center_price, 
                        color='green', alpha=0.1, label='Long Zone')
        
        # 绘制权益曲线
        ax1_twin = ax1.twinx()
        equity_curve = pd.DataFrame(results['equity_curve'])
        equity_curve['date_num'] = mdates.date2num(equity_curve['timestamp'])
        ax1_twin.plot(equity_curve['date_num'], equity_curve['equity'], 
                     label='Portfolio Value', color='blue')
        
        # 设置标题和标签
        ax1.set_title('Price, Grid Lines and Trading Points')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1_twin.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # 合并两个轴的图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 绘制交易分布
        if len(results['trades']) > 0:
            returns = pd.DataFrame(results['trades'])['return']
            ax2.hist(returns, bins=50, alpha=0.75)
            ax2.set_title('Trade Returns Distribution')
            ax2.set_xlabel('Return')
            ax2.set_ylabel('Frequency')
            ax2.grid(True)
            
        # 绘制回撤
        equity_series = equity_curve['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        
        ax3.fill_between(equity_curve['date_num'], drawdown, 0, 
                        color='red', alpha=0.3)
        ax3.set_title('Drawdown')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown %')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show() 