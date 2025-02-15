from typing import Dict, List
import numpy as np
import pandas as pd
from src.trading.engine.backtest_engine import BacktestEngine, Order, Position
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from src.utils.logger import Logger

class GridStrategy(BacktestEngine):
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000,
        commission_rate: float = 0.001,
        leverage: float = 1.0,
        grid_num: int = 10,
        price_range: float = 0.1,
        position_ratio: float = 0.8,
        long_pos_limit: float = 0.8,
        short_pos_limit: float = 0.8,
        min_price_precision: float = 0.0001,
        min_qty_precision: float = 0.001,
        dynamic_grid: bool = True,  # 是否启用动态网格
        ma_window: int = 50,       # 移动平均窗口
        vol_window: int = 30,      # 波动率计算窗口
        adjust_threshold: float = 0.03,  # 调整阈值
        vol_multiplier: float = 3.0,     # 波动率乘数
        min_grid: int = 20,        # 最小网格数
        max_grid: int = 100,       # 最大网格数
        base_vol: float = 0.01,    # 基准波动率
        **kwargs
    ):
        super().__init__(
            data=data,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            leverage=leverage,
            min_price_precision=min_price_precision,
            min_qty_precision=min_qty_precision,
            **kwargs
        )
        
        # 网格参数
        self.grid_num = grid_num
        self.price_range = price_range
        self.position_ratio = position_ratio
        self.long_pos_limit = long_pos_limit
        self.short_pos_limit = short_pos_limit
        
        # 精度参数（从父类继承）
        self.price_precision = self._get_precision(min_price_precision)
        self.qty_precision = self._get_precision(min_qty_precision)
        self.min_order_qty = min_qty_precision
        
        # 动态网格参数
        self.dynamic_grid = dynamic_grid
        self.ma_window = ma_window
        self.vol_window = vol_window
        self.adjust_threshold = adjust_threshold
        self.vol_multiplier = vol_multiplier
        self.min_grid = min_grid
        self.max_grid = max_grid
        self.base_vol = base_vol
        
        # 网格状态
        self.center_price = None
        self.grid_orders = {}  # price -> order
        self.grid_positions = {}  # price -> position
        
        self.symbol = 'GRID'  # 使用固定symbol
        
    def _check_position_limits(self, side: str, new_size: float) -> bool:
        """检查是否超出仓位限制（修复保证金计算）"""
        # 使用当前价格而不是中心价格
        current_price = self.data.iloc[-1]['close']
        new_margin = (new_size * current_price) / self.leverage
        
        # 计算当前保证金占用
        current_long = sum(
            (p.quantity * p.entry_price) / self.leverage 
            for p in self.positions.values() 
            if p.side == 'LONG'
        )
        current_short = sum(
            (p.quantity * p.entry_price) / self.leverage 
            for p in self.positions.values() 
            if p.side == 'SHORT'
        )
        
        # 计算总保证金占用比例
        total_margin_ratio = (current_long + current_short) / self.initial_capital
        if total_margin_ratio >= 1.0:
            self.logger.warning("总保证金占用已达100%")
            return False
        
        # 基于保证金计算仓位比例
        if side == 'BUY':
            new_long_ratio = (current_long + new_margin) / self.initial_capital
            self.logger.info(f"多头仓位检查 - 当前: {current_long:.2f}, 新增: {new_margin:.2f}, 比例: {new_long_ratio:.2%}")
            return new_long_ratio <= self.long_pos_limit
        else:
            new_short_ratio = (current_short + new_margin) / self.initial_capital
            self.logger.info(f"空头仓位检查 - 当前: {current_short:.2f}, 新增: {new_margin:.2f}, 比例: {new_short_ratio:.2%}")
            return new_short_ratio <= self.short_pos_limit

    def calculate_position_size(self, price: float) -> float:
        """计算仓位大小（添加风险控制）"""
        # 计算可用保证金（考虑未实现盈亏）
        equity = self._calculate_equity(self.data.iloc[-1])
        available = equity * self.position_ratio - sum(
            (p.quantity * p.entry_price) / self.leverage 
            for p in self.positions.values()
        )
        
        # 确保可用保证金为正
        available = max(available, 0)
        
        # 计算单个网格仓位
        size = (available / self.grid_num) * self.leverage / price
        
        # 应用数量精度
        size = round(size // self.min_order_qty * self.min_order_qty, self.qty_precision)
        return size

    def _place_grid_order(self, price: float, side: str, quantity: float):
        """放置网格订单"""
        # 添加价格精度处理
        price = round(price, self.price_precision)
        
        # 检查是否已存在相同价格的订单
        if price in self.grid_orders:
            self.logger.debug(f"已存在相同价格的订单: {price}")
            return
        
        # 检查最小下单量
        if quantity < self.min_order_qty:
            self.logger.debug(f"订单数量小于最小值: {quantity} < {self.min_order_qty}")
            return
        
        # 检查仓位限制
        if not self._check_position_limits(side, quantity):
            self.logger.warning(f"超出仓位限制 - {side} {quantity}@{price}")
            return
        
        # 使用引擎的place_order方法下单
        order = self.place_order(
            symbol=self.symbol,
            side=side,
            order_type='LIMIT',
            price=price,
            quantity=quantity
        )
        
        if order:
            # 记录到网格订单字典中
            self.grid_orders[price] = order
            self.logger.info(f"放置网格订单 - {side} {quantity}@{price}")
        
        return order

    def initialize_grid(self, center_price: float):
        """初始化网格"""
        self.center_price = center_price
        grid_step = center_price * (self.price_range / self.grid_num)
        
        # 在中心价格上下创建网格
        for i in range(1, self.grid_num + 1):
            # 上方网格（做空）
            sell_price = center_price + i * grid_step
            # 下方网格（做多）
            buy_price = center_price - i * grid_step
            
            # 计算网格订单数量
            quantity = self.calculate_position_size(center_price)
            
            # 放置网格订单
            self._place_grid_order(sell_price, 'SELL', quantity)
            self._place_grid_order(buy_price, 'BUY', quantity)

    def on_order_filled(self, order):
        """订单成交回调"""
        # 修正后的平仓逻辑
        price = round(float(order.price), self.price_precision)
        
        if order.side == 'BUY':
            if self._is_closing_short(price):
                # 精确匹配最接近的持仓
                short_positions = [(float(p_price), p) for p_price, p in self.grid_positions.items() 
                                 if p.side == 'SHORT' and float(p_price) > price]
                if short_positions:
                    # 找到价格最接近的持仓
                    closest = min(short_positions, key=lambda x: abs(x[0] - price))
                    pos_price, pos = closest
                    profit = (pos.entry_price - price) * order.quantity
                    self._record_trade(pos, order, profit)
                    del self.grid_positions[str(pos_price)]
            else:  # 开多单
                self.grid_positions[str(price)] = Position(
                    symbol=order.symbol,
                    side='LONG',
                    entry_price=price,
                    quantity=order.quantity,
                    entry_time=order.filled_time
                )
        
        else:  # 'SELL'
            if self._is_closing_long(price):  # 如果是平多单
                # 找到对应的多头持仓
                long_positions = [(float(p_price), p) for p_price, p in self.grid_positions.items() 
                                if p.side == 'LONG' and float(p_price) < price]
                if long_positions:
                    # 找到价格最接近的持仓
                    closest = min(long_positions, key=lambda x: abs(x[0] - price))
                    pos_price, pos = closest
                    profit = (price - pos.entry_price) * order.quantity
                    self._record_trade(pos, order, profit)
                    del self.grid_positions[str(pos_price)]
            else:  # 开空单
                self.grid_positions[str(price)] = Position(
                    symbol=order.symbol,
                    side='SHORT',
                    entry_price=price,
                    quantity=order.quantity,
                    entry_time=order.filled_time
                )
        
        # 移除已触发的订单
        if price in self.grid_orders:
            del self.grid_orders[price]
        
        # 在相反方向放置新的网格订单
        grid_step = self.center_price * (self.price_range / self.grid_num)
        new_price = price + grid_step if order.side == 'BUY' else price - grid_step
        new_price = round(new_price, self.price_precision)
        
        # 检查新价格是否在网格范围内
        upper_bound = self.center_price * (1 + self.price_range)
        lower_bound = self.center_price * (1 - self.price_range)
        
        if lower_bound <= new_price <= upper_bound:
            new_side = 'SELL' if order.side == 'BUY' else 'BUY'
            self._place_grid_order(new_price, new_side, order.quantity)

    def _is_closing_long(self, price: float) -> bool:
        """检查是否是平多单"""
        return any(pos.side == 'LONG' and float(pos_price) < price 
                  for pos_price, pos in self.grid_positions.items())

    def _is_closing_short(self, price: float) -> bool:
        """检查是否是平空单"""
        return any(pos.side == 'SHORT' and float(pos_price) > price 
                  for pos_price, pos in self.grid_positions.items())

    def _record_trade(self, position: Position, order, profit: float):
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

    def on_bar(self, bar: pd.Series):
        """策略主逻辑"""
        # 只在第一次初始化网格
        if self.center_price is None:
            self.logger.info(f"初始化网格，中心价格: {bar['close']}")
            self.initialize_grid(bar['close'])
            return
        
        current_price = bar['close']
        
        # 记录当前资金状态
        equity = self._calculate_equity(bar)
        # self.logger.info(f"""
        # ====== 资金状态 [{bar['timestamp']}] ======
        # 当前价格: {current_price:.8f}
        # 账户权益: ${equity:.2f}
        # 可用资金: ${self.available_capital:.2f}
        # 持仓数量: {len(self.grid_positions)}
        # 挂单数量: {len(self.grid_orders)}
        # 多头持仓: {sum(1 for p in self.grid_positions.values() if p.side == 'LONG')}
        # 空头持仓: {sum(1 for p in self.grid_positions.values() if p.side == 'SHORT')}
        # 多头占用保证金: ${sum((p.quantity * current_price) / self.leverage for p in self.grid_positions.values() if p.side == 'LONG'):.2f}
        # 空头占用保证金: ${sum((p.quantity * current_price) / self.leverage for p in self.grid_positions.values() if p.side == 'SHORT'):.2f}
        # ================================
        # """)
        
        # 计算网格边界
        grid_step = self.center_price * (self.price_range / self.grid_num)
        upper_boundary = self.center_price * (1 + self.price_range)
        lower_boundary = self.center_price * (1 - self.price_range)
        
        # 检查价格是否超出网格边界
        if current_price > upper_boundary:
            self.logger.warning(f"""
            价格超出上边界!
            当前价格: {current_price:.8f}
            上边界: {upper_boundary:.8f}
            超出幅度: {((current_price - upper_boundary) / upper_boundary * 100):.2f}%
            """)
        elif current_price < lower_boundary:
            self.logger.warning(f"""
            价格超出下边界!
            当前价格: {current_price:.8f}
            下边界: {lower_boundary:.8f}
            超出幅度: {((lower_boundary - current_price) / lower_boundary * 100):.2f}%
            """)
        
        # 检查订单成交情况
        for order in self.pending_orders[:]:
            if order.status == 'FILLED':
                self.on_order_filled(order)

        # 动态调整网格中心价格（新增）
        if self.dynamic_grid and len(self.data) >= self.ma_window:
            # 使用移动平均线作为新的中心价格
            new_center = self.data['close'].rolling(self.ma_window).mean().iloc[-1]
            if abs(new_center - self.center_price) > self.center_price * self.adjust_threshold:
                self.logger.info(f"动态调整网格中心价格 {self.center_price} -> {new_center}")
                self.center_price = new_center
                self.initialize_grid(new_center)
        
        # 新增波动率适应逻辑
        if len(self.data) >= self.vol_window:
            returns = self.data['close'].pct_change().dropna()
            current_vol = returns[-self.vol_window:].std()
            new_grid_num = int(self.grid_num * (current_vol / self.base_vol))
            new_grid_num = max(min(new_grid_num, self.max_grid), self.min_grid)
            
            if new_grid_num != self.grid_num:
                self.logger.info(f"调整网格数量 {self.grid_num} -> {new_grid_num} (波动率: {current_vol:.4f})")
                self.grid_num = new_grid_num
                self.initialize_grid(self.center_price)

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
            
            # 开多标记（绿色三角）
            long_entries = trades_df[trades_df['side'] == 'LONG']
            ax1.scatter(long_entries['entry_num'], long_entries['entry_price'], 
                       marker='^', color='lime', s=150, label='Long Entry', alpha=1)
            
            # 平多标记（红色横线）
            ax1.scatter(long_entries['exit_num'], long_entries['exit_price'], 
                       marker='_', color='red', s=150, label='Long Exit', alpha=1)
            
            # 开空标记（红色三角）
            short_entries = trades_df[trades_df['side'] == 'SHORT']
            ax1.scatter(short_entries['entry_num'], short_entries['entry_price'], 
                       marker='v', color='red', s=150, label='Short Entry', alpha=1)
            
            # 平空标记（绿色横线）
            ax1.scatter(short_entries['exit_num'], short_entries['exit_price'], 
                       marker='_', color='lime', s=150, label='Short Exit', alpha=1)
            
            # 用线连接开仓和平仓点
            for _, trade in trades_df.iterrows():
                ax1.plot([trade['entry_num'], trade['exit_num']], 
                        [trade['entry_price'], trade['exit_price']], 
                        color='gray', linestyle='--', alpha=0.3)
        
        # 添加所有开仓点（包括未平仓的）
        for price, position in self.grid_positions.items():
            entry_num = mdates.date2num(position.entry_time)
            if position.side == 'LONG':
                ax1.scatter(entry_num, price, 
                           marker='^', color='lime', s=150, 
                           label='Open Long', alpha=0.5)
            else:  # SHORT
                ax1.scatter(entry_num, price, 
                           marker='v', color='red', s=150, 
                           label='Open Short', alpha=0.5)
        
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

    def _get_precision(self, value: float) -> int:
        """计算精度位数"""
        return int(round(-np.log10(value))) 