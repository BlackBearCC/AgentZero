import backtrader as bt
import numpy as np
from typing import List, Dict
from .base import BaseStrategy

class AutoGridStrategy(BaseStrategy):
    """自适应网格交易策略"""
    
    # 优化后的参数配置
    params = (
        ('base_spacing', 0.02),      # 基础网格间距3%
        ('dynamic_ratio', 1.5),      # 动态扩展系数
        ('leverage', 20),            # 杠杆倍数
        ('max_grids', 50),           # 网格数量
        ('rebalance_bars', 96),      # 96根K线重新平衡
        ('max_drawdown', 20),       # 最大回撤限制
        ('atr_period', 14),         # ATR周期
        ('ema_fast', 20),           # 快速EMA
        ('ema_slow', 50),           # 慢速EMA
    )

    def __init__(self):
        super().__init__()
        # 添加订单跟踪字典
        self.order_info = {}  # 用于存储订单创建时间
    
    def _init_indicators(self):
        """初始化技术指标"""
        # 核心指标
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
        
        # 动态参数
        self.price_center = None
        self.grid_levels = []
        self.active_orders = []
        self.last_update_bar = 0
        self.spacing = self.p.base_spacing
        
        # 风险控制
        self.drawdown = bt.analyzers.DrawDown()
        
        self.logger.info(f"策略初始化 - 基础间距: {self.p.base_spacing}, "
                        f"杠杆倍数: {self.p.leverage}, "
                        f"网格数量: {self.p.max_grids}")

    def next(self):
        """策略主逻辑"""
        try:
            # 初始化价格中枢
            if self.price_center is None:
                self.price_center = self.data.close[0]
                self.adaptive_grid_adjustment()
                self.logger.info(f"初始价格中枢设置为: {self.price_center:.2f}")
                return

            # 定期网格再平衡
            if len(self.data) - self.last_update_bar >= self.p.rebalance_bars:
                self.adaptive_grid_adjustment()
                self.last_update_bar = len(self.data)
                self.logger.debug("执行网格再平衡")

            # 执行网格交易
            self.execute_grid_orders()

            # 动态止损检查
            self._check_risk_control()
            
        except Exception as e:
            self.logger.error(f"策略执行错误: {str(e)}")

    def adaptive_grid_adjustment(self):
        """增强版自适应网格算法"""
        try:
            current_price = self.data.close[0]
            
            # 计算趋势方向
            trend = (self.ema_fast[0] - self.ema_slow[0]) / self.ema_slow[0]
            
            # 基于波动率的动态调整
            atr_ratio = self.atr[0] / current_price
            
            # 更保守的间距计算
            self.spacing = np.clip(
                (atr_ratio * 2) * 100,  # 降低间距
                0.5,   # 最小0.5%
                2.0    # 最大2.0%
            )
            
            # 根据趋势调整网格范围
            if abs(trend) > 0.01:  # 明显趋势
                # 顺势设置网格
                expansion = 1.2
                upper = current_price * (1 + self.spacing/100 * expansion)
                lower = current_price * (1 - self.spacing/100 * expansion)
            else:
                # 震荡市场，缩小网格范围
                expansion = 0.8
                upper = current_price * (1 + self.spacing/100 * expansion)
                lower = current_price * (1 - self.spacing/100 * expansion)
            
            # 重置网格
            self.grid_levels = []
            
            # 获取当前持仓
            position = self.getposition()
            current_size = position.size if position else 0
            
            # 根据持仓设置网格
            for level in [0.2, 0.4, 0.6, 0.8]:
                price = lower + (upper - lower) * level
                
                # 计算基础仓位
                base_size = self._calculate_position_size(price)
                
                # 根据当前持仓调整方向和大小
                if current_size > 0:
                    # 持多仓时，更多卖单
                    size = base_size * (2.0 if price > current_price else 0.5)
                elif current_size < 0:
                    # 持空仓时，更多买单
                    size = base_size * (2.0 if price < current_price else 0.5)
                else:
                    # 无持仓时，均衡配置
                    size = base_size
                
                self.grid_levels.append({
                    'price': price,
                    'side': 'buy' if price < current_price else 'sell',
                    'size': size
                })
            
            self.logger.info(f"网格更新 - 间距: {self.spacing:.2f}%, "
                           f"趋势: {trend:.2f}, "
                           f"当前持仓: {current_size}")
            
        except Exception as e:
            self.logger.error(f"网格调整错误: {str(e)}")

    def execute_grid_orders(self):
        """高频订单管理系统"""
        try:
            current_bar = len(self.data)
            current_price = self.data.close[0]
            
            # 检查当前持仓
            position = self.getposition()
            total_position = abs(position.size) if position else 0
            
            # # 添加持仓限制
            # max_total_position = self.broker.getvalue()*20  # 最大总持仓
            
            # # 如果总持仓超过限制，不再开新仓
            # if total_position * current_price >= max_total_position:
            #     self.logger.warning(f"总持仓达到限制: {total_position:.2f}")
            #     return
            
            # 清理过期订单
            active_orders = self.active_orders.copy()  # 创建副本避免修改迭代对象
            for order in active_orders:
                if order.status == bt.Order.Submitted:  # 只处理未完成的订单
                    created_bar = self.order_info.get(order.ref, 0)
                    if current_bar - created_bar > self.p.rebalance_bars:
                        self.cancel(order)
                        self.active_orders.remove(order)
                        self.logger.debug(f"取消过期订单: {order.ref}")
            
            # 动态调整订单
            for grid in self.grid_levels:
                if grid['side'] == 'buy':
                    # 买单主动靠近当前价格
                    order_price = current_price * (1 - 0.1 * self.spacing/100)
                else:
                    # 卖单主动靠近当前价格
                    order_price = current_price * (1 + 0.1 * self.spacing/100)
                
                # 创建新订单
                order = self.buy if grid['side'] == 'buy' else self.sell
                new_order = order(
                    price=order_price,
                    size=grid['size'],
                    exectype=bt.Order.Limit,
                    valid=None  # 不设置过期时间，由我们自己管理
                )
                
                # 记录订单创建时间
                self.order_info[new_order.ref] = current_bar
                self.active_orders.append(new_order)
                
            self.logger.debug(f"订单更新 - 活跃订单数: {len(self.active_orders)}")
            
        except Exception as e:
            self.logger.error(f"订单执行错误: {str(e)}")

    def _calculate_position_size(self, price: float) -> float:
        """计算杠杆后的仓位大小"""
        try:
            portfolio_value = self.broker.getvalue()
            # 降低单个网格使用的资金比例
            position_value = portfolio_value * 0.02  # 从10%改为2%
            
            # 添加最小和最大仓位限制
            min_position = 100  # 最小100USDT
            max_position = portfolio_value * 0.1  # 最大10%资金
            
            # 计算基础仓位
            position = (position_value * self.p.leverage) / price
            
            # 限制仓位范围
            position = max(min_position/price, min(position, max_position/price))
            
            return position
            
        except Exception as e:
            self.logger.error(f"仓位计算错误: {str(e)}")
            return 0.0

    def _check_risk_control(self):
        """风险控制"""
        try:
            # 获取当前回撤
            dd_analysis = self.drawdown.get_analysis()
            current_dd = dd_analysis.get('drawdown', 0)
            
            # 分阶段止损
            if current_dd > self.p.max_drawdown * 0.7:  # 达到警戒线
                # 减仓20%
                position = self.getposition()
                if position.size > 0:
                    self.close(size=position.size * 0.2)
                    self.logger.warning(f"触发风险控制 - 减仓20%, 当前回撤: {current_dd:.2f}%")
            elif current_dd > self.p.max_drawdown:  # 达到止损线
                self.close()  # 全部平仓
                self.logger.warning(f"触发风险控制 - 全部平仓, 当前回撤: {current_dd:.2f}%")
                
        except Exception as e:
            self.logger.error(f"风险控制错误: {str(e)}")

    def get_analysis(self) -> Dict:
        """获取策略分析结果"""
        analysis = super().get_analysis()
        analysis.update({
            'grid_count': len(self.grid_levels),
            'last_update_bar': self.last_update_bar,
            'price_center': self.price_center,
            'drawdown': self.drawdown.get_analysis()
        })
        return analysis

    def notify_trade(self, trade):
        """交易通知"""
        if trade.isclosed:
            self.logger.info(
                f"交易结束 - 毛利润: {trade.pnl:.2f}, "
                f"净利润: {trade.pnlcomm:.2f}"
            )

    def notify_order(self, order):
        """订单状态更新通知"""
        if order.status in [bt.Order.Completed, bt.Order.Canceled, bt.Order.Rejected]:
            # 订单完成或取消，从跟踪字典中移除
            self.order_info.pop(order.ref, None)
            if order in self.active_orders:
                self.active_orders.remove(order) 