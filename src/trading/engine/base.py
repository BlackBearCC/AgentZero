from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Order:
    id: str
    timestamp: datetime
    symbol: str
    side: str  # 'BUY'/'SELL'
    order_type: str  # 'LIMIT'/'MARKET'
    price: float
    quantity: float
    status: str = 'PENDING'  # PENDING/FILLED/CANCELLED
    filled_price: Optional[float] = None
    filled_time: Optional[datetime] = None

@dataclass
class Position:
    symbol: str
    side: str  # 'LONG'/'SHORT'
    entry_price: float
    quantity: float
    entry_time: datetime

class BaseStrategy(ABC):
    """策略基类"""
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """数据预处理模板方法"""
        self._clean_data()
        self._generate_features()
        self._validate_data()
    
    @abstractmethod
    def _clean_data(self):
        """数据清洗"""
        pass
    
    @abstractmethod
    def _generate_features(self):
        """特征工程"""
        pass
    
    def _validate_data(self):
        """数据验证"""
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(self.data.columns):
            missing = required_columns - set(self.data.columns)
            raise ValueError(f"缺失必要列: {missing}")
    
    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """生成交易信号"""
        pass

class BacktestResult:
    """回测结果分析类"""
    def __init__(self, equity_curve: pd.DataFrame, trades: pd.DataFrame):
        self.equity = equity_curve['equity']
        self.trades = trades
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """计算核心指标"""
        returns = self.equity.pct_change().dropna()
        
        self.metrics = {
            'total_return': self.equity.iloc[-1] / self.equity.iloc[0] - 1,
            'annualized_return': self._annualize(returns),
            'max_drawdown': self._max_drawdown(),
            'sharpe_ratio': self._sharpe_ratio(returns),
            'sortino_ratio': self._sortino_ratio(returns),
            'win_rate': self.trades['profit'].gt(0).mean(),
            'profit_factor': abs(self.trades[self.trades['profit'] > 0]['profit'].sum() / 
                               self.trades[self.trades['profit'] < 0]['profit'].sum()),
            'turnover': self.trades['quantity'].sum() / self.equity.mean()
        }
    
    def _annualize(self, returns: pd.Series) -> float:
        days = (returns.index[-1] - returns.index[0]).days
        return (1 + returns.mean()) ** (365 / days) - 1 if days > 0 else 0
    
    def _max_drawdown(self) -> float:
        peak = self.equity.expanding().max()
        drawdown = (self.equity - peak) / peak
        return drawdown.min()
    
    def _sharpe_ratio(self, returns: pd.Series) -> float:
        return returns.mean() / returns.std() * np.sqrt(252)
    
    def _sortino_ratio(self, returns: pd.Series) -> float:
        downside = returns[returns < 0].std()
        return returns.mean() / downside * np.sqrt(252) if downside != 0 else 0
    
    def report(self) -> str:
        """生成文字报告"""
        report = [
            "====== 回测结果报告 ======",
            f"累计收益率: {self.metrics['total_return']:.2%}",
            f"年化收益率: {self.metrics['annualized_return']:.2%}",
            f"最大回撤: {self.metrics['max_drawdown']:.2%}",
            f"夏普比率: {self.metrics['sharpe_ratio']:.2f}",
            f"索提诺比率: {self.metrics['sortino_ratio']:.2f}",
            f"胜率: {self.metrics['win_rate']:.2%}",
            f"盈亏比: {self.metrics['profit_factor']:.2f}",
            f"换手率: {self.metrics['turnover']:.2f}"
        ]
        return "\n".join(report)