from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Position:
    """持仓类"""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    quantity: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    
    def update_exit(self, exit_price: float, exit_time: datetime):
        """更新平仓信息"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        
        # 计算盈亏
        if self.side == 'LONG':
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:  # 'SHORT'
            self.pnl = (self.entry_price - exit_price) * self.quantity 