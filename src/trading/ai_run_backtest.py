from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch

from src.trading.feeds.crypto_feed import DataManager
from src.trading.models.datasets import MarketMicrostructureDataset
from src.trading.models.transformer import AlphaTransformer

class TransformerBacktester:
    """Transformer模型回测器"""
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.load_model(model_path)
        
    def load_model(self, path: str):
        """加载预训练模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model = AlphaTransformer(**checkpoint['model_config']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler_state']
        self.model.eval()
        
    def run_backtest(self, data: pd.DataFrame, initial_capital: float = 1_000_000):
        """运行回测"""
        print("开始回测...")
        
        # 准备数据
        dataset = MarketMicrostructureDataset(data)
        features = dataset.features
        
        # 初始化回测变量
        positions = []
        equity = [initial_capital]
        current_position = 0
        
        # 遍历每个时间点
        window_size = 60
        for i in range(window_size, len(features) - 10):  # -10是预测长度
            # 获取当前窗口的特征
            window = features[i-window_size:i]
            x = torch.FloatTensor(window).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                pred = self.model(x)
                
            # 计算信号
            future_returns = pred[0].cpu().numpy()
            expected_return = np.mean(future_returns)
            volatility = np.std(future_returns)
            
            # 计算目标仓位
            target_position = self._calculate_position(expected_return, volatility)
            
            # 执行交易
            delta = target_position - current_position
            current_position = target_position
            positions.append(current_position)
            
            # 更新权益
            price_change = data['close'].iloc[i] / data['close'].iloc[i-1] - 1
            equity.append(equity[-1] * (1 + current_position * price_change))
            
            # 打印进度
            if i % 100 == 0:
                print(f"进度: {i}/{len(features)}, 当前权益: {equity[-1]:,.2f}")
        
        # 计算回测指标
        self._calculate_metrics(equity, data.index[window_size:window_size+len(equity)])
        
        return pd.DataFrame({
            'equity': equity,
            'position': positions + [0]  # 添加最后一个0仓位
        }, index=data.index[window_size:window_size+len(equity)])
    
    def _calculate_position(self, mu: float, sigma: float) -> float:
        """计算目标仓位"""
        if sigma < 1e-6:
            return 0
        kelly = mu / (sigma**2)
        return np.clip(kelly, -1, 1)  # 限制仓位在[-1, 1]之间
    
    def _calculate_metrics(self, equity: list, dates: pd.DatetimeIndex):
        """计算回测指标"""
        equity_series = pd.Series(equity, index=dates)
        returns = equity_series.pct_change().dropna()
        
        # 计算主要指标
        total_return = (equity[-1] / equity[0] - 1) * 100
        annual_return = (1 + total_return/100) ** (252/len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol != 0 else 0
        max_drawdown = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100
        
        print(f"""
        ====== 回测结果 ======
        总收益率: {total_return:.2f}%
        年化收益: {annual_return*100:.2f}%
        年化波动: {annual_vol*100:.2f}%
        夏普比率: {sharpe:.2f}
        最大回撤: {max_drawdown:.2f}%
        =====================
        """)
        
        # # 绘制权益曲线
        # plt.figure(figsize=(12, 6))
        # plt.plot(dates, equity)
        # plt.title('Backtest Equity Curve')
        # plt.xlabel('Date')
        # plt.ylabel('Equity')
        # plt.grid(True)
        # plt.show()

if __name__ == "__main__":
    # 加载回测数据
    data_mgr = DataManager()
    data = data_mgr.get_feed(
        symbol='BTC/USDT',
        timeframe='15m',
        start=datetime(2024, 2, 1),
        end=datetime(2024, 2, 15)
    )['indicator']
    
    # 运行回测
    backtester = TransformerBacktester('models/alpha_v1.pth')
    results = backtester.run_backtest(data)