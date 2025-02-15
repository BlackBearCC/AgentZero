from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from src.trading.feeds.crypto_feed import DataManager

class MarketMicrostructureDataset(Dataset):
    """市场微观结构数据集"""
    def __init__(self, data, sequence_length=60, pred_length=10):
        self.data = data
        self.seq_len = sequence_length
        self.pred_len = pred_length
        self.scaler = StandardScaler()
        
        # 生成高级特征
        data = self._create_features(data)
        self.features = self.scaler.fit_transform(data)
        
    def _create_features(self, df):
        """生成微观结构特征"""
        # 订单簿特征
        df['bid_ask_spread'] = df['ask1'] - df['bid1']
        df['mid_price'] = (df['ask1'] + df['bid1'])/2
        df['order_imbalance'] = (df['bid_size1'] - df['ask_size1'])/(df['bid_size1'] + df['ask_size1'])
        
        # 流动性特征
        df['weighted_depth'] = (df['bid_size1']*df['bid1'] + df['ask_size1']*df['ask1'])/(df['bid_size1'] + df['ask_size1'])
        
        # 波动率特征
        df['volatility'] = df['mid_price'].pct_change().rolling(20).std()
        
        # 量价特征
        df['volume_imbalance'] = df['volume'] * df['return']
        
        return df.dropna().values
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len
    
    def __getitem__(self, idx):
        seq_x = self.features[idx:idx+self.seq_len]
        seq_y = self.features[idx+self.seq_len:idx+self.seq_len+self.pred_len, 3]  # 预测mid_price变化
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)

class AlphaTransformer(nn.Module):
    """基于Transformer的Alpha因子生成器"""
    def __init__(self, input_dim, num_heads=8, num_layers=6, d_model=64):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, 
            dim_feedforward=d_model*4, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads)
        self.proj = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, x):
        # x shape: (seq_len, batch, input_dim)
        x = self.embedding(x)
        memory = self.transformer(x)
        
        # 时间序列注意力
        attn_out, _ = self.temporal_attn(memory, memory, memory)
        combined = torch.cat([memory, attn_out], dim=-1)
        
        # 多尺度特征融合
        return self.proj(combined)

class DeepTradingStrategy:
    """深度学习交易策略"""
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)
        self.scaler = StandardScaler()
        
        if model_path:
            self.load_model(model_path)
            
    def _build_model(self):
        return AlphaTransformer(
            input_dim=15,  # 根据特征数量调整
            num_heads=8,
            num_layers=6,
            d_model=64
        )
    
    def train(self, train_data, epochs=100):
        self.model.train()
        dataset = MarketMicrostructureDataset(train_data)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        criterion = nn.HuberLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, 
            steps_per_epoch=len(loader), 
            epochs=epochs
        )
        
        for epoch in range(epochs):
            total_loss = 0
            for x, y in tqdm(loader, desc=f'Epoch {epoch+1}'):
                x, y = x.to(self.device), y.to(self.device)
                
                # 多任务学习：预测价格变化和波动率
                pred = self.model(x.permute(1,0,2))
                loss = criterion(pred, y.unsqueeze(-1))
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
    
    def predict_signal(self, market_state):
        """生成交易信号"""
        self.model.eval()
        with torch.no_grad():
            features = self.scaler.transform(market_state)
            seq = torch.FloatTensor(features).unsqueeze(1).to(self.device)
            pred = self.model(seq.permute(1,0,2))
            
        # 计算预期收益和风险
        expected_return = pred[:, -1].mean().item()
        risk = pred[:, -1].std().item()
        
        # 生成动态仓位
        position = self._calculate_position(expected_return, risk)
        return position
    
    def _calculate_position(self, mu, sigma):
        """凯利公式动态仓位管理"""
        f = mu / (sigma**2 + 1e-6)
        return np.tanh(f)  # 限制仓位在[-1, 1]之间

class AITradingBacktest:
    """AI交易回测框架"""
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data
        self.positions = []
        self.equity = [1e6]  # 初始资金
        
    def run_backtest(self):
        window_size = 60  # 与模型输入序列长度一致
        for i in tqdm(range(window_size, len(self.data))):
            market_state = self.data[i-window_size:i]
            
            # 获取AI信号
            position = self.strategy.predict_signal(market_state)
            
            # 执行交易
            self._execute_trade(position)
            
            # 记录权益
            self._update_equity()
    
    def _execute_trade(self, target_pos):
        current_pos = self.positions[-1] if self.positions else 0
        delta = target_pos - current_pos
        
        # 考虑交易成本后的实际成交
        self.positions.append(current_pos + delta*0.999)  # 考虑0.1%的交易成本
    
    def _update_equity(self):
        # 计算持仓收益
        price_change = self.data['close'].pct_change().iloc[-1]
        self.equity.append(self.equity[-1] * (1 + self.positions[-1]*price_change))

def prepare_ai_data(symbol: str, start: str, end: str, timeframe: str = '1m'):
    """准备AI训练数据"""
    data_mgr = DataManager()
    
    try:
        # 获取原始数据
        feeds = data_mgr.get_feed(
            symbol=symbol,
            timeframe=timeframe,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end)
        )
        
        # 合并两个数据源
        df = pd.concat([feeds['indicator'].dataname, feeds['execution'].dataname], axis=1)
        df.columns = ['ind_open', 'ind_high', 'ind_low', 'ind_close', 'ind_volume',
                     'exe_open', 'exe_high', 'exe_low', 'exe_close', 'exe_volume']
        
        # 生成特征
        df['return'] = np.log(df['exe_close'] / df['exe_close'].shift(1))
        df['spread'] = df['exe_high'] - df['exe_low']
        df['volume_imbalance'] = df['exe_volume'] * df['return']
        
        # 添加市场微观结构特征
        df['order_flow'] = (df['exe_volume'] * df['return']).rolling(10).mean()
        df['volatility'] = df['return'].rolling(20).std()
        
        return df.dropna()
    
    except Exception as e:
        print(f"数据准备失败: {str(e)}")
        raise

# 使用示例
if __name__ == "__main__":
    # 获取BTC/USDT数据
    data = prepare_ai_data(
        symbol='BTC/USDT',
        start='2024-01-01',
        end='2025-01-01',
        timeframe='5m'
    )
    
    # 初始化策略
    strategy = DeepTradingStrategy()
    
    # 划分训练/测试集
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # 训练模型
    strategy.train(train_data)
    
    # 回测
    backtester = AITradingBacktest(strategy, test_data)
    backtester.run_backtest()
    
    # 可视化
    plt.plot(backtester.equity)
    plt.title('AI Trading Performance')
    plt.show()