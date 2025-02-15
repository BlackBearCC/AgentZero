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
import os

class MarketMicrostructureDataset(Dataset):
    """市场微观结构数据集"""
    def __init__(self, data, sequence_length=60, pred_length=10):
        self.data = data.copy()
        self.seq_len = sequence_length
        self.pred_len = pred_length
        self.scaler = StandardScaler()
        
        # 生成高级特征
        data = self._create_features(self.data)
        
        # 打印特征信息，帮助调试
        print(f"""
        ====== 特征信息 ======
        原始特征数: {len(self.data.columns)}
        处理后特征数: {len(data.columns)}
        特征列表: {data.columns.tolist()}
        ==================
        """)
        
        self.features = self.scaler.fit_transform(data)
        
    def _create_features(self, df):
        """生成微观结构特征"""
        feature_df = pd.DataFrame()
        
        # 价格特征
        feature_df['close'] = df['close']
        feature_df['return'] = df['return']
        feature_df['volatility'] = df['volatility']
        
        # 订单簿特征
        feature_df['spread'] = df['spread']
        feature_df['mid_price'] = df['mid_price']
        feature_df['imbalance'] = df['imbalance']
        feature_df['weighted_depth'] = (df['vwap_bid'] * df['bid_depth'] + df['vwap_ask'] * df['ask_depth'])/(df['bid_depth'] + df['ask_depth'])
        
        # 交易量特征
        feature_df['volume'] = df['volume']
        feature_df['volume_imbalance'] = df['volume'] * df['return']
        
        return feature_df.dropna()
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len
    
    def __getitem__(self, idx):
        seq_x = self.features[idx:idx+self.seq_len]
        seq_y = self.features[idx+self.seq_len:idx+self.seq_len+self.pred_len, 0]  # 使用close价格作为预测目标
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)

class AlphaTransformer(nn.Module):
    """基于Transformer的Alpha因子生成器"""
    def __init__(self, input_dim=9, num_heads=8, num_layers=6, d_model=64, pred_len=10):
        super().__init__()
        self.pred_len = pred_len
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, pred_len)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        memory = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # 使用最后一个时间步的特征进行预测
        last_hidden = memory[:, -1]  # (batch_size, d_model)
        pred = self.proj(last_hidden)  # (batch_size, pred_len)
        
        return pred

class DeepTradingStrategy:
    """深度学习交易策略"""
    def __init__(self, model_path=None):
        """初始化策略"""
        # 检查CUDA可用性
        print(f"""
        ====== CUDA 信息 ======
        PyTorch版本: {torch.__version__}
        CUDA是否可用: {torch.cuda.is_available()}
        GPU数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}
        当前GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}
        ===================
        """)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.model = self._build_model().to(self.device)
        self.scaler = None
        
        if model_path:
            self.load_model(model_path)
    
    def _build_model(self):
        return AlphaTransformer(
            input_dim=9,
            num_heads=8,
            num_layers=6,
            d_model=64,
            pred_len=10
        )
    
    def save_model(self, path='models/alpha_transformer.pth'):
        """保存模型和scaler"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型状态和scaler
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_state': self.scaler,
            'model_config': {
                'input_dim': 9,
                'num_heads': 8,
                'num_layers': 6,
                'd_model': 64,
                'pred_len': 10
            }
        }, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path):
        """加载模型和scaler"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到模型文件: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载模型配置和状态
        self.model = AlphaTransformer(**checkpoint['model_config']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler_state']
        
        print(f"模型已从 {path} 加载")
    
    def train(self, train_data, epochs=100, save_path='models/alpha_transformer.pth'):
        self.model.train()
        dataset = MarketMicrostructureDataset(train_data)
        self.scaler = dataset.scaler
        
        loader = DataLoader(
            dataset, 
            batch_size=256, 
            shuffle=True,
            num_workers=4 if self.device.type == 'cuda' else 0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, 
            steps_per_epoch=len(loader), 
            epochs=epochs
        )
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            self.model.train()
            
            for x, y in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}'):
                x, y = x.to(self.device), y.to(self.device)
                
                pred = self.model(x)
                loss = criterion(pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(save_path)
                print(f"发现更好的模型，已保存 (Loss: {best_loss:.4f})")
        
        print(f"训练完成！最佳Loss: {best_loss:.4f}")
    
    def predict_signal(self, market_state):
        """生成交易信号"""
        self.model.eval()
        with torch.no_grad():
            # 使用与训练时相同的特征处理
            dataset = MarketMicrostructureDataset(market_state, sequence_length=60, pred_length=10)
            features = dataset.features  # 已经标准化的特征
            
            seq = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # 添加batch维度
            pred = self.model(seq)
            
            # 计算预期收益和风险
            expected_return = pred[0, -1].item()  # 使用最后一个预测值
            risk = pred[0].std().item()
            
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
        # 获取原始数据（包含K线和订单簿数据）
        feeds = data_mgr.get_feed(
            symbol=symbol,
            timeframe=timeframe,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end)
        )
        
        # 直接使用DataFrame
        df = feeds['indicator']  # 不再需要.dataname
        
        # 生成特征
        df['return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 使用已有的订单簿特征
        # spread, mid_price, bid_depth, ask_depth, imbalance 已在_fetch_data中计算
        
        # 添加额外的市场微观结构特征
        df['order_flow'] = (df['volume'] * df['return']).rolling(10).mean()
        df['volatility'] = df['return'].rolling(20).std()
        
        print(f"""
        ====== 数据准备完成 ======
        数据范围: {df.index.min()} - {df.index.max()}
        数据点数: {len(df)}
        特征列表: {df.columns.tolist()}
        缺失值统计:
        {df.isnull().sum()}
        ========================
        """)
        
        return df.dropna()
    
    except Exception as e:
        print(f"数据准备失败: {str(e)}")
        raise

# 使用示例
if __name__ == "__main__":
    # 获取数据
    data = prepare_ai_data(
        symbol='BTC/USDT',
        start='2024-01-01',
        end='2025-01-01',
        timeframe='5m'
    )
    
    # 划分训练集和测试集
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # 初始化策略
    strategy = DeepTradingStrategy()
    
    # 训练模型（会自动保存最佳模型）
    strategy.train(train_data, epochs=100)
    
    # 加载最佳模型进行回测
    strategy = DeepTradingStrategy(model_path='models/alpha_transformer.pth')
    backtester = AITradingBacktest(strategy, test_data)
    backtester.run_backtest()
    
    # 可视化
    plt.plot(backtester.equity)
    plt.title('AI Trading Performance')
    plt.show()