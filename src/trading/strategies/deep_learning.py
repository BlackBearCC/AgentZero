import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from src.trading.engine.base import BaseStrategy

class LSTMModel(torch.nn.Module):
    """LSTM预测模型"""
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=0.2
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(out[:, -1, :])

class DeepLearningStrategy(BaseStrategy):
    """生产级深度学习策略"""
    def __init__(self, data: pd.DataFrame, model_path: str = None):
        super().__init__(data)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
        
        if model_path:
            self.load_model(model_path)
        else:
            raise ValueError("必须提供训练好的模型路径")
    
    def load_model(self, path: str):
        """加载生产环境模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model = LSTMModel(
            input_size=checkpoint['input_dim'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_dim']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        
    def create_dataset(self, data: pd.DataFrame) -> Dataset:
        """创建生产环境数据集"""
        features = data[self._get_feature_columns()]
        scaled = self.scaler.transform(features)
        return TradingDataset(scaled, seq_length=60)
    
    def generate_signals(self) -> pd.DataFrame:
        """生产环境信号生成"""
        dataset = self.create_dataset(self.data)
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        
        signals = []
        self.model.eval()
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                pred = self.model(x).cpu().numpy()
                signals.extend(pred)
        
        return self._postprocess_signals(signals)
    
    def _postprocess_signals(self, raw_pred: list) -> pd.DataFrame:
        """后处理预测结果"""
        signals = pd.DataFrame(
            index=self.data.index[60:],
            data={
                'pred': raw_pred[:, 0],
                'confidence': raw_pred[:, 1]
            }
        )
        signals['action'] = np.where(
            signals['pred'] > 0.5, 'BUY',
            np.where(signals['pred'] < -0.5, 'SELL', 'HOLD')
        )
        signals['size'] = signals['confidence'].clip(0, 1)
        return signals
    
    def _clean_data(self):
        """数据清洗"""
        # 处理缺失值
        self.data.ffill(inplace=True)
        self.data.dropna(inplace=True)
        
        # 过滤异常值
        vol_mean = self.data['volume'].rolling(20).mean()
        self.data = self.data[(self.data['volume'] < 5 * vol_mean) & 
                             (self.data['volume'] > 0.2 * vol_mean)]
    
    def _generate_features(self):
        """生成特征"""
        # 基础特征
        self.data['returns'] = np.log(self.data['close'] / self.data['close'].shift(1))
        self.data['volatility'] = self.data['returns'].rolling(20).std()
        
        # 技术指标
        self.data['ma10'] = self.data['close'].rolling(10).mean()
        self.data['ma50'] = self.data['close'].rolling(50).mean()
        self.data['rsi'] = self._calculate_rsi(14)
        
        # 订单簿特征（示例）
        self.data['spread'] = self.data['ask1'] - self.data['bid1']
        self.data['depth_imbalance'] = (self.data['bid_size1'] - self.data['ask_size1']) / \
                                      (self.data['bid_size1'] + self.data['ask_size1'])
        
        self.data.dropna(inplace=True)
    
    def _calculate_rsi(self, window: int) -> pd.Series:
        """计算RSI"""
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _get_feature_columns(self):
        return ['returns', 'volatility', 'ma10', 'ma50', 'rsi', 
               'spread', 'depth_imbalance', 'volume'] 