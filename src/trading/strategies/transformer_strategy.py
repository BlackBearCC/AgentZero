import torch
import torch.nn as nn
from .deep_learning import DeepLearningStrategy

class TransformerStrategy(DeepLearningStrategy):
    """基于Transformer的量化策略"""
    
    def __init__(self, 
                 lookback_window: int = 60,
                 pred_length: int = 5,
                 d_model: int = 64,
                 nhead: int = 8,
                 num_layers: int = 6):
        
        model = TransformerModel(
            input_dim=5,  # OHLCV
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            pred_length=pred_length
        )
        
        super().__init__(
            model=model,
            lookback_window=lookback_window
        )
        
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建高级特征"""
        features = super()._create_features(data)
        
        # 添加订单簿特征
        features['spread'] = data['ask1'] - data['bid1']
        features['imbalance'] = (data['bid_depth'] - data['ask_depth']) / (data['bid_depth'] + data['ask_depth'])
        
        # 添加时序特征
        features['ma_ratio'] = data['close'] / data['close'].rolling(20).mean()
        features['momentum'] = data['close'].pct_change(5)
        
        return features.dropna()
        
    def _parse_prediction(self, prediction: torch.Tensor) -> float:
        """解析Transformer输出"""
        # 预测未来N个时间步的收益
        future_returns = prediction.squeeze()[-self.model.pred_length:]
        expected_return = torch.mean(future_returns).item()
        volatility = torch.std(future_returns).item()
        
        # 使用夏普比率确定仓位
        if volatility < 1e-6:
            return 0.0
        return expected_return / volatility

class TransformerModel(nn.Module):
    """时间序列Transformer模型"""
    
    def __init__(self, input_dim, d_model, nhead, num_layers, pred_length):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model*4,
                dropout=0.1
            ),
            num_layers=num_layers
        )
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, pred_length)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        encoded = self.encoder(x)
        output = self.output_proj(encoded[-1])  # (batch_size, pred_length)
        return output 