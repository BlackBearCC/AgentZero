import torch
import torch.nn as nn

class AlphaTransformer(nn.Module):
    """基于Transformer的Alpha因子生成器"""
    def __init__(self, input_dim=11, num_heads=8, num_layers=6, d_model=64, pred_len=10):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        
        # 特征投影层
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 时序注意力
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # 输出投影
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
    
    def get_config(self):
        """获取模型配置"""
        return {
            'input_dim': self.embedding.in_features,
            'num_heads': self.transformer.layers[0].self_attn.num_heads,
            'num_layers': len(self.transformer.layers),
            'd_model': self.d_model,
            'pred_len': self.pred_len
        } 