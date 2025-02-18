import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        feature_df = pd.DataFrame(index=df.index)
        
        # 基础价格特征
        feature_df['close'] = df['close']
        feature_df['return'] = np.log(df['close'] / df['close'].shift(1))
        feature_df['volatility'] = feature_df['return'].rolling(20).std()
        
        # 订单簿特征 (使用可用字段，添加错误处理)
        try:
            # 如果有订单簿数据
            if 'bid1' in df.columns and 'ask1' in df.columns:
                feature_df['spread'] = df['ask1'] - df['bid1']
                feature_df['mid_price'] = (df['ask1'] + df['bid1']) / 2
            else:
                # 使用K线数据的替代特征
                feature_df['spread'] = df['high'] - df['low']
                feature_df['mid_price'] = (df['high'] + df['low']) / 2
            
            # 深度特征
            if 'bid_size1' in df.columns and 'ask_size1' in df.columns:
                feature_df['depth_imbalance'] = (df['bid_size1'] - df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
            else:
                # 使用成交量作为替代
                feature_df['depth_imbalance'] = df['volume'].pct_change()
            
        except Exception as e:
            print(f"警告：创建订单簿特征时出错: {str(e)}")
            # 使用基础特征作为替代
            feature_df['spread'] = df['high'] - df['low']
            feature_df['mid_price'] = (df['high'] + df['low']) / 2
            feature_df['depth_imbalance'] = df['volume'].pct_change()
        
        # 交易量特征
        feature_df['volume'] = df['volume']
        feature_df['volume_ma'] = df['volume'].rolling(20).mean()
        feature_df['volume_std'] = df['volume'].rolling(20).std()
        
        # 技术指标
        feature_df['ma_ratio'] = df['close'] / df['close'].rolling(20).mean()
        feature_df['rsi'] = self._calculate_rsi(df['close'])
        
        # 打印可用的特征列表
        print("可用特征列表:", feature_df.columns.tolist())
        
        return feature_df.dropna()
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len
    
    def __getitem__(self, idx):
        seq_x = self.features[idx:idx+self.seq_len]
        seq_y = self.features[idx+self.seq_len:idx+self.seq_len+self.pred_len, 0]  # 使用close价格作为预测目标
        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y) 