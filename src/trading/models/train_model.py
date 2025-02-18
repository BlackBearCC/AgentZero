import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.trading.engine.base import BaseStrategy
from src.trading.feeds.crypto_feed import DataManager
from src.trading.strategies.deep_learning import DeepLearningStrategy
from src.utils.logger import Logger





class ModelTrainer:
    """模型训练管道"""
    def __init__(self, symbol: str, model_name: str):
        self.symbol = symbol
        self.model_name = model_name
        self.logger = Logger("model_trainer")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, start: str, end: str, timeframe: str = '1m') -> pd.DataFrame:
        """准备训练数据"""
        data_mgr = DataManager()
        feeds = data_mgr.get_feed(
            symbol=self.symbol,
            timeframe=timeframe,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end)
        )
        return feeds['indicator']
    
    def train_pipeline(
        self,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        epochs: int = 100,
        batch_size: int = 256
    ):
        """完整训练流程"""
        # 准备数据
        full_data = self.prepare_data(train_start, val_end)
        train_data = full_data.loc[train_start:train_end]
        val_data = full_data.loc[val_start:val_end]
        
        # 初始化策略
        strategy = DeepLearningStrategy(train_data)
        
        # 训练模型
        strategy.train(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            model_path=f"models/{self.model_name}.pth"
        )
        
        # 验证模型
        val_results = self.evaluate_model(strategy, val_data)
        self.logger.info(f"验证结果: {val_results}")
        
        return strategy

    def evaluate_model(self, strategy: BaseStrategy, data: pd.DataFrame) -> dict:
        """模型评估"""
        strategy.model.eval()
        dataset = strategy.create_dataset(data)
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = strategy.model(x)
                loss = strategy.criterion(pred, y)
                total_loss += loss.item()
        
        return {
            'avg_loss': total_loss / len(loader),
            'model_size': sum(p.numel() for p in strategy.model.parameters())
        }

if __name__ == "__main__":
    trainer = ModelTrainer(symbol='BTC/USDT', model_name='alpha_v1')
    trainer.train_pipeline(
        train_start='2023-01-01',
        train_end='2023-06-30',
        val_start='2023-07-01',
        val_end='2023-12-31'
    ) 