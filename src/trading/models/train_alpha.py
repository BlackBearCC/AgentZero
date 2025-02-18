from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from src.trading.models.datasets import MarketMicrostructureDataset
from src.trading.feeds.crypto_feed import DataManager
from src.trading.models.transformer import AlphaTransformer



class ModelTrainer:
    """Alpha模型训练器"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._print_gpu_info()
        
    def _print_gpu_info(self):
        print(f"""
        ====== CUDA 信息 ======
        PyTorch版本: {torch.__version__}
        CUDA是否可用: {torch.cuda.is_available()}
        GPU数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}
        当前GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}
        ===================
        """)
    
    def train(self, train_data, val_data=None, epochs=100):
        """训练模型"""
        model = AlphaTransformer().to(self.device)
        dataset = MarketMicrostructureDataset(train_data)
        print("特征维度:", dataset.features.shape)
        loader = DataLoader(
            dataset, 
            batch_size=256, 
            shuffle=True,
            num_workers=4 if self.device.type == 'cuda' else 0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, 
            steps_per_epoch=len(loader), 
            epochs=epochs
        )
        
        best_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for x, y in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}'):
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                loss = criterion(pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
            
            # 验证集评估
            if val_data is not None:
                val_loss = self._evaluate(model, val_data)
                print(f"Validation Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_model(model, dataset.scaler)
                print(f"发现更好的模型，已保存 (Loss: {best_loss:.4f})")
    
    def _evaluate(self, model, val_data):
        """评估模型"""
        model.eval()
        dataset = MarketMicrostructureDataset(val_data)
        loader = DataLoader(dataset, batch_size=512)
        criterion = nn.MSELoss()
        
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                loss = criterion(pred, y)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _save_model(self, model, scaler, path=None):
        """保存模型"""
        if path is None:
            path = f'models/{self.model_name}.pth'
            
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_state': scaler,
            'model_config': model.get_config()
        }, path)

if __name__ == "__main__":
    # 准备训练数据
    data_mgr = DataManager()
    data = data_mgr.get_feed(
        symbol='BTC/USDT',
        timeframe='15m',
        start=datetime(2024, 1, 1),
        end=datetime(2025, 2, 1)
    )['indicator']
    
    # 划分训练集和验证集
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # 训练模型
    trainer = ModelTrainer('alpha_v1')
    trainer.train(train_data, val_data, epochs=100) 