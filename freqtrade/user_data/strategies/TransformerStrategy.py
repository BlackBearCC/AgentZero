import numpy as np
import pandas as pd
import talib.abstract as ta
import torch
from freqtrade.strategy import IStrategy, DecimalParameter
from typing import Dict, List
import logging
from datetime import datetime, timedelta
import os
import torch.nn as nn
import math

logger = logging.getLogger(__name__)

# 添加位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# 添加Transformer模型类 - 与src/trading/models/transformer.py完全一致
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

class TransformerStrategy(IStrategy):
    """
    基于Transformer模型的交易策略
    保持与ai_run_backtest.py中相同的逻辑
    """
    
    # 定义ROI表 - 随时间递减的利润目标
    minimal_roi = {
        "0": 0.05,    # 0分钟后，如果有5%的利润就卖出
        "30": 0.025,  # 30分钟后，如果有2.5%的利润就卖出
        "60": 0.01,   # 60分钟后，如果有1%的利润就卖出
        "120": 0      # 120分钟后，任何利润都卖出
    }
    
    # 止损设置
    stoploss = -0.02  # 2%止损
    
    # 时间周期
    timeframe = '15m'
    
    # 订单类型
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    # 参数优化范围 - 与ai_run_backtest.py中保持一致
    position_limit = DecimalParameter(0.3, 0.7, default=0.6217296565132786, space="buy", optimize=True)
    volatility_threshold = DecimalParameter(0.01, 0.02, default=0.01364604981391187, space="buy", optimize=True)
    signal_threshold = DecimalParameter(0.2, 0.4, default=0.244313868705759, space="buy", optimize=True)
    smoothing_factor = DecimalParameter(0.5, 0.9, default=0.7607211409966947, space="buy", optimize=True)
    stop_loss_pct = DecimalParameter(0.01, 0.03, default=0.010137904412717525, space="sell", optimize=True)
    take_profit_pct = DecimalParameter(0.03, 0.07, default=0.055370509955518045, space="sell", optimize=True)
    
    # 在类级别初始化model属性
    model = None
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # 确保初始化self.model
        self.model = None
        self.last_signal = 0
        self.scaler = None
        self.prediction_horizon = 20  # 设置预测长度
        
        # 修正模型路径
        self.model_path = '/freqtrade/user_data/models/alpha_v1.pth'
        
        try:
            # 添加更多调试信息
            logger.info(f"当前工作目录: {os.getcwd()}")
            logger.info(f"尝试加载模型: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                # 列出可能的位置
                possible_locations = [
                    '/freqtrade/user_data/models/',
                    '/freqtrade/models/',
                    './models/'
                ]
                for loc in possible_locations:
                    if os.path.exists(loc):
                        logger.info(f"目录存在: {loc}, 文件列表: {os.listdir(loc)}")
            else:
                # 正确加载字典格式的模型
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                # 从checkpoint中提取模型配置和权重
                model_config = checkpoint['model_config']
                model_state = checkpoint['model_state_dict']
                self.scaler = checkpoint['scaler_state']
                
                # 重建模型 - 使用本地定义的AlphaTransformer类
                self.model = AlphaTransformer(**model_config)
                self.model.load_state_dict(model_state)
                self.model.eval()
                
                logger.info(f"成功加载模型: {self.model_path}")
                
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            # 设置为None以便策略可以回退到基本逻辑
            self.model = None
            
        # 记录当前持仓
        self.current_position = 0
        
        # 记录止损止盈价格
        self.stop_loss_price = None
        self.take_profit_price = None
        
        # 记录初始化完成
        logger.info("TransformerStrategy初始化完成, model属性状态: " + str(self.model is not None))
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        计算技术指标
        与ai_run_backtest.py中保持一致的指标计算
        """
        # 复制数据框以避免修改原始数据
        df = dataframe.copy()
        
        # 计算基本价格指标
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 计算波动率
        df['volatility'] = df['returns'].rolling(window=14).std()
        
        # 计算RSI
        df['rsi'] = ta.RSI(df, timeperiod=14)
        
        # 计算MACD
        macd = ta.MACD(df)
        df['macd'] = macd['macd']
        df['macdsignal'] = macd['macdsignal']
        df['macdhist'] = macd['macdhist']
        
        # 计算布林带
        bollinger = ta.BBANDS(
            df, 
            timeperiod=20,
            nbdevup=2.0,
            nbdevdn=2.0
        )
        df['bb_upperband'] = bollinger['upperband']
        df['bb_middleband'] = bollinger['middleband']
        df['bb_lowerband'] = bollinger['lowerband']
        df['bb_percent'] = (df['close'] - df['bb_lowerband']) / (df['bb_upperband'] - df['bb_lowerband'])
        
        # 计算EMA
        df['ema_short'] = ta.EMA(df, timeperiod=10)
        df['ema_medium'] = ta.EMA(df, timeperiod=50)
        df['ema_long'] = ta.EMA(df, timeperiod=200)
        
        # 计算ATR
        df['atr'] = ta.ATR(df, timeperiod=14)
        
        # 计算OBV
        df['obv'] = ta.OBV(df)
        
        # 计算Stochastic
        stoch = ta.STOCH(df)
        df['slowk'] = stoch['slowk']
        df['slowd'] = stoch['slowd']
        
        # 计算ADX
        adx = ta.ADX(df)
        df['adx'] = adx
        
        # 计算CCI
        df['cci'] = ta.CCI(df)
        
        # 计算MFI
        df['mfi'] = ta.MFI(df)
        
        # 计算Williams %R
        df['willr'] = ta.WILLR(df)
        
        # 计算ROC
        df['roc'] = ta.ROC(df)
        
        # 计算Awesome Oscillator
        df['ao'] = self.awesome_oscillator(df)
        
        # 计算Ichimoku云
        df = self.ichimoku(df)
        
        # 计算价格与移动平均线的距离
        df['dist_ema_short'] = (df['close'] - df['ema_short']) / df['close']
        df['dist_ema_medium'] = (df['close'] - df['ema_medium']) / df['close']
        df['dist_ema_long'] = (df['close'] - df['ema_long']) / df['close']
        
        # 计算交易量指标
        df['volume_mean'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_z'] = (df['volume'] - df['volume_mean']) / df['volume_std']
        df['volume_change'] = df['volume'].pct_change()
        
        # 计算价格动量
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        # 计算高低价差异
        df['high_low_diff'] = (df['high'] - df['low']) / df['close']
        
        # 计算日内位置
        df['intraday_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # 计算价格趋势
        df['price_trend'] = df['close'].diff(5)
        
        # 添加时间特征
        df['hour'] = pd.to_datetime(df['date']).dt.hour if 'date' in df.columns else 0
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek if 'date' in df.columns else 0
        
        # 生成模型预测
        if len(df) >= 30:  # 确保有足够的数据进行预测
            df = self._add_predictions(df)
        
        return df
    
    def awesome_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """计算Awesome Oscillator"""
        median_price = (df['high'] + df['low']) / 2
        ao = ta.SMA(median_price, timeperiod=5) - ta.SMA(median_price, timeperiod=34)
        return ao
    
    def ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算Ichimoku云指标"""
        # 转换线
        df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        
        # 基准线
        df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        
        # 先行带A
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # 先行带B
        df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
        
        # 滞后带
        df['chikou_span'] = df['close'].shift(-26)
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame, lookback: int = 30) -> torch.Tensor:
        """
        准备模型输入特征
        与ai_run_backtest.py中保持一致的特征处理
        """
        if len(df) < lookback:
            return None
        
        # 选择特征列 - 与模型训练时保持一致
        # 根据错误信息，模型期望11个特征，而不是42个
        feature_columns = [
            'close', 'volume',  # 基础价格和成交量
            'rsi', 'macd', 'macdsignal',  # 技术指标
            'bb_percent', 'ema_short',  # 布林带和均线
            'volatility', 'momentum',  # 波动率和动量
            'high_low_diff', 'intraday_position'  # 价格结构
        ]
        
        # 确保所有特征列都存在
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"特征列 {col} 不存在，将被跳过")
                feature_columns.remove(col)
        
        # 确保特征数量正确
        if len(feature_columns) != 11:
            logger.warning(f"特征数量不匹配：期望11个，实际{len(feature_columns)}个")
            # 如果特征不足，可以添加一些派生特征或重复特征
            while len(feature_columns) < 11:
                # 添加一些基本特征的派生版本
                if 'close' in df.columns and 'close_pct' not in feature_columns and len(feature_columns) < 11:
                    df['close_pct'] = df['close'].pct_change()
                    feature_columns.append('close_pct')
                elif 'volume' in df.columns and 'volume_pct' not in feature_columns and len(feature_columns) < 11:
                    df['volume_pct'] = df['volume'].pct_change()
                    feature_columns.append('volume_pct')
                else:
                    # 如果还是不够，复制一些已有特征
                    df[f'{feature_columns[0]}_copy'] = df[feature_columns[0]]
                    feature_columns.append(f'{feature_columns[0]}_copy')
        
        # 获取最近的数据
        recent_data = df[feature_columns].iloc[-lookback:].values
        
        # 处理缺失值
        recent_data = np.nan_to_num(recent_data, nan=0.0)
        
        # 归一化 - 与模型训练时保持一致的归一化方法
        mean = np.mean(recent_data, axis=0)
        std = np.std(recent_data, axis=0)
        std[std == 0] = 1  # 避免除以零
        normalized_data = (recent_data - mean) / std
        
        # 转换为张量
        features = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0)
        return features
    
    def _add_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加模型预测
        与ai_run_backtest.py中保持一致的预测逻辑
        """
        if self.model is None:
            # 如果模型未加载，使用简单的技术指标作为替代
            df['prediction'] = 0.0
            df.loc[df['rsi'] < 30, 'prediction'] = 0.5
            df.loc[df['rsi'] > 70, 'prediction'] = -0.5
            return df
        
        try:
            # 准备模型输入
            features = self._prepare_features(df)
            if features is None:
                logger.warning("数据不足，无法生成预测")
                df['prediction'] = 0.0
                return df
            
            # 使用模型预测
            with torch.no_grad():
                predictions = self.model(features).cpu().numpy().flatten()
            
            # 确保预测长度正确
            if len(predictions) < self.prediction_horizon:
                # 如果预测长度不足，填充为0
                predictions = np.pad(predictions, (0, self.prediction_horizon - len(predictions)))
            elif len(predictions) > self.prediction_horizon:
                # 如果预测长度过长，截断
                predictions = predictions[:self.prediction_horizon]
            
            # 计算短期、中期、长期预测
            short_term = np.mean(predictions[:5])  # 短期预测 (1-5个周期)
            medium_term = np.mean(predictions[5:10])  # 中期预测 (6-10个周期)
            long_term = np.mean(predictions[10:])  # 长期预测 (11+个周期)
            
            # 保存预测结果
            df['prediction_short'] = short_term
            df['prediction_medium'] = medium_term
            df['prediction_long'] = long_term
            
            # 计算综合信号
            signal = short_term * 0.5 + medium_term * 0.3 + long_term * 0.2
            
            # 应用平滑因子
            signal = self.smoothing_factor.value * signal + (1 - self.smoothing_factor.value) * self.last_signal
            self.last_signal = signal
            
            # 保存综合信号
            df['prediction'] = signal
            
            logger.debug(f"生成预测: 短期={short_term:.4f}, 中期={medium_term:.4f}, 长期={long_term:.4f}, 综合={signal:.4f}")
            
        except Exception as e:
            logger.error(f"预测生成失败: {e}")
            df['prediction'] = 0.0
        
        return df
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        生成买入信号
        与ai_run_backtest.py中保持一致的买入逻辑
        """
        df = dataframe.copy()
        
        # 初始化买入列
        df['enter_long'] = 0
        df['enter_short'] = 0
        
        if len(df) < 2:
            return df
        
        # 获取最新的预测和波动率
        signal = df['prediction'].iloc[-1]
        volatility = df['volatility'].iloc[-1]
        
        # 检查波动率
        if pd.isna(volatility) or volatility > self.volatility_threshold.value:
            logger.debug(f"波动率过高 ({volatility:.4f} > {self.volatility_threshold.value})，跳过交易")
            return df
        
        # 生成买入信号
        if signal > self.signal_threshold.value:
            # 多头信号
            df.loc[df.index[-1], 'enter_long'] = 1
            logger.debug(f"生成多头信号: 信号={signal:.4f}, 阈值={self.signal_threshold.value}")
            
            # 更新当前持仓
            self.current_position = self.position_limit.value
            
            # 设置止损止盈
            current_price = df['close'].iloc[-1]
            self.stop_loss_price = current_price * (1 - self.stop_loss_pct.value)
            self.take_profit_price = current_price * (1 + self.take_profit_pct.value)
            
        elif signal < -self.signal_threshold.value:
            # 空头信号
            df.loc[df.index[-1], 'enter_short'] = 1
            logger.debug(f"生成空头信号: 信号={signal:.4f}, 阈值={-self.signal_threshold.value}")
            
            # 更新当前持仓
            self.current_position = -self.position_limit.value
            
            # 设置止损止盈
            current_price = df['close'].iloc[-1]
            self.stop_loss_price = current_price * (1 + self.stop_loss_pct.value)
            self.take_profit_price = current_price * (1 - self.take_profit_pct.value)
        
        return df
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        生成卖出信号
        与ai_run_backtest.py中保持一致的卖出逻辑
        """
        df = dataframe.copy()
        
        # 初始化卖出列
        df['exit_long'] = 0
        df['exit_short'] = 0
        
        if len(df) < 2:
            return df
        
        # 获取最新的预测和价格
        signal = df['prediction'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # 检查止损止盈
        if self.current_position > 0 and self.stop_loss_price is not None:
            # 多头止损
            if current_price <= self.stop_loss_price:
                df.loc[df.index[-1], 'exit_long'] = 1
                logger.debug(f"触发多头止损: 价格={current_price:.4f}, 止损价={self.stop_loss_price:.4f}")
                self.current_position = 0
                self.stop_loss_price = None
                self.take_profit_price = None
                return df
            
            # 多头止盈
            if self.take_profit_price is not None and current_price >= self.take_profit_price:
                df.loc[df.index[-1], 'exit_long'] = 1
                logger.debug(f"触发多头止盈: 价格={current_price:.4f}, 止盈价={self.take_profit_price:.4f}")
                self.current_position = 0
                self.stop_loss_price = None
                self.take_profit_price = None
                return df
        
        elif self.current_position < 0 and self.stop_loss_price is not None:
            # 空头止损
            if current_price >= self.stop_loss_price:
                df.loc[df.index[-1], 'exit_short'] = 1
                logger.debug(f"触发空头止损: 价格={current_price:.4f}, 止损价={self.stop_loss_price:.4f}")
                self.current_position = 0
                self.stop_loss_price = None
                self.take_profit_price = None
                return df
            
            # 空头止盈
            if self.take_profit_price is not None and current_price <= self.take_profit_price:
                df.loc[df.index[-1], 'exit_short'] = 1
                logger.debug(f"触发空头止盈: 价格={current_price:.4f}, 止盈价={self.take_profit_price:.4f}")
                self.current_position = 0
                self.stop_loss_price = None
                self.take_profit_price = None
                return df
        
        # 信号反转退出
        if self.current_position > 0 and signal < -self.signal_threshold.value:
            # 多头持仓，但收到空头信号
            df.loc[df.index[-1], 'exit_long'] = 1
            logger.debug(f"信号反转退出多头: 信号={signal:.4f}")
            self.current_position = 0
        
        elif self.current_position < 0 and signal > self.signal_threshold.value:
            # 空头持仓，但收到多头信号
            df.loc[df.index[-1], 'exit_short'] = 1
            logger.debug(f"信号反转退出空头: 信号={signal:.4f}")
            self.current_position = 0
        
        return df
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           **kwargs) -> float:
        """
        自定义仓位大小
        实现动态仓位控制
        """
        # 获取当前信号强度
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 2:
            return proposed_stake
        
        signal = dataframe['prediction'].iloc[-1]
        
        # 计算信号强度
        if signal > self.signal_threshold.value:
            # 多头信号
            strength = min((signal - self.signal_threshold.value) / (1 - self.signal_threshold.value), 1)
            stake = proposed_stake * strength
        elif signal < -self.signal_threshold.value:
            # 空头信号
            strength = min((abs(signal) - self.signal_threshold.value) / (1 - self.signal_threshold.value), 1)
            stake = proposed_stake * strength
        else:
            # 中性信号
            stake = 0
        
        # 确保仓位在允许范围内
        stake = max(min_stake, min(stake, max_stake))
        
        return stake
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        """
        自定义杠杆大小
        默认不使用杠杆
        """
        return 1.0 