import ccxt
import pandas as pd
import backtrader as bt
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Optional, Union
import logging
import time

class CCXTFeed:
    """CCXT数据源"""
    
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 打印调试信息
        self.logger.info(f"CCXTFeed初始化参数: {kwargs}")
        
        self.symbol = kwargs.get('symbol', 'BTC/USDT')
        self.timeframe = kwargs.get('timeframe', '1h')
        self.exchange_id = kwargs.get('exchange', 'binance')
        
        # 初始化交易所
        exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'timeout': 30000,
        })
        
        # 获取数据
        self.df = self._fetch_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start=kwargs.get('start'),
            end=kwargs.get('end')
        )
        
        # 打印DataFrame信息
        self.logger.info(f"DataFrame信息:\n{self.df.info()}")
        self.logger.info(f"DataFrame前5行:\n{self.df.head()}")

    def _fetch_data(self, 
                   symbol: str, 
                   timeframe: str, 
                   start: Optional[datetime] = None,
                   end: Optional[datetime] = None) -> pd.DataFrame:
        """获取历史数据"""
        try:
            # 确保时间范围有效
            now = datetime.now()
            end = min(end or now, now)
            start = start or (end - timedelta(days=30))
            
            since = int(start.timestamp() * 1000)
            
            for attempt in range(3):
                try:
                    self.logger.info(f"正在获取 {symbol} 的历史数据...")
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=since,
                        limit=1000
                    )
                    
                    if not ohlcv:
                        raise ValueError(f"未获取到 {symbol} 的数据")
                    
                    df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                    df.set_index('datetime', inplace=True)
                    
                    df = df[df.index <= end]
                    df = df[df.index >= start]
                    
                    if df.empty:
                        raise ValueError(f"过滤后的数据为空: {symbol}")
                    
                    self.logger.info(f"成功获取 {len(df)} 条数据")
                    return df
                    
                except Exception as e:
                    self.logger.warning(f"第 {attempt+1} 次获取失败: {str(e)}")
                    if attempt < 2:
                        time.sleep(1)
                    else:
                        raise
                        
        except Exception as e:
            self.logger.error(f"获取数据失败: {str(e)}")
            raise

    def get_data(self) -> bt.feeds.PandasData:
        """返回backtrader可用的数据源"""
        return bt.feeds.PandasData(
            dataname=self.df,
            datetime=None,  # 使用索引作为日期时间
            open=0,        # DataFrame中的列索引
            high=1,
            low=2,
            close=3,
            volume=4,
            openinterest=-1
        )

class DataManager:
    """数据管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._feeds = {}

    def get_feed(self, 
                symbol: str, 
                timeframe: str = '1d',
                exchange: str = 'binance',
                start: Optional[datetime] = None,
                end: Optional[datetime] = None) -> bt.feeds.PandasData:
        """获取数据源"""
        try:
            feed_key = f"{exchange}_{symbol}_{timeframe}"
            
            if feed_key not in self._feeds:
                feed = CCXTFeed(
                    symbol=symbol,
                    timeframe=timeframe,
                    exchange=exchange,
                    start=start,
                    end=end
                )
                self._feeds[feed_key] = feed.get_data()
                
            return self._feeds[feed_key]
            
        except Exception as e:
            self.logger.error(f"获取数据源失败: {str(e)}")
            raise 