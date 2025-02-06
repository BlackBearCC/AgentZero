import ccxt
import pandas as pd
import backtrader as bt
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Optional, Union
import logging
import time
from pathlib import Path
from src.utils.logger import Logger

class CCXTFeed:
    """CCXT数据源"""
    
    def __init__(self, **kwargs):
        # 使用统一的日志管理
        self.logger = Logger()
        
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
        self.logger.info(f"DataFrame最后5行:\n{self.df.tail()}")

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
            
            # 打印关键时间信息
            self.logger.info(f"""
            ====== 数据获取参数 ======
            当前时间: {now}
            请求开始: {start}
            请求结束: {end}
            时间戳开始: {int(start.timestamp() * 1000)}
            时间戳结束: {int(end.timestamp() * 1000)}
            交易对: {symbol}
            时间周期: {timeframe}
            ========================
            """)
            
            since = int(start.timestamp() * 1000)
            all_data = []
            
            # 分批获取数据
            current_since = since
            while current_since < end.timestamp() * 1000:
                try:
                    self.logger.info(f"获取数据批次 - 开始时间: {datetime.fromtimestamp(current_since/1000)}")
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_since,
                        limit=1000
                    )
                    
                    if not ohlcv:
                        self.logger.warning(f"未获取到数据 - since: {datetime.fromtimestamp(current_since/1000)}")
                        break
                        
                    # 打印每批数据的时间范围
                    batch_start = datetime.fromtimestamp(ohlcv[0][0]/1000)
                    batch_end = datetime.fromtimestamp(ohlcv[-1][0]/1000)
                    self.logger.info(f"获取到数据 - 从 {batch_start} 到 {batch_end}, 条数: {len(ohlcv)}")
                    
                    all_data.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1
                    
                    if current_since >= end.timestamp() * 1000:
                        break
                        
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                except Exception as e:
                    self.logger.error(f"获取数据失败: {str(e)}")
                    break
            
            if not all_data:
                raise ValueError(f"未获取到 {symbol} 的数据")
            
            # 转换为DataFrame
            df = pd.DataFrame(all_data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # 过滤时间范围
            df = df[df.index <= end]
            df = df[df.index >= start]
            
            # 打印最终数据信息
            self.logger.info(f"""
            ====== 最终数据信息 ======
            数据条数: {len(df)}
            时间范围: {df.index[0]} 到 {df.index[-1]}
            时间间隔: {timeframe}
            ========================
            """)
            
            return df
            
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
        self.logger = Logger()
        self._feeds = {}

    def get_feed(self,
                symbol: str,
                timeframe: str,
                start: datetime,
                end: datetime) -> bt.feeds.PandasData:
        """获取回测数据"""
        try:
            # 确保时间参数正确
            if start is None or end is None:
                raise ValueError("开始和结束时间不能为空")
            
            if start >= end:
                raise ValueError("开始时间必须早于结束时间")
                
            # 调整时区（如果需要）
            start = pd.Timestamp(start).tz_localize(None)
            end = pd.Timestamp(end).tz_localize(None)
            
            self.logger.info(f"加载数据 - 交易对: {symbol}, "
                           f"时间周期: {timeframe}, "
                           f"开始: {start}, "
                           f"结束: {end}")
            
            # 创建CCXT数据源
            feed = CCXTFeed(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end
            )
            
            # 获取DataFrame数据
            df = feed.df
            
            if df.empty:
                raise ValueError(f"获取到的数据为空: {symbol}")
            
            # 创建 backtrader 数据源
            data = bt.feeds.PandasData(
                dataname=df,
                datetime=None,  # 使用索引作为日期时间
                open=0,        # DataFrame中的列索引
                high=1,
                low=2,
                close=3,
                volume=4,
                openinterest=-1
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"数据加载错误: {str(e)}")
            raise 