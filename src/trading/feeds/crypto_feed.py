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
    
    def __init__(self, symbol: str, timeframe: str, start: datetime, end: datetime):
        self.logger = Logger()
        self._symbol = symbol
        self._timeframe = timeframe
        self._start = start
        self._end = end
        
        # 初始化交易所
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # 使用合约市场
            }
        })
        
        # 设置缓存目录
        self.cache_dir = Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 添加1分钟级别数据
        self._df_1m = None  # 1分钟数据用于订单执行
        self.df = None      # 原始timeframe数据用于指标计算
        
        self._load_data()

    def _load_data(self):
        try:
            # 加载原始timeframe数据用于指标计算
            self.df = self._get_data(
                symbol=self._symbol,
                timeframe=self._timeframe,
                start=self._start,
                end=self._end
            )
            
            # 同时加载1分钟数据用于订单执行
            if self._timeframe != '1m':
                self._df_1m = self._get_data(
                    symbol=self._symbol,
                    timeframe='1m',
                    start=self._start,
                    end=self._end
                )
            else:
                self._df_1m = self.df

        except Exception as e:
            self.logger.error(f"获取数据错误: {str(e)}")
            raise

    def _fetch_data(self, 
                   symbol: str, 
                   timeframe: str, 
                   start: Optional[datetime] = None,
                   end: Optional[datetime] = None) -> pd.DataFrame:
        """获取历史数据"""
        try:
            since = int(start.timestamp() * 1000) if start else None
            end_ts = int(end.timestamp() * 1000) if end else None
            
            self.logger.info(f"""
            ====== 开始获取数据 ======
            交易对: {symbol}
            时间周期: {timeframe}
            开始时间: {start} ({since})
            结束时间: {end} ({end_ts})
            ========================
            """)
            
            all_data = []
            current_since = since
            
            while True:
                # 获取数据
                self.logger.info(f"获取数据片段 - since: {current_since}")
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000
                )
                
                if not ohlcv:
                    self.logger.warning(f"未获取到数据 - since: {current_since}")
                    break
                    
                # 添加到列表
                all_data.extend(ohlcv)
                
                # 更新时间戳
                last_timestamp = ohlcv[-1][0]
                
                # 检查是否达到结束时间
                if end_ts and last_timestamp >= end_ts:
                    self.logger.info(f"达到结束时间: {last_timestamp} >= {end_ts}")
                    break
                    
                # 如果获取的数据少于1000条，说明已经没有更多数据
                if len(ohlcv) < 1000:
                    self.logger.info("获取的数据少于1000条，结束获取")
                    break
                    
                current_since = last_timestamp + 1
                self.logger.info(f"更新since为: {current_since}")
                
                # 添加延时避免超过频率限制
                time.sleep(self.exchange.rateLimit / 1000)
            
            if not all_data:
                raise ValueError(f"未获取到 {symbol} 的数据")
            
            # 转换为DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()  # 确保数据按时间排序
            
            # 确保数据在时间范围内
            df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
            
            self.logger.info(f"""
            ====== 数据获取完成 ======
            数据点数: {len(df)}
            数据范围: {df.index.min()} - {df.index.max()}
            ========================
            """)
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取数据失败: {str(e)}")
            raise

    def _get_data(self, symbol: str, timeframe: str, 
                  start: Optional[datetime] = None,
                  end: Optional[datetime] = None) -> pd.DataFrame:
        """获取数据，优先使用缓存"""
        try:
            # 生成缓存文件名
            cache_file = self.cache_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
            
            # 确保时间范围有效
            now = datetime.now()
            end = min(end or now, now)
            start = start or (end - timedelta(days=30))
            
            self.logger.info(f"""
                ====== 数据获取参数 ======
                缓存文件: {cache_file}
                交易对: {symbol}
                时间周期: {timeframe}
                开始时间: {start}
                结束时间: {end}
                ========================
            """)
            
            # 检查是否存在缓存文件
            if cache_file.exists():
                self.logger.info(f"发现缓存文件: {cache_file}")
                cached_df = pd.read_parquet(cache_file)
                cached_df.index = pd.to_datetime(cached_df.index)
                
                # 检查缓存数据是否覆盖所需时间范围
                if not cached_df.empty:
                    cache_start = cached_df.index.min()
                    cache_end = cached_df.index.max()
                    
                    self.logger.info(f"缓存数据范围: {cache_start} 到 {cache_end}")
                    
                    # 如果缓存完全覆盖所需范围，直接使用缓存
                    if cache_start <= start and cache_end >= end:
                        self.logger.info("缓存数据完全覆盖所需范围，使用缓存数据")
                        return cached_df[(cached_df.index >= start) & (cached_df.index <= end)]
                    else:
                        self.logger.info("缓存数据不完全覆盖所需范围，需要重新获取数据")
                        # 删除缓存文件
                        cache_file.unlink()
            
            # 如果没有缓存或缓存无效，获取新数据
            self.logger.info("开始获取新数据")
            df = self._fetch_data(symbol, timeframe, start, end)
            
            # 检查获取的数据
            if df.empty:
                raise ValueError(f"获取到的数据为空")
            
            self.logger.info(f"获取到数据范围: {df.index.min()} 到 {df.index.max()}")
            
            # 保存到缓存
            df.to_parquet(cache_file)
            self.logger.info(f"保存数据到缓存: {cache_file}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取数据错误: {str(e)}")
            raise

    def get_data(self) -> Dict[str, bt.feeds.PandasData]:
        """返回两个数据源"""
        return {
            'indicator': bt.feeds.PandasData(  # 用于指标计算的数据
                dataname=self.df,
                datetime=None,
                open=0,
                high=1,
                low=2,
                close=3,
                volume=4,
                openinterest=-1
            ),
            'execution': bt.feeds.PandasData(  # 用于订单执行的数据
                dataname=self._df_1m,
                datetime=None,
                open=0,
                high=1,
                low=2,
                close=3,
                volume=4,
                openinterest=-1
            )
        }

class DataManager:
    """数据管理器"""
    
    def __init__(self):
        self.logger = Logger()
        self._feeds = {}

    def get_feed(self,
                symbol: str,
                timeframe: str,
                start: datetime,
                end: datetime) -> Dict[str, bt.feeds.PandasData]:
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
            
            # 获取两个数据源
            return feed.get_data()
            
        except Exception as e:
            self.logger.error(f"数据加载错误: {str(e)}")
            raise 