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
            
            # 检查是否存在缓存文件
            if cache_file.exists():
                self.logger.info(f"发现缓存文件: {cache_file}")
                cached_df = pd.read_parquet(cache_file)
                cached_df.index = pd.to_datetime(cached_df.index)
                
                # 检查缓存数据是否覆盖所需时间范围
                if not cached_df.empty:
                    cache_start = cached_df.index.min()
                    cache_end = cached_df.index.max()
                    
                    # 如果缓存完全覆盖所需范围，直接使用缓存
                    if cache_start <= start and cache_end >= end:
                        self.logger.info(f"使用缓存数据 - 范围: {cache_start} 到 {cache_end}")
                        return cached_df[(cached_df.index >= start) & (cached_df.index <= end)]
                    
                    # 如果需要补充数据
                    df_list = []
                    
                    # 获取开始日期之前的数据
                    if start < cache_start:
                        self.logger.info(f"获取早期数据: {start} 到 {cache_start}")
                        early_df = self._fetch_data(symbol, timeframe, start, cache_start)
                        if not early_df.empty:
                            df_list.append(early_df)
                    
                    # 添加缓存数据
                    cache_data = cached_df[(cached_df.index >= start) & (cached_df.index <= end)]
                    if not cache_data.empty:
                        df_list.append(cache_data)
                    
                    # 获取结束日期之后的数据
                    if end > cache_end:
                        self.logger.info(f"获取最新数据: {cache_end} 到 {end}")
                        late_df = self._fetch_data(symbol, timeframe, cache_end, end)
                        if not late_df.empty:
                            df_list.append(late_df)
                    
                    if df_list:
                        final_df = pd.concat(df_list, axis=0).sort_index()
                        final_df = final_df[~final_df.index.duplicated(keep='first')]
                        
                        # 更新缓存
                        final_df.to_parquet(cache_file)
                        self.logger.info(f"更新缓存文件: {cache_file}")
                        
                        return final_df
            
            # 如果没有缓存或缓存无效，获取新数据
            self.logger.info("获取新数据并创建缓存")
            df = self._fetch_data(symbol, timeframe, start, end)
            
            # 保存到缓存
            if not df.empty:
                df.to_parquet(cache_file)
                self.logger.info(f"创建缓存文件: {cache_file}")
            
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