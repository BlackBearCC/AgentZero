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
        """获取历史K线和订单簿数据"""
        try:
            since = int(start.timestamp() * 1000) if start else None
            end_ts = int(end.timestamp() * 1000) if end else None
            
            self.logger.info(f"""
            ====== 开始获取数据 ======
            交易对: {symbol}
            时间周期: {timeframe}
            开始时间: {start} ({since})
            结束时间: {end} ({end_ts})
            API延迟限制: {self.exchange.rateLimit}ms
            ============================
            """)
            
            # 获取K线数据
            all_ohlcv = []
            current_since = since
            page_count = 0
            
            while True:
                page_count += 1
                self.logger.info(f"正在获取第 {page_count} 页K线数据...")
                
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                last_timestamp = ohlcv[-1][0]
                if last_timestamp >= end_ts or len(ohlcv) < 1000:
                    break
                    
                current_since = last_timestamp + 1
                time.sleep(self.exchange.rateLimit / 1000)
                
                self.logger.info(f"""
                当前进度:
                - 已获取数据点数: {len(all_ohlcv)}
                - 最新数据时间: {datetime.fromtimestamp(last_timestamp/1000)}
                """)
            
            # 转换为DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 获取订单簿数据（一次性获取当前状态）
            self.logger.info("正在获取订单簿数据...")
            try:
                orderbook = self.exchange.fetch_order_book(symbol, limit=20)  # 获取更深的订单簿数据
                
                if orderbook['bids'] and orderbook['asks']:
                    # 提取买卖盘数据
                    bids = pd.DataFrame(orderbook['bids'], columns=['price', 'volume'])
                    asks = pd.DataFrame(orderbook['asks'], columns=['price', 'volume'])
                    
                    # 计算订单簿特征
                    df['bid1'] = bids['price'].iloc[0]
                    df['ask1'] = asks['price'].iloc[0]
                    df['spread'] = df['ask1'] - df['bid1']
                    df['mid_price'] = (df['ask1'] + df['bid1']) / 2
                    
                    # 计算累计深度
                    df['bid_depth'] = bids['volume'].sum()
                    df['ask_depth'] = asks['volume'].sum()
                    df['imbalance'] = (df['bid_depth'] - df['ask_depth']) / (df['bid_depth'] + df['ask_depth'])
                    
                    # 计算加权平均价格
                    df['vwap_bid'] = (bids['price'] * bids['volume']).sum() / bids['volume'].sum()
                    df['vwap_ask'] = (asks['price'] * asks['volume']).sum() / asks['volume'].sum()
                
            except Exception as e:
                self.logger.warning(f"获取订单簿数据失败: {str(e)}")
                # 使用K线数据计算替代指标
                df['spread'] = df['high'] - df['low']
                df['mid_price'] = (df['high'] + df['low']) / 2
                df['imbalance'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            
            self.logger.info(f"""
            ====== 数据获取完成 ======
            时间范围: {df.index.min()} - {df.index.max()}
            数据点数: {len(df)}
            特征数量: {len(df.columns)}
            内存占用: {df.memory_usage().sum() / 1024 / 1024:.2f}MB
            ========================
            """)
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取数据失败: {str(e)}")
            raise

    def _get_data(self, 
                  symbol: str, 
                  timeframe: str,
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

    def get_data(self) -> Dict[str, Union[bt.feeds.PandasData, pd.DataFrame]]:
        """返回数据源"""
        return {
            'indicator': self.df,  # 直接返回DataFrame
            'execution': bt.feeds.PandasData(  # 保持backtrader格式用于回测
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