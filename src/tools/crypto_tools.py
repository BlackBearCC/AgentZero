from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import asyncio
import aiohttp

class BaseCryptoTool(ABC):
    """加密货币工具基类"""
    
    _exchange = None  # 类级别的单例 exchange
    _markets_initialized = False
    
    def __init__(self):
        if not BaseCryptoTool._exchange:
            BaseCryptoTool._exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'defaultMarket': 'futures',
                    'fetchMarkets': True,
                    'warnOnFetchOHLCVLimitArgument': False,
                },
                'urls': {
                    'api': {
                        'public': 'https://fapi.binance.com/fapi/v1',
                        'private': 'https://fapi.binance.com/fapi/v1',
                        'fapiPublic': 'https://fapi.binance.com/fapi/v1',
                        'fapiPrivate': 'https://fapi.binance.com/fapi/v1',
                        'dapiPublic': 'https://dapi.binance.com/dapi/v1',
                        'dapiPrivate': 'https://dapi.binance.com/dapi/v1'
                    }
                }
            })
        self.exchange = BaseCryptoTool._exchange
        
        # 仅在开发环境启用详细日志
        self.exchange.verbose = False  # 关闭 CCXT 详细日志
        self.cache = {}
        self.cache_ttl = 300
        
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass
        
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """工具参数描述"""
        pass
        
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具功能"""
        if 'symbol' not in params:
            raise ValueError("缺少必需参数: symbol")
            
        # 统一处理交易对格式
        symbol = f"{params['symbol']}/USDT:USDT"
            
        # 验证交易对是否存在
        if symbol not in self.exchange.markets:
            raise ValueError(f"不支持的交易对: {symbol}")
            
        params['symbol'] = symbol
        return await self._run(params)
        
    @abstractmethod
    async def _run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """具体工具实现"""
        pass

    async def initialize(self):
        """初始化交易所连接和市场数据"""
        try:
            if not BaseCryptoTool._markets_initialized:
                self.exchange.options['defaultType'] = 'future'
                await self.exchange.load_markets(True)
                
                if not self.exchange.markets:
                    raise ValueError("无法加载合约市场数据")
                    
                BaseCryptoTool._markets_initialized = True
                
        except Exception as e:
            BaseCryptoTool._markets_initialized = False
            raise ValueError(f"初始化市场数据失败: {str(e)}")

    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'exchange'):
            await self.exchange.close()

# class MarketDataTool(BaseCryptoTool):
#     """市场数据工具"""
    
#     @property
#     def name(self) -> str:
#         return "market_data"
        
#     @property
#     def description(self) -> str:
#         return "获取市场深度数据"
        
#     @property
#     def parameters(self) -> Dict[str, Any]:
#         return {
#             "symbol": {
#                 "type": "string",
#                 "description": "交易对名称",
#                 "required": True
#             }
#         }
        
#     async def _run(self, params: Dict[str, Any]) -> Dict[str, Any]:
#         """获取市场数据"""
#         try:
#             # 使用统一的 API 方法获取订单簿数据
#             orderbook = await self.exchange.fetch_order_book(
#                 params['symbol'],
#                 limit=5  # 只获取前5档深度
#             )
            
#             if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
#                 raise ValueError("获取深度数据失败")
                
#             return {
#                 "bid": float(orderbook['bids'][0][0]),  # 最优买价
#                 "ask": float(orderbook['asks'][0][0]),  # 最优卖价
#                 "spread": float(orderbook['asks'][0][0]) - float(orderbook['bids'][0][0])  # 买卖价差
#             }
            
#         except Exception as e:
#             raise ValueError(f"获取 {params['symbol']} 深度数据失败")

class NewsAggregatorTool(BaseCryptoTool):
    """新闻聚合工具"""
    
    _news_cache = {}
    
    def __init__(self):
        super().__init__()
        self.crypto_panic_url = "https://cryptopanic.com/api/v1/posts/"
        self.crypto_panic_key = "7b9a7b637b6f6b394d8cf09c6619d52ea45f2cee"
        self.cache_ttl = 300  # 5分钟缓存
        
        # 初始化内置 LLM
        from src.llm.deepseek import DeepSeekLLM
        import os
        
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY not found")
            
        self.llm = DeepSeekLLM(
            model_name="deepseek-chat",
            temperature=0.3,  # 降低温度以获得更稳定的分析
            api_key=deepseek_api_key
        )
    
    @property
    def name(self) -> str:
        return "news"
        
    @property
    def description(self) -> str:
        return "获取加密货币相关新闻"
        
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "symbol": {
                "type": "string",
                "description": "交易对名称",
                "required": True
            },
            "limit": {
                "type": "integer",
                "description": "新闻条数",
                "default": 20,
                "required": True
            }
        }
        
    async def analyze_news(self, news_data: List[Dict[str, Any]]) -> str:
        """使用 LLM 分析新闻"""
        # 构建新闻时间线
        news_timeline = "\n".join([
            f"- [{item['published_at']}] {item['title']} (来源: {item['source']})"
            for item in sorted(news_data, key=lambda x: x['published_at'], reverse=True)
        ])
        
        prompt = f"""作为加密货币分析师，请分析以下最新新闻事件的市场影响。

    新闻时间线：
    {news_timeline}

    请按以下格式输出分析报告：

    ### 最新新闻概要
    [请列出3-5条最重要的新闻，用简短的一句话概括每条新闻的核心内容]

    ### 市场影响分析
    1. 核心事件分析
    - 最重要的2-3个新闻要点及其关联性
    - 事件影响链分析

    2. 市场影响评估
    - 市场情绪影响：[积极/中性/消极]
    - 可能的价格影响方向
    - 影响持续时间评估

    3. 风险与机会分析
    - 潜在市场风险
    - 值得关注的机会
    - 需要持续跟踪的指标

    请用专业、客观的语言进行分析，避免过度推测，注重数据支持。
    """
        try:
            response = await self.llm.agenerate([[
                {
                    "role": "system", 
                    "content": """你是一位专业的加密货币市场分析师，擅长新闻分析和事件影响评估。
                    - 分析始终基于事实，注重客观性和谨慎性
                    - 使用清晰的结构化格式输出
                    - 重点突出关键信息和实际影响
                    - 避免过度推测和主观判断"""
                },
                {"role": "user", "content": prompt}
            ]])
            return response.generations[0][0].text
        except Exception as e:
            return f"新闻分析失败: {str(e)}"
            
    async def _run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取并分析新闻数据"""
        symbol = params['symbol']
        limit = params.get('limit', 20)
        
        # 检查缓存
        cache_key = f"{symbol}_{limit}"
        if cache_key in self._news_cache:
            cached_data, timestamp = self._news_cache[cache_key]
            if datetime.now().timestamp() - timestamp < self.cache_ttl:
                return cached_data
                
        if not self.crypto_panic_key:
            raise ValueError("未正确配置新闻 API key")
            
        # 获取新闻数据
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    api_params = {
                        "auth_token": self.crypto_panic_key,
                        "currencies": symbol.split("/")[0],
                        "kind": "news",
                        "limit": limit
                    }
                    async with session.get(self.crypto_panic_url, params=api_params) as response:
                        if response.status != 200:
                            raise ValueError(f"新闻 API 请求失败: {await response.text()}")
                            
                        data = await response.json()
                        if not data.get("results"):
                            raise ValueError(f"未找到 {symbol} 相关新闻")
                            
                        news_list = [
                            {
                                "title": item["title"],
                                "url": item["url"],
                                "source": item["source"]["title"],
                                "published_at": item["published_at"]
                            }
                            for item in data["results"][:limit]
                        ]
                        
                        # 使用 LLM 分析新闻
                        analysis = await self.analyze_news(news_list)
                        
                        result = {
                            "news": news_list,
                            "analysis": analysis  # 添加 LLM 分析结果
                        }
                        
                        # 更新缓存
                        self._news_cache[cache_key] = (result, datetime.now().timestamp())
                        return result
                        
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)

class TechnicalAnalysisTool(BaseCryptoTool):
    """技术分析工具"""
    
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.cache = {}  # 添加缓存
        self.cache_ttl = 300  # 缓存时间5分钟
        
    @property
    def name(self) -> str:
        return "technical"
        
    @property
    def description(self) -> str:
        return "进行技术指标分析"
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "symbol": {
                "type": "string",
                "description": "交易对名称，如 BTC、ETH 等",
                "required": True
            },
            "timeframe": {
                "type": "string",
                "description": "时间周期，如 1m、5m、1h、4h、1d 等",
                "default": "1h",
                "required": False
            },
            "limit": {
                "type": "integer",
                "description": "获取的数据点数量",
                "default": 100,
                "required": False
            },
            "indicators": {
                "type": "array",
                "description": "需要计算的指标列表，如 ['ma', 'rsi', 'macd']",
                "default": ["ma", "rsi", "macd"],
                "required": False
            }
        }        
    async def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """获取并处理K线数据"""
        try:
            cache_key = f"{symbol}_{timeframe}_{limit}"
            current_time = datetime.now().timestamp()
            
            # 检查缓存
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if current_time - timestamp < self.cache_ttl:
                    return cached_data
                    
            # 获取K线数据
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe,
                limit=limit,
                params={
                    'limit': limit,
                    'warnOnFetchOHLCVLimitArgument': False
                }
            )
            
            if not ohlcv:
                raise ValueError(f"无法获取 {symbol} 的K线数据")
                
            # 转换为DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 更新缓存
            self.cache[cache_key] = (df, current_time)
            
            return df
            
        except Exception as e:
            raise ValueError(f"获取K线数据失败: {str(e)}")
            
    async def _run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """进行技术分析"""
        if not self.initialized:
            await self.initialize()
            self.initialized = True
            
        symbol = params['symbol']
        timeframe = params.get("timeframe", "1h")
        limit = params.get("limit", 100)
        
        # 使用新的获取数据方法
        df = await self._fetch_ohlcv(symbol, timeframe, limit)
        
        # 计算指标
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.sma(length=7, append=True)
        df.ta.sma(length=25, append=True)
        df.ta.sma(length=99, append=True)
        
        latest = df.iloc[-1]
        
        return {
            "ma": {
                "ma7": float(latest['SMA_7']),
                "ma25": float(latest['SMA_25']),
                "ma99": float(latest['SMA_99'])
            },
            "rsi": float(latest['RSI_14']),
            "macd": {
                "macd": float(latest['MACD_12_26_9']),
                "signal": float(latest['MACDs_12_26_9']),
                "hist": float(latest['MACDh_12_26_9'])
            }
        } 