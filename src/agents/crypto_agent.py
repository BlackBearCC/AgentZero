from typing import Dict, Any, List, Optional, AsyncIterator
from src.agents.base_agent import BaseAgent
from src.tools.crypto_tools import (
    NewsAggregatorTool,
    TechnicalAnalysisTool
)

import json
from datetime import datetime

class CryptoAgent(BaseAgent):
    """加密货币分析 Agent"""
    
    def __init__(self, config: Dict[str, Any], llm=None, memory_llm=None, tools=None):
        """初始化加密货币 Agent
        
        Args:
            config: 配置信息
            llm: 语言模型实例
            memory_llm: 记忆系统语言模型
            tools: 自定义工具列表
        """
        # 初始化默认工具集
        default_tools = [
            NewsAggregatorTool(),  # 新闻聚合
            TechnicalAnalysisTool()  # 技术分析
        ]
        
        # 合并自定义工具
        if tools:
            default_tools.extend(tools)
            
        super().__init__(config, llm, memory_llm, tools=default_tools)
        
        # 初始化市场数据缓存
        self.market_cache = {}
        self.cache_ttl = config.get("cache_ttl", 300)  # 缓存时效,默认5分钟
        
    async def _build_context(self, input_text: str, remark: str = '') -> Dict[str, Any]:
        """构建上下文信息
        
        Args:
            input_text: 用户输入文本
            remark: 备注信息
            
        Returns:
            包含消息历史和分析上下文的字典
        """
        # 获取基础上下文（包含历史消息等）
        context = await super()._build_context(input_text, remark)
        
        # 添加加密货币分析相关的上下文
        context.update({
            "market_cache": self.market_cache,
            "cache_ttl": self.cache_ttl,
            "tools": {
                tool.name: tool.description 
                for tool in self.tools
            }
        })
        
        return context

    async def load_prompt(self) -> str:
        """加载角色提示词"""
        base_prompt = self.config.get("system_prompt", "")
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in self.tools
        ])
        
        return f"""{base_prompt}

可用工具:
{tool_descriptions}

分析步骤:
1. 收集相关市场数据和新闻
2. 进行技术分析
3. 综合分析并给出建议
"""

    async def update_prompt(self, **kwargs) -> str:
        """更新角色提示词"""
        if kwargs.get("system_prompt"):
            self.config["system_prompt"] = kwargs["system_prompt"]
        return await self.load_prompt()

    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析用户意图并决定使用哪些工具"""
        self._logger.debug(f"[CryptoAgent] 开始思考分析，上下文: {context}")
        
        try:
            # 1. 构建工具信息
            tools_info = []
            self._logger.debug("[CryptoAgent] 正在构建工具信息...")
            for tool in self.tools:
                tool_info = (
                    f"- {tool.name}: {tool.description}\n"
                    f"  参数:\n" + 
                    "\n".join(
                        f"    {name}: {param['description']} "
                        f"(类型: {param['type']}, "
                        f"必需: {param['required']}, "
                        f"默认值: {param.get('default', 'None')})"
                        for name, param in tool.parameters.items()
                    )
                )
                tools_info.append(tool_info)
            
            # 加载并填充思考提示词
            think_prompt = await self._load_prompt_template("crypto-think")
            think_prompt = think_prompt.replace(
                "{{tools_info}}", 
                "\n\n".join(tools_info)
            )
            
            # 3. 调用 LLM 分析
            self._logger.debug("[CryptoAgent] 正在调用 LLM 分析...")
            response = await self.llm.agenerate([[
                {
                    "role": "system",
                    "content": think_prompt
                },
                {
                    "role": "user",
                    "content": context["input_text"]
                }
            ]])
            
            # 4. 提取并解析 JSON
            text = response.generations[0][0].text
            self._logger.debug(f"[CryptoAgent] LLM 返回结果: {text}")
            
            # 移除可能的 markdown 代码块标记
            text = text.replace("```json", "").replace("```", "").strip()
            
            try:
                tools_config = json.loads(text)
                self._logger.debug(f"[CryptoAgent] 解析后的工具配置: {tools_config}")
            except json.JSONDecodeError as e:
                self._logger.error(f"[CryptoAgent] JSON 解析失败: {text}")
                raise ValueError(f"无效的工具配置: {str(e)}")
            
            # 5. 验证工具配置
            self._logger.debug("[CryptoAgent] 正在验证工具配置...")
            validated_config = self._validate_tools_config(tools_config)
            self._logger.debug(f"[CryptoAgent] 验证后的工具配置: {validated_config}")
            
            return validated_config
                
        except Exception as e:
            self._logger.error(f"[CryptoAgent] 思考过程出错: {str(e)}")
            raise

    def _validate_tools_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证工具配置"""
        validated = {"tools": []}
        available_tools = {tool.name: tool for tool in self.tools}
        
        for tool_config in config.get("tools", []):
            tool_name = tool_config.get("name")
            if tool_name not in available_tools:
                continue
            
            tool = available_tools[tool_name]
            params = tool_config.get("params", {})
            
            # 验证并填充默认参数
            validated_params = {}
            for param_name, param_spec in tool.parameters.items():
                if param_name in params:
                    validated_params[param_name] = params[param_name]
                elif param_spec.get("required", False):
                    raise ValueError(f"工具 {tool_name} 缺少必需参数 {param_name}")
                elif "default" in param_spec:
                    validated_params[param_name] = param_spec["default"]
                
            validated["tools"].append({
                "name": tool_name,
                "params": validated_params
            })
        
        return validated

    async def _load_prompt_template(self, template_name: str) -> str:
        """加载提示词模板"""
        try:
            template_path = f"src/prompts/system/{template_name}.txt"
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self._logger.error(f"加载提示词模板失败: {str(e)}")
            raise

    async def _analyze_market(self, tools_config: Dict[str, Any]) -> Dict[str, Any]:
        """根据工具配置分析市场数据"""
        self._logger.debug(f"[CryptoAgent] 开始分析市场数据，工具配置: {tools_config}")
        
        results = {
            "data": {},
            "status": "success",
            "message": ""
        }
        
        try:
            available_tools = {tool.name: tool for tool in self.tools}
            self._logger.debug(f"[CryptoAgent] 可用工具: {list(available_tools.keys())}")
            
            for tool_config in tools_config["tools"]:
                tool_name = tool_config["name"]
                params = tool_config["params"]
                symbol = params["symbol"]  # 先获取 symbol
                
                if tool_name not in available_tools:
                    self._logger.warning(f"[CryptoAgent] 工具 {tool_name} 不可用，跳过")
                    continue
                    
                tool = available_tools[tool_name]
                self._logger.debug(f"[CryptoAgent] 正在执行工具 {tool_name}，参数: {params}")
                
                try:
                    async with tool:
                        result = await tool.run(params)
                        self._logger.debug(f"[CryptoAgent] 工具 {tool_name} 执行结果: {result}")
                        
                        # 按交易对组织数据
                        if symbol not in results["data"]:
                            results["data"][symbol] = {}
                        
                        results["data"][symbol][tool_name] = result
                        
                except Exception as e:
                    self._logger.error(f"[CryptoAgent] 工具 {tool_name} 执行失败: {str(e)}")
                    if symbol not in results["data"]:
                        results["data"][symbol] = {}
                    results["data"][symbol][tool_name] = {
                        "error": str(e)
                    }
                    results["status"] = "partial_error"
                    results["message"] += f"{symbol} {tool_name}: {str(e)}; "
                    
        except Exception as e:
            self._logger.error(f"[CryptoAgent] 市场分析失败: {str(e)}")
            results["status"] = "error"
            results["message"] = f"分析失败: {str(e)}"
            
        self._logger.debug(f"[CryptoAgent] 市场分析完成，结果: {results}")
        return results

    async def generate_response(self, input_text: str, remark: str = "") -> str:
        """生成市场分析回复"""
        self._logger.debug(f"[CryptoAgent] 开始生成回复，输入: {input_text}")
        
        try:
            # 1. 构建基础上下文
            self._logger.debug("[CryptoAgent] 正在构建上下文...")
            context = await self._build_context(input_text, remark)
            
            # 2. 分析用户意图，确定需要使用的工具
            self._logger.debug("[CryptoAgent] 正在分析用户意图...")
            tools_config = await self.think(context)
            self._logger.debug(f"[CryptoAgent] 工具配置: {tools_config}")
            
            # 3. 调用工具获取数据
            self._logger.debug("[CryptoAgent] 正在获取市场数据...")
            market_data = await self._analyze_market(tools_config)
            self._logger.debug(f"[CryptoAgent] 市场数据: {market_data}")
            
            # 4. 格式化市场数据为易读格式
            self._logger.debug("[CryptoAgent] 正在格式化市场数据...")
            formatted_data = self._format_market_data(market_data)
            
            # 5. 更新系统提示词，插入市场数据
            self._logger.debug("[CryptoAgent] 正在更新系统提示词...")
            system_prompt = self.config["system_prompt"].replace(
                "{{market_data}}", 
                formatted_data
            )
            
            # 6. 生成回复
            self._logger.debug("[CryptoAgent] 正在生成回复...")
            response = await self.llm.agenerate([[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ]])
            
            result = response.generations[0][0].text
            self._logger.debug(f"[CryptoAgent] 生成回复完成: {result}")
            
            # 7. 保存交互记录
            self._logger.debug("[CryptoAgent] 正在保存交互记录...")
            await self._save_interaction(input_text, result, {
                **context,
                "tools_config": tools_config,
                "market_data": market_data
            })
            
            return result
            
        except Exception as e:
            self._logger.error(f"[CryptoAgent] 生成回复失败: {str(e)}")
            raise

    def _format_market_data(self, market_data: Dict[str, Any]) -> str:
        """格式化市场数据为易读格式"""
        try:
            self._logger.debug(f"[CryptoAgent] 开始格式化市场数据: {market_data}")
            formatted = []
            
            for symbol, data in market_data.get("data", {}).items():
                self._logger.debug(f"[CryptoAgent] 处理 {symbol} 的数据: {data}")
                section = [f"## {symbol} 市场数据"]
                
                # 处理每个工具的数据
                for tool_name, tool_data in data.items():
                    self._logger.debug(f"[CryptoAgent] {symbol} 的 {tool_name} 工具数据: {tool_data}")
                    
                    if "error" in tool_data:
                        error_msg = f"{tool_name} 数据获取失败: {tool_data['error']}"
                        self._logger.debug(f"[CryptoAgent] {symbol} 工具执行错误: {error_msg}")
                        section.append(error_msg)
                        continue
                        
                    if tool_name == "price_tracker":
                        section.append(f"当前价格: ${tool_data['price']:,.2f}")
                        section.append(f"24h 涨跌幅: {tool_data['change_24h']}%")
                        section.append(f"24h 成交量: ${tool_data['volume_24h']:,.2f}")
                        
                    elif tool_name == "market_data":
                        section.append(f"最佳买价: ${tool_data['bid']:,.2f}")
                        section.append(f"最佳卖价: ${tool_data['ask']:,.2f}")
                        section.append(f"买卖价差: ${tool_data['spread']:,.2f}")
                        
                    elif tool_name == "technical":
                        section.append("\n技术指标:")
                        section.append(f"RSI(14): {tool_data['rsi']:.2f}")
                        ma_data = tool_data['ma']
                        section.append(f"MA7/25/99: ${ma_data['ma7']:,.2f} / ${ma_data['ma25']:,.2f} / ${ma_data['ma99']:,.2f}")
                        macd_data = tool_data['macd']
                        section.append(f"MACD: {macd_data['macd']:.2f} (Signal: {macd_data['signal']:.2f}, Hist: {macd_data['hist']:.2f})")
                        
                    elif tool_name == "news" and tool_data.get("news"):
                        section.append("\n最新新闻:")
                        for news in tool_data["news"][:3]:
                            section.append(f"- {news['title']}")
                            
                formatted_section = "\n".join(section)
                self._logger.debug(f"[CryptoAgent] {symbol} 格式化结果:\n{formatted_section}")
                formatted.append(formatted_section)
                
            final_result = "\n\n".join(formatted) if formatted else "无可用市场数据"
            self._logger.debug(f"[CryptoAgent] 最终格式化结果:\n{final_result}")
            return final_result
            
        except Exception as e:
            self._logger.error(f"[CryptoAgent] 格式化市场数据失败: {str(e)}")
            return "数据格式化失败"

    async def astream_response(self, input_text: str, remark: str = "") -> AsyncIterator[str]:
        """流式生成市场分析回复"""
        try:
            # 1. 构建基础上下文
            context = await self._build_context(input_text, remark)
            
            # 2. 分析用户意图，确定需要使用的工具
            tools_config = await self.think(context)
            
            # 3. 调用工具获取数据
            market_data = await self._analyze_market(tools_config)
            
            # 4. 格式化市场数据
            formatted_data = self._format_market_data(market_data)
            
            # 5. 更新系统提示词
            system_prompt = self.config["system_prompt"].replace(
                "{{market_data}}", 
                formatted_data
            )
            
            # 6. 流式生成回复
            async for chunk in self.llm.astream([
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ]):
                yield chunk.text
            
            # 7. 保存交互记录
            await self._save_interaction(input_text, "流式回复", {
                **context,
                "tools_config": tools_config,
                "market_data": market_data
            })
            
        except Exception as e:
            self._logger.error(f"生成回复失败: {str(e)}")
            raise 

    async def _save_interaction(self, input_text: str, response: str, context: Dict[str, Any]) -> None:
        """保存交互记录
        
        Args:
            input_text: 用户输入
            response: AI 回复
            context: 上下文信息
        """
        try:
            # 构建交互记录
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "input": input_text,
                "response": response,
                "context": context
            }
            
            # 如果需要持久化存储，可以在这里添加存储逻辑
            self._logger.debug(f"保存交互记录: {interaction}")
            
        except Exception as e:
            self._logger.error(f"保存交互记录失败: {str(e)}")
            # 不抛出异常，因为这是非关键操作 