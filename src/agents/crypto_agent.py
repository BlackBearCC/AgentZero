import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator, Union
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
            
        super().__init__(config, llm, memory_llm, tools=tools)
        
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
        })
        
        return context

    async def load_prompt(self) -> str:
        """加载角色提示词"""
        base_prompt = self.config.get("system_prompt", "")
        
        return base_prompt

    async def update_prompt(self, **kwargs) -> str:
        """更新角色提示词"""
        if kwargs.get("system_prompt"):
            self.config["system_prompt"] = kwargs["system_prompt"]
        return await self.load_prompt()

    async def think(self, context: Dict[str, Any]) -> Union[Dict[str, Any], str]:
        """分析用户意图并决定是否使用工具"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # 1. 构建工具信息为标准 JSON 格式
                tools_info = {
                    "available_tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": {
                                name: {
                                    "description": param["description"],
                                    "type": param["type"],
                                    "required": param["required"],
                                    "default": param.get("default", None)
                                }
                                for name, param in tool.parameters.items()
                            }
                        }
                        for tool in self.tools
                    ]
                }
                
                # 转换为格式化的 JSON 字符串
                formatted_tools = json.dumps(tools_info, indent=2, ensure_ascii=False)
                
                # 2. 加载并填充提示词
                system_prompt = await self._load_prompt_template("crypto-analyst")
                system_prompt = system_prompt.replace("{{tools_info}}", formatted_tools)
                
                self._logger.debug(f"[CryptoAgent] 工具信息:\n{formatted_tools}")
                
                # 3. 调用 LLM 分析
                response = await self.llm.agenerate([[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": context["input_text"]
                    }
                ]])
                
                text = response.generations[0][0].text.strip()
                self._logger.debug(f"[CryptoAgent] LLM 返回结果: {text}")
                
                # 4. 尝试解析为 JSON
                try:
                    # 提取非 JSON 部分（如果有）
                    pre_tool_message = ""
                    json_text = text
                    
                    # 如果包含 JSON 格式的工具调用
                    if "```json" in text:
                        parts = text.split("```json")
                        pre_tool_message = parts[0].strip()
                        json_text = parts[1].split("```")[0].strip()
                    elif text.find("{") > 0:  # JSON 前有文本
                        json_start = text.find("{")
                        pre_tool_message = text[:json_start].strip()
                        json_text = text[json_start:]
                    
                    # 如果文本中包含 JSON
                    if "{" in json_text and "}" in json_text:
                        # 提取 JSON 部分
                        json_start = json_text.find("{")
                        json_end = json_text.rfind("}") + 1
                        json_text = json_text[json_start:json_end]
                        
                        # 解析 JSON
                        tools_config = json.loads(json_text)
                        validated_config = self._validate_tools_config(tools_config)
                        
                        # 添加工具调用前的消息
                        if pre_tool_message:
                            validated_config["pre_tool_message"] = pre_tool_message
                        else:
                            # 如果没有前置消息，生成一个默认的
                            symbol = context.get("symbol", "加密货币")
                            validated_config["pre_tool_message"] = f"让我来帮您分析 {symbol} 的市场情况~"
                            
                        return validated_config
                    else:
                        # 直接返回文本回答
                        return text
                        
                except json.JSONDecodeError as e:
                    if attempt == max_retries - 1:
                        return text
                    self._logger.warning(f"[CryptoAgent] JSON 解析失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    await asyncio.sleep(1)
                    continue
                
            except Exception as e:
                if attempt == max_retries - 1:
                    self._logger.error(f"[CryptoAgent] 思考过程出错: {str(e)}")
                    raise
                self._logger.warning(f"[CryptoAgent] 思考失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                await asyncio.sleep(1)  # 添加延迟
                continue
        
        raise ValueError("达到最大重试次数，思考过程失败")

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

    async def generate_response(self, input_text: str, remark: str = "") -> AsyncIterator[Dict[str, Any]]:
        """生成市场分析回复，支持详细的阶段性返回"""
        self._logger.debug(f"[CryptoAgent] 开始生成回复，输入: {input_text}")
        
        try:
            # 1. 构建基础上下文
            context = await self._build_context(input_text, remark)
            
            # 2. 分析用户意图，确定需要使用的工具
            yield {
                "stage": "think_start",
                "message": "正在思考如何分析您的问题..."
            }
            
            think_result = await self.think(context)
            self._logger.debug(f"[CryptoAgent] 思考结果: {think_result}")
            
            if isinstance(think_result, dict):
                # 返回思考阶段结果，包含前置消息
                pre_tool_message = think_result.get("pre_tool_message", "")
                tools_info = [
                    {
                        "name": tool["name"],
                        "params": tool["params"]
                    }
                    for tool in think_result.get("tools", [])
                ]
                
                # 先返回 AI 的前置消息
                if pre_tool_message:
                    yield {
                        "stage": "pre_tool_message",
                        "message": pre_tool_message
                    }
                
                # 然后返回工具使用计划
                yield {
                    "stage": "think_complete",
                    "type": "tool_call",
                    "tools": think_result.get("tools", []),
                    "tools_info": tools_info
                }
                
                # 3. 获取市场数据
                for tool in think_result.get("tools", []):
                    yield {
                        "stage": "fetch_data",
                        "tool": tool["name"],
                        "params": tool["params"],
                        "message": f"正在获取 {tool['params'].get('symbol', '未知')} 的{tool['name']}数据..."
                    }
                
                market_data = await self._analyze_market(think_result)
                
                # 4. 分析数据
                yield {
                    "stage": "analysis_start",
                    "message": "数据获取完成，开始技术分析...",
                    "market_data": market_data
                }
                
                formatted_data = self._format_market_data(market_data)
                
                yield {
                    "stage": "analysis_complete",
                    "message": "技术分析完成，正在整理分析结果...",
                    "formatted_data": formatted_data
                }
                
                # 5. LLM 分析
                yield {
                    "stage": "llm_analysis_start",
                    "message": "正在生成详细分析报告..."
                }
                
                # 构建分析提示词
                analysis_prompt = [
                    "=== 工具调用结果开始 ===",
                    formatted_data,
                    "=== 工具调用结果结束 ===",
                    "",
                    f"请基于以上数据，分析：{input_text}",
                    "",
                    "请注意：",
                    "1. 分析要基于实际数据",
                    "2. 关注数据的时效性",
                    "3. 保持客观专业的态度"
                ]
                
                analysis_prompt = "\n".join(analysis_prompt)
                system_prompt = await self._load_prompt_template("crypto-analyst")
                
                yield {
                    "stage": "llm_processing",
                    "message": "AI 正在思考分析结果..."
                }
                
                response = await self.llm.agenerate([[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ]])
                
                final_analysis = response.generations[0][0].text
                
                # 6. 返回最终结果
                yield {
                    "stage": "complete",
                    "type": "tool_call",
                    "final_response": final_analysis,
                    "market_data": market_data,
                    "formatted_data": formatted_data
                }
                
                # 保存交互记录
                await self._save_interaction(input_text, final_analysis, {
                    **context,
                    "market_data": market_data,
                    "used_tools": True
                })
                
            else:
                # 直接文本回答
                yield {
                    "stage": "think_complete",
                    "type": "direct",
                    "message": "这个问题我可以直接回答..."
                }
                
                yield {
                    "stage": "complete",
                    "type": "direct",
                    "response": think_result
                }
                
                await self._save_interaction(input_text, think_result, {
                    **context,
                    "used_tools": False
                })
            
        except Exception as e:
            self._logger.error(f"[CryptoAgent] 生成回复失败: {str(e)}")
            yield {
                "stage": "error",
                "error": str(e),
                "message": "处理过程中出现错误"
            }

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
                        
                    elif tool_name == "technical":
                        section.append("\n### 技术指标:")
                        indicators = tool_data['indicators']
                        ma_data = indicators['ma']
                        section.append(f"MA7/25/99: ${ma_data['ma7']:,.2f} / ${ma_data['ma25']:,.2f} / ${ma_data['ma99']:,.2f}")
                        section.append(f"RSI(14): {indicators['rsi']:.2f}")
                        macd_data = indicators['macd']
                        section.append(f"MACD: {macd_data['macd']:.2f} (Signal: {macd_data['signal']:.2f}, Hist: {macd_data['hist']:.2f})")
                        
                        # 价格信息
                        price_data = tool_data['price']
                        section.append(f"\n价格区间:")
                        section.append(f"当前价格: ${price_data['current']:,.2f}")
                        section.append(f"近期最高: ${price_data['high']:,.2f}")
                        section.append(f"近期最低: ${price_data['low']:,.2f}")
                        
                        # 成交量分析
                        volume_data = tool_data['volume']
                        section.append("\n成交量分析:")
                        section.append(f"当前成交量: {volume_data['current']:,.2f}")
                        section.append(f"20日均量: {volume_data['sma20']:,.2f}")
                        section.append(f"量价比: {volume_data['ratio']:.2f}")
                        section.append(f"成交量趋势: {volume_data['trend']}")
                        section.append(f"量价相关性: {volume_data['correlation']:.2f}")
                        
                        # 背离分析
                        divergence_data = tool_data['divergence']
                        section.append("\n背离分析:")
                        section.append(f"RSI背离: {divergence_data['rsi']}")
                        section.append(f"MACD背离: {divergence_data['macd']}")
                        
                        # 波动率
                        section.append(f"\n波动率: {tool_data['volatility']*100:.2f}%")
                        
                        # 分析结果
                        analysis = tool_data['analysis']
                        
                        # 程序分析结果
                        section.append("\n### 程序分析结果:")
                        section.append(analysis['program_analysis'])
                        
                        # LLM独立分析和对比验证
                        section.append("\n### LLM分析结果:")
                        section.append(analysis['llm_analysis'])
                        
                        
                    elif tool_name == "news":
                        section.append("\n### 加密货币新闻分析")
                        if "analysis" in tool_data:
                            section.append(tool_data["analysis"])
                            
                section.append("\n")  # 添加空行分隔
                formatted.extend(section)
                
            return "\n".join(formatted)
            
        except Exception as e:
            self._logger.error(f"[CryptoAgent] 格式化市场数据失败: {str(e)}")
            return f"格式化数据时出错: {str(e)}"

    async def astream_response(self, input_text: str, remark: str = "") -> AsyncIterator[str]:
        """流式生成市场分析回复"""
        try:
            # 1. 构建基础上下文
            context = await self._build_context(input_text, remark)
            
            # 2. 分析用户意图，确定需要使用的工具
            think_result = await self.think(context)
            
            # 3. 如果是工具配置，则调用工具

            market_data = None
            if isinstance(think_result, dict):

                # 获取市场数据
                market_data = await self._analyze_market(think_result)
                formatted_data = self._format_market_data(market_data)
                
                # 构建完整的系统提示词
                system_prompt = await self._load_prompt_template("crypto-analyst")
                tools_info = {
                    "available_tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        }
                        for tool in self.tools
                    ]
                }
                formatted_tools = json.dumps(tools_info, indent=2, ensure_ascii=False)
                
                # 替换提示词中的占位符
                system_prompt = system_prompt.replace("{{tools_info}}", formatted_tools)
                system_prompt = system_prompt.replace("{{market_data}}", formatted_data)
            else:
                # 如果不需要工具，使用原始提示词
                system_prompt = await self._load_prompt_template("crypto-analyst")
                system_prompt = system_prompt.replace("{{market_data}}", "")
            
            # 4. 流式生成回复
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
            
            # 5. 保存交互记录
            await self._save_interaction(input_text, "流式回复", {
                **context,
                "think_result": think_result,
                "market_data": market_data
            })
            
        except Exception as e:
            self._logger.error(f"[CryptoAgent] 生成回复失败: {str(e)}")
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