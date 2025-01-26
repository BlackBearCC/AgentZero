from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from src.services.chat_service import ChatService
from src.services.agent_service import get_agent_service
from src.utils.logger import Logger
import logging  # 添加日志导入
import asyncio

class TelegramBotService:
    def __init__(self, token: str):
        # 配置日志
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore.http11').setLevel(logging.WARNING)
        logging.getLogger('telegram').setLevel(logging.INFO)
        
        self.token = token
        self.logger = Logger()
        self.application = Application.builder().token(token).build()
        self._chat_service = None
        
        # 添加启动日志
        self.logger.info("Telegram Bot Service initialized")
        
    async def _ensure_chat_service(self):
        """确保 ChatService 已初始化"""
        if not self._chat_service:
            agent_service = await get_agent_service()
            self._chat_service = ChatService(agent_service)
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /start 命令"""
        # 创建固定在输入框旁边的键盘
        keyboard = [
            ["BTC", "ETH"],
            ["加密货币新闻"]
        ]
        reply_markup = ReplyKeyboardMarkup(
            keyboard,
            resize_keyboard=True,  # 自适应大小
            one_time_keyboard=False  # 保持键盘显示
        )
        
        welcome_text = """
✨ 欢迎来到 Crypto-chan 的加密市场分析室！

我是您的加密市场分析助手 Crypto-chan~ 🌟
让我来帮您分析市场、追踪行情！

🎮 您可以这样和我互动：

📝 直接问我问题：
- "比特币现在是牛市吗？"
- "以太坊最近的趋势如何？"
- "现在适合投资吗？"

🎯 使用命令：
/analyze <币种> - 获取详细分析报告
例如：/analyze ETH

⚡️ 快捷按钮：
BTC - 比特币市场分析
ETH - 以太坊市场分析
加密货币新闻 - 最新市场动态

让我们一起探索加密货币的世界吧！💫
"""
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup  # 使用 ReplyKeyboardMarkup
        )
    
    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理 /analyze 命令"""
        try:
            await self._ensure_chat_service()
            
            if not context.args:
                await update.message.reply_text(
                    "✨ 呐呐~需要告诉 Crypto-chan 要分析哪个加密货币哦！\n"
                    "例如：/analyze ETH 💫"
                )
                return
                
            symbol = context.args[0].upper()
            message = f"请分析 {symbol} 的市场情况"
            
            # 发送"正在分析"消息，使用可爱的表情和语气
            processing_message = await update.message.reply_text(
                f"✨ 好的呢~让 Crypto-chan 来看看 {symbol} 的情况！\n"
                "🔍 正在获取最新数据，请稍等一下下哦... 💫"
            )
            
            # 调用 ChatService 处理消息
            response = await self._chat_service.process_message(
                agent_id="crypto_001",
                message=message,
                context={"symbol": symbol}
            )
            
            # 如果响应包含问候语和工具调用
            if isinstance(response, dict):
                # 如果有工具调用前的问候语
                if "pre_tool_message" in response:
                    await processing_message.edit_text(response["pre_tool_message"])
                    # 发送新的"正在分析"消息
                    processing_message = await update.message.reply_text(
                        f"🔍 Crypto-chan 正在认真分析 {symbol} 的市场数据...请稍等一下下哦！✨"
                    )
                
                # 获取完整分析结果
                final_response = await self._chat_service.process_tool_response(response)
                await processing_message.edit_text(final_response)
            else:
                # 直接发送响应
                await processing_message.edit_text(response)
            
        except Exception as e:
            self.logger.error(f"分析失败: {str(e)}")
            await update.message.reply_text(
                "😢 呜呜...Crypto-chan 分析的时候遇到了一点小问题呢...\n"
                f"错误信息: {str(e)}\n"
                "让我们稍后再试试看吧！💪"
            )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理普通消息"""
        try:
            await self._ensure_chat_service()
            processing_message = None
            last_update_time = 0
            
            # 处理快捷指令
            message_text = update.message.text
            if message_text == "BTC":
                message_text = "使用技术分析和新闻工具分析BTC市场情况"
            elif message_text == "ETH":
                message_text = "使用技术分析和新闻工具分析ETH市场情况"
            elif message_text == "加密货币新闻":
                message_text = "获取最新的加密货币市场新闻"
            
            async def update_message(message_obj, new_text: str, min_display_time: int = 5, reply_markup=None):
                """更新消息，确保最小显示时间"""
                nonlocal last_update_time
                current_time = asyncio.get_event_loop().time()
                time_since_last_update = current_time - last_update_time
                
                if time_since_last_update < min_display_time:
                    await asyncio.sleep(min_display_time - time_since_last_update)
                
                await message_obj.edit_text(
                    new_text,
                    reply_markup=reply_markup
                )
                last_update_time = asyncio.get_event_loop().time()
            
            async for response in self._chat_service.process_telegram_message(
                agent_id="crypto_001",
                message=message_text
            ):
                stage = response.get("stage")
                
                if stage == "think_start":
                    processing_message = await update.message.reply_text(
                        "💭 让 Crypto-chan 想想看...\n"
                        f"问题：{message_text}"
                    )
                    last_update_time = asyncio.get_event_loop().time()
                
                elif stage == "pre_tool_message":
                    await update_message(
                        processing_message,
                        f"💭 {response['message']}"
                    )
                    
                elif stage == "think_complete":
                    if response.get("type") == "tool_call":
                        tools_info = response.get("tools_info", [])
                        tools_text = "\n".join([
                            f"- {tool['name']}: {tool['params'].get('symbol', '未知')}\n"
                            f"  参数: {', '.join([f'{k}={v}' for k, v in tool['params'].items()])}"
                            for tool in tools_info
                        ])
                        
                        await update_message(
                            processing_message,
                            f"💡 分析计划：\n{tools_text}"
                        )
                    
                elif stage == "fetch_data":
                    tool_name = response.get("tool")
                    params = response.get("params", {})
                    params_text = "\n".join([f"  - {k}: {v}" for k, v in params.items()])
                    
                    await update_message(
                        processing_message,
                        f"🔍 正在获取数据：{tool_name}\n"
                        f"参数：\n{params_text}\n"
                        "请稍等片刻~ 💫"
                    )
                    
                elif stage == "analysis_start":
                    market_data = response.get("market_data", {})
                    data_summary = []
                    
                    for symbol, data in market_data.get("data", {}).items():
                        data_summary.append(f"📊 {symbol} 数据获取完成：")
                        for tool_name in data.keys():
                            data_summary.append(f"  ✓ {tool_name}")
                    
                    await update_message(
                        processing_message,
                        "📈 数据获取完成！\n" + 
                        "\n".join(data_summary) + "\n\n"
                        "正在进行技术分析...\n"
                        "（计算指标、检测形态、分析背离等）"
                    )
                    
                elif stage == "analysis_complete":
                    formatted_data = response.get("formatted_data", "")
                    
                    # 确保数据是人类可读的格式
                    if formatted_data:
                        try:
                            # 尝试美化数据展示
                            data_lines = []
                            for line in formatted_data.split('\n'):
                                # 跳过原始数值
                                if not line.replace('.', '').replace('-', '').isdigit():
                                    # 添加适当的缩进和图标
                                    if line.strip().startswith('价格'):
                                        data_lines.append(f"💰 {line.strip()}")
                                    elif line.strip().startswith('成交量'):
                                        data_lines.append(f"📊 {line.strip()}")
                                    elif line.strip().startswith('趋势'):
                                        data_lines.append(f"📈 {line.strip()}")
                                    else:
                                        data_lines.append(f"  {line.strip()}")
                            
                            formatted_data = "\n".join(data_lines)
                            
                            # 如果数据太长，只显示摘要
                            if len(formatted_data) > 500:
                                formatted_data = formatted_data[:500] + "...\n(数据太长已省略)"
                        except Exception:
                            formatted_data = "数据已获取，正在分析中..."
                    
                    await update_message(
                        processing_message,
                        "📊 技术分析完成！关键数据：\n\n" +
                        formatted_data + "\n\n" +
                        "正在生成分析报告..."
                    )
                    
                elif stage == "llm_analysis_start":
                    await update_message(
                        processing_message,
                        "🧠 数据分析完成！\n"
                        "Crypto-chan 正在整合以上数据，生成详细分析报告...\n"
                        "（考虑技术面、市场情绪等多个维度）"
                    )
                    
                elif stage == "llm_processing":
                    await update_message(
                        processing_message,
                        "✍️ 正在撰写分析报告...\n"
                        "整合各项指标数据，形成最终结论...\n"
                        "马上就好！"
                    )
                    
                elif stage == "complete":
                    if response.get("type") == "tool_call":
                        formatted_data = response.get("formatted_data", "")
                        
                        # 创建带有查看数据按钮的消息
                        keyboard = [[
                            InlineKeyboardButton("📊 查看详细数据", callback_data=f"view_data_{update.message.message_id}")
                        ]]
                        reply_markup = InlineKeyboardMarkup(keyboard)
                        
                        # 保存数据到 context.bot_data 中，以便回调时使用
                        if not context.bot_data.get('formatted_data'):
                            context.bot_data['formatted_data'] = {}
                        context.bot_data['formatted_data'][str(update.message.message_id)] = formatted_data
                        
                        await update_message(
                            processing_message,
                            "✨ 分析完成啦！以下是详细分析报告 📝\n\n" + 
                            response["final_response"],
                            reply_markup=reply_markup
                        )
                    else:
                        await update_message(
                            processing_message,
                            response["response"]
                        )
                        
                elif stage == "error":
                    error_msg = response.get("error", "未知错误")
                    error_details = response.get("details", {})
                    
                    error_text = [
                        "😢 呜呜...Crypto-chan 遇到了一点小问题呢...",
                        f"错误信息: {error_msg}"
                    ]
                    
                    if error_details:
                        error_text.append("\n详细信息：")
                        for k, v in error_details.items():
                            error_text.append(f"- {k}: {v}")
                    
                    error_text.append("\n让我们稍后再试试看吧！💪")
                    
                    await update.message.reply_text("\n".join(error_text))
                    
        except Exception as e:
            self.logger.error(f"消息处理失败: {str(e)}")
            await update.message.reply_text(
                "😢 呜呜...Crypto-chan 遇到了一点小问题呢...\n"
                f"错误信息: {str(e)}\n"
                "让我们稍后再试试看吧！💪"
            )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理按钮回调"""
        query: CallbackQuery = update.callback_query
        await query.answer()  # 确认回调查询
        
        try:
            if query.data.startswith("view_data_"):
                message_id = query.data.split("_")[2]
                formatted_data = context.bot_data['formatted_data'].get(str(message_id))
                
                if formatted_data:
                    await query.message.reply_text(
                        "📊 详细数据参考：\n\n" + formatted_data,
                        quote=True  # 引用原消息
                    )
                else:
                    await query.message.reply_text(
                        "😢 抱歉，数据已过期或不可用",
                        quote=True
                    )
        except Exception as e:
            await query.message.reply_text(
                "😢 获取数据时出现错误，请重试",
                quote=True
            )
    
    def run(self):
        """启动机器人"""
        try:
            # 注册命令处理器
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("analyze", self.analyze))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            # 添加回调查询处理器
            self.application.add_handler(CallbackQueryHandler(self.handle_callback))
            
            # 启动机器人
            self.application.run_polling()
            
        except Exception as e:
            self.logger.error(f"机器人启动失败: {str(e)}")
            raise 

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    import os
    
    # 加载环境变量
    load_dotenv()
    
    # 获取 token
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
    
    # 创建并运行 bot
    bot = TelegramBotService(token)
    print("Telegram Bot is starting...")  # 添加启动日志
    bot.run() 