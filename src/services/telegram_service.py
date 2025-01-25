from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.services.chat_service import ChatService
from src.services.agent_service import get_agent_service
from src.utils.logger import Logger
import logging  # 添加日志导入

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
        welcome_text = """
欢迎使用加密货币分析机器人！

可用命令：
/analyze <symbol> - 分析指定加密货币
例如：/analyze ETH

直接发送消息询问任何加密货币相关问题。
"""
        await update.message.reply_text(welcome_text)
    
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
            
            # 发送"正在处理"消息，使用可爱的表情和语气
            processing_message = await update.message.reply_text(
                "✨ 哈喽哈喽~让 Crypto-chan 想想看呢... 💭"
            )
            
            # 调用 ChatService 处理消息
            response = await self._chat_service.process_message(
                agent_id="crypto_001",
                message=update.message.text
            )
            
            # 如果响应包含问候语和工具调用
            if isinstance(response, dict):
                # 如果有工具调用前的问候语
                if "pre_tool_message" in response:
                    await processing_message.edit_text(response["pre_tool_message"])
                    # 发送新的"正在分析"消息
                    processing_message = await update.message.reply_text(
                        "🔍 Crypto-chan 正在认真分析市场数据中...请稍等一下下哦！✨"
                    )
                
                # 获取完整分析结果
                final_response = await self._chat_service.process_tool_response(response)
                await processing_message.edit_text(final_response)
            else:
                # 直接发送响应
                await processing_message.edit_text(response)
            
        except Exception as e:
            self.logger.error(f"消息处理失败: {str(e)}")
            await update.message.reply_text(
                "😢 呜呜...Crypto-chan 遇到了一点小问题呢...\n"
                f"错误信息: {str(e)}\n"
                "让我们稍后再试试看吧！💪"
            )
    
    def run(self):
        """启动机器人"""
        try:
            # 注册命令处理器
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("analyze", self.analyze))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            
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