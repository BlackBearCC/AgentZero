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
                await update.message.reply_text("请指定要分析的加密货币符号，例如：/analyze ETH")
                return
                
            symbol = context.args[0].upper()
            message = f"请分析 {symbol} 的市场情况"
            
            # 发送"正在分析"消息
            processing_message = await update.message.reply_text(f"正在分析 {symbol} 的市场情况，请稍候...")
            
            # 调用 ChatService 处理消息
            response = await self._chat_service.process_message(
                agent_id="crypto_001",
                message=message,
                context={"symbol": symbol}
            )
            
            # 发送分析结果
            await processing_message.edit_text(response)
            
        except Exception as e:
            self.logger.error(f"分析失败: {str(e)}")
            await update.message.reply_text(f"分析过程中出现错误: {str(e)}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """处理普通消息"""
        try:
            await self._ensure_chat_service()
            
            # 发送"正在处理"消息
            processing_message = await update.message.reply_text("正在处理您的问题，请稍候...")
            
            # 调用 ChatService 处理消息
            response = await self._chat_service.process_message(
                agent_id="crypto_001",
                message=update.message.text
            )
            
            # 更新消息内容
            await processing_message.edit_text(response)
            
        except Exception as e:
            self.logger.error(f"消息处理失败: {str(e)}")
            await update.message.reply_text(f"处理消息时出现错误: {str(e)}")
    
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