import logging
import os
import sys

class Logger:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self):
        if not self._initialized:
            # 获取环境变量中的日志级别，默认为 INFO
            log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
            
            # 配置根日志记录器
            logging.basicConfig(
                level=getattr(logging, log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(sys.stdout),  # 输出到标准输出
                    logging.FileHandler('app.log')  # 同时写入文件
                ]
            )
            
            # 创建日志记录器
            self.logger = logging.getLogger('ai-chat')
            self.logger.setLevel(getattr(logging, log_level))
            
            self._initialized = True
            
    def debug(self, msg: str):
        """调试日志"""
        self.logger.debug(msg)
        
    def info(self, msg: str):
        """信息日志"""
        self.logger.info(msg)
        
    def warning(self, msg: str):
        """警告日志"""
        self.logger.warning(msg)
        
    def error(self, msg: str):
        """错误日志"""
        self.logger.error(msg) 