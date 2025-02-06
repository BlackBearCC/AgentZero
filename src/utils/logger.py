import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class Logger:
    _instances = {}  # 存储不同模块的logger实例
    
    def __new__(cls, module_name: str = 'default'):
        """单例模式，但按模块名区分实例"""
        if module_name not in cls._instances:
            cls._instances[module_name] = super().__new__(cls)
            cls._instances[module_name]._initialized = False
        return cls._instances[module_name]
        
    def __init__(self, module_name: str = 'default'):
        """初始化logger
        Args:
            module_name: 模块名称，用于区分不同模块的日志
        """
        if not self._initialized:
            self.module_name = module_name
            
            # 创建logs目录结构
            log_dir = Path(__file__).parent.parent.parent / 'logs' / module_name
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成日志文件名（包含模块名和时间戳）
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_file = log_dir / f"{module_name}_{current_time}.log"
            
            # 获取日志级别
            log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
            
            # 自定义格式化类
            class ModuleFormatter(logging.Formatter):
                def format(self, record):
                    record.module_name = module_name
                    return super().format(record)
            
            # 创建格式化器
            formatter = ModuleFormatter(
                '%(asctime)s - %(name)s - [%(module_name)s] - %(levelname)s - %(message)s'
            )
            
            # 创建文件处理器
            file_handler = logging.FileHandler(
                str(self._log_file), 
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            
            # 创建控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            
            # 创建logger
            self.logger = logging.getLogger(f'{module_name}')
            self.logger.setLevel(getattr(logging, log_level))
            
            # 清除现有处理器
            self.logger.handlers.clear()
            
            # 添加处理器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            # 设置不传播到父记录器
            self.logger.propagate = False
            
            # 记录初始化信息
            self.logger.info(f"日志初始化 - 模块: {module_name}, 文件: {self._log_file}")
            
            self._initialized = True
    
    @property
    def log_file(self) -> Path:
        """获取日志文件路径"""
        return self._log_file
    
    def get_logger(self) -> logging.Logger:
        """获取logger实例"""
        return self.logger
            
    def debug(self, msg: str, *args, **kwargs):
        """调试日志"""
        self.logger.debug(msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs):
        """信息日志"""
        self.logger.info(msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs):
        """警告日志"""
        self.logger.warning(msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs):
        """错误日志"""
        self.logger.error(msg, *args, **kwargs) 