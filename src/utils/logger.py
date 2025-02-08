import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import inspect

# 添加彩色日志支持
class ColorFormatter(logging.Formatter):
    """自定义的彩色日志格式化器"""
    
    # ANSI转义序列颜色代码
    COLORS = {
        'GREEN': '\033[32m',
        'YELLOW': '\033[33m',
        'RED': '\033[31m',
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        # 添加颜色属性
        if hasattr(record, 'color') and record.color in self.COLORS:
            color_code = self.COLORS[record.color]
            reset_code = self.COLORS['RESET']
            record.msg = f"{color_code}{record.msg}{reset_code}"
        return super().format(record)

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
            
            # 修改日志格式，移除 filename 和 funcName
            log_format = '%(asctime)s - [%(module_name)s] - %(levelname)s - %(message)s'
            
            # 使用ColorFormatter
            formatter = ColorFormatter(log_format)
            
            # 文件处理器使用相同格式
            file_formatter = logging.Formatter(log_format)
            
            file_handler = logging.FileHandler(
                str(self._log_file), 
                encoding='utf-8'
            )
            file_handler.setFormatter(file_formatter)
            
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            
            self.logger = logging.getLogger(f'{module_name}')
            self.logger.setLevel(getattr(logging, log_level))
            
            self.logger.handlers.clear()
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            self.logger.propagate = False
            
            # 初始化日志时添加 extra 参数
            self.logger.info(
                f"日志初始化 - 模块: {module_name}, 文件: {self._log_file}", 
                extra={'module_name': module_name}
            )
            
            self._initialized = True
    
    @property
    def log_file(self) -> Path:
        """获取日志文件路径"""
        return self._log_file
    
    def get_logger(self) -> logging.Logger:
        """获取logger实例"""
        return self.logger
            
    def _log(self, level: str, msg: str, color: Optional[str] = None, *args, **kwargs):
        """统一的日志记录方法"""
        # 获取调用栈信息
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back.f_back  # 向上查找两层，跳过 debug/info/warning/error 方法
        
        # 构建日志前缀
        if caller_frame:
            filename = os.path.basename(caller_frame.f_code.co_filename)
            funcname = caller_frame.f_code.co_name
            msg = f"[{filename}:{funcname}] {msg}"
        
        # 确保每条日志都有 module_name
        extra = kwargs.get('extra', {})
        extra.update({
            'module_name': self.module_name
        })
        if color:
            extra['color'] = color.upper()
            
        kwargs['extra'] = extra
        getattr(self.logger, level)(msg, *args, **kwargs)
            
    def debug(self, msg: str, color: Optional[str] = None, *args, **kwargs):
        """调试日志"""
        self._log('debug', msg, color, *args, **kwargs)
        
    def info(self, msg: str, color: Optional[str] = None, *args, **kwargs):
        """信息日志"""
        self._log('info', msg, color, *args, **kwargs)
        
    def warning(self, msg: str, color: Optional[str] = None, *args, **kwargs):
        """警告日志"""
        self._log('warning', msg, color, *args, **kwargs)
        
    def error(self, msg: str, color: Optional[str] = None, *args, **kwargs):
        """错误日志"""
        self._log('error', msg, color, *args, **kwargs) 