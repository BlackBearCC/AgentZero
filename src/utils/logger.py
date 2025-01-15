import logging
from typing import Optional

class Logger:
    _instance = None
    _initialized = False

    def __new__(cls, name: str = "AgentZero"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name: str = "AgentZero"):
        if not self._initialized:
            self._initialized = True
            self.logger = logging.getLogger(name)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

    def error(self, msg: str):
        self.logger.error(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def warning(self, msg: str):
        self.logger.warning(msg) 