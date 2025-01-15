import yaml
from typing import Dict, Any
import os

class Config:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        env = os.getenv("ENV", "dev")
        config_path = f"config/{env}.yaml"
        
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)
        
        # 环境变量覆盖
        self._override_from_env()
    
    def _override_from_env(self):
        for key, value in self._config.items():
            env_value = os.getenv(key.upper())
            if env_value:
                self._config[key] = env_value
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default) 