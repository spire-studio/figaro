import yaml
import os
from typing import Dict, Any


class ConfigManager:
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                if config is None:
                    raise ValueError("配置文件为空")
                return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML配置文件格式错误: {e}")
    
    def get_dataset_config(self) -> Dict[str, Any]:
        return self.config.get('dataset', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.config.get('model', {})
    
    def get_federated_config(self) -> Dict[str, Any]:
        return self.config.get('federated', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        return self.config.get('logging', {})
    
    def get_web_config(self) -> Dict[str, Any]:
        return self.config.get('web', {})
    
    def get_system_config(self) -> Dict[str, Any]:
        return self.config.get('system', {})
    
    def get_distributed_config(self) -> Dict[str, Any]:
        return self.config.get('distributed', {})
    
    def get_compression_config(self) -> Dict[str, Any]:
        return self.config.get('compression', {})
    
    def get_privacy_config(self) -> Dict[str, Any]:
        return self.config.get('privacy', {})
    
    def get_config(self, key: str = None) -> Any:
        if key is None:
            return self.config
        return self.config.get(key)
    
    def validate_config(self) -> bool:
        required_sections = ['dataset', 'model', 'federated', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"缺少必需的配置节: {section}")
        
        # 验证数据集配置
        dataset_config = self.get_dataset_config()
        if 'name' not in dataset_config:
            raise ValueError("数据集配置缺少name参数")
        
        # 验证模型配置
        model_config = self.get_model_config()
        if 'name' not in model_config:
            raise ValueError("模型配置缺少name参数")
        
        # 验证联邦学习配置
        federated_config = self.get_federated_config()
        required_fed_params = ['num_clients', 'num_rounds', 'local_epochs', 'learning_rate']
        for param in required_fed_params:
            if param not in federated_config:
                raise ValueError(f"联邦学习配置缺少{param}参数")
        
        return True