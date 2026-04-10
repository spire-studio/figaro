import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from abc import ABC, abstractmethod


class AggregationStrategy(ABC):
    @abstractmethod
    def aggregate(self, 
                 client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: Optional[List[float]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class FedAvgAggregation(AggregationStrategy):
    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('FedAvgAggregation')
    
    def aggregate(self, 
                 client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: Optional[List[float]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        if not client_models:
            raise ValueError("客户端模型列表不能为空")
        
        num_clients = len(client_models)
        
        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        
        total_weight = sum(client_weights)
        if total_weight > 0:
            client_weights = [w / total_weight for w in client_weights]
        else:
            client_weights = [1.0 / num_clients] * num_clients
        
        if len(client_weights) != num_clients:
            raise ValueError(f"权重数量 ({len(client_weights)}) 与客户端数量 ({num_clients}) 不匹配")
        
        self.logger.debug(f"开始FedAvg聚合，客户端数量: {num_clients}")
        
        param_keys = client_models[0].keys()
        
        aggregated_params = {}
        
        for param_name in param_keys:
            client_params = [model[param_name] for model in client_models]
            
            weighted_param = torch.zeros_like(client_params[0], device=self.device)
            for param, weight in zip(client_params, client_weights):
                weighted_param += param.to(self.device) * weight
            
            aggregated_params[param_name] = weighted_param
        
        self.logger.debug(f"FedAvg聚合完成")
        
        return aggregated_params
    
    def get_name(self) -> str:
        return "fedavg"


class WeightedAggregation(AggregationStrategy):
    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('WeightedAggregation')
    
    def aggregate(self, 
                 client_models: List[Dict[str, torch.Tensor]], 
                 client_weights: Optional[List[float]] = None,
                 client_info: Optional[List[Dict[str, Any]]] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        if not client_models:
            raise ValueError("客户端模型列表不能为空")
        
        num_clients = len(client_models)
        
        if client_weights is None and client_info is not None:
            client_weights = [info.get('samples', 1) for info in client_info]
            total_samples = sum(client_weights)
            if total_samples > 0:
                client_weights = [w / total_samples for w in client_weights]
            else:
                client_weights = [1.0 / num_clients] * num_clients
        elif client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        
        self.logger.debug(f"开始加权聚合，客户端数量: {num_clients}")
        
        # 使用FedAvg的聚合逻辑
        fedavg_aggregator = FedAvgAggregation(self.device)
        return fedavg_aggregator.aggregate(client_models, client_weights)
    
    def get_name(self) -> str:
        return "weighted_avg"


class AggregationFactory:
    
    _strategies = {
        'fedavg': FedAvgAggregation,
        'weighted_avg': WeightedAggregation,
        'simple_avg': FedAvgAggregation, 
    }
    
    @classmethod
    def create_aggregator(cls,
                         strategy_name: str,
                         device: torch.device = None,
                         config: Optional[Dict[str, Any]] = None,
                         **kwargs) -> AggregationStrategy:
        strategy_name = strategy_name.lower()

        if strategy_name not in cls._strategies:
            raise ValueError(f"不支持的聚合策略: {strategy_name}. "
                           f"支持的策略: {list(cls._strategies.keys())}")

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(device=device, **kwargs)
    
    @classmethod
    def get_supported_strategies(cls) -> List[str]:
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(cls, 
                         name: str, 
                         strategy_class: type) -> None:
        if not issubclass(strategy_class, AggregationStrategy):
            raise ValueError("策略类必须继承自AggregationStrategy")
        
        cls._strategies[name.lower()] = strategy_class
    
def create_aggregator(strategy_name: str,
                     device: torch.device = None,
                     config: Optional[Dict[str, Any]] = None,
                     **kwargs) -> AggregationStrategy:
    return AggregationFactory.create_aggregator(strategy_name, device, config, **kwargs)


def get_supported_aggregation_methods() -> List[str]:
    return AggregationFactory.get_supported_strategies()