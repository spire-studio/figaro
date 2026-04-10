import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import copy
import logging
from collections import OrderedDict

from .aggregation import AggregationFactory

from fl_core.privacy.encryption import CKKSManager


class FederatedServer:
    def __init__(self, 
                 global_model: nn.Module,
                 aggregation_method: str = "fedavg",
                 device: torch.device = None,
                 config: Optional[Dict[str, Any]] = None,
                 ckks_manager: Optional[CKKSManager] = None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = global_model.to(self.device)
        self.aggregation_method = aggregation_method.lower()
        self.config = config
        
        self.supported_methods = AggregationFactory.get_supported_strategies()
        
        if self.aggregation_method not in self.supported_methods:
            raise ValueError(f"不支持的聚合方法: {aggregation_method}. "
                           f"支持的方法: {self.supported_methods}")
        
        self.aggregator = AggregationFactory.create_aggregator(
            self.aggregation_method, 
            self.device, 
            config=self.config
        )
        
        self.current_round = 0
        self.client_updates_history = []
        self.global_model_history = []
        
        self.logger = logging.getLogger('FederatedServer')

        self.ckks_manager = ckks_manager
        
        self.logger.info(f"联邦学习服务器初始化完成，聚合方法: {aggregation_method}")
    
    def get_global_model_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: param.clone().detach() for name, param in self.global_model.named_parameters()}
    
    def get_global_model_state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: param.clone().detach() for name, param in self.global_model.state_dict().items()}
    
    def set_global_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        model_dict = self.global_model.state_dict()
        
        for name, param in parameters.items():
            if name in model_dict:
                model_dict[name].copy_(param.to(self.device))
            else:
                self.logger.warning(f"参数键 {name} 在全局模型中不存在")
    
    def broadcast_model(self) -> Dict[str, torch.Tensor]:
        global_params = self.get_global_model_parameters()
        self.logger.debug(f"广播全局模型参数，轮次: {self.current_round}")
        return global_params
    
    def aggregate_models(self, 
                        client_models: List[Dict[str, torch.Tensor]], 
                        client_weights: Optional[List[float]] = None,
                        client_info: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        if not client_models:
            raise ValueError("客户端模型列表不能为空")
        
        num_clients = len(client_models)
        
        if client_weights is None:
            if self.aggregation_method == "weighted_avg" and client_info is not None:
                client_weights = [info.get('samples', 1) for info in client_info]
                total_samples = sum(client_weights)
                client_weights = [w / total_samples for w in client_weights]
            else:
                client_weights = [1.0 / num_clients] * num_clients
        
        if len(client_weights) != num_clients:
            raise ValueError(f"权重数量 ({len(client_weights)}) 与客户端数量 ({num_clients}) 不匹配")
        
        total_weight = sum(client_weights)
        if total_weight > 0:
            client_weights = [w / total_weight for w in client_weights]
        else:
            client_weights = [1.0 / num_clients] * num_clients
        
        # self.logger.info(f"开始聚合 {num_clients} 个客户端模型，聚合方法: {self.aggregation_method}")
        
        aggregated_updates = {} 

        if self.ckks_manager:
            # CKKS 密文聚合
            self.logger.info(f"开始 CKKS 密文聚合 {num_clients} 个客户端...")
            
            # 密文聚合
            encrypted_agg = self.ckks_manager.aggregate_encrypted_models(
                client_models, 
                client_weights
            )
            
            # 解密回 Tensor
            self.logger.info("解密聚合结果...")
            target_shapes = {k: v.shape for k, v in self.global_model.state_dict().items()}
            
            aggregated_updates = self.ckks_manager.decrypt_model(
                encrypted_agg, target_shapes, self.device
            )
            
        else:
            # 普通聚合 (包含 稀疏化处理)
            self.logger.info(f"开始聚合 {num_clients} 个客户端模型 (Delta模式)，聚合方法: {self.aggregation_method}")
            
            aggregated_updates = self.aggregator.aggregate(
                client_models, 
                client_weights=client_weights,
                client_info=client_info
            )
        
        self.client_updates_history.append({
            'round': self.current_round,
            'num_clients': num_clients,
            'client_weights': client_weights.copy(),
            'aggregation_method': self.aggregation_method,
            'mode': 'ckks' if self.ckks_manager else 'plaintext'
        })
        
        self.logger.info(f"模型聚合完成，轮次: {self.current_round}")
        
        return aggregated_updates
    
    def update_global_model(self, aggregated_updates: Dict[str, torch.Tensor]) -> None:
        current_state = self.get_global_model_state_dict()
        
        self.global_model_history.append({
            'round': self.current_round,
            'state_dict': copy.deepcopy(current_state) 
        })
        
        new_state = {}
        
        for name, param in current_state.items():
            if name in aggregated_updates:
                update = aggregated_updates[name].to(self.device)
                
                if param.is_floating_point():
                    new_state[name] = param + update
                else:
                    new_state[name] = param 
            else:
                new_state[name] = param
        
        self.global_model.load_state_dict(new_state)
        
        self.logger.info(f"全局模型已更新 (应用聚合更新量), 轮次: {self.current_round}")
    
    def evaluate_global_model(self, 
                            test_data: np.ndarray, 
                            test_labels: np.ndarray, 
                            batch_size: int = 128) -> Tuple[float, float]:
        self.global_model.eval()
        
        test_data_tensor = torch.FloatTensor(test_data).to(self.device)
        test_labels_tensor = torch.LongTensor(test_labels).to(self.device)
        test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.global_model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        self.logger.info(f"全局模型评估完成 - 轮次: {self.current_round}, "
                        f"损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
        
        return avg_loss, accuracy
    
    def next_round(self) -> None:
        self.current_round += 1
        self.logger.info(f"进入联邦学习轮次: {self.current_round}")
    
    def get_server_statistics(self) -> Dict[str, Any]:
        return {
            'current_round': self.current_round,
            'aggregation_method': self.aggregation_method,
            'total_rounds_completed': len(self.client_updates_history),
            'global_model_parameters': sum(p.numel() for p in self.global_model.parameters()),
            'device': str(self.device),
            'supported_methods': self.supported_methods
        }
    
    def get_aggregation_history(self) -> List[Dict[str, Any]]:
        return self.client_updates_history.copy()
    
    def save_global_model(self, filepath: str) -> None:
        torch.save({
            'global_model_state_dict': self.global_model.state_dict(),
            'current_round': self.current_round,
            'aggregation_method': self.aggregation_method,
            'client_updates_history': self.client_updates_history,
            'server_statistics': self.get_server_statistics()
        }, filepath)
        
        self.logger.info(f"全局模型已保存到: {filepath}")
    
    def load_global_model(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.global_model.load_state_dict(checkpoint['global_model_state_dict'])
        self.current_round = checkpoint.get('current_round', 0)
        self.aggregation_method = checkpoint.get('aggregation_method', 'fedavg')
        self.client_updates_history = checkpoint.get('client_updates_history', [])
        
        self.logger.info(f"全局模型已从 {filepath} 加载，当前轮次: {self.current_round}")

    
    def __str__(self) -> str:
        return (f"FederatedServer(round={self.current_round}, "
                f"method={self.aggregation_method}, "
                f"device={self.device})")
    
    def __repr__(self) -> str:
        return self.__str__()