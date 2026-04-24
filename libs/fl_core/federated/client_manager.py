import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import copy
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from .client import FederatedClient

from fl_core.compression.sparsification import GlobalTopKSparsifier
from fl_core.privacy.encryption import CKKSManager


class ClientManager:
    
    def __init__(self,
                 clients: List[FederatedClient],
                 selection_strategy: str = "random",
                 max_workers: Optional[int] = None,
                 device: torch.device = None,
                 sparsifier: Optional[GlobalTopKSparsifier] = None,
                 ckks_manager: Optional[CKKSManager] = None):
        # if not clients:
        #     raise ValueError("客户端列表不能为空")
        
        self.logger = logging.getLogger('ClientManager')
        
        self.clients = clients if clients else []
        self.num_clients = len(self.clients)
        self.selection_strategy = selection_strategy.lower()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.max_workers = max_workers if max_workers else max(1, min(4, self.num_clients))
        self.enable_parallel = True
        
        self.round_robin_index = 0
        self.client_weights = None
        
        self.supported_strategies = ["random", "all", "round_robin", "weighted"]
        if self.selection_strategy not in self.supported_strategies:
            raise ValueError(f"不支持的选择策略: {selection_strategy}. "
                           f"支持的策略: {self.supported_strategies}")

        self.sparsifier = sparsifier
        self.ckks_manager = ckks_manager
        
        self.training_stats = {
            'total_rounds': 0,
            'total_training_time': 0.0,
            'client_participation': {client.client_id: 0 for client in clients},
            'parallel_training_enabled': self.enable_parallel
        }

        if self.clients:
            self.training_stats['client_participation'] = {client.client_id: 0 for client in self.clients}
        else:
            self.training_stats['client_participation'] = {}

        self.logger.info(f"客户端管理器初始化完成，客户端数量: {self.num_clients}, "
                        f"选择策略: {selection_strategy}, 最大并行数: {self.max_workers}")
    
        if self.sparsifier:
            self.logger.info(f"稀疏化通信已启用 (Ratio: {self.sparsifier.ratio})")
        if self.ckks_manager:
            self.logger.info(f"CKKS同态加密通信已启用")
    
    def set_client_weights(self, weights: List[float]) -> None:
        if len(weights) != self.num_clients:
            raise ValueError(f"权重数量 ({len(weights)}) 与客户端数量 ({self.num_clients}) 不匹配")
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("权重总和必须大于0")
        
        self.client_weights = [w / total_weight for w in weights]
        self.logger.info(f"客户端权重已设置: {self.client_weights}")
    
    def select_clients(self, 
                      num_clients: int, 
                      strategy: Optional[str] = None,
                      exclude_clients: Optional[List[int]] = None) -> List[FederatedClient]:
        if num_clients <= 0:
            raise ValueError("选择的客户端数量必须大于0")
        
        strategy = strategy or self.selection_strategy
        exclude_clients = exclude_clients or []
        
        available_clients = [client for client in self.clients 
                           if client.client_id not in exclude_clients]
        
        if not available_clients:
            raise ValueError("没有可用的客户端")
        
        if num_clients > len(available_clients):
            self.logger.warning(f"请求的客户端数量 ({num_clients}) 超过可用数量 ({len(available_clients)})")
            num_clients = len(available_clients)
        
        if strategy == "all":
            selected_clients = available_clients[:num_clients]
        elif strategy == "random":
            selected_clients = random.sample(available_clients, num_clients)
        elif strategy == "round_robin":
            selected_clients = self._select_round_robin(available_clients, num_clients)
        elif strategy == "weighted":
            selected_clients = self._select_weighted(available_clients, num_clients)
        else:
            raise ValueError(f"不支持的选择策略: {strategy}")
        
        for client in selected_clients:
            self.training_stats['client_participation'][client.client_id] += 1
        
        selected_ids = [client.client_id for client in selected_clients]
        self.logger.info(f"选择了 {len(selected_clients)} 个客户端: {selected_ids}, 策略: {strategy}")
        
        return selected_clients
    
    def _select_round_robin(self, 
                           available_clients: List[FederatedClient], 
                           num_clients: int) -> List[FederatedClient]:
        selected_clients = []
        num_available = len(available_clients)
        
        for i in range(num_clients):
            client_index = (self.round_robin_index + i) % num_available
            selected_clients.append(available_clients[client_index])
        
        self.round_robin_index = (self.round_robin_index + num_clients) % num_available
        return selected_clients
    
    def _select_weighted(self, 
                        available_clients: List[FederatedClient], 
                        num_clients: int) -> List[FederatedClient]:
        if self.client_weights is None:
            self.logger.warning("未设置客户端权重，使用随机选择")
            return random.sample(available_clients, num_clients)
        
        available_weights = []
        for client in available_clients:
            client_index = next(i for i, c in enumerate(self.clients) if c.client_id == client.client_id)
            available_weights.append(self.client_weights[client_index])
        
        total_weight = sum(available_weights)
        if total_weight <= 0:
            return random.sample(available_clients, num_clients)
        
        normalized_weights = [w / total_weight for w in available_weights]
        
        selected_indices = np.random.choice(
            len(available_clients), 
            size=num_clients, 
            replace=False, 
            p=normalized_weights
        )
        
        return [available_clients[i] for i in selected_indices]
    
    def train_clients_sequential(self, 
                               selected_clients: List[FederatedClient],
                               epochs: int = 5,
                               learning_rate: float = 0.01,
                               batch_size: int = 32,
                               round_num: int = 0) -> List[Dict[str, Any]]:
        training_results = []
        start_time = time.time()
        
        self.logger.info(f"开始顺序训练 {len(selected_clients)} 个客户端，轮次: {round_num}")
        
        for i, client in enumerate(selected_clients):
            try:
                self.logger.debug(f"训练客户端 {client.client_id} ({i+1}/{len(selected_clients)})")
                
                result = client.train(
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    round_num=round_num
                )
                
                training_results.append(result)
                
            except Exception as e:
                self.logger.error(f"客户端 {client.client_id} 训练失败: {str(e)}")
                error_result = {
                    'client_id': client.client_id,
                    'round': round_num,
                    'loss': float('inf'),
                    'accuracy': 0.0,
                    'samples': 0,
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'error': str(e)
                }
                training_results.append(error_result)
        
        training_time = time.time() - start_time
        self.training_stats['total_training_time'] += training_time
        
        self.logger.info(f"顺序训练完成，耗时: {training_time:.2f}秒")
        
        return training_results
    
    def train_clients_parallel(self, 
                             selected_clients: List[FederatedClient],
                             epochs: int = 5,
                             learning_rate: float = 0.01,
                             batch_size: int = 32,
                             round_num: int = 0) -> List[Dict[str, Any]]:
        training_results = []
        start_time = time.time()
        
        self.logger.info(f"开始并行训练 {len(selected_clients)} 个客户端，轮次: {round_num}, "
                        f"最大并行数: {self.max_workers}")
        
        def train_single_client(client: FederatedClient) -> Dict[str, Any]:
            try:
                return client.train(
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    round_num=round_num
                )
            except Exception as e:
                self.logger.error(f"客户端 {client.client_id} 训练失败: {str(e)}")
                return {
                    'client_id': client.client_id,
                    'round': round_num,
                    'loss': float('inf'),
                    'accuracy': 0.0,
                    'samples': 0,
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'error': str(e)
                }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_client = {
                executor.submit(train_single_client, client): client 
                for client in selected_clients
            }
            
            for future in as_completed(future_to_client):
                client = future_to_client[future]
                try:
                    result = future.result()
                    training_results.append(result)
                    self.logger.debug(f"客户端 {client.client_id} 训练完成")
                except Exception as e:
                    self.logger.error(f"客户端 {client.client_id} 训练异常: {str(e)}")
                    error_result = {
                        'client_id': client.client_id,
                        'round': round_num,
                        'loss': float('inf'),
                        'accuracy': 0.0,
                        'samples': 0,
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'error': str(e)
                    }
                    training_results.append(error_result)
        
        training_results.sort(key=lambda x: x['client_id'])
        
        training_time = time.time() - start_time
        self.training_stats['total_training_time'] += training_time
        
        self.logger.info(f"并行训练完成，耗时: {training_time:.2f}秒")
        
        return training_results
    
    def train_clients(self, 
                     selected_clients: List[FederatedClient],
                     epochs: int = 5,
                     learning_rate: float = 0.01,
                     batch_size: int = 32,
                     round_num: int = 0,
                     parallel: Optional[bool] = None) -> List[Dict[str, Any]]:
        if not selected_clients:
            raise ValueError("选中的客户端列表不能为空")
        
        use_parallel = parallel if parallel is not None else self.enable_parallel
        
        # 更新统计
        self.training_stats['total_rounds'] += 1
        
        if use_parallel and len(selected_clients) > 1:
            if self.max_workers <= 0:
                self.max_workers = min(4, len(selected_clients))
            return self.train_clients_parallel(
                selected_clients, epochs, learning_rate, batch_size, round_num
            )
        else:
            return self.train_clients_sequential(
                selected_clients, epochs, learning_rate, batch_size, round_num
            )
    
    def evaluate_clients(self, 
                        selected_clients: List[FederatedClient],
                        use_test_data: bool = True,
                        parallel: Optional[bool] = None) -> List[Dict[str, Any]]:
        if not selected_clients:
            raise ValueError("客户端列表不能为空")
        
        use_parallel = parallel if parallel is not None else self.enable_parallel
        
        def evaluate_single_client(client: FederatedClient) -> Dict[str, Any]:
            try:
                loss, accuracy = client.evaluate(use_test_data=use_test_data)
                return {
                    'client_id': client.client_id,
                    'loss': loss,
                    'accuracy': accuracy,
                    'samples': len(client.test_dataset if use_test_data else client.train_dataset)
                }
            except Exception as e:
                self.logger.error(f"客户端 {client.client_id} 评估失败: {str(e)}")
                return {
                    'client_id': client.client_id,
                    'loss': float('inf'),
                    'accuracy': 0.0,
                    'samples': 0,
                    'error': str(e)
                }
        
        evaluation_results = []
        
        if use_parallel and len(selected_clients) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_client = {
                    executor.submit(evaluate_single_client, client): client 
                    for client in selected_clients
                }
                
                for future in as_completed(future_to_client):
                    result = future.result()
                    evaluation_results.append(result)
        else:
            for client in selected_clients:
                result = evaluate_single_client(client)
                evaluation_results.append(result)
        
        evaluation_results.sort(key=lambda x: x['client_id'])
        
        data_type = "测试" if use_test_data else "训练"
        self.logger.info(f"客户端{data_type}数据评估完成，评估了 {len(selected_clients)} 个客户端")
        
        return evaluation_results
    
    def get_client_models(self, selected_clients: List[FederatedClient],
                          global_model_params: Optional[Dict[str, torch.Tensor]] = None) -> List[Dict[str, torch.Tensor]]:

        client_models = []

        for client in selected_clients:
            try:
                model_params = client.get_model_parameters()

                delta = {}
                if global_model_params:
                    for k, v in model_params.items():
                        if v.is_floating_point():
                            v = v.to(self.device) 
                            global_v = global_model_params[k].to(self.device)
                            
                            delta[k] = v - global_v
                else:
                    delta = {k: v.to(self.device) for k, v in model_params.items()}

                payload = delta

                if self.ckks_manager:
                    # 加密
                    payload = self.ckks_manager.encrypt_model(delta)
                elif self.sparsifier:
                    # 稀疏化
                    payload = self.sparsifier.sparsify(delta)
                
                client_models.append(payload)
                
            except Exception as e:
                self.logger.error(f"获取客户端 {client.client_id} 模型参数失败: {str(e)}")
                client_models.append({})
        
        return client_models
    
    def broadcast_model_to_clients(self, 
                                  selected_clients: List[FederatedClient],
                                  global_model_params: Dict[str, torch.Tensor]) -> None:
        for client in selected_clients:
            try:
                client.set_model_parameters(global_model_params)
            except Exception as e:
                self.logger.error(f"向客户端 {client.client_id} 广播模型参数失败: {str(e)}")
        
        client_ids = [client.client_id for client in selected_clients]
        self.logger.info(f"已向客户端 {client_ids} 广播全局模型参数")
    
    def get_client_statistics(self) -> Dict[str, Any]:
        total_train_samples = sum(len(client.train_dataset) for client in self.clients)
        total_test_samples = sum(len(client.test_dataset) for client in self.clients)

        participation_stats = self.training_stats['client_participation']
        avg_participation = np.mean(list(participation_stats.values())) if participation_stats else 0
        max_participation = max(participation_stats.values()) if participation_stats else 0
        min_participation = min(participation_stats.values()) if participation_stats else 0

        return {
            'num_clients': self.num_clients,
            'selection_strategy': self.selection_strategy,
            'max_workers': self.max_workers,
            'parallel_enabled': self.enable_parallel,
            'total_train_samples': total_train_samples,
            'total_test_samples': total_test_samples,
            'training_stats': self.training_stats.copy(),
            'participation_stats': {
                'average': avg_participation,
                'maximum': max_participation,
                'minimum': min_participation,
                'per_client': participation_stats.copy()
            }
        }
    
    
    def set_parallel_training(self, enable: bool, max_workers: Optional[int] = None) -> None:
        self.enable_parallel = enable
        
        if max_workers is not None:
            self.max_workers = max_workers
        
        self.training_stats['parallel_training_enabled'] = enable
        
        self.logger.info(f"并行训练设置已更新: 启用={enable}, 最大线程数={self.max_workers}")
    
    def get_client_by_id(self, client_id: int) -> Optional[FederatedClient]:
        for client in self.clients:
            if client.client_id == client_id:
                return client
        return None
    
    def get_clients_by_ids(self, client_ids: List[int]) -> List[FederatedClient]:
        selected_clients = []
        for client_id in client_ids:
            client = self.get_client_by_id(client_id)
            if client:
                selected_clients.append(client)
            else:
                self.logger.warning(f"未找到客户端ID: {client_id}")
        
        return selected_clients
    
    def __str__(self) -> str:
        return (f"ClientManager(clients={self.num_clients}, "
                f"strategy={self.selection_strategy}, "
                f"parallel={self.enable_parallel}, "
                f"max_workers={self.max_workers})")
    
    def __repr__(self) -> str:
        return self.__str__()