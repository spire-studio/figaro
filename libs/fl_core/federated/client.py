import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import copy
import logging


class FederatedClient:

    def __init__(self, 
                 client_id: int,
                 train_data: np.ndarray,
                 train_labels: np.ndarray,
                 test_data: np.ndarray,
                 test_labels: np.ndarray,
                 model: nn.Module,
                 device: torch.device = None):
        self.client_id = client_id
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_data = torch.FloatTensor(train_data).to(self.device)
        self.train_labels = torch.LongTensor(train_labels).to(self.device)
        self.test_data = torch.FloatTensor(test_data).to(self.device)
        self.test_labels = torch.LongTensor(test_labels).to(self.device)
        
        self.train_dataset = TensorDataset(self.train_data, self.train_labels)
        self.test_dataset = TensorDataset(self.test_data, self.test_labels)
        
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'rounds': []
        }
        
        self.is_malicious = False
        self.attack_config = None
        
        self.logger = logging.getLogger(f'Client_{client_id}')
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        model_dict = self.model.state_dict()
        
        for name, param in parameters.items():
            if name in model_dict:
                model_dict[name].copy_(param.to(self.device))
            else:
                self.logger.warning(f"参数键 {name} 在模型中不存在")
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: param.clone().detach() for name, param in self.model.named_parameters()}
    
    def get_model_state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: param.clone().detach() for name, param in self.model.state_dict().items()}
    
    def train(self, 
              epochs: int = 5, 
              learning_rate: float = 0.01, 
              batch_size: int = 32,
              round_num: int = 0) -> Dict[str, Any]:
        self.model.train()
        
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=False
        )
        
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
            
            epoch_avg_loss = epoch_loss / epoch_samples
            epoch_accuracy = epoch_correct / epoch_samples
            
            self.logger.debug(f"客户端 {self.client_id}, 轮次 {round_num}, Epoch {epoch+1}/{epochs}: "
                            f"Loss: {epoch_avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            correct_predictions += epoch_correct
        
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            avg_accuracy = correct_predictions / total_samples
        else:
            avg_loss = 0.0
            avg_accuracy = 0.0
        
        self.training_history['loss'].append(avg_loss)
        self.training_history['accuracy'].append(avg_accuracy)
        self.training_history['rounds'].append(round_num)
        
        training_result = {
            'client_id': self.client_id,
            'round': round_num,
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'samples': len(self.train_dataset),
            'epochs': epochs,
            'learning_rate': learning_rate
        }
        
        self.logger.info(f"客户端 {self.client_id} 训练完成, "
                        f"训练损失: {avg_loss:.4f}, 训练准确率: {avg_accuracy:.4f}")
        
        return training_result
    
    def evaluate(self, use_test_data: bool = True) -> Tuple[float, float]:
        self.model.eval()
        
        if use_test_data:
            eval_dataset = self.test_dataset
            data_type = "测试"
        else:
            eval_dataset = self.train_dataset
            data_type = "训练"
        
        eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        
        self.logger.debug(f"客户端 {self.client_id} {data_type}数据评估: "
                         f"损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
        
        return avg_loss, accuracy
    

    def update_train_data(self, new_data, new_labels):
        self.train_data = new_data
        self.train_labels = new_labels
        
        self.train_dataset = TensorDataset(torch.from_numpy(new_data), torch.from_numpy(new_labels))
        print(f"Client {self.client_id}: 训练数据已更新 (Poisoned)")
    def get_training_history(self) -> Dict[str, List]:
        return self.training_history.copy()
    
    def get_data_statistics(self) -> Dict[str, Any]:
        train_labels_np = self.train_labels.cpu().numpy()
        test_labels_np = self.test_labels.cpu().numpy()
        
        unique_train, counts_train = np.unique(train_labels_np, return_counts=True)
        unique_test, counts_test = np.unique(test_labels_np, return_counts=True)
        
        train_distribution = dict(zip(unique_train.tolist(), counts_train.tolist()))
        test_distribution = dict(zip(unique_test.tolist(), counts_test.tolist()))
        
        return {
            'client_id': self.client_id,
            'train_samples': len(self.train_dataset),
            'test_samples': len(self.test_dataset),
            'train_label_distribution': train_distribution,
            'test_label_distribution': test_distribution,
            'is_malicious': self.is_malicious,
            'attack_config': self.attack_config
        }
    
    def save_model(self, filepath: str) -> None:
        torch.save({
            'client_id': self.client_id,
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'is_malicious': self.is_malicious,
            'attack_config': self.attack_config
        }, filepath)
        
        self.logger.info(f"客户端 {self.client_id} 模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)
        self.is_malicious = checkpoint.get('is_malicious', False)
        self.attack_config = checkpoint.get('attack_config', None)
        
        self.logger.info(f"客户端 {self.client_id} 模型已从 {filepath} 加载")
    
    def __str__(self) -> str:
        return (f"FederatedClient(id={self.client_id}, "
                f"train_samples={len(self.train_dataset)}, "
                f"test_samples={len(self.test_dataset)}, "
                f"malicious={self.is_malicious})")
    
    def __repr__(self) -> str:
        return self.__str__()