import torch
import torch.nn as nn
import copy
import pickle
from typing import Dict, Any, Optional, Union
from collections import OrderedDict

from .lenet import LeNet, LeNetCIFAR
from .cnn import SimpleCNN, DeepCNN, CNNMNIST
from .resnet import (
    ResNet18, ResNet34, ResNet50,
    ResNet18CIFAR, ResNet34CIFAR,
    ResNet18MNIST
)


class ModelManager:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_registry = {
            'lenet': LeNet,
            'lenet_cifar': LeNetCIFAR,
            
            'cnn': SimpleCNN,
            'simple_cnn': SimpleCNN,
            'deep_cnn': DeepCNN,
            'cnn_mnist': CNNMNIST,
            
            'resnet18': ResNet18,
            'resnet34': ResNet34,
            'resnet50': ResNet50,
            'resnet18_cifar': ResNet18CIFAR,
            'resnet34_cifar': ResNet34CIFAR,
            'resnet18_mnist': ResNet18MNIST,
        }
    
    def create_model(self, model_name: str, input_shape: tuple, num_classes: int) -> nn.Module:
        model_name_lower = model_name.lower()
        
        if model_name_lower not in self.model_registry:
            raise ValueError(f"不支持的模型类型: {model_name}. 支持的模型: {list(self.model_registry.keys())}")
        
        model_class = self.model_registry[model_name_lower]
        input_channels = input_shape[0] if len(input_shape) == 3 else 1
        
        if model_name_lower == 'lenet':
            if input_shape[1:] == (32, 32):  # CIFAR数据集
                model = LeNetCIFAR(num_classes=num_classes, input_channels=input_channels)
            else:  # MNIST数据集
                model = LeNet(num_classes=num_classes, input_channels=input_channels)
        elif model_name_lower in ['cnn', 'simple_cnn']:
            if input_shape[1:] == (28, 28):  # MNIST数据集
                model = CNNMNIST(num_classes=num_classes, input_channels=input_channels)
            else:  # CIFAR数据集
                model = SimpleCNN(num_classes=num_classes, input_channels=input_channels)
        elif model_name_lower == 'resnet18':
            if input_shape[1:] == (32, 32):  # CIFAR数据集
                model = ResNet18CIFAR(num_classes=num_classes, input_channels=input_channels)
            elif input_shape[1:] == (28, 28):  # MNIST数据集
                model = ResNet18MNIST(num_classes=num_classes, input_channels=input_channels)
            else:  # 标准ImageNet尺寸
                model = ResNet18(num_classes=num_classes, input_channels=input_channels)
        else:
            # 直接使用指定的模型类
            model = model_class(num_classes=num_classes, input_channels=input_channels)
        
        model.to(self.device)
        return model
    
    def get_model_parameters(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {name: param.clone().detach() for name, param in model.named_parameters()}
    
    def get_model_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {name: param.clone().detach() for name, param in model.state_dict().items()}
    
    def set_model_parameters(self, model: nn.Module, parameters: Dict[str, torch.Tensor]) -> None:
        model_dict = model.state_dict()
        
        # 验证参数键是否匹配
        param_keys = set(parameters.keys())
        model_keys = set(model_dict.keys())
        
        if param_keys != model_keys:
            missing_keys = model_keys - param_keys
            unexpected_keys = param_keys - model_keys
            
            if missing_keys:
                print(f"警告: 缺少参数键: {missing_keys}")
            if unexpected_keys:
                print(f"警告: 意外的参数键: {unexpected_keys}")
        
        for name, param in parameters.items():
            if name in model_dict:
                model_dict[name].copy_(param.to(self.device))
    
    def set_model_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
        device_state_dict = {name: param.to(self.device) for name, param in state_dict.items()}
        model.load_state_dict(device_state_dict)
    
    def get_model_gradients(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone().detach()
            else:
                gradients[name] = torch.zeros_like(param)
        return gradients
    
    def set_model_gradients(self, model: nn.Module, gradients: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in gradients:
                if param.grad is None:
                    param.grad = gradients[name].clone().to(self.device)
                else:
                    param.grad.copy_(gradients[name].to(self.device))
    
    def serialize_parameters(self, parameters: Dict[str, torch.Tensor]) -> bytes:
        cpu_parameters = {name: param.cpu() for name, param in parameters.items()}
        return pickle.dumps(cpu_parameters)
    
    def deserialize_parameters(self, data: bytes) -> Dict[str, torch.Tensor]:
        parameters = pickle.loads(data)
        return {name: param.to(self.device) for name, param in parameters.items()}
    
    def save_model(self, model: nn.Module, filepath: str) -> None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': self.config,
            'model_class': model.__class__.__name__
        }, filepath)
    
    def load_model(self, filepath: str, model_name: str, input_shape: tuple, num_classes: int) -> nn.Module:
        checkpoint = torch.load(filepath, map_location=self.device)
        
        model = self.create_model(model_name, input_shape, num_classes)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def clone_model(self, model: nn.Module) -> nn.Module:
        cloned_model = copy.deepcopy(model)
        cloned_model.to(self.device)
        return cloned_model
    
    def get_model_size(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())
    
    def get_trainable_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def freeze_layers(self, model: nn.Module, layer_names: list) -> None:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
    
    def unfreeze_layers(self, model: nn.Module, layer_names: list) -> None:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
    
    def get_supported_models(self) -> list:
        return list(self.model_registry.keys())
    
    def model_summary(self, model: nn.Module, input_shape: tuple) -> str:
        total_params = self.get_model_size(model)
        trainable_params = self.get_trainable_parameters(model)
        
        summary = f"""
模型摘要:
========
模型类型: {model.__class__.__name__}
输入形状: {input_shape}
总参数数: {total_params:,}
可训练参数: {trainable_params:,}
设备: {self.device}
========
        """
        
        return summary.strip()