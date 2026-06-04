from __future__ import annotations

import copy
import pickle
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from fl_core.simulation_registry import canonical_model_name, list_model_names

from .basic import (
    CharLSTM,
    DNN,
    FedAvgCNN,
    MclrLogistic,
    TextCNN,
    TextDNN,
    TextLogistic,
)
from .cnn import CNNMNIST, DeepCNN, SimpleCNN
from .lenet import LeNet, LeNetCIFAR
from .resnet import (
    ResNet18,
    ResNet18CIFAR,
    ResNet18MNIST,
    ResNet34,
    ResNet34CIFAR,
    ResNet50,
)


class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_registry: dict[str, Any] = {
            "mclr_logistic": MclrLogistic,
            "mclr": MclrLogistic,
            "logistic": MclrLogistic,
            "logistic_regression": MclrLogistic,
            "dnn": DNN,
            "mlp": DNN,
            "fedavgcnn": FedAvgCNN,
            "fedavg_cnn": FedAvgCNN,
            "lenet": LeNet,
            "lenet_cifar": LeNetCIFAR,
            "cnn": SimpleCNN,
            "simple_cnn": SimpleCNN,
            "deep_cnn": DeepCNN,
            "cnn_mnist": CNNMNIST,
            "resnet18": ResNet18,
            "resnet34": ResNet34,
            "resnet50": ResNet50,
            "resnet18_cifar": ResNet18CIFAR,
            "resnet34_cifar": ResNet34CIFAR,
            "resnet18_mnist": ResNet18MNIST,
            "alexnet": "torchvision_alexnet",
            "mobilenet": "torchvision_mobilenet_v2",
            "mobilenet_v2": "torchvision_mobilenet_v2",
            "mobilenetv2": "torchvision_mobilenet_v2",
            "googlenet": "torchvision_googlenet",
            "google_net": "torchvision_googlenet",
            "textlogistic": TextLogistic,
            "text_logistic": TextLogistic,
            "textdnn": TextDNN,
            "text_dnn": TextDNN,
            "textcnn": TextCNN,
            "text_cnn": TextCNN,
            "charlstm": CharLSTM,
            "char_lstm": CharLSTM,
            "char_rnn": CharLSTM,
        }

    def create_model(
        self,
        model_name: str,
        input_shape: tuple,
        num_classes: int,
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        canonical_name = canonical_model_name(model_name)
        registry_key = canonical_name.lower().replace("-", "_")

        if registry_key not in self.model_registry:
            raise ValueError(
                f"Unsupported model type: {model_name}. "
                f"Supported models: {list(self.model_registry.keys())}"
            )

        model_class = self.model_registry[registry_key]
        dataset_info = dataset_info or {}
        input_shape_tuple = tuple(input_shape or ())
        input_channels = input_shape_tuple[0] if len(input_shape_tuple) == 3 else 1
        input_dim = self._flatten_dim(input_shape_tuple)
        modality = dataset_info.get("modality", "image")

        if registry_key == "lenet":
            if len(input_shape_tuple) == 3 and input_shape_tuple[1:] == (32, 32):
                model = LeNetCIFAR(num_classes=num_classes, input_channels=input_channels)
            else:
                model = LeNet(num_classes=num_classes, input_channels=input_channels)
        elif registry_key == "fedavgcnn":
            model = FedAvgCNN(num_classes=num_classes, input_channels=input_channels)
        elif registry_key in {"cnn", "simple_cnn"}:
            if len(input_shape_tuple) == 3 and input_shape_tuple[1:] == (28, 28):
                model = CNNMNIST(num_classes=num_classes, input_channels=input_channels)
            else:
                model = SimpleCNN(num_classes=num_classes, input_channels=input_channels)
        elif registry_key == "resnet18":
            if len(input_shape_tuple) == 3 and input_shape_tuple[1:] == (32, 32):
                model = ResNet18CIFAR(num_classes=num_classes, input_channels=input_channels)
            elif len(input_shape_tuple) == 3 and input_shape_tuple[1:] == (28, 28):
                model = ResNet18MNIST(num_classes=num_classes, input_channels=input_channels)
            else:
                model = ResNet18(num_classes=num_classes, input_channels=input_channels)
        elif registry_key in {"mclr_logistic", "mclr", "logistic", "logistic_regression"}:
            model = MclrLogistic(num_classes=num_classes, input_shape=input_shape_tuple, input_dim=input_dim)
        elif registry_key in {"dnn", "mlp"}:
            model = DNN(
                num_classes=num_classes,
                input_shape=input_shape_tuple,
                input_dim=input_dim,
                hidden_dim=int(self.config.get("hidden_dim", 100)),
            )
        elif registry_key in {"textlogistic", "text_logistic"}:
            model = TextLogistic(
                num_classes=num_classes,
                input_dim=input_dim or int(dataset_info.get("feature_dim", 5000)),
            )
        elif registry_key in {"textdnn", "text_dnn"}:
            model = TextDNN(
                num_classes=num_classes,
                input_dim=input_dim or int(dataset_info.get("feature_dim", 5000)),
                hidden_dim=int(self.config.get("hidden_dim", 100)),
            )
        elif registry_key in {"textcnn", "text_cnn"}:
            model = TextCNN(
                num_classes=num_classes,
                input_dim=input_dim or int(dataset_info.get("feature_dim", 5000)),
            )
        elif registry_key in {"charlstm", "char_lstm", "char_rnn"}:
            model = CharLSTM(
                num_classes=num_classes,
                vocab_size=int(dataset_info.get("vocab_size", num_classes)),
                hidden_dim=int(self.config.get("hidden_dim", 128)),
            )
        elif isinstance(model_class, str) and model_class.startswith("torchvision_"):
            if modality != "image":
                raise ValueError(f"{canonical_name} only supports image datasets")
            model = self._create_torchvision_model(model_class, num_classes, input_channels)
        else:
            try:
                model = model_class(num_classes=num_classes, input_channels=input_channels)
            except TypeError:
                model = model_class(num_classes=num_classes)

        model.to(self.device)
        return model

    @staticmethod
    def _flatten_dim(input_shape: tuple) -> Optional[int]:
        if not input_shape:
            return None
        total = 1
        for dim in input_shape:
            total *= int(dim)
        return total

    def _create_torchvision_model(self, model_key: str, num_classes: int, input_channels: int) -> nn.Module:
        import torchvision.models as tv_models

        if model_key == "torchvision_alexnet":
            model = tv_models.alexnet(weights=None, num_classes=num_classes)
            if input_channels != 3:
                model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
            return model

        if model_key == "torchvision_mobilenet_v2":
            model = tv_models.mobilenet_v2(weights=None, num_classes=num_classes)
            if input_channels != 3:
                model.features[0][0] = nn.Conv2d(
                    input_channels,
                    32,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            return model

        if model_key == "torchvision_googlenet":
            model = tv_models.googlenet(weights=None, aux_logits=False, num_classes=num_classes)
            if input_channels != 3:
                model.conv1.conv = nn.Conv2d(
                    input_channels,
                    64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                )
            return model

        raise ValueError(f"Unsupported torchvision model key: {model_key}")

    def get_model_parameters(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {name: param.clone().detach() for name, param in model.named_parameters()}

    def get_model_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {name: param.clone().detach() for name, param in model.state_dict().items()}

    def set_model_parameters(self, model: nn.Module, parameters: Dict[str, torch.Tensor]) -> None:
        model_dict = model.state_dict()

        param_keys = set(parameters.keys())
        model_keys = set(model_dict.keys())
        if param_keys != model_keys:
            missing_keys = model_keys - param_keys
            unexpected_keys = param_keys - model_keys
            if missing_keys:
                print(f"Warning: missing parameter keys: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: unexpected parameter keys: {unexpected_keys}")

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
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": self.config,
                "model_class": model.__class__.__name__,
            },
            filepath,
        )

    def load_model(self, filepath: str, model_name: str, input_shape: tuple, num_classes: int) -> nn.Module:
        checkpoint = torch.load(filepath, map_location=self.device)
        model = self.create_model(model_name, input_shape, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
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
        return list_model_names()

    def model_summary(self, model: nn.Module, input_shape: tuple) -> str:
        total_params = self.get_model_size(model)
        trainable_params = self.get_trainable_parameters(model)

        summary = f"""
Model summary:
========
Model type: {model.__class__.__name__}
Input shape: {input_shape}
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Device: {self.device}
========
        """

        return summary.strip()
