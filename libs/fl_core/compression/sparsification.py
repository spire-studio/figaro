from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


def _is_trainable_float(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.is_floating_point()
        and tensor.dim() > 0
        and "num_batches_tracked" not in name
        and "running_" not in name
    )


class CompressionStrategy(ABC):
    method = "none"

    @abstractmethod
    def sparsify(self, update_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass


class GlobalTopKSparsifier(CompressionStrategy):
    method = "global_topk"

    def __init__(self, ratio: float = 0.5, **_: Any):
        self.ratio = ratio
        self.logger = logging.getLogger("GlobalTopKSparsifier")

    def sparsify(self, update_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.ratio >= 1.0:
            return update_dict

        trainable_tensors = [
            tensor.abs().view(-1)
            for name, tensor in update_dict.items()
            if _is_trainable_float(name, tensor)
        ]
        if not trainable_tensors:
            return update_dict

        all_abs_params = torch.cat(trainable_tensors)
        k = int(all_abs_params.numel() * self.ratio)
        if k == 0:
            return {
                name: torch.zeros_like(tensor) if _is_trainable_float(name, tensor) else tensor
                for name, tensor in update_dict.items()
            }

        threshold = torch.kthvalue(all_abs_params, all_abs_params.numel() - k + 1).values.item()
        sparse_update = {}
        for name, tensor in update_dict.items():
            if not _is_trainable_float(name, tensor):
                sparse_update[name] = tensor
                continue
            sparse_update[name] = tensor * (torch.abs(tensor) >= threshold).to(tensor.dtype)
        return sparse_update


class RandomKSparsifier(CompressionStrategy):
    method = "random_k"

    def __init__(self, ratio: float = 0.5, **_: Any):
        self.ratio = ratio
        self.logger = logging.getLogger("RandomKSparsifier")

    def sparsify(self, update_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.ratio >= 1.0:
            return update_dict
        sparse_update = {}
        for name, tensor in update_dict.items():
            if not _is_trainable_float(name, tensor):
                sparse_update[name] = tensor
                continue
            flat = tensor.view(-1)
            k = int(flat.numel() * self.ratio)
            if k <= 0:
                sparse_update[name] = torch.zeros_like(tensor)
                continue
            mask = torch.zeros(flat.numel(), device=tensor.device, dtype=torch.bool)
            indices = torch.randperm(flat.numel(), device=tensor.device)[:k]
            mask[indices] = True
            sparse_update[name] = (flat * mask.to(flat.dtype)).view_as(tensor)
        return sparse_update


class ThresholdSparsifier(CompressionStrategy):
    method = "threshold"

    def __init__(self, threshold: float = 1e-3, **_: Any):
        self.threshold = float(threshold)
        self.ratio = 1.0
        self.logger = logging.getLogger("ThresholdSparsifier")

    def sparsify(self, update_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sparse_update = {}
        for name, tensor in update_dict.items():
            if not _is_trainable_float(name, tensor):
                sparse_update[name] = tensor
                continue
            sparse_update[name] = tensor * (torch.abs(tensor) >= self.threshold).to(tensor.dtype)
        return sparse_update


class SignCompressor(CompressionStrategy):
    method = "sign"

    def __init__(self, **_: Any):
        self.ratio = 1.0
        self.logger = logging.getLogger("SignCompressor")

    def sparsify(self, update_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        compressed = {}
        for name, tensor in update_dict.items():
            if not _is_trainable_float(name, tensor):
                compressed[name] = tensor
                continue
            mean_abs = tensor.abs().mean().clamp_min(1e-12)
            compressed[name] = tensor.sign() * mean_abs
        return compressed


class QuantInt8Compressor(CompressionStrategy):
    method = "quant_int8"

    def __init__(self, **_: Any):
        self.ratio = 1.0
        self.logger = logging.getLogger("QuantInt8Compressor")

    def sparsify(self, update_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        compressed = {}
        for name, tensor in update_dict.items():
            if not _is_trainable_float(name, tensor):
                compressed[name] = tensor
                continue
            scale = tensor.abs().max().clamp_min(1e-12) / 127.0
            quantized = torch.clamp(torch.round(tensor / scale), -127, 127)
            compressed[name] = quantized * scale
        return compressed


class CompressionFactory:
    _strategies = {
        "global_topk": GlobalTopKSparsifier,
        "topk": GlobalTopKSparsifier,
        "random_k": RandomKSparsifier,
        "randomk": RandomKSparsifier,
        "threshold": ThresholdSparsifier,
        "sign": SignCompressor,
        "quant_int8": QuantInt8Compressor,
        "int8": QuantInt8Compressor,
    }

    @classmethod
    def create(cls, method: str = "global_topk", **kwargs: Any) -> CompressionStrategy:
        key = method.lower()
        if key not in cls._strategies:
            raise ValueError(f"Unsupported compression method: {method}. Supported: {list(cls._strategies.keys())}")
        return cls._strategies[key](**kwargs)

    @classmethod
    def supported_methods(cls) -> list[str]:
        return ["global_topk", "random_k", "threshold", "sign", "quant_int8"]
