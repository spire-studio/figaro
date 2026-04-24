import torch
from typing import Dict
import logging

class GlobalTopKSparsifier:
    def __init__(self, ratio: float = 0.5):
        self.ratio = ratio
        self.logger = logging.getLogger("Sparsifier")

    def sparsify(self, update_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        if self.ratio >= 1.0:
            return update_dict

        # 1. 收集所有参与训练的参数（排除BN层统计量和标量）
        trainable_tensors = []
        for name, tensor in update_dict.items():
            if tensor.dim() > 0 and 'num_batches_tracked' not in name and 'running_' not in name:
                trainable_tensors.append(tensor.abs().view(-1))
        
        if not trainable_tensors:
            return update_dict

        # 2. 全局拼接并计算阈值
        all_abs_params = torch.cat(trainable_tensors)
        k = int(all_abs_params.numel() * self.ratio)
        
        if k == 0:
            return {k: torch.zeros_like(v) for k, v in update_dict.items()}

        # 获取第 k 大的值作为阈值
        threshold = torch.kthvalue(all_abs_params, all_abs_params.numel() - k + 1).values.item()

        # 3. 应用掩码
        sparse_update = {}
        for name, tensor in update_dict.items():
            # 不处理统计量
            if tensor.dim() == 0 or 'num_batches_tracked' in name or 'running_' in name:
                sparse_update[name] = tensor
                continue

            # 生成掩码
            mask = torch.abs(tensor) >= threshold
            sparse_update[name] = tensor * mask.float()
            
        return sparse_update