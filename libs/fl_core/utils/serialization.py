import io
import torch
from collections import OrderedDict

def tensor_to_bytes(state_dict: OrderedDict) -> bytes:
    buffer = io.BytesIO()
    # 转换为 CPU 张量以确保分布式兼容性
    cpu_dict = {k: v.cpu() for k, v in state_dict.items()}
    torch.save(cpu_dict, buffer)
    return buffer.getvalue()

def bytes_to_tensor(data: bytes, device: torch.device = None) -> OrderedDict:
    buffer = io.BytesIO(data)
    # 加载时指定映射到当前设备的内存
    return torch.load(buffer, map_location=device if device else 'cpu', weights_only=True)