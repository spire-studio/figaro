import tenseal as ts
import torch
import numpy as np
from typing import Dict, List, Any
import logging

class CKKSManager:
    def __init__(self, 
                 poly_modulus_degree: int = 8192, 
                 coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60],
                 global_scale: float = 2**40):
        self.logger = logging.getLogger("CKKSManager")
        
        try:
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes
            )
            self.context.global_scale = global_scale
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
            self.logger.info(f"CKKS Context 初始化成功: Poly={poly_modulus_degree}")
        except Exception as e:
            self.logger.error(f"CKKS Context 初始化失败: {e}")
            raise e

    def encrypt_model(self, model_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        encrypted_dict = {}
        for name, tensor in model_dict.items():
            if 'num_batches_tracked' in name or not tensor.is_floating_point():
                continue
            data = tensor.flatten().detach().cpu().numpy()
            try:
                encrypted_dict[name] = ts.ckks_vector(self.context, data)
            except Exception as e:
                self.logger.warning(f"层 {name} 加密失败: {e}")
        return encrypted_dict

    def decrypt_model(self, 
                      encrypted_dict: Dict[str, Any], 
                      target_shapes: Dict[str, torch.Size], 
                      device: torch.device) -> Dict[str, torch.Tensor]:
        decrypted_dict = {}
        for name, enc_vector in encrypted_dict.items():
            if name not in target_shapes: continue
            try:
                dec_list = enc_vector.decrypt()
                target_len = np.prod(target_shapes[name])
                dec_data = dec_list[:target_len]
                tensor = torch.tensor(dec_data, dtype=torch.float32).view(target_shapes[name]).to(device)
                decrypted_dict[name] = tensor
            except Exception as e:
                self.logger.error(f"层 {name} 解密失败: {e}")
                decrypted_dict[name] = torch.zeros(target_shapes[name]).to(device)
        return decrypted_dict

    def aggregate_encrypted_models(self, 
                                   encrypted_models: List[Dict[str, Any]], 
                                   weights: List[float]) -> Dict[str, Any]:
        if not encrypted_models: return {}
        ref_model = encrypted_models[0]
        num_clients = len(encrypted_models)
        aggregated_dict = {}
        
        for name in ref_model.keys():
            try:
                agg_vector = encrypted_models[0][name] * weights[0]
                for i in range(1, num_clients):
                    if name in encrypted_models[i]:
                        client_vec = encrypted_models[i][name]
                        agg_vector = agg_vector + (client_vec * weights[i])
                aggregated_dict[name] = agg_vector
            except Exception as e:
                self.logger.error(f"聚合层 {name} 时出错: {e}")
        return aggregated_dict