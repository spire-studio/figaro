import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict


class DataSplitter:
    def __init__(self, save_dir: str = "./client_data"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    
    def split_data_iid(self, data: np.ndarray, labels: np.ndarray, num_clients: int) -> List[Dict[str, np.ndarray]]:

        print(f"执行IID数据分割，客户端数量: {num_clients}")
        
        num_samples = len(data)
        samples_per_client = num_samples // num_clients
        
        indices = np.random.permutation(num_samples)
        
        client_data_list = []
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            if i == num_clients - 1:  
                end_idx = num_samples
            else:
                end_idx = (i + 1) * samples_per_client
            
            client_indices = indices[start_idx:end_idx]
            
            client_data = {
                'data': data[client_indices],
                'labels': labels[client_indices],
                'client_id': i,
                'num_samples': len(client_indices)
            }
            
            client_data_list.append(client_data)
            
            print(f"客户端 {i}: {len(client_indices)} 个样本")
        
        return client_data_list
    
    def split_data_non_iid(self, data: np.ndarray, labels: np.ndarray, 
                          num_clients: int, alpha: float = 0.5) -> List[Dict[str, np.ndarray]]:
        print(f"执行Non-IID数据分割，客户端数量: {num_clients}, alpha: {alpha}")
        
        num_classes = len(np.unique(labels))
        print(f"数据集类别数: {num_classes}")
        
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        client_data_indices = [[] for _ in range(num_clients)]
        
        for class_id in range(num_classes):
            indices = np.array(class_indices[class_id])
            num_samples_class = len(indices)
            
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            proportions = np.cumsum(proportions)
            
            np.random.shuffle(indices)
            
            start_idx = 0
            for client_id in range(num_clients):
                end_proportion = proportions[client_id]
                end_idx = int(end_proportion * num_samples_class)
                
                end_idx = min(end_idx, num_samples_class)
                
                if start_idx < end_idx:
                    client_data_indices[client_id].extend(indices[start_idx:end_idx])
                
                start_idx = end_idx
        
        client_data_list = []
        
        for client_id in range(num_clients):
            indices = client_data_indices[client_id]
            
            if len(indices) == 0:
                print(f"警告: 客户端 {client_id} 没有分配到数据")
                indices = np.random.choice(len(data), size=10, replace=False)
            
            indices = np.array(indices)
            
            client_data = {
                'data': data[indices],
                'labels': labels[indices],
                'client_id': client_id,
                'num_samples': len(indices)
            }
            
            client_data_list.append(client_data)
            
            unique_labels, counts = np.unique(labels[indices], return_counts=True)
            label_dist = dict(zip(unique_labels, counts))
            print(f"客户端 {client_id}: {len(indices)} 个样本, 类别分布: {label_dist}")
        
        return client_data_list
    
    def save_client_data(self, client_data_list: List[Dict[str, np.ndarray]], 
                        experiment_name: str = "default", 
                        dataset_info: Dict[str, Any] = None,
                        split_config: Dict[str, Any] = None) -> str:
        import json
        from datetime import datetime
        
        save_path = os.path.join(self.save_dir, experiment_name)
        os.makedirs(save_path, exist_ok=True)
        
        for client_data in client_data_list:
            client_id = client_data['client_id']
            filename = f"client_{client_id}.pkl"
            filepath = os.path.join(save_path, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(client_data, f)
        
        stats = self.get_split_statistics(client_data_list)
        
        metadata = {
            'experiment_info': {
                'name': experiment_name,
                'created_time': datetime.now().isoformat(),
                'total_clients': len(client_data_list),
                'total_samples': stats['total_samples']
            },
            'dataset_info': dataset_info or {},
            'split_config': split_config or {},
            'statistics': {
                'samples_per_client': stats['samples_per_client'],
                'avg_samples_per_client': float(stats['avg_samples_per_client']),
                'std_samples_per_client': float(stats['std_samples_per_client']),
                'min_samples_per_client': int(stats['min_samples_per_client']),
                'max_samples_per_client': int(stats['max_samples_per_client']),
                'global_label_distribution': {str(k): int(v) for k, v in stats['global_label_distribution'].items()}
            },
            'client_details': []
        }
        
        for i, client_data in enumerate(client_data_list):
            labels = client_data['labels']
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_dist = {str(int(label)): int(count) for label, count in zip(unique_labels, counts)}
            
            proportions = counts / len(labels)
            gini_coefficient = 1 - np.sum(proportions ** 2)
            
            client_info = {
                'client_id': int(client_data['client_id']),
                'num_samples': int(client_data['num_samples']),
                'data_shape': list(client_data['data'].shape),
                'label_distribution': label_dist,
                'num_classes': len(unique_labels),
                'gini_coefficient': float(gini_coefficient),
                'data_statistics': {
                    'mean': float(client_data['data'].mean()),
                    'std': float(client_data['data'].std()),
                    'min': float(client_data['data'].min()),
                    'max': float(client_data['data'].max())
                }
            }
            
            metadata['client_details'].append(client_info)
        
        meta_json_path = os.path.join(save_path, 'meta.json')
        with open(meta_json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        simple_metadata = {
            'num_clients': len(client_data_list),
            'total_samples': stats['total_samples'],
            'experiment_name': experiment_name
        }
        
        metadata_path = os.path.join(save_path, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(simple_metadata, f)
        
        print(f"客户端数据已保存到: {save_path}")
        print(f"元数据文件: meta.json, metadata.pkl")
        return save_path
    
    def load_client_data(self, experiment_name: str = "default") -> List[Dict[str, np.ndarray]]:
        load_path = os.path.join(self.save_dir, experiment_name)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"找不到实验数据: {load_path}")
        
        metadata_path = os.path.join(load_path, 'metadata.pkl')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"找不到元数据文件: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        num_clients = metadata['num_clients']
        
        client_data_list = []
        for client_id in range(num_clients):
            filename = f"client_{client_id}.pkl"
            filepath = os.path.join(load_path, filename)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"找不到客户端数据文件: {filepath}")
            
            with open(filepath, 'rb') as f:
                client_data = pickle.load(f)
            
            client_data_list.append(client_data)
        
        print(f"已加载 {num_clients} 个客户端的数据")
        return client_data_list
    
    def get_split_statistics(self, client_data_list: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        num_clients = len(client_data_list)
        total_samples = sum(cd['num_samples'] for cd in client_data_list)
        
        samples_per_client = [cd['num_samples'] for cd in client_data_list]
        
        all_labels = []
        client_label_distributions = []
        
        for client_data in client_data_list:
            labels = client_data['labels']
            all_labels.extend(labels)
            
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_dist = dict(zip(unique_labels.astype(int), counts.astype(int)))
            client_label_distributions.append(label_dist)
        
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        global_label_dist = dict(zip(unique_labels.astype(int), counts.astype(int)))
        
        statistics = {
            'num_clients': num_clients,
            'total_samples': total_samples,
            'samples_per_client': samples_per_client,
            'avg_samples_per_client': np.mean(samples_per_client),
            'std_samples_per_client': np.std(samples_per_client),
            'min_samples_per_client': np.min(samples_per_client),
            'max_samples_per_client': np.max(samples_per_client),
            'global_label_distribution': global_label_dist,
            'client_label_distributions': client_label_distributions
        }
        
        return statistics
    
    def list_saved_experiments(self) -> List[str]:
        if not os.path.exists(self.save_dir):
            return []
        
        experiments = []
        for item in os.listdir(self.save_dir):
            item_path = os.path.join(self.save_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.pkl')
                if os.path.exists(metadata_path):
                    experiments.append(item)
        
        return experiments


class FederatedDataManager:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.splitter = DataSplitter()
    
    def prepare_federated_data(self, train_data: np.ndarray, train_labels: np.ndarray,
                             test_data: np.ndarray, test_labels: np.ndarray,
                             save_data: bool = False, experiment_name: str = None) -> List[Dict]:
        
        num_clients = self.config.get('num_clients', 10)
        distribution = self.config.get('distribution', 'iid')
        alpha = self.config.get('alpha', 0.5)

        # 1. 分割训练集
        if distribution.lower() == 'iid':
            train_splits = self.splitter.split_data_iid(train_data, train_labels, num_clients)
            test_splits = self.splitter.split_data_iid(test_data, test_labels, num_clients)
        else:
            train_splits = self.splitter.split_data_non_iid(train_data, train_labels, num_clients, alpha)
            test_splits = self.splitter.split_data_non_iid(test_data, test_labels, num_clients, alpha)

        # 2. 合并训练集和测试集到同一个 client 字典中
        combined_client_data = []
        for i in range(num_clients):
            combined_node = {
                'client_id': i,
                'data': train_splits[i]['data'],      # 训练数据
                'labels': train_splits[i]['labels'],  # 训练标签
                'test_x': test_splits[i]['data'],     # 测试数据 
                'test_y': test_splits[i]['labels'],   # 测试标签
                'num_samples': train_splits[i]['num_samples']
            }
            combined_client_data.append(combined_node)

        # 3. 保存逻辑
        if save_data:
            if experiment_name is None:
                experiment_name = f"{self.config.get('name', 'dataset')}_{distribution}_clients_{num_clients}"
            
            dataset_info = {
                'name': self.config.get('name', 'unknown'),
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'input_shape': self.config.get('input_shape')
            }

            self.splitter.save_client_data(combined_client_data, experiment_name, dataset_info)
        
        return combined_client_data