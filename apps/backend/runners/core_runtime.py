import sys
import os
import time
import threading
import webbrowser
import pickle
import argparse
import traceback
import socket
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FL_CORE_PARENT = PROJECT_ROOT / "libs"
if str(FL_CORE_PARENT) not in sys.path:
    sys.path.insert(0, str(FL_CORE_PARENT))

from fl_core.utils.config import ConfigManager
from fl_core.utils.logger import Logger
from fl_core.data.data_loader import DataManager
from fl_core.data.data_splitter import DataSplitter, FederatedDataManager
from fl_core.models.model_manager import ModelManager
from fl_core.federated.server import FederatedServer
from fl_core.federated.client import FederatedClient
from fl_core.federated.client_manager import ClientManager
from fl_core.compression.sparsification import GlobalTopKSparsifier
from fl_core.privacy.encryption import CKKSManager
from fl_core.federated.communication import GRPCServer, GRPCWorker, GRPCClientProxy
WebApp = None


class FederatedLearningFramework:
    
    def __init__(self, config_path: str = "config.yaml", on_round_end_callback=None):
        self.config_path = config_path
        self.on_round_end_callback = on_round_end_callback
        self.config_manager = None
        self.logger = None
        self.data_manager = None
        self.data_splitter = None
        self.model_manager = None
        self.server = None
        self.client_manager = None
        self.web_app = None
        
        self.training_completed = False
        self.results_file = None
        
    def initialize_system(self) -> bool:
        try:
            print("联邦学习训练框架")
            print("Federated Learning Training Framework")
            print("=" * 60)
            
            print("正在加载配置文件...")
            self.config_manager = ConfigManager(self.config_path)
            self.config_manager.validate_config()
            print("✓ 配置文件加载成功")

            fed_cfg_early = self.config_manager.get_federated_config()
            seed = int(fed_cfg_early.get('seed', 42))
            self._apply_seed(seed)
            print(f"✓ 随机种子已固定: seed={seed}")

            system_cfg = self.config_manager.config.get('system', {})
            dist_cfg = self.config_manager.config.get('distributed', {})
            mode = system_cfg.get('mode', 'simulation')
            role = system_cfg.get('role', 'server')
            if mode == "distributed":
                if role == "server":
                    identity = "server"
                else:
                    client_id = dist_cfg.get('client_id', 'unknown')
                    identity = f"client_{client_id}"
            else:
                identity = "simulation"
            logging_config = self.config_manager.get_logging_config()
            self.logger = Logger(
                log_dir=logging_config.get('log_dir', './logs'),
                results_dir=logging_config.get('results_dir', './results'),
                identity=identity,
                on_round_end=self.on_round_end_callback
            )
            print("✓ 日志记录器初始化成功")
            
            experiment_info = self._get_experiment_info()
            self.logger.set_experiment_info(experiment_info)
            
            self._log_experiment_config(experiment_info)


            compression_config = self.config_manager.config.get('compression', {})
            sparsify_conf = compression_config.get('sparsification', {})
            
            self.sparsifier = None
            if sparsify_conf.get('enable', False):
                ratio = sparsify_conf.get('ratio', 0.5)
                self.sparsifier = GlobalTopKSparsifier(ratio=ratio)
                print(f"✓ 通信压缩: 稀疏化已启用 (Ratio: {ratio})")
            

            privacy_config = self.config_manager.config.get('privacy', {})
            ckks_conf = privacy_config.get('homomorphic_encryption', {})
            
            self.ckks_manager = None
            if ckks_conf.get('enable', False):
                # 暂不支持同时开启
                if self.sparsifier:
                    print("! 警告: 检测到同时开启了稀疏化和同态加密。优先使用加密，稀疏化将禁用。")
                    self.sparsifier = None
                
                self.ckks_manager = CKKSManager(
                    poly_modulus_degree=ckks_conf.get('poly_modulus_degree', 8192)
                )
                print(f"✓ 隐私保护: CKKS同态加密已启用")

            
            print("✓ 系统初始化完成")
            return True
            
        except Exception as e:
            import traceback
            print(f"✗ 系统初始化失败: {e}")
            traceback.print_exc()
            return False
    
    @staticmethod
    def _apply_seed(seed: int) -> None:
        """Fix all RNG sources so runs with the same config reproduce exactly."""
        import os
        import random
        import numpy as np
        import torch

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _get_experiment_info(self) -> Dict[str, Any]:
        dataset_cfg = self.config_manager.get_dataset_config()
        model_cfg = self.config_manager.get_model_config()
        fed_cfg = self.config_manager.get_federated_config()
        compression_cfg = self.config_manager.get_compression_config()
        privacy_cfg = self.config_manager.get_privacy_config()

        return {
            "basic": {
                "dataset_name": dataset_cfg.get('name', 'Unknown'),
                "distribution": dataset_cfg.get('distribution', 'iid'),
                "alpha": dataset_cfg.get('alpha', 'N/A'),
                "model_name": model_cfg.get('name', 'Unknown'),
                "num_classes": model_cfg.get('num_classes', 10)
            },
            "federated": {
                "num_rounds": fed_cfg.get('num_rounds', 0),
                "num_clients": fed_cfg.get('num_clients', 0),
                "clients_per_round": fed_cfg.get('clients_per_round', 0),
                "local_epochs": fed_cfg.get('local_epochs', 1),
                "learning_rate": fed_cfg.get('learning_rate', 0.01),
                "aggregation": fed_cfg.get('aggregation', 'fedavg')
            },
            "security": {
                "encryption": {
                    "enabled": privacy_cfg.get('homomorphic_encryption', {}).get('enable', False),
                    "type": "CKKS",
                    "poly_modulus_degree": privacy_cfg.get('homomorphic_encryption', {}).get('poly_modulus_degree', 0)
                },
                "compression": {
                    "enabled": compression_cfg.get('sparsification', {}).get('enable', False),
                    "type": "Global Top-K",
                    "ratio": compression_cfg.get('sparsification', {}).get('ratio', 0.0)
                }
            }
        }
    
    def _log_experiment_config(self, experiment_info: Dict[str, Any]) -> None:
        basic = experiment_info.get('basic', {})
        fed = experiment_info.get('federated', {})

        self.logger.log_info("=" * 50)
        self.logger.log_info("实验配置信息")
        self.logger.log_info("=" * 50)

        self.logger.log_info(f"数据集: {basic.get('dataset_name')}")
        self.logger.log_info(f"数据分布: {basic.get('distribution')} (Alpha: {basic.get('alpha')})")
        self.logger.log_info(f"模型: {basic.get('model_name')}")

        self.logger.log_info(f"客户端总数: {fed.get('num_clients')}")
        self.logger.log_info(f"训练轮次: {fed.get('num_rounds')}")
        self.logger.log_info(f"每轮参与客户端: {fed.get('clients_per_round')}")
        self.logger.log_info(f"本地训练轮数: {fed.get('local_epochs')}")
        self.logger.log_info(f"学习率: {fed.get('learning_rate')}")
        self.logger.log_info(f"聚合方法: {fed.get('aggregation')}")

        self.logger.log_info("=" * 50)

    def run_distributed_server(self):
        distributed_config = self.config_manager.get_distributed_config()
        port = distributed_config.get('port')

        self.logger.log_info(f"正在启动分布式服务器 (Port: {port})...")
        
        # 1. FederatedDataManager
        self.logger.log_info("\n[数据管理] 正在加载并分发数据到本地磁盘...")
        dataset_config = self.config_manager.get_dataset_config()
        
        # 初始化数据管理器
        self.data_manager = DataManager(dataset_config)
        fed_data_manager = FederatedDataManager(dataset_config)
        
        # 加载原始数据集
        train_data, train_labels, test_data, test_labels = self.data_manager.load_dataset()
        
        # 执行切分并保存
        fed_data_manager.prepare_federated_data(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            save_data=True
        )
        self.logger.log_info(f"✓ 所有客户端数据已就绪并保存")

        # 2. 设置模型和聚合服务器
        self.setup_models()
        federated_config = self.config_manager.get_federated_config()

        self.server = FederatedServer(
            global_model=self.global_model,
            aggregation_method=federated_config.get('aggregation', 'fedavg'),
            ckks_manager=self.ckks_manager,
        )

        # 3. 启动 gRPC 通信服务
        self.grpc_service = GRPCServer(port=port)
        # grpc_thread = threading.Thread(target=self.grpc_service.start, daemon=True)
        # grpc_thread.start()
        self.grpc_service.start()

        # 4. 初始化客户端管理器
        self.client_manager = ClientManager(
            clients=[],
            sparsifier=self.sparsifier,
            ckks_manager=self.ckks_manager
        )

        self.logger.log_info(f"等待 {federated_config.get('num_clients')} 个远程客户端连接...")
        # 阻塞直到所有物理客户端通过 gRPC 注册
        while len(self.grpc_service.registered_proxies) < federated_config.get('num_clients'):
            # print('server wait')
            time.sleep(1)
        
        connected_clients = list(self.grpc_service.registered_proxies.values())
        self.client_manager.clients = connected_clients
        self.client_manager.num_clients = len(connected_clients)

        self.client_manager.max_workers = max(1, min(4, len(connected_clients)))

        self.client_manager.training_stats['client_participation'] = {
            c.client_id: 0 for c in connected_clients
        }
        self.logger.log_info("✓ 所有客户端已连接，开始联邦训练。")
        
        # 5. 进入标准联邦训练循环
        self.run_federated_training()

        for proxy in self.client_manager.clients:
            if hasattr(proxy, 'terminate'):
                proxy.terminate()
        
        time.sleep(2)

        if hasattr(self, 'grpc_service'):
            self.logger.log_info("[系统] 正在关闭服务器...")
            self.grpc_service.stop()

    def run_distributed_client(self):
        
        distributed_config = self.config_manager.get_distributed_config()
        client_id = distributed_config.get('client_id')
        server_ip = distributed_config.get('server_ip')
        port = distributed_config.get('port')

        self.logger.log_info(f"正在启动客户端 ID: {client_id}...")
        
        # 1. 加载数据
        dataset_config = self.config_manager.get_dataset_config()
        self.data_manager = DataManager(dataset_config)
        fed_config = self.config_manager.get_federated_config()

        # 2. 设置模型
        self.setup_models()

        experiment_name = f"{dataset_config.get('name')}_{dataset_config.get('distribution')}_clients_{fed_config.get('num_clients')}"
        data_path = os.path.join("./client_data", experiment_name, f"client_{client_id}.pkl")
        self.logger.log_info(f"正在从路径加载数据: {data_path}")
        self._wait_for_client_data_file(client_id=client_id, data_path=data_path)

        with open(data_path, 'rb') as f:
            client_bundle = pickle.load(f)
        
        train_data = client_bundle.get('data')
        train_labels = client_bundle.get('labels')
        test_data = client_bundle.get('test_x')
        test_labels = client_bundle.get('test_y')

        self.logger.log_info(f"✓ 数据加载成功: client_{client_id}")
        self.logger.log_info(f"  - 训练样本数: {len(train_data)}")
        self.logger.log_info(f"  - 测试样本数: {len(test_data)}")

        # 4. 实例化真正的本地客户端
        local_client = FederatedClient(
            client_id=client_id,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            model=self.model_manager.clone_model(self.global_model)
        )

        # 5. 启动 Worker 连接服务器
        worker = GRPCWorker(
            client_id=client_id,
            local_client=local_client,
            server_address=f"{server_ip}:{port}"
        )
        self._wait_for_server_ready(client_id=client_id, server_ip=server_ip, port=port)
        self.logger.log_info(f"正在连接服务器 {server_ip}:{port} ...")
        worker.run() 

    def _wait_for_client_data_file(self, *, client_id: int, data_path: str) -> None:
        if os.path.exists(data_path):
            return

        timeout_seconds = int(os.environ.get("DIST_CLIENT_DATA_WAIT_SECONDS", "300"))
        self.logger.log_info(f"[dist] client-{client_id} waiting for data file {data_path}")
        deadline = time.time() + max(timeout_seconds, 1)
        while time.time() < deadline:
            if os.path.exists(data_path):
                return
            time.sleep(1)
        raise FileNotFoundError(
            f"Client data file not found after {timeout_seconds}s: {data_path}"
        )

    def _wait_for_server_ready(self, *, client_id: int, server_ip: str, port: int) -> None:
        timeout_seconds = int(os.environ.get("DIST_CLIENT_SERVER_WAIT_SECONDS", "300"))
        strict_wait = os.environ.get("DIST_CLIENT_SERVER_WAIT_STRICT", "0") == "1"
        self.logger.log_info(f"[dist] client-{client_id} waiting for server {server_ip}:{port}")
        deadline = time.time() + max(timeout_seconds, 1)
        last_error: OSError | None = None
        while time.time() < deadline:
            try:
                with socket.create_connection((server_ip, int(port)), timeout=1.0):
                    return
            except OSError as exc:
                last_error = exc
                time.sleep(1)
        detail = f", last_error={last_error}" if last_error else ""
        message = f"Server {server_ip}:{port} not reachable after {timeout_seconds}s{detail}"
        if strict_wait:
            raise TimeoutError(message)
        self.logger.log_warning(f"{message}; continue to gRPC register retries.")

    def prepare_data(self) -> bool:
        try:
            print("\n正在准备数据集...")
            
            dataset_config = self.config_manager.get_dataset_config()
            self.data_manager = DataManager(dataset_config)
            
            train_data, train_labels, test_data, test_labels = self.data_manager.load_dataset()
            self.logger.log_info(f"数据集加载完成 - 训练集: {len(train_data)}, 测试集: {len(test_data)}")
            
            federated_config = self.config_manager.get_federated_config()
            self.data_splitter = DataSplitter(save_dir="./client_data")
            
            num_clients = federated_config.get('num_clients')
            distribution = dataset_config.get('distribution', 'iid')
            
            if distribution.lower() == 'iid':
                train_client_data = self.data_splitter.split_data_iid(train_data, train_labels, num_clients)
                test_client_data = self.data_splitter.split_data_iid(test_data, test_labels, num_clients)
            else:  # non_iid
                alpha = dataset_config.get('alpha', 0.5)
                train_client_data = self.data_splitter.split_data_non_iid(train_data, train_labels, num_clients, alpha)
                test_client_data = self.data_splitter.split_data_non_iid(test_data, test_labels, num_clients, alpha)
            
            client_data = []
            for i in range(num_clients):
                client_data.append((
                    train_client_data[i]['data'],
                    train_client_data[i]['labels'],
                    test_client_data[i]['data'],
                    test_client_data[i]['labels']
                ))
            
            self.client_data = client_data
            
            self.logger.log_info(f"数据分割完成 - {len(client_data)} 个客户端")
            print("✓ 数据集准备完成")
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"数据准备失败: {e}")
            print(f"✗ 数据准备失败: {e}")
            return False
    
    def setup_models(self) -> bool:
        try:
            print("正在设置模型...")
            
            model_config = self.config_manager.get_model_config()
            self.model_manager = ModelManager(model_config)
            
            dataset_info = self.data_manager.get_dataset_info()
            
            global_model = self.model_manager.create_model(
                model_name=model_config.get('name'),
                input_shape=dataset_info['input_shape'],
                num_classes=dataset_info['num_classes']
            )
            
            self.global_model = global_model
            
            model_summary = self.model_manager.model_summary(global_model, dataset_info['input_shape'])
            self.logger.log_info("模型创建完成:")
            for line in model_summary.splitlines():
                if line.strip():
                    self.logger.log_info(line)
            
            print("✓ 模型设置完成")
            return True
            
        except Exception as e:
            self.logger.log_error(f"模型设置失败: {e}")
            print(f"✗ 模型设置失败: {e}")
            return False
    
    def setup_federated_components(self) -> bool:
        try:
            print("正在设置联邦学习组件...")
            
            federated_config = self.config_manager.get_federated_config()

            aggregation_method = federated_config.get('aggregation', 'fedavg')

            self.server = FederatedServer(
                global_model=self.global_model,
                aggregation_method=aggregation_method,
                ckks_manager=self.ckks_manager,
            )

            clients = []

            for client_id, (train_data, train_labels, test_data, test_labels) in enumerate(self.client_data):
                client_model = self.model_manager.clone_model(self.global_model)

                client = FederatedClient(
                    client_id=client_id,
                    train_data=train_data,
                    train_labels=train_labels,
                    test_data=test_data,
                    test_labels=test_labels,
                    model=client_model
                )

                clients.append(client)

            self.client_manager = ClientManager(
                clients=clients,
                sparsifier=self.sparsifier,
                ckks_manager=self.ckks_manager,
                selection_strategy="random"
            )
            # Parallel client training races the global torch RNG inside
            # DataLoader shuffle, which defeats seeding. Force sequential
            # so same seed → bit-exact reproducible runs.
            self.client_manager.set_parallel_training(False)

            self.logger.log_info(f"联邦学习组件设置完成 - 服务器: {aggregation_method}, 客户端: {len(clients)}")
            
            print("✓ 联邦学习组件设置完成")
            return True
            
        except Exception as e:
            self.logger.log_error(f"联邦学习组件设置失败: {e}")
            print(f"✗ 联邦学习组件设置失败: {e}")
            return False
    
    def run_federated_training(self) -> bool:
        try:
            print("\n开始联邦学习训练...")
            
            federated_config = self.config_manager.get_federated_config()
            num_rounds = federated_config.get('num_rounds')
            clients_per_round = federated_config.get('clients_per_round')
            local_epochs = federated_config.get('local_epochs')
            learning_rate = federated_config.get('learning_rate')
            
            all_clients = self.client_manager.clients
            
            for round_num in range(1, num_rounds + 1):
                print(f"\n轮次 {round_num}/{num_rounds}")
                self.logger.log_info(f"开始第 {round_num} 轮训练")
                
                # global_params = self.global_model.state_dict()

                # self.client_manager.clients = list(self.grpc_service.registered_proxies.values())
                # self.client_manager.num_clients = len(self.client_manager.clients)

                selected_clients = self.client_manager.select_clients(clients_per_round)         
                selected_ids = [client.client_id for client in selected_clients]
                self.logger.log_info(f"选择客户端: {selected_ids}")  
                global_params = self.server.get_global_model_state_dict()
                self.client_manager.broadcast_model_to_clients(all_clients, global_params)
                
                training_results = self.client_manager.train_clients(
                    selected_clients=all_clients,
                    epochs=local_epochs,
                    learning_rate=learning_rate,
                    round_num=round_num
                )
                
                client_models = self.client_manager.get_client_models(
                    selected_clients=selected_clients,
                    global_model_params=global_params)
                
                aggregated_params = self.server.aggregate_models(
                    client_models=client_models,
                    client_info=training_results
                )
                
                self.server.update_global_model(aggregated_params)
                
                client_eval_results = self.client_manager.evaluate_clients(all_clients)

                total_clients = 0
                total_loss = 0.0
                total_accuracy = 0.0
                for result in client_eval_results:
                    if 'error' not in result and result['samples'] > 0:
                        total_clients += 1
                        total_loss += result['loss']
                        total_accuracy += result['accuracy']

                if total_clients > 0:
                    global_loss = total_loss / total_clients
                    global_accuracy = total_accuracy / total_clients
                else:
                    self.logger.log_warning("所有客户端评估失败或样本数为0，无法计算全局指标。")
                    global_loss = float('inf')
                    global_accuracy = 0.0
                

                self.logger.log_round_results(
                    round_num=round_num,
                    global_loss=global_loss,
                    global_acc=global_accuracy,
                    client_train_results=training_results,
                    client_eval_results=client_eval_results
                )
                
                avg_client_loss = sum(r['loss'] for r in training_results) / len(training_results)
                avg_client_acc = sum(r['accuracy'] for r in training_results) / len(training_results)
                
                print(f"  全局模型 - 损失: {global_loss:.4f}, 准确率: {global_accuracy:.4f}")
                print(f"  客户端平均 - 损失: {avg_client_loss:.4f}, 准确率: {avg_client_acc:.4f}")

                self.server.next_round()
            
            self.results_file = self.logger.save_final_results()
            self.training_completed = True
            
            print(f"\n✓ 联邦学习训练完成！结果已保存到: {self.results_file}")
            self.logger.log_info("联邦学习训练完成")
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"联邦学习训练失败: {e}")
            traceback.print_exc()
            return False
    
    def start_web_interface(self) -> None:
        try:
            if WebApp is None:
                print("Web可视化模块不可用，已跳过。")
                if self.logger:
                    self.logger.log_warning("Web visualization module unavailable; skipped.")
                return

            web_config = self.config_manager.get_web_config()
            
            if not web_config.get('enable', True):
                print("Web界面已禁用")
                return
            
            print("\n正在启动Web可视化界面...")
            
            results_dir = self.logger.results_dir if self.logger else "./results"
            self.web_app = WebApp(results_dir=results_dir)
            
            host = web_config.get('host', 'localhost')
            port = web_config.get('port', 5000)
            
            port = self._find_available_port(host, port)
            
            def run_web_server():
                try:
                    self.web_app.run_server(host=host, port=port, debug=False)
                except Exception as e:
                    print(f"Web服务器启动失败: {e}")
                    if self.logger:
                        self.logger.log_error(f"Web服务器启动失败: {e}")
            
            web_thread = threading.Thread(target=run_web_server, daemon=True)
            web_thread.start()
            
            time.sleep(3)
            
            if self._check_web_server_health(host, port):
                try:
                    url = f"http://{host}:{port}"
                    webbrowser.open(url)
                    print(f"✓ Web界面已启动: {url}")
                    if self.logger:
                        self.logger.log_info(f"Web界面已启动: {url}")
                except Exception as e:
                    print(f"无法自动打开浏览器: {e}")
                    print(f"请手动访问: http://{host}:{port}")
            else:
                print("Web服务器启动可能失败，请检查日志")
            
        except Exception as e:
            print(f"Web界面启动失败: {e}")
            if self.logger:
                self.logger.log_error(f"Web界面启动失败: {e}")
    
    def _find_available_port(self, host: str, start_port: int) -> int:
        import socket
        
        for port in range(start_port, start_port + 10):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind((host, port))
                    return port
            except OSError:
                continue
        
        print(f"警告: 无法找到可用端口，使用默认端口 {start_port}")
        return start_port
    
    def _check_web_server_health(self, host: str, port: int) -> bool:
        import socket
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                return result == 0
        except Exception:
            return False
        
    def cleanup_resources(self) -> None:
        try:
            if hasattr(self, 'global_model') and self.global_model is not None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            
            temp_dirs = ['./temp', './cache']
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    try:
                        import shutil
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass
            
            if self.logger:
                self.logger.log_info("系统资源清理完成")
            print("✓ 资源清理完成")
            
        except Exception as e:
            print(f"资源清理失败: {e}")
    
    def run_simulation(self):
        try:
            # if not self.initialize_system():
            #     return False
            
            if not self.prepare_data():
                return False
            
            if not self.setup_models():
                return False
            
            if not self.setup_federated_components():
                return False
            
            if not self.run_federated_training():
                return False
            
            # self.start_web_interface()
            
            return True
            
        except KeyboardInterrupt:
            print("\n用户中断训练")
            self.logger.log_info("用户中断训练") if self.logger else None
            return False
        except Exception as e:
            print(f"运行失败: {e}")
            if self.logger:
                self.logger.log_error(f"运行失败: {e}")
            return False
        finally:
            self.cleanup_resources()
    def run(self):
        if not self.initialize_system(): return False
        
        sytem_config = self.config_manager.get_system_config()
        mode = sytem_config.get('mode', 'simulation')
        role = sytem_config.get('role', 'server')
        try:
            if mode == "simulation":
                return self.run_simulation()
            elif mode == "distributed":
                if role == "server":
                    self.run_distributed_server()
                else:
                    self.run_distributed_client()
                return True
        except KeyboardInterrupt:
            print("\n用户中断运行")
            return False
        finally:
            self.cleanup_resources()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()

    try:
        framework = FederatedLearningFramework(config_path=args.config)
        
        success = framework.run()
        
        if success:
            print("\n" + "=" * 60)
            print("联邦学习训练流程完成！")
            print("=" * 60)
            
            if framework.training_completed:
                print("训练已完成")
            
            sys.exit(0)
        else:
            print("\n联邦学习训练流程失败")
            sys.exit(1)
            
    except Exception as e:
        print(f"程序运行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
