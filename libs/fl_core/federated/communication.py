import grpc
import threading
import queue
import time
import logging
import os
import torch
from concurrent import futures
from typing import Dict, Any, List, Tuple

try:
    import federated_comm_pb2 as pb2
    import federated_comm_pb2_grpc as pb2_grpc
except ImportError:
    from . import federated_comm_pb2 as pb2
    from . import federated_comm_pb2_grpc as pb2_grpc

from fl_core.utils.serialization import tensor_to_bytes, bytes_to_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GRPCComm")

class GRPCClientProxy:
    def __init__(self, client_id):
        self.client_id = client_id
        self.instruction_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_malicious = False 
        self.latest_global_params = None
        self._test_samples = 0
        self._train_samples = 0

    @property
    def test_dataset(self):
        class MockDataset:
            def __init__(self, count): self.count = count
            def __len__(self): return self.count
        return MockDataset(self._test_samples)
    
    @property
    def train_dataset(self):
        class MockDataset:
            def __init__(self, count): self.count = count
            def __len__(self): return self.count
        return MockDataset(self._train_samples)

    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        self.latest_global_params = parameters

    def train(self, epochs, learning_rate, batch_size, round_num, **kwargs):
        if self.latest_global_params is None:
            raise ValueError(f"客户端 {self.client_id} 尚未收到全局模型参数")
        
        param_bytes = tensor_to_bytes(self.latest_global_params)
        self.instruction_queue.put({
            'type': 'TRAIN', 'params': param_bytes, 'lr': learning_rate,
            'epochs': epochs, 'batch_size': batch_size, 'round': round_num
        })
        
        result = self.result_queue.get()
        if result.get('error'):
            raise Exception(f"Remote Client Error: {result['error']}")
        
        if 'updated_params' in result and isinstance(result['updated_params'], bytes):
            result['updated_params'] = bytes_to_tensor(result['updated_params'])
            self.latest_global_params = result['updated_params']

        self._train_samples = result.get('samples', 0)
        return result

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        return self.latest_global_params

    def evaluate(self, use_test_data: bool = True) -> Tuple[float, float]:
        self.instruction_queue.put({
            'type': 'EVALUATE', 'params': b'', 'use_test': use_test_data,
            'lr': 0, 'epochs': 0, 'batch_size': 0, 'round': 0
        })
        result = self.result_queue.get()
        self._test_samples = result.get('samples', 0)
        return result.get('loss', 0.0), result.get('accuracy', 0.0)

    def terminate(self):
        self.instruction_queue.put({
            'type': 'FINISH', 'params': b'', 'lr': 0, 'epochs': 0, 'batch_size': 0, 'round': 0
        })


class FederatedServiceServicer(pb2_grpc.FederatedServiceServicer):
    def __init__(self, on_client_connected=None):
        self.registered_proxies = {}
        self.on_client_connected = on_client_connected 

    def Register(self, request, context):
        client_id = request.client_id
        print(f"DEBUG: [Server] 收到注册请求: Client {client_id}", flush=True)
        
        if client_id not in self.registered_proxies:
            # 创建代理对象
            self.registered_proxies[client_id] = GRPCClientProxy(client_id)
            logger.info(f"Server: 客户端 {client_id} 注册成功")
        
        return pb2.RegistrationResponse(success=True, message="OK")

    def TaskStream(self, request_iterator, context):
        # 此时我们不再尝试从流里 next() 识别 ID，而是通过 Metadata 或者让客户端第一个包只发 ID
        client_id = None
        try:
            # 获取第一个包（握手包）
            print("DEBUG: [Server] 等待流初始化...", flush=True)
            first_msg = next(request_iterator)
            client_id = first_msg.client_id
            
            if client_id not in self.registered_proxies:
                # 防御性编程：如果没调 Register 就直接连 Stream
                self.registered_proxies[client_id] = GRPCClientProxy(client_id)

            proxy = self.registered_proxies[client_id]
            # 立即发送确认，打破客户端的 for 循环阻塞
            yield pb2.TaskRequest(task_type="CONNECTED")

            while context.is_active():
                try:
                    task = proxy.instruction_queue.get(timeout=0.2)
                    yield pb2.TaskRequest(
                        round_num=task.get('round', 0),
                        task_type=task['type'],
                        model_params=task['params'],
                        learning_rate=task['lr'],
                        epochs=task['epochs'],
                        batch_size=task['batch_size']
                    )
                    # 等待该任务的响应
                    response = next(request_iterator)
                    proxy.result_queue.put(self._parse_response(response))
                except queue.Empty:
                    continue
        except Exception as e:
            logger.error(f"Server: 客户端 {client_id} 流中断: {e}")

    def _parse_response(self, response):
        return {
            'client_id': response.client_id,
            'updated_params': response.updated_params,
            'loss': response.loss,
            'accuracy': response.accuracy,
            'samples': response.samples,
            'error': response.error
        }


class GRPCServer:
    def __init__(self, port=50051, on_client_connected=None):
        self.port = port
        self.servicer = FederatedServiceServicer(on_client_connected=on_client_connected)
        self.registered_proxies = self.servicer.registered_proxies
        self.server = None

    def start(self):
        options = [
            ('grpc.max_send_message_length', 128 * 1024 * 1024),
            ('grpc.max_receive_message_length', 128 * 1024 * 1024)
        ]
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=50), options=options)
        pb2_grpc.add_FederatedServiceServicer_to_server(self.servicer, self.server)
        # 使用 0.0.0.0 确保外网/内网均可连接
        bound_port = self.server.add_insecure_port(f'0.0.0.0:{self.port}')
        if bound_port == 0:
            raise RuntimeError(f"gRPC Server 端口绑定失败: 0.0.0.0:{self.port}")
        logger.info(f"gRPC Server 启动成功，监听端口 {bound_port}...")
        self.server.start()

    def stop(self):
        if self.server:
            self.server.stop(5)
            logger.info("gRPC Server 已停止。")


class GRPCWorker:
    def __init__(self, client_id, local_client, server_address):
        self.client_id = client_id
        self.local_client = local_client
        self.server_address = server_address
        self.response_queue = queue.Queue()

    def run(self):
        options = [
            ('grpc.max_send_message_length', 128 * 1024 * 1024),
            ('grpc.max_receive_message_length', 128 * 1024 * 1024),
            ('grpc.enable_retries', 1),
        ]
        register_timeout_seconds = int(os.environ.get("DIST_CLIENT_REGISTER_WAIT_SECONDS", "300"))
        retry_interval_seconds = float(os.environ.get("DIST_CLIENT_REGISTER_RETRY_INTERVAL_SECONDS", "2"))
        
        print(f"DEBUG: [Client {self.client_id}] 尝试建立到 {self.server_address} 的通道...", flush=True)
        
        with grpc.insecure_channel(self.server_address, options=options) as channel:
            stub = pb2_grpc.FederatedServiceStub(channel)
            
            deadline = time.time() + max(register_timeout_seconds, 1)
            registered = False
            while time.time() < deadline:
                try:
                    grpc.channel_ready_future(channel).result(timeout=2)
                    print(f"DEBUG: [Client {self.client_id}] 正在执行预注册...", flush=True)
                    reg_resp = stub.Register(pb2.ClientInfo(client_id=self.client_id), timeout=10)
                    print(f"DEBUG: [Client {self.client_id}] 注册响应: {reg_resp.success}")
                    if reg_resp.success:
                        registered = True
                        break
                    logger.warning(f"Client {self.client_id} 注册返回失败，稍后重试。")
                except grpc.FutureTimeoutError:
                    logger.info(f"Client {self.client_id} 等待服务端就绪中: {self.server_address}")
                except grpc.RpcError as e:
                    code = e.code().name if hasattr(e.code(), "name") else str(e.code())
                    logger.warning(
                        f"Client {self.client_id} 预注册失败: code={code}, details={e.details()}; 稍后重试。"
                    )
                except Exception as e:
                    logger.warning(f"Client {self.client_id} 预注册异常: {e}; 稍后重试。")

                time.sleep(max(retry_interval_seconds, 0.1))

            if not registered:
                logger.error(
                    f"Client {self.client_id} 在 {register_timeout_seconds}s 内未能注册到 {self.server_address}"
                )
                return

            def request_generator():
                # 发送第一个握手包
                yield pb2.TaskResponse(client_id=self.client_id)
                while True:
                    res = self.response_queue.get()
                    yield res

            responses = stub.TaskStream(request_generator())
            try:
                for request in responses:
                    if request.task_type == "CONNECTED":
                        print(f"DEBUG: [Client {self.client_id}] 流通道已激活。")
                        continue
                    self._handle_request(request)
            except Exception as e:
                logger.error(f"Client {self.client_id} 运行中断: {e}")

    def _handle_request(self, request):
        try:
            if request.task_type == "CONNECTED":
                print(f"DEBUG: [Client {self.client_id}] 服务器握手确认已收到。", flush=True)
                return
            
            elif request.task_type == "TRAIN":
                params = bytes_to_tensor(request.model_params)
                self.local_client.set_model_parameters(params)
                res = self.local_client.train(
                    epochs=request.epochs, 
                    learning_rate=request.learning_rate,
                    batch_size=request.batch_size if request.batch_size > 0 else 32,
                    round_num=request.round_num
                )
                updated_state_dict = self.local_client.get_model_state_dict()
                self.response_queue.put(pb2.TaskResponse(
                    client_id=self.client_id, 
                    updated_params=tensor_to_bytes(updated_state_dict),
                    loss=res['loss'], accuracy=res['accuracy'], samples=res['samples']
                ))
            
            elif request.task_type == "EVALUATE":
                if request.model_params:
                    self.local_client.set_model_parameters(bytes_to_tensor(request.model_params))
                loss, acc = self.local_client.evaluate(use_test_data=True)
                self.response_queue.put(pb2.TaskResponse(
                    client_id=self.client_id, loss=loss, accuracy=acc, 
                    samples=len(self.local_client.test_dataset)
                ))
                
            elif request.task_type == "FINISH":
                raise InterruptedError("Finished by server")
                
        except Exception as e:
            self.response_queue.put(pb2.TaskResponse(client_id=self.client_id, error=str(e)))
