import os
import json
import logging
import datetime
from typing import Dict, List, Any
from pathlib import Path


class Logger:
    def __init__(self, log_dir: str = "./logs", results_dir: str = "./results", identity: str = "node", on_round_end=None):
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        self.identity = identity
        self.on_round_end = on_round_end
        
        # 创建目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志记录器
        self._setup_logger()
        
        # 初始化结果存储
        self.training_results = {
            "experiment_info": {},
            "global_results": {
                "rounds": [],
                "global_loss": [],
                "global_accuracy": []
            },
            "client_results": {}
        }

        override_name = os.environ.get("FIGARO_RESULTS_FILE", "").strip()
        if override_name:
            safe_name = Path(override_name).name
            if safe_name.endswith(".json"):
                self.current_session_filename = safe_name
            else:
                self.current_session_filename = f"{safe_name}.json"
        else:
            self.current_session_filename = f"live_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        self.save_results(self.current_session_filename)
    
    def _setup_logger(self):
        # 可通过环境变量关闭文件日志（由上层 runtime 统一落盘）。
        disable_file_log = os.environ.get("FIGARO_DISABLE_FILE_LOG", "0") == "1"
        # 支持显式指定日志文件路径（相对路径基于 log_dir）。
        override_log_file = os.environ.get("FIGARO_LOG_FILE", "").strip()
        log_filename: Path | None = None
        if not disable_file_log:
            if override_log_file:
                override_path = Path(override_log_file)
                log_filename = override_path if override_path.is_absolute() else self.log_dir / override_path
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = self.log_dir / f"training_{self.identity}_{timestamp}.log"
        
        self.logger = logging.getLogger(f"Figaro_{self.identity}")
        self.logger.setLevel(logging.INFO)
        # Do not propagate to the root logger — otherwise Python's lastResort
        # handler re-emits every record as "INFO:Figaro_x:..." which duplicates
        # every subprocess log line in the backend's captured stream.
        self.logger.propagate = False

        # 防止重复添加 handler（在某些环境中多次初始化会导致日志翻倍）
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 创建控制台处理器
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        if log_filename is not None:
            log_filename.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_filename, encoding='utf-8', delay=False)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'))
            self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        
        self.logger.info(f"日志系统启动，身份标识: {self.identity}")
        if log_filename is not None:
            self.logger.info(f"日志文件路径: {log_filename}")
        else:
            self.logger.info("日志文件写入已禁用（FIGARO_DISABLE_FILE_LOG=1）")
    
    def set_experiment_info(self, experiment_info: Dict[str, Any]):
        self.training_results["experiment_info"] = experiment_info
        self.logger.info(f"实验信息设置: {experiment_info}")
    
    def log_round_results(self, round_num: int, global_loss: float, 
                         global_acc: float, client_train_results: List[Dict[str, float]],
                         client_eval_results: List[Dict[str, float]] = None):
        
        self.training_results["global_results"]["rounds"].append(round_num)
        self.training_results["global_results"]["global_loss"].append(global_loss)
        self.training_results["global_results"]["global_accuracy"].append(global_acc)
        
        for i in range(len(client_train_results)):
            train_res = client_train_results[i]
            eval_res = client_eval_results[i] if client_eval_results and i < len(client_eval_results) else {}
            
            client_id = f"client_{train_res['client_id']}"
            
            if client_id not in self.training_results["client_results"]:
                self.training_results["client_results"][client_id] = {
                    "train_loss": [], "train_acc": [],
                    "test_loss": [], "test_acc": []
                }
            
            # 存储训练指标
            self.training_results["client_results"][client_id]["train_loss"].append(train_res.get('loss', 0))
            self.training_results["client_results"][client_id]["train_acc"].append(train_res.get('accuracy', 0))
            
            # 存储测试指标
            self.training_results["client_results"][client_id]["test_loss"].append(eval_res.get('loss', 0))
            self.training_results["client_results"][client_id]["test_acc"].append(eval_res.get('accuracy', 0))

        self.save_results(self.current_session_filename)

        # if self.on_round_end:
        #     latest_data = {
        #         "round": round_num,
        #         "global_loss": global_loss,
        #         "global_accuracy": global_acc,
        #         "client_train": client_train_results, 
        #         "client_test": client_eval_results 
        #     }
        #     self.on_round_end(latest_data)

        self.logger.info(f"轮次 {round_num}: 全局损失={global_loss:.4f}, 全局准确率={global_acc:.4f}")
    
    def save_results(self, filename: str = None) -> str:
        filename = filename or self.current_session_filename
        filepath = self.results_dir / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.training_results, f, ensure_ascii=False, indent=2)
            return str(filepath)
        except Exception as e:
            self.logger.error(f"保存中间结果失败: {e}")
            return ""
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        # 如果是相对路径，则在results_dir中查找
        if not os.path.isabs(filename):
            filepath = self.results_dir / filename
        else:
            filepath = Path(filename)
        
        if not filepath.exists():
            raise FileNotFoundError(f"结果文件不存在: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            self.logger.info(f"成功加载结果文件: {filepath}")
            return results
            
        except Exception as e:
            self.logger.error(f"加载结果文件失败: {e}")
            raise
    
    def get_results_files(self) -> List[str]:
        try:
            json_files = list(self.results_dir.glob("*.json"))
            return [f.name for f in json_files]
        except Exception as e:
            self.logger.error(f"获取结果文件列表失败: {e}")
            return []
    
    def get_results_files_with_info(self) -> List[Dict[str, Any]]:
        try:
            json_files = list(self.results_dir.glob("*.json"))
            files_info = []
            
            for file_path in json_files:
                try:
                    # 获取文件基本信息
                    stat = file_path.stat()
                    file_info = {
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "modified_time": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "experiment_info": {}
                    }
                    
                    # 尝试读取实验信息
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if "experiment_info" in data:
                                file_info["experiment_info"] = data["experiment_info"]
                    except:
                        # 如果读取失败，保持空的experiment_info
                        pass
                    
                    files_info.append(file_info)
                    
                except Exception as e:
                    self.logger.warning(f"获取文件 {file_path.name} 信息失败: {e}")
                    continue
            
            # 按修改时间排序（最新的在前）
            files_info.sort(key=lambda x: x["modified_time"], reverse=True)
            return files_info
            
        except Exception as e:
            self.logger.error(f"获取结果文件信息失败: {e}")
            return []
    
    def log_info(self, message: str):
        self.logger.info(message)
        self._flush_handlers()
    
    def log_warning(self, message: str):
        self.logger.warning(message)
        self._flush_handlers()
    
    def log_error(self, message: str):
        self.logger.error(message)
        self._flush_handlers()
    
    def log_debug(self, message: str):
        self.logger.debug(message)
        self._flush_handlers()

    def _flush_handlers(self):
        for handler in self.logger.handlers:
            handler.flush()
    
    def save_final_results(self, filename: str = None) -> str:
        return self.save_results(filename)
