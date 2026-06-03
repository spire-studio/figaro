from datetime import datetime, timezone
from types import SimpleNamespace

from app.services.agent.objectives import AgentOptimizationObjective
from app.services.agent.state import AgentState, ExperimentRecord


def _metrics_payload(accuracy: float) -> dict:
    return {
        "experiment_info": {},
        "global_results": {
            "rounds": [1],
            "global_loss": [0.1],
            "global_accuracy": [accuracy],
        },
        "client_results": {},
    }


def test_agent_config_schema_endpoint_returns_ui_metadata(client):
    response = client.get("/api/v1/agent/config/schema")

    assert response.status_code == 200
    payload = response.json()
    assert "Auto" in payload["model"]["name"]["options"]
    assert "FedAvgCNN" in payload["model"]["name"]["options"]
    assert "TextDNN" in payload["model"]["name"]["options"]
    assert "CharLSTM" in payload["model"]["name"]["options"]
    llm_task_ui = payload["task"]["type"]["ui"]["options"]["llm_peft_sft"]
    assert llm_task_ui["badge"] == "simulation"
    assert "disabled" not in llm_task_ui
    assert payload["llm"]["base_model"]["type"] == "text"
    assert "Qwen/Qwen2.5-0.5B-Instruct" in payload["llm"]["base_model"]["options"]
    assert payload["llm"]["base_model"]["ui"]["option_source"]["path"] == "./models/llm"
    assert payload["sft"]["dataset_path"]["default"] == "./datasets/llm/train.jsonl"
    assert "./datasets/llm/train.jsonl" in payload["sft"]["dataset_path"]["options"]
    assert payload["sft"]["dataset_path"]["ui"]["option_source"]["path"] == "./datasets/llm"
    assert payload["sft"]["file_format"]["options"] == ["auto", "jsonl", "parquet"]
    assert payload["sft"]["validation_split"]["default"] == 0.0
    assert "alpaca" in payload["sft"]["format"]["options"]
    assert payload["evaluation"]["dataset_path"]["default"] == "./datasets/llm/validation.jsonl"
    assert "./datasets/llm/validation.jsonl" in payload["evaluation"]["dataset_path"]["options"]
    assert payload["evaluation"]["dataset_path"]["ui"]["option_source"]["path"] == "./datasets/llm"
    aggregation_ui = payload["federated"]["aggregation"]["ui"]
    assert aggregation_ui["featured"] is True
    assert aggregation_ui["options"]["fedprox"]["disabled"] is True
    assert aggregation_ui["options"]["scaffold"]["badge"] == "experimental"


def test_agent_optimize_endpoint_handles_agent_state_result(client, monkeypatch):
    import app.api.v1.endpoints.agent as agent_module

    class _FakeCompiledGraph:
        async def ainvoke(self, _state):
            return AgentState(
                goal="maximize accuracy",
                job_name="opt-job-a",
                current_job_name="opt-job-a",
                max_iterations=2,
                iteration=2,
                objective=AgentOptimizationObjective.ACCURACY,
                resolved_objective=AgentOptimizationObjective.ACCURACY,
                history=[
                    ExperimentRecord(
                        iteration=1,
                        run_id="run-1",
                        job_id=10,
                        name="exp-1",
                        iteration_goal="baseline",
                        plan_summary="run baseline config",
                        config_patch={},
                        config_diff=[],
                        score=0.8,
                        result_summary="baseline complete",
                        decision="recorded",
                        lessons_learned=[],
                        config={"federated": {"num_clients": 10}},
                        metrics=_metrics_payload(0.8),
                    )
                ],
                summary="done",
            )

    class _FakeGraph:
        @staticmethod
        def compile():
            return _FakeCompiledGraph()

    class _FakeBuilder:
        def __init__(self, *, llm_service, session):
            self.llm_service = llm_service
            self.session = session

        @staticmethod
        def build():
            return _FakeGraph()

    monkeypatch.setattr(agent_module, "FederatedAgentGraphBuilder", _FakeBuilder)

    response = client.post(
        "/api/v1/agent/optimize",
        json={
            "goal": "maximize accuracy",
            "max_iterations": 2,
            "system_mode": "simulation",
            "model_name": "gpt-test",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["goal"] == "maximize accuracy"
    assert payload["job_name"] == "opt-job-a"
    assert payload["iterations_executed"] == 2
    assert payload["experiments"][0]["run_id"] == "run-1"
    assert payload["best_config"] == {"federated": {"num_clients": 10}}
    assert payload["best_metrics"]["global_results"]["global_accuracy"] == [0.8]
    assert payload["summary_text"] == "done"


def test_agent_optimize_endpoint_handles_dict_result(client, monkeypatch):
    import app.api.v1.endpoints.agent as agent_module

    class _FakeCompiledGraph:
        async def ainvoke(self, _state):
            return {
                "goal": "stability",
                "job_name": "opt-job-b",
                "current_job_name": "opt-job-b",
                "max_iterations": 1,
                "iteration": 1,
                "objective": AgentOptimizationObjective.AUTO,
                "resolved_objective": AgentOptimizationObjective.ACCURACY,
                "history": [
                    ExperimentRecord(
                        iteration=1,
                        run_id="run-2",
                        job_id=11,
                        name="exp-1",
                        iteration_goal="single run",
                        plan_summary="run default config",
                        config_patch={},
                        config_diff=[],
                        score=0.88,
                        result_summary="single-run",
                        decision="recorded",
                        lessons_learned=[],
                        config={"federated": {"num_clients": 10}},
                        metrics=_metrics_payload(0.88),
                    )
                ],
                "summary": "single-run",
            }

    class _FakeGraph:
        @staticmethod
        def compile():
            return _FakeCompiledGraph()

    class _FakeBuilder:
        def __init__(self, *, llm_service, session):
            self.llm_service = llm_service
            self.session = session

        @staticmethod
        def build():
            return _FakeGraph()

    monkeypatch.setattr(agent_module, "FederatedAgentGraphBuilder", _FakeBuilder)

    response = client.post(
        "/api/v1/agent/optimize",
        json={"goal": "stability", "max_iterations": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["goal"] == "stability"
    assert payload["job_name"] == "opt-job-b"
    assert payload["objective"] == "auto"
    assert payload["resolved_objective"] == "accuracy"
    assert payload["iterations_executed"] == 1
    assert payload["experiments"][0]["job_id"] == 11
    assert payload["best_config"] == {"federated": {"num_clients": 10}}
    assert payload["best_metrics"]["global_results"]["global_accuracy"] == [0.88]
    assert payload["summary_text"] == "single-run"


def test_agent_optimize_start_endpoint_returns_live_progress(client, monkeypatch):
    import app.api.v1.endpoints.agent as agent_module

    # ADD `planned_experiments` here 👇
    async def _fake_start_optimization(*, goal, max_iterations, system_mode, model_name, job_name, objective, planned_experiments, config_constraints):
        assert goal == "live optimize"
        assert max_iterations == 3
        assert system_mode == "simulation"
        assert model_name == "gpt-live"
        assert job_name == "opt-job-live"
        assert planned_experiments is None # Optional: verify the default value is passed
        assert config_constraints == {}
        return {
            "task_id": "task-1",
            "status": "running",
            "goal": goal,
            "job_name": job_name,
            "max_iterations": max_iterations,
            "model_name": model_name,
            "objective": objective,
            "resolved_objective": AgentOptimizationObjective.ACCURACY,
            "current_phase": "running",
            "current_iteration": 1,
            "completed_iterations": 0,
            "current_plan": {
                "iteration": 1,
                "iteration_goal": "run first experiment",
                "plan_summary": "testing alpha=0.1",
                "hypothesis": None,
                "rationale": [],
                "config_patch": {"federated": {"num_rounds": 10}},
                "config_diff": [],
            },
            "current_experiment": {
                "iteration": 1,
                "phase": "running",
                "job_id": 42,
                "job_name": job_name,
                "run_id": "run-live",
                "run_status": "running",
                "config": {"federated": {"num_rounds": 10}},
                "metrics": None,
            },
            "best_config": None,
            "best_metrics": None,
            "experiments": [],
            "summary_text": None,
            "error_message": None,
        }

    monkeypatch.setattr(agent_module.agent_runtime_service, "start_optimization", _fake_start_optimization)

    response = client.post(
        "/api/v1/agent/optimize/start",
        json={
            "goal": "live optimize",
            "max_iterations": 3,
            "system_mode": "simulation",
            "model_name": "gpt-live",
            "job_name": "opt-job-live",
            "objective": "accuracy",
        },
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["task_id"] == "task-1"
    assert payload["job_name"] == "opt-job-live"
    assert payload["status"] == "running"
    assert payload["current_plan"]["iteration_goal"] == "run first experiment"
    assert payload["current_experiment"]["run_id"] == "run-live"


def test_agent_optimize_progress_endpoint_returns_snapshot(client, monkeypatch):
    import app.api.v1.endpoints.agent as agent_module

    async def _fake_get_task(task_id):
        assert task_id == "task-2"
        return {
            "task_id": task_id,
            "status": "completed",
            "goal": "progress goal",
            "job_name": "opt-job-progress",
            "max_iterations": 2,
            "model_name": None,
            "objective": "auto",
            "resolved_objective": "accuracy",
            "current_phase": "completed",
            "current_iteration": 2,
            "completed_iterations": 2,
            "current_plan": {
                "iteration": 2,
                "iteration_goal": "run second experiment",
                "plan_summary": "test alpha=0.5",
                "hypothesis": None,
                "rationale": [],
                "config_patch": {"federated": {"num_rounds": 20}},
                "config_diff": [],
            },
            "current_experiment": {
                "iteration": 2,
                "phase": "completed",
                "job_id": 99,
                "job_name": "opt-job-progress",
                "run_id": "run-final",
                "run_status": "finished",
                "config": {"federated": {"num_rounds": 20}},
                "metrics": _metrics_payload(0.95),
            },
            "best_config": None,
            "best_metrics": None,
            "experiments": [
                {
                    "run_id": "run-final",
                    "job_id": 99,
                    "iteration": 2,
                    "iteration_goal": "run second experiment",
                    "plan_summary": "test alpha=0.5",
                    "hypothesis": None,
                    "rationale": [],
                    "config_patch": {},
                    "config_diff": [],
                    "config": {"federated": {"num_rounds": 20}},
                    "metrics": _metrics_payload(0.95),
                    "score": 0.95,
                    "result_summary": "completed",
                    "decision": "recorded",
                    "lessons_learned": [],
                }
            ],
            "summary_text": "completed live task",
            "error_message": None,
        }

    monkeypatch.setattr(agent_module.agent_runtime_service, "get_task", _fake_get_task)

    response = client.get("/api/v1/agent/optimize/task-2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["job_name"] == "opt-job-progress"
    assert payload["resolved_objective"] == "accuracy"
    assert payload["completed_iterations"] == 2
    assert payload["experiments"][0]["job_id"] == 99
    assert payload["best_config"] == {"federated": {"num_rounds": 20}}
    assert payload["best_metrics"]["global_results"]["global_accuracy"] == [0.95]


def test_agent_optimization_jobs_history_endpoints(client, monkeypatch):
    import app.api.v1.endpoints.agent as agent_module

    now = datetime.now(timezone.utc)
    snapshot = {
        "optimization_job_id": 7,
        "task_id": "task-history-1",
        "status": "completed",
        "goal": "benchmark alpha sweep",
        "job_name": "cifar10-alpha-sweep",
        "max_iterations": 4,
        "model_name": "gpt-5.4",
        "objective": "auto",
        "resolved_objective": "accuracy",
        "current_phase": "completed",
        "current_iteration": 4,
        "completed_iterations": 4,
        "current_plan": None,
        "current_experiment": None,
        "best_config": None,
        "best_metrics": None,
        "experiments": [],
        "summary_text": "done",
        "error_message": None,
        "created_at": now,
        "updated_at": now,
        "finished_at": now,
    }
    history_job = SimpleNamespace(
        id=7,
        task_id="task-history-1",
        job_name="cifar10-alpha-sweep",
        status="completed",
        goal="benchmark alpha sweep",
        model_name="gpt-5.4",
        current_phase="completed",
        max_iterations=4,
        current_iteration=4,
        completed_iterations=4,
        best_score=0.93,
        created_at=now,
        updated_at=now,
        finished_at=now,
        snapshot_json=snapshot,
    )

    class _FakeHistoryService:
        def __init__(self, _session):
            self._session = _session

        async def list_jobs(self, **_filters):
            return [history_job]

        async def get_job_or_raise(self, optimization_job_id):
            assert optimization_job_id == 7
            return history_job

    monkeypatch.setattr(agent_module, "AgentOptimizationHistoryService", _FakeHistoryService)

    list_response = client.get("/api/v1/agent/optimization-jobs")
    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert list_payload[0]["optimization_job_id"] == 7
    assert list_payload[0]["job_name"] == "cifar10-alpha-sweep"
    assert list_payload[0]["resolved_objective"] == "accuracy"
    assert list_payload[0]["best_score"] == 0.93

    detail_response = client.get("/api/v1/agent/optimization-jobs/7")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["optimization_job_id"] == 7
    assert detail_payload["job_name"] == "cifar10-alpha-sweep"
    assert detail_payload["resolved_objective"] == "accuracy"


def test_agent_optimization_jobs_history_filters_are_forwarded(client, monkeypatch):
    import app.api.v1.endpoints.agent as agent_module

    captured = {}

    class _FakeHistoryService:
        def __init__(self, _session):
            self._session = _session

        async def list_jobs(self, **filters):
            captured.update(filters)
            return []

    monkeypatch.setattr(agent_module, "AgentOptimizationHistoryService", _FakeHistoryService)

    response = client.get(
        "/api/v1/agent/optimization-jobs"
        "?status=completed"
        "&q=cifar"
        "&model_name=gpt-test"
        "&objective=accuracy"
        "&best_score_min=0.8"
        "&best_score_max=0.95"
        "&dataset=CIFAR-10"
        "&config_model=FedAvgCNN"
        "&aggregation=fedavg"
        "&num_clients=3"
        "&num_rounds=10"
    )

    assert response.status_code == 200
    assert captured["status"] == "completed"
    assert captured["q"] == "cifar"
    assert captured["model_name"] == "gpt-test"
    assert captured["objective"] == "accuracy"
    assert captured["best_score_min"] == 0.8
    assert captured["best_score_max"] == 0.95
    assert captured["config_filters"] == {
        "dataset.name": "CIFAR-10",
        "model.name": "FedAvgCNN",
        "federated.aggregation": "fedavg",
        "federated.num_clients": 3,
        "federated.num_rounds": 10,
    }


def test_agent_config_version_endpoints(client, monkeypatch):
    import app.api.v1.endpoints.agent as agent_module

    now = datetime.now(timezone.utc)
    version = SimpleNamespace(
        id=3,
        optimization_job_id=7,
        run_id="run-1",
        iteration=1,
        source="experiment",
        label="alpha 0.1",
        config_hash="abc123",
        config_json={"dataset": {"alpha": 0.1}},
        diff_json=[
            {
                "path": "dataset.alpha",
                "change_type": "updated",
                "old_value": 0.5,
                "new_value": 0.1,
            }
        ],
        created_at=now,
    )

    class _FakeHistoryService:
        def __init__(self, _session):
            self._session = _session

        async def list_config_versions(self, optimization_job_id):
            assert optimization_job_id == 7
            return [version]

        async def diff_config_versions(self, *, optimization_job_id, from_version_id, to_version_id):
            assert optimization_job_id == 7
            assert from_version_id is None
            assert to_version_id == 3
            return version.diff_json

    monkeypatch.setattr(agent_module, "AgentOptimizationHistoryService", _FakeHistoryService)

    versions_response = client.get("/api/v1/agent/optimization-jobs/7/config-versions")
    assert versions_response.status_code == 200
    versions_payload = versions_response.json()
    assert versions_payload[0]["id"] == 3
    assert versions_payload[0]["label"] == "alpha 0.1"
    assert versions_payload[0]["diff_json"][0]["path"] == "dataset.alpha"

    diff_response = client.get("/api/v1/agent/optimization-jobs/7/config-diff?to_version_id=3")
    assert diff_response.status_code == 200
    diff_payload = diff_response.json()
    assert diff_payload["optimization_job_id"] == 7
    assert diff_payload["to_version_id"] == 3
    assert diff_payload["changes"][0]["new_value"] == 0.1
