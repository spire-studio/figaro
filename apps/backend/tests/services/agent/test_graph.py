import asyncio
from types import SimpleNamespace

from app.services.agent.graph import FederatedAgentGraphBuilder
from app.services.agent.state import AgentState


class _FakeLLMService:
    def __init__(self):
        self.model = None
        self.instructions = None
        self.input_text = None

    async def generate_text(self, *, model, instructions, input_text):
        self.model = model
        self.instructions = instructions
        self.input_text = input_text
        return (
            '{"iteration_goal":"baseline","plan_summary":"disable attack","config_patch":{"attack":{"enable":false}}}',
            {},
        )


class _FakeJobService:
    def __init__(self, _session):
        self._session = _session
        self.last_created_name = None
        self.last_updated_config = None

    def get_config_schema(self):
        return {
            "dataset": {"name": {"default": "mnist"}},
            "model": {"name": {"default": "cnn"}},
            "federated": {},
            "attack": {},
            "defense": {},
        }

    @staticmethod
    def normalize_simulation_config(candidate):
        return candidate

    async def create_job(self, name, description):
        self.last_created_name = name
        return SimpleNamespace(id=7, name=name, description=description, config_json={"created": True})

    async def update_job_config(self, job_id, config):
        self.last_updated_config = config
        return SimpleNamespace(
            id=job_id,
            name=self.last_created_name or "generated-job",
            config_json={
                "federated": {
                    "num_clients": 3,
                    "num_rounds": 10,
                    "clients_per_round": 2,
                    "local_epochs": 5,
                    "learning_rate": 0.01,
                },
                "attack": {"enable": False},
                "defense": {"enable": True, "strategy": "median"},
            },
        )


class _FakeRunService:
    def __init__(self, _session):
        self._session = _session

    async def start_run(self, _job_id):
        return SimpleNamespace(id="run-1", status="queued")


class _SyncNumClientsJobService(_FakeJobService):
    @staticmethod
    def normalize_simulation_config(candidate):
        normalized = dict(candidate)
        federated = dict(normalized.get("federated") or {})
        dataset = dict(normalized.get("dataset") or {})
        if "num_clients" in federated:
            dataset["num_clients"] = federated["num_clients"]
        normalized["federated"] = federated
        normalized["dataset"] = dataset
        return normalized


def test_plan_node_uses_default_llm_config(monkeypatch):
    async def _run():
        import app.services.agent.graph as graph_module

        monkeypatch.setattr(graph_module, "SimulationJobService", _FakeJobService)
        monkeypatch.setattr(graph_module, "get_platform_capabilities", lambda: {"datasets": ["mnist"]})

        llm_service = _FakeLLMService()
        builder = FederatedAgentGraphBuilder(llm_service=llm_service, session=object())  # type: ignore[arg-type]
        state = AgentState(goal="optimize", model_name="default-test-model")

        updated = await builder._node_plan(state)
        assert updated.iteration == 1
        assert updated.current_plan is not None
        assert updated.current_plan.iteration_goal == "baseline"
        assert updated.current_plan.plan_summary == "disable attack"
        assert updated.current_plan.config_patch == {}
        assert updated.current_config["attack"]["enable"] is False
        assert updated.current_config["dataset"]["name"] == "mnist"
        assert updated.current_config["model"]["name"] == "cnn"
        assert llm_service.model == "default-test-model"

    asyncio.run(_run())


def test_launch_node_uses_persisted_normalized_config_and_job_name(monkeypatch):
    async def _run():
        import app.services.agent.graph as graph_module

        fake_job_service = _FakeJobService(object())
        monkeypatch.setattr(graph_module, "SimulationJobService", lambda _session: fake_job_service)
        monkeypatch.setattr(graph_module, "SimulationRunService", _FakeRunService)

        builder = FederatedAgentGraphBuilder(llm_service=object(), session=object())  # type: ignore[arg-type]
        state = AgentState(
            goal="optimize",
            job_name="named-opt-job",
            current_config={"attack": {"enable": False}},
            iteration=1,
        )

        updated = await builder._node_launch(state)
        assert updated.current_job_id == 7
        assert updated.current_job_name == "named-opt-job"
        assert updated.current_run_id == "run-1"
        assert updated.current_config["federated"]["num_rounds"] == 10
        assert fake_job_service.last_created_name == "named-opt-job"
        assert fake_job_service.last_updated_config == {"attack": {"enable": False}}

    asyncio.run(_run())


def test_plan_node_hides_dataset_num_clients_from_agent_diff(monkeypatch):
    async def _run():
        import app.services.agent.graph as graph_module

        class _ClientCountLLMService(_FakeLLMService):
            async def generate_text(self, *, model, instructions, input_text):
                self.model = model
                self.instructions = instructions
                self.input_text = input_text
                return (
                    '{"iteration_goal":"increase clients","plan_summary":"only change federated count","config_patch":{"federated":{"num_clients":6}}}',
                    {},
                )

        monkeypatch.setattr(graph_module, "SimulationJobService", _SyncNumClientsJobService)
        monkeypatch.setattr(graph_module, "get_platform_capabilities", lambda: {"datasets": ["mnist"]})

        llm_service = _ClientCountLLMService()
        builder = FederatedAgentGraphBuilder(llm_service=llm_service, session=object())  # type: ignore[arg-type]
        state = AgentState(
            goal="optimize",
            current_config={
                "dataset": {"name": "mnist", "num_clients": 5},
                "federated": {"num_clients": 5},
                "attack": {"enable": False},
                "defense": {"enable": True, "strategy": "median"},
            },
        )

        updated = await builder._node_plan(state)
        assert updated.current_plan is not None
        assert "dataset.num_clients" not in llm_service.input_text
        diff_paths = [change.path for change in updated.current_plan.config_diff]
        assert "federated.num_clients" in diff_paths
        assert "dataset.num_clients" not in diff_paths

    asyncio.run(_run())
