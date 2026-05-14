from app.models.distributed import DistributedJob, DistributedSession
from app.models.simulation import SimulationJob, SimulationRun, SimulationRunStatus
from app.schemas.distributed.jobs import (
    DistributedJobConfigUpdateRequest,
    DistributedJobResponse,
)
from app.schemas.distributed.runtime import DistributedRuntimeStatusResponse
from app.schemas.distributed.sessions import (
    DistributedSessionCreateRequest,
    DistributedSessionProgressResponse,
    DistributedSessionResponse,
)
from app.schemas.message import Message
from app.schemas.simulation.jobs import (
    SimulationJobConfigUpdateRequest,
    SimulationJobCopyRequest,
    SimulationJobCreateRequest,
    SimulationJobResponse,
    SimulationJobUpdateRequest,
)
from app.schemas.simulation.runs import (
    SimulationRunMetricsResponse,
    SimulationRunResponse,
)


def test_basic_request_schema_defaults():
    dist_session = DistributedSessionCreateRequest()
    dist_cfg = DistributedJobConfigUpdateRequest()
    message = Message(message="ok")
    sim_create = SimulationJobCreateRequest(name="job", description=None)
    sim_update = SimulationJobUpdateRequest()
    sim_cfg = SimulationJobConfigUpdateRequest(config={"x": 1})
    sim_copy = SimulationJobCopyRequest()

    assert dist_session.server_ip == "localhost"
    assert dist_session.server_port == 50052
    assert dist_cfg.config == {}
    assert message.detail is None
    assert sim_create.name == "job"
    assert sim_update.name is None
    assert sim_update.description is None
    assert sim_update.status is None
    assert sim_cfg.config == {"x": 1}
    assert sim_copy.name is None


def test_from_attributes_response_models_round_trip():
    sim_job = SimulationJob(id=1, name="job-a", description="desc")
    sim_run = SimulationRun(
        id="run00001",
        job_id=1,
        status=SimulationRunStatus.RUNNING,
        process_id=321,
        command="python runner.py",
    )
    dist_job = DistributedJob(
        id=2,
        name="dist-a",
        description="distributed",
        expected_clients=2,
        config_json={"federated": {"num_clients": 2}},
    )
    dist_session = DistributedSession(
        id="sess0001",
        job_id=2,
        server_ip="localhost",
        server_port=50052,
    )

    sim_job_response = SimulationJobResponse.model_validate(sim_job)
    sim_run_response = SimulationRunResponse.model_validate(sim_run)
    dist_job_response = DistributedJobResponse.model_validate(dist_job)
    dist_session_response = DistributedSessionResponse.model_validate(dist_session)

    assert sim_job_response.id == 1
    assert sim_run_response.id == "run00001"
    assert sim_run_response.process_id == 321
    assert dist_job_response.expected_clients == 2
    assert dist_session_response.id == "sess0001"


def test_metrics_and_progress_schema_defaults_and_parsing():
    metrics = SimulationRunMetricsResponse()
    assert metrics.global_results.rounds == []
    assert metrics.global_results.global_loss == []
    assert metrics.global_results.global_accuracy == []
    assert metrics.llm_results.rounds == []
    assert metrics.llm_dataset == {}
    assert metrics.client_results == {}

    parsed = SimulationRunMetricsResponse.model_validate(
        {
            "global_results": {
                "rounds": [1, 2],
                "global_loss": [0.8, 0.4],
                "global_accuracy": [0.6, 0.75],
            },
            "client_results": {
                "c0": {
                    "train_loss": [0.5],
                    "train_acc": [0.7],
                }
            },
        }
    )
    assert parsed.client_results["c0"].test_acc == []
    assert parsed.global_results.global_accuracy[-1] == 0.75

    runtime = DistributedRuntimeStatusResponse(
        role="server",
        runtime_id="rt-1",
        status="running",
        pid=99,
        started_at=None,
        ended_at=None,
        exit_code=None,
        config_path="/tmp/config.json",
        log_path="/tmp/log.txt",
        results_path=None,
        total_rounds=10,
    )
    progress = DistributedSessionProgressResponse(
        session_id="sess-1",
        status="running",
        pid=99,
        started_at=None,
        ended_at=None,
        exit_code=None,
        total_rounds=10,
        last_round=3,
        latest_global_loss=0.3,
        latest_global_accuracy=0.8,
    )

    assert runtime.role == "server"
    assert progress.metrics_json == {}
    assert progress.log_path is None
