from app.models.base import utcnow
from app.models.distributed import (
    DistributedJob,
    DistributedJobStatus,
    DistributedParticipant,
    DistributedParticipantStatus,
    DistributedSession,
    DistributedSessionStatus,
)
from app.models.simulation import (
    SimulationJob,
    SimulationJobStatus,
    SimulationRun,
    SimulationRunLog,
    SimulationRunResult,
    SimulationRunStatus,
)


def test_utcnow_returns_timezone_aware_datetime():
    now = utcnow()
    assert now.tzinfo is not None
    assert now.utcoffset() is not None


def test_simulation_models_have_expected_defaults():
    job = SimulationJob(name="job-a")
    run = SimulationRun(job_id=1, command="python runner.py")
    log = SimulationRunLog(run_id=run.id, message="hello")
    result = SimulationRunResult(run_id=run.id, artifact_type="metrics", path="/tmp/m.json")

    assert job.status == SimulationJobStatus.DRAFT
    assert job.config_json == {}
    assert run.status == SimulationRunStatus.QUEUED
    assert len(run.id) == 8
    assert log.level == "INFO"
    assert result.metadata_json == {}
    assert run.created_at.tzinfo is not None


def test_distributed_models_have_expected_defaults():
    job = DistributedJob(name="dist-job")
    session = DistributedSession(job_id=1)
    participant = DistributedParticipant(session_id=session.id, participant_name="client-1")

    assert job.status == DistributedJobStatus.DRAFT
    assert job.expected_clients == 1
    assert session.status == DistributedSessionStatus.WAITING_CLIENTS
    assert session.server_ip == "127.0.0.1"
    assert session.server_port == 50052
    assert len(session.id) == 8
    assert participant.status == DistributedParticipantStatus.PENDING_APPROVAL
    assert participant.metadata_json == {}
    assert participant.assigned_participant_id is None
