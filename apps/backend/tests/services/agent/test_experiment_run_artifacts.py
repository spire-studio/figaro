from app.services.agent.experiment_service import AgentExperimentRunService


class _DummySession:
    pass


def test_agent_experiment_run_artifact_paths_share_timestamp(tmp_path):
    service = AgentExperimentRunService(session=_DummySession())  # type: ignore[arg-type]
    service._job_config_dir = lambda: tmp_path / "configs"  # type: ignore[method-assign]
    service._results_dir = lambda: tmp_path / "results"  # type: ignore[method-assign]
    service._runtime_log_dir = lambda: tmp_path / "logs"  # type: ignore[method-assign]
    timestamp = "20260603_151122Z"
    run_id = "abc12345"

    assert service._run_config_path(run_id, artifact_timestamp=timestamp).name == f"{timestamp}_{run_id}.json"
    assert service._build_live_results_filename(run_id, artifact_timestamp=timestamp) == (
        f"{timestamp}_{run_id}_live_results.json"
    )
    assert service._run_log_path(run_id, artifact_timestamp=timestamp).name == f"{timestamp}_{run_id}_server.log"
