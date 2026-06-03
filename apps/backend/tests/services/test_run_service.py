import asyncio
import re
from pathlib import Path

from app.models.simulation import SimulationJob, SimulationJobStatus, SimulationRun, SimulationRunStatus
from app.services import SimulationRunService


class _DummySession:
    async def commit(self):
        return None

    async def refresh(self, _obj):
        return None


def test_run_service_start_run_flow_without_real_subprocess(monkeypatch, tmp_path):
    async def _run_test():
        service = SimulationRunService(session=_DummySession())
        service._job_config_dir = lambda: Path(tmp_path)  # type: ignore[method-assign]
        service._results_dir = lambda: Path(tmp_path) / "results"  # type: ignore[method-assign]
        service._runtime_log_dir = lambda: Path(tmp_path) / "logs"  # type: ignore[method-assign]
        service._results_dir().mkdir(parents=True, exist_ok=True)
        service._runtime_log_dir().mkdir(parents=True, exist_ok=True)
        captured: dict[str, object] = {"results": []}

        class _FakeProcess:
            def __init__(self):
                self.pid = 111
                self.stdout = _FakeStream([b"\n"])
                self.stderr = _FakeStream([])
                self.returncode = None

            async def wait(self):
                self.returncode = 0
                return 0

            def terminate(self):
                self.returncode = -15

        class _FakeStream:
            def __init__(self, lines):
                self.lines = list(lines)

            async def readline(self):
                if self.lines:
                    return self.lines.pop(0)
                return b""

        async def _fake_spawn(self, _command, env=None):
            captured["env"] = env
            return _FakeProcess()

        async def _fake_watch(self, run_id, process, *, artifact_timestamp=None):
            captured["artifact_timestamp"] = artifact_timestamp
            self._processes.pop(run_id, None)
            self._tasks.pop(run_id, None)

        # Fake repositories methods used by start_run
        job = SimulationJob(id=1, name="j1", description=None, status=SimulationJobStatus.READY, config_json={"x": 1})

        async def _get_job(_job_id):
            return job

        async def _create_run(run: SimulationRun):
            return run

        async def _update_run_status(run: SimulationRun, **kwargs):
            run.status = kwargs["status"]
            run.process_id = kwargs.get("process_id")
            return run

        async def _add_log(_run_id, _msg, level="INFO"):
            return None

        async def _add_result(**_kwargs):
            captured["results"].append(_kwargs)
            return None

        monkeypatch.setattr(SimulationRunService, "_spawn_subprocess", _fake_spawn)
        monkeypatch.setattr(SimulationRunService, "_watch_process", _fake_watch)

        service.job_repository.get_job = _get_job  # type: ignore[assignment]
        service.run_repository.create_run = _create_run  # type: ignore[assignment]
        service.run_repository.update_run_status = _update_run_status  # type: ignore[assignment]
        service.run_repository.add_log = _add_log  # type: ignore[assignment]
        service.run_repository.add_result = _add_result  # type: ignore[assignment]

        run = await service.start_run(job_id=1)
        assert run is not None
        assert run.status == SimulationRunStatus.RUNNING
        assert run.process_id == 111
        env = captured["env"]
        assert isinstance(env, dict)
        result_filename = env["FIGARO_RESULTS_FILE"]
        result_match = re.fullmatch(rf"(\d{{8}}_\d{{6}}Z)_{run.id}_live_results\.json", result_filename)
        assert result_match is not None
        timestamp = result_match.group(1)

        results = captured["results"]
        assert isinstance(results, list)
        config_artifact = results[0]
        config_path = Path(config_artifact["path"])
        assert config_path.name == f"{timestamp}_{run.id}.json"
        assert config_artifact["metadata_json"]["artifact_timestamp"] == timestamp
        assert service._simulation_log_path(run.id, artifact_timestamp=timestamp).name == f"{timestamp}_{run.id}_server.log"

    asyncio.run(_run_test())
