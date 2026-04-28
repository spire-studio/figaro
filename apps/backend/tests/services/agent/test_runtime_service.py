import asyncio

from app.services.agent.runtime_service import AgentRuntimeService
from app.services.agent.state import AgentState, ExperimentRecord


def test_runtime_service_marks_stale_queued_task_as_failed(monkeypatch):
    async def _run():
        service = AgentRuntimeService()
        task_id = "task-stale"
        persisted_snapshots: list[dict] = []

        async def _fake_persist_snapshot(snapshot):  # noqa: ANN001
            persisted_snapshots.append(snapshot)

        monkeypatch.setattr(service, "_persist_snapshot", _fake_persist_snapshot)

        async with service._lock:
            service._snapshots.clear()
            service._tasks.clear()
            service._snapshots[task_id] = {
                "task_id": task_id,
                "status": "queued",
                "current_phase": "queued",
                "error_message": None,
                "finished_at": None,
            }

        async def _explode():
            raise RuntimeError("startup boom")

        task = asyncio.create_task(_explode())
        await asyncio.sleep(0)
        service._on_task_done(task_id, task)
        await asyncio.sleep(0)

        snapshot = await service.get_task(task_id)
        assert snapshot["status"] == "failed"
        assert snapshot["current_phase"] == "failed"
        assert snapshot["error_message"] == "startup boom"
        assert snapshot["finished_at"] is not None
        assert persisted_snapshots[-1]["status"] == "failed"

    asyncio.run(_run())


def test_runtime_service_persists_best_config_and_metrics(monkeypatch):
    async def _run():
        service = AgentRuntimeService()
        task_id = "task-best"
        persisted_snapshots: list[dict] = []

        async def _fake_persist_snapshot(snapshot):  # noqa: ANN001
            persisted_snapshots.append(snapshot)

        monkeypatch.setattr(service, "_persist_snapshot", _fake_persist_snapshot)

        async with service._lock:
            service._snapshots.clear()
            service._tasks.clear()
            service._snapshots[task_id] = {
                "task_id": task_id,
                "status": "running",
            }

        state = AgentState(goal="compare configs")
        state.history = [
            ExperimentRecord(
                iteration=1,
                run_id="run-low",
                job_id=1,
                name="low",
                config={"model": {"name": "CNN"}},
                metrics={"global_results": {"global_accuracy": [0.7]}},
                score=0.7,
            ),
            ExperimentRecord(
                iteration=2,
                run_id="run-high",
                job_id=2,
                name="high",
                config={"model": {"name": "ResNet"}},
                metrics={"global_results": {"global_accuracy": [0.9]}},
                score=0.9,
            ),
        ]

        await service._update_from_state(task_id, state, status="completed")

        snapshot = await service.get_task(task_id)
        assert snapshot["best_config"] == {"model": {"name": "ResNet"}}
        assert snapshot["best_metrics"] == {"global_results": {"global_accuracy": [0.9]}}
        assert persisted_snapshots[-1]["best_config"] == {"model": {"name": "ResNet"}}

    asyncio.run(_run())
