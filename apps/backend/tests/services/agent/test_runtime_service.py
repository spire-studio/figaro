import asyncio

from app.services.agent.runtime_service import AgentRuntimeService


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
