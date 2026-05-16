import asyncio

import pytest
from sqlalchemy.exc import IntegrityError

from app.core import exceptions
from app.models.distributed import DistributedJob
from app.services.distributed.job_service import DistributedJobService


class _DummySession:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0
        self.refreshed_objects: list[object] = []

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def refresh(self, obj) -> None:
        self.refreshed_objects.append(obj)


def test_create_job_success_commits_and_refreshes(monkeypatch):
    async def _run():
        session = _DummySession()
        service = DistributedJobService(session=session)  # type: ignore[arg-type]

        default_config = {"federated": {"num_clients": 3}}
        created = DistributedJob(
            id=1,
            name="dist-job",
            description="desc",
            expected_clients=3,
            config_json=default_config,
        )

        async def _get_job_by_name(_name):
            return None

        async def _create_job(**_kwargs):
            return created

        monkeypatch.setattr(
            service.config_service,
            "build_default_job_config",
            lambda: default_config,
        )
        monkeypatch.setattr(
            service.config_service,
            "extract_expected_clients",
            lambda _cfg: 3,
        )
        monkeypatch.setattr(service.job_repository, "get_job_by_name", _get_job_by_name)
        monkeypatch.setattr(service.job_repository, "create_job", _create_job)

        job = await service.create_job(name="dist-job", description="desc")
        assert job.id == 1
        assert session.commits == 1
        assert session.rollbacks == 0
        assert session.refreshed_objects == [job]

    asyncio.run(_run())


def test_create_job_raises_when_name_exists(monkeypatch):
    async def _run():
        session = _DummySession()
        service = DistributedJobService(session=session)  # type: ignore[arg-type]

        async def _existing(_name):
            return DistributedJob(name="dup")

        monkeypatch.setattr(service.job_repository, "get_job_by_name", _existing)

        with pytest.raises(exceptions.JobAlreadyExists, match="already exists"):
            await service.create_job(name="dup", description=None)

    asyncio.run(_run())


def test_create_job_rolls_back_on_integrity_error(monkeypatch):
    async def _run():
        session = _DummySession()
        service = DistributedJobService(session=session)  # type: ignore[arg-type]

        async def _get_job_by_name(_name):
            return None

        async def _create_job(**_kwargs):
            raise IntegrityError("insert", {}, Exception("duplicate key"))

        monkeypatch.setattr(
            service.config_service,
            "build_default_job_config",
            lambda: {"federated": {"num_clients": 1}},
        )
        monkeypatch.setattr(
            service.config_service,
            "extract_expected_clients",
            lambda _cfg: 1,
        )
        monkeypatch.setattr(service.job_repository, "get_job_by_name", _get_job_by_name)
        monkeypatch.setattr(service.job_repository, "create_job", _create_job)

        with pytest.raises(exceptions.JobAlreadyExists, match="already exists"):
            await service.create_job(name="dup", description=None)

        assert session.commits == 0
        assert session.rollbacks == 1

    asyncio.run(_run())


def test_update_job_config_rejects_invalid_payload():
    async def _run():
        session = _DummySession()
        service = DistributedJobService(session=session)  # type: ignore[arg-type]

        with pytest.raises(exceptions.BadRequestError, match="non-empty JSON object"):
            await service.update_job_config(job_id=1, config_json={})

        with pytest.raises(exceptions.BadRequestError, match="non-empty JSON object"):
            await service.update_job_config(job_id=1, config_json="bad")  # type: ignore[arg-type]

    asyncio.run(_run())


def test_normalize_server_config_rejects_llm_peft_route():
    service = DistributedJobService(session=_DummySession())  # type: ignore[arg-type]

    with pytest.raises(exceptions.BadRequestError, match="simulation mode only"):
        service.config_service.normalize_server_config(
            {
                "task": {"type": "llm_peft_sft"},
                "federated": {"num_clients": 2},
            }
        )


def test_update_job_config_normalizes_and_persists(monkeypatch):
    async def _run():
        session = _DummySession()
        service = DistributedJobService(session=session)  # type: ignore[arg-type]

        target = DistributedJob(
            id=3,
            name="dist",
            expected_clients=1,
            config_json={"federated": {"num_clients": 1}},
        )
        normalized = {"federated": {"num_clients": 4}}
        calls = {}

        async def _get_job_or_raise(_job_id):
            return target

        async def _update_job_config(job, *, config_json, expected_clients):
            calls["job"] = job
            calls["config_json"] = config_json
            calls["expected_clients"] = expected_clients
            return job

        monkeypatch.setattr(service, "get_job_or_raise", _get_job_or_raise)
        monkeypatch.setattr(
            service.config_service,
            "normalize_server_config",
            lambda _cfg: normalized,
        )
        monkeypatch.setattr(
            service.config_service,
            "extract_expected_clients",
            lambda _cfg: 4,
        )
        monkeypatch.setattr(
            service.job_repository,
            "update_job_config",
            _update_job_config,
        )

        updated = await service.update_job_config(
            job_id=3,
            config_json={"federated": {"num_clients": 4}},
        )

        assert updated is target
        assert calls["job"] is target
        assert calls["config_json"] == normalized
        assert calls["expected_clients"] == 4
        assert session.commits == 1
        assert session.refreshed_objects == [target]

    asyncio.run(_run())


def test_get_job_or_raise_returns_job_or_raises(monkeypatch):
    async def _run():
        session = _DummySession()
        service = DistributedJobService(session=session)  # type: ignore[arg-type]

        existing = DistributedJob(id=8, name="exists")

        async def _get_job(_job_id):
            return existing

        monkeypatch.setattr(service.job_repository, "get_job", _get_job)
        assert await service.get_job_or_raise(8) is existing

        async def _missing(_job_id):
            return None

        monkeypatch.setattr(service.job_repository, "get_job", _missing)
        with pytest.raises(exceptions.ResourceNotFound, match="Distributed job not found"):
            await service.get_job_or_raise(9)

    asyncio.run(_run())
