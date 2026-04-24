import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.models.distributed import DistributedJobStatus
from app.repositories.distributed.job_repository import DistributedJobRepository


async def _create_repository():
    import app.models.distributed  # noqa: F401

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_maker = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    session = session_maker()
    return engine, session


def test_repository_create_and_query_job_by_name_case_insensitive():
    async def _run():
        engine, session = await _create_repository()
        try:
            repo = DistributedJobRepository(session)
            created = await repo.create_job(
                name="CaseName",
                description="job",
                expected_clients=2,
                config_json={"federated": {"num_clients": 2}},
            )
            await session.commit()

            by_id = await repo.get_job(created.id)
            by_name = await repo.get_job_by_name("casename")

            assert by_id is not None
            assert by_name is not None
            assert by_name.id == created.id
        finally:
            await session.close()
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(_run())


def test_repository_updates_status_and_config():
    async def _run():
        engine, session = await _create_repository()
        try:
            repo = DistributedJobRepository(session)
            job = await repo.create_job(
                name="dist-a",
                description=None,
                expected_clients=1,
                config_json={"federated": {"num_clients": 1}},
            )
            await session.commit()

            await repo.update_job_status(job, DistributedJobStatus.READY)
            await repo.update_job_config(
                job,
                config_json={"federated": {"num_clients": 3}},
                expected_clients=3,
            )
            await session.commit()

            listed = await repo.list_jobs()
            assert len(listed) == 1
            assert listed[0].status == DistributedJobStatus.READY
            assert listed[0].expected_clients == 3
            assert listed[0].config_json["federated"]["num_clients"] == 3
        finally:
            await session.close()
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(_run())
