"""
Distributed job service.

Handles distributed job lifecycle and job-level configuration updates.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import exceptions
from app.models.distributed import DistributedJob
from app.repositories.distributed import DistributedJobRepository
from app.services.distributed.config_service import DistributedConfigService


class DistributedJobService:
    """
    Application service for distributed job operations.
    """

    JOB_NAME_CONFLICT_MESSAGE = "Distributed job name already exists"


    def __init__(self, session: AsyncSession):
        self.session = session
        self.job_repository = DistributedJobRepository(session)
        self.config_service = DistributedConfigService()


    async def create_job(
        self,
        *,
        name: str,
        description: str | None,
    ) -> DistributedJob:
        """
        Create a distributed job with normalized default server config.
        """
        config_json = self.config_service.build_default_job_config()
        expected_clients = self.config_service.extract_expected_clients(config_json)

        existing = await self.job_repository.get_job_by_name(name)
        if existing:
            raise exceptions.JobAlreadyExists(self.JOB_NAME_CONFLICT_MESSAGE)

        try:
            job = await self.job_repository.create_job(
                name=name,
                description=description,
                expected_clients=expected_clients,
                config_json=config_json,
            )
            await self.session.commit()
            await self.session.refresh(job)
            return job
        except IntegrityError as exc:
            await self.session.rollback()
            raise exceptions.JobAlreadyExists(self.JOB_NAME_CONFLICT_MESSAGE) from exc


    async def update_job_config(self, *, job_id: int, config_json: dict[str, Any]) -> DistributedJob:
        """
        Update distributed job config and enforce server-side constraints.
        """
        if not isinstance(config_json, dict) or not config_json:
            raise exceptions.BadRequestError("Config must be a non-empty JSON object")

        normalized = self.config_service.normalize_server_config(config_json)
        expected_clients = self.config_service.extract_expected_clients(normalized)

        job = await self.get_job_or_raise(job_id)
        await self.job_repository.update_job_config(
            job,
            config_json=normalized,
            expected_clients=expected_clients,
        )
        await self.session.commit()
        await self.session.refresh(job)
        return job


    async def list_jobs(self) -> list[DistributedJob]:
        """
        List distributed jobs ordered by creation time.
        """
        return await self.job_repository.list_jobs()


    async def get_job_or_raise(self, job_id: int) -> DistributedJob:
        """
        Get distributed job by ID or raise not-found.
        """
        job = await self.job_repository.get_job(job_id)
        if not job:
            raise exceptions.ResourceNotFound("Distributed job not found")
        return job
