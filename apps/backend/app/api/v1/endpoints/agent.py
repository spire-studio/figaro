"""
Agent API endpoints for autonomous experiment optimization.
"""

from __future__ import annotations

from fastapi import APIRouter, status, HTTPException
import json

from app.api.deps import AsyncSessionDep
from app.schemas.agent import (
    AgentConfigChangeResponse,
    AgentCurrentExperimentResponse,
    AgentCurrentPlanResponse,
    AgentExperimentResponse,
    AgentExperimentSummary,
    AgentModelsResponse,
    AgentOptimizationJobSummaryResponse,
    AgentOptimizeRequest,
    AgentOptimizeProgressResponse,
    AgentOptimizeResponse,
    AgentRunLogResponse,
    AgentRunMetricsResponse,
    AgentRunResponse,
)
from app.schemas.simulation import SimulationRunMetricsResponse
from app.services.agent import AgentExperimentRunService, AgentExperimentService, AgentOptimizationHistoryService, AgentRuntimeService, AgentState
from app.services.agent.graph import FederatedAgentGraphBuilder
from app.services.agent.objectives import AgentOptimizationObjective, resolve_objective
from app.services.llm import LLMRegistry, LLMService
from app.schemas.agent import AgentPlanPreviewRequest, AgentPlanPreviewResponse, ExperimentPlanPreview, AgentPlanReviseRequest
import uuid

agent_router = APIRouter()
agent_runtime_service = AgentRuntimeService()


@agent_router.get(
    "/models",
    response_model=AgentModelsResponse,
    status_code=status.HTTP_200_OK,
    summary="List available LLM model names for Agent",
)
async def list_agent_models() -> AgentModelsResponse:
    """Return available model names exposed by the LLM registry."""
    return AgentModelsResponse(
        models=LLMRegistry.get_all_names(),
        default_model=LLMRegistry.get_default_name(),
    )


def _build_experiments(history: list) -> list[AgentExperimentSummary]:
    """
    Convert internal history snapshots into API experiment summaries.
    """
    experiments: list[AgentExperimentSummary] = []
    for record in history:
        metrics_payload = record.metrics if hasattr(record, "metrics") else record.get("metrics")
        iteration = record.iteration if hasattr(record, "iteration") else record.get("iteration", 0)
        run_id = record.run_id if hasattr(record, "run_id") else record.get("run_id")
        job_id = record.job_id if hasattr(record, "job_id") else record.get("job_id")
        config = record.config if hasattr(record, "config") else record.get("config")
        config_diff_payload = record.config_diff if hasattr(record, "config_diff") else record.get("config_diff", [])
        metrics = SimulationRunMetricsResponse.model_validate(metrics_payload)
        experiments.append(
            AgentExperimentSummary(
                iteration=iteration,
                run_id=run_id,
                job_id=job_id,
                iteration_goal=record.iteration_goal if hasattr(record, "iteration_goal") else record.get("iteration_goal"),
                plan_summary=record.plan_summary if hasattr(record, "plan_summary") else record.get("plan_summary"),
                hypothesis=record.hypothesis if hasattr(record, "hypothesis") else record.get("hypothesis"),
                rationale=record.rationale if hasattr(record, "rationale") else record.get("rationale", []),
                config_patch=record.config_patch if hasattr(record, "config_patch") else record.get("config_patch", {}),
                config_diff=[
                    AgentConfigChangeResponse.model_validate(item)
                    for item in config_diff_payload
                ],
                config=config,
                metrics=metrics,
                score=record.score if hasattr(record, "score") else record.get("score"),
                result_summary=record.result_summary if hasattr(record, "result_summary") else record.get("result_summary"),
                decision=record.decision if hasattr(record, "decision") else record.get("decision"),
                lessons_learned=record.lessons_learned if hasattr(record, "lessons_learned") else record.get("lessons_learned", []),
            )
        )
    return experiments


def _build_progress_response(snapshot: dict) -> AgentOptimizeProgressResponse:
    """
    Convert a runtime snapshot into the progress response schema.
    """
    best_metrics = (
        SimulationRunMetricsResponse.model_validate(snapshot["best_metrics"])
        if snapshot.get("best_metrics") is not None
        else None
    )

    current_experiment_payload = snapshot.get("current_experiment")
    current_experiment = None
    if current_experiment_payload is not None:
        current_metrics = current_experiment_payload.get("metrics")
        current_experiment = AgentCurrentExperimentResponse(
            iteration=current_experiment_payload.get("iteration", 0),
            phase=current_experiment_payload.get("phase"),
            job_id=current_experiment_payload.get("job_id"),
            job_name=current_experiment_payload.get("job_name"),
            run_id=current_experiment_payload.get("run_id"),
            run_status=current_experiment_payload.get("run_status"),
            config=current_experiment_payload.get("config"),
            metrics=SimulationRunMetricsResponse.model_validate(current_metrics) if current_metrics is not None else None,
        )

    current_plan_payload = snapshot.get("current_plan")
    current_plan = None
    if current_plan_payload is not None:
        current_plan = AgentCurrentPlanResponse(
            iteration=current_plan_payload.get("iteration", 0),
            iteration_goal=current_plan_payload.get("iteration_goal", ""),
            plan_summary=current_plan_payload.get("plan_summary"),
            hypothesis=current_plan_payload.get("hypothesis"),
            rationale=current_plan_payload.get("rationale") or [],
            config_patch=current_plan_payload.get("config_patch") or {},
            config_diff=[
                AgentConfigChangeResponse.model_validate(item)
                for item in current_plan_payload.get("config_diff", [])
            ],
        )

    return AgentOptimizeProgressResponse(
        optimization_job_id=snapshot.get("optimization_job_id"),
        task_id=snapshot["task_id"],
        status=snapshot["status"],
        goal=snapshot["goal"],
        job_name=snapshot.get("job_name"),
        max_iterations=snapshot["max_iterations"],
        model_name=snapshot.get("model_name"),
        objective=snapshot.get("objective", AgentOptimizationObjective.AUTO.value),
        resolved_objective=snapshot.get("resolved_objective", AgentOptimizationObjective.ACCURACY.value),
        current_phase=snapshot.get("current_phase"),
        current_iteration=snapshot.get("current_iteration", 0),
        completed_iterations=snapshot.get("completed_iterations", 0),
        current_plan=current_plan,
        current_experiment=current_experiment,
        best_config=snapshot.get("best_config"),
        best_metrics=best_metrics,
        experiments=_build_experiments(snapshot.get("experiments", [])),
        draft_experiments=snapshot.get("draft_experiments", []),
        summary_text=snapshot.get("summary_text"),
        error_message=snapshot.get("error_message"),
        created_at=snapshot.get("created_at"),
        updated_at=snapshot.get("updated_at"),
        finished_at=snapshot.get("finished_at"),
    )


def _build_history_summary(item) -> AgentOptimizationJobSummaryResponse:
    """
    Convert one persisted optimization job row into a summary payload.
    """
    snapshot = item.snapshot_json if isinstance(item.snapshot_json, dict) else {}
    return AgentOptimizationJobSummaryResponse(
        optimization_job_id=item.id,
        task_id=item.task_id,
        job_name=item.job_name,
        status=item.status.value if hasattr(item.status, "value") else str(item.status),
        goal=item.goal,
        model_name=item.model_name,
        objective=snapshot.get("objective", AgentOptimizationObjective.AUTO.value),
        resolved_objective=snapshot.get("resolved_objective", AgentOptimizationObjective.ACCURACY.value),
        current_phase=item.current_phase,
        max_iterations=item.max_iterations,
        current_iteration=item.current_iteration,
        completed_iterations=item.completed_iterations,
        best_score=item.best_score,
        created_at=item.created_at,
        updated_at=item.updated_at,
        finished_at=item.finished_at,
    )


@agent_router.get(
    "/optimization-jobs",
    response_model=list[AgentOptimizationJobSummaryResponse],
    status_code=status.HTTP_200_OK,
    summary="List persisted agent optimization jobs",
)
async def list_optimization_jobs(session: AsyncSessionDep) -> list[AgentOptimizationJobSummaryResponse]:
    """
    Return persisted Agent optimization jobs ordered by latest update time.
    """
    service = AgentOptimizationHistoryService(session)
    jobs = await service.list_jobs()
    return [_build_history_summary(item) for item in jobs]


@agent_router.get(
    "/optimization-jobs/{optimization_job_id}",
    response_model=AgentOptimizeProgressResponse,
    status_code=status.HTTP_200_OK,
    summary="Get one persisted agent optimization job with full history",
)
async def get_optimization_job(
    optimization_job_id: int,
    session: AsyncSessionDep,
) -> AgentOptimizeProgressResponse:
    """
    Return the persisted snapshot for one Agent optimization job.
    """
    service = AgentOptimizationHistoryService(session)
    job = await service.get_job_or_raise(optimization_job_id)
    snapshot = dict(job.snapshot_json if isinstance(job.snapshot_json, dict) else {})
    snapshot.setdefault("optimization_job_id", job.id)
    snapshot.setdefault("task_id", job.task_id)
    snapshot.setdefault("status", job.status.value if hasattr(job.status, "value") else str(job.status))
    snapshot.setdefault("goal", job.goal)
    snapshot.setdefault("job_name", job.job_name)
    snapshot.setdefault("max_iterations", job.max_iterations)
    snapshot.setdefault("model_name", job.model_name)
    snapshot.setdefault("current_phase", job.current_phase)
    snapshot.setdefault("current_iteration", job.current_iteration)
    snapshot.setdefault("completed_iterations", job.completed_iterations)
    snapshot.setdefault("experiments", [])
    snapshot.setdefault("created_at", job.created_at)
    snapshot.setdefault("updated_at", job.updated_at)
    snapshot.setdefault("finished_at", job.finished_at)
    return _build_progress_response(snapshot)


@agent_router.post(
    "/optimize/start",
    response_model=AgentOptimizeProgressResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start an asynchronous agent optimization task",
)
async def start_optimization(
    payload: AgentOptimizeRequest,
) -> AgentOptimizeProgressResponse:
    """
    Start the optimization task in the background and return its initial snapshot.
    """
    snapshot = await agent_runtime_service.start_optimization(
        goal=payload.goal,
        job_name=payload.job_name,
        max_iterations=payload.max_iterations,
        system_mode=payload.system_mode,
        model_name=payload.model_name,
        objective=payload.objective,
        planned_experiments=payload.planned_experiments,
    )
    return _build_progress_response(snapshot)


@agent_router.get(
    "/optimize/{task_id}",
    response_model=AgentOptimizeProgressResponse,
    status_code=status.HTTP_200_OK,
    summary="Get live progress for an agent optimization task",
)
async def get_optimization_progress(task_id: str) -> AgentOptimizeProgressResponse:
    """
    Return the latest snapshot for one asynchronous optimization task.
    """
    snapshot = await agent_runtime_service.get_task(task_id)
    return _build_progress_response(snapshot)


@agent_router.post(
    "/optimize",
    response_model=AgentOptimizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Run agent-driven optimization loop",
)
async def optimize(
    payload: AgentOptimizeRequest,
    session: AsyncSessionDep,
) -> AgentOptimizeResponse:
    """
    Run a bounded optimization loop where the agent:

    - proposes a configuration;
    - creates/updates a simulation job;
    - starts a run and waits for it to complete;
    - reads normalized metrics and updates its internal state;
    - repeats until the iteration budget is exhausted.

    This first version runs synchronously for simplicity and is intended
    as a building block for more advanced, asynchronous orchestration.
    """
    llm_service = LLMService()
    resolved_objective = resolve_objective(
        goal=payload.goal,
        requested_objective=payload.objective,
    )
    builder = FederatedAgentGraphBuilder(llm_service=llm_service, session=session)
    graph = builder.build()

    state = AgentState(
        goal=payload.goal,
        max_iterations=payload.max_iterations,
        system_mode=payload.system_mode,
        model_name=payload.model_name,
        job_name=payload.job_name,
        objective=payload.objective,
        resolved_objective=resolved_objective,
    )

    # LangGraph executor expects a dict-like object; dataclass is fine.
    # We pass the SQLAlchemy session via config so each node can reuse it.
    app = graph.compile()
    # LangGraph returns the state as a plain dict; convert it back to
    # AgentState for downstream processing.
    raw_state = await app.ainvoke(state)  # type: ignore[arg-type]
    if isinstance(raw_state, AgentState):
        final_state = raw_state
    else:
        final_state = AgentState(**raw_state)  # type: ignore[arg-type]

    experiments = _build_experiments(final_state.history)

    return AgentOptimizeResponse(
        goal=final_state.goal,
        job_name=final_state.current_job_name or final_state.job_name,
        max_iterations=final_state.max_iterations,
        objective=final_state.objective,
        resolved_objective=final_state.resolved_objective,
        iterations_executed=final_state.iteration,
        best_config=None,
        best_metrics=None,
        experiments=experiments,
        summary_text=final_state.summary,
    )


# ===================================================================
#  Agent Experiment / Run endpoints (DB-backed tables)
# ===================================================================


def _to_experiment_response(exp) -> AgentExperimentResponse:
    return AgentExperimentResponse(
        id=exp.id,
        name=exp.name,
        description=exp.description,
        status=exp.status.value if hasattr(exp.status, "value") else str(exp.status),
        config_json=exp.config_json if isinstance(exp.config_json, dict) else {},
        created_at=exp.created_at.isoformat() if exp.created_at else "",
        updated_at=exp.updated_at.isoformat() if exp.updated_at else "",
    )


def _to_run_response(run) -> AgentRunResponse:
    return AgentRunResponse(
        id=run.id,
        experiment_id=run.experiment_id,
        status=run.status.value if hasattr(run.status, "value") else str(run.status),
        config_json=run.config_json if isinstance(run.config_json, dict) else {},
        metrics_json=run.metrics_json if isinstance(run.metrics_json, dict) else {},
        started_at=run.started_at.isoformat() if run.started_at else None,
        ended_at=run.ended_at.isoformat() if run.ended_at else None,
        created_at=run.created_at.isoformat() if run.created_at else "",
    )


@agent_router.get(
    "/experiments",
    response_model=list[AgentExperimentResponse],
    status_code=status.HTTP_200_OK,
    summary="List agent experiments",
)
async def list_agent_experiments(session: AsyncSessionDep) -> list[AgentExperimentResponse]:
    """Return all agent experiments ordered by creation time."""
    service = AgentExperimentService(session)
    experiments = await service.list_experiments()
    return [_to_experiment_response(exp) for exp in experiments]


@agent_router.get(
    "/experiments/{experiment_id}",
    response_model=AgentExperimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get agent experiment",
)
async def get_agent_experiment(experiment_id: int, session: AsyncSessionDep) -> AgentExperimentResponse:
    """Return a single agent experiment with its config."""
    service = AgentExperimentService(session)
    exp = await service.get_experiment_or_raise(experiment_id)
    return _to_experiment_response(exp)


@agent_router.get(
    "/experiments/{experiment_id}/runs",
    response_model=list[AgentRunResponse],
    status_code=status.HTTP_200_OK,
    summary="List runs for an agent experiment",
)
async def list_agent_experiment_runs(experiment_id: int, session: AsyncSessionDep) -> list[AgentRunResponse]:
    """Return all runs for a specific agent experiment."""
    # Ensure experiment exists
    exp_service = AgentExperimentService(session)
    await exp_service.get_experiment_or_raise(experiment_id)
    # List runs via repository
    run_service = AgentExperimentRunService(session)
    runs = await run_service.repo.list_runs_for_experiment(experiment_id)
    return [_to_run_response(run) for run in runs]


@agent_router.get(
    "/runs",
    response_model=list[AgentRunResponse],
    status_code=status.HTTP_200_OK,
    summary="List all agent runs",
)
async def list_agent_runs(session: AsyncSessionDep, limit: int = 200) -> list[AgentRunResponse]:
    """Return agent experiment runs ordered by most recent first."""
    service = AgentExperimentRunService(session)
    runs = await service.list_runs(limit=limit)
    return [_to_run_response(run) for run in runs]


@agent_router.get(
    "/runs/{run_id}",
    response_model=AgentRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Get agent run",
)
async def get_agent_run(run_id: str, session: AsyncSessionDep) -> AgentRunResponse:
    """Return details for a single agent run."""
    service = AgentExperimentRunService(session)
    run = await service.get_run(run_id)
    return _to_run_response(run)


@agent_router.get(
    "/runs/{run_id}/metrics",
    response_model=AgentRunMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get agent run metrics",
)
async def get_agent_run_metrics(run_id: str, session: AsyncSessionDep) -> AgentRunMetricsResponse:
    """Return normalized training metrics for an agent run."""
    service = AgentExperimentRunService(session)
    metrics = await service.get_run_metrics(run_id)
    return AgentRunMetricsResponse(run_id=run_id, metrics=metrics)


@agent_router.get(
    "/runs/{run_id}/logs",
    response_model=list[AgentRunLogResponse],
    status_code=status.HTTP_200_OK,
    summary="Get agent run logs",
)
async def get_agent_run_logs(
    run_id: str,
    session: AsyncSessionDep,
    limit: int = 500,
) -> list[AgentRunLogResponse]:
    """Return log entries for an agent run."""
    service = AgentExperimentRunService(session)
    logs = await service.get_run_logs(run_id, limit=limit)
    return [
        AgentRunLogResponse(
            id=log.id,
            level=log.level,
            message=log.message,
            created_at=log.created_at.isoformat() if log.created_at else "",
        )
        for log in logs
    ]


@agent_router.post(
    "/optimize/plan",
    response_model=AgentPlanPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a plan preview and save as draft",
)
async def generate_plan_preview(
    payload: AgentPlanPreviewRequest,
    session: AsyncSessionDep,
) -> AgentPlanPreviewResponse:
    """
    1. Call LLM to generate the plan.
    2. Create a Job record with PENDING_REVIEW status.
    """
    # Borrow the graph builder to do the LLM parsing
    llm_service = LLMService()
    builder = FederatedAgentGraphBuilder(llm_service=llm_service, session=session)
    
    # Construct a dummy initial state to run through the parse node
    temp_state = AgentState(
        goal=payload.goal,
        job_name=payload.job_name,
        model_name=payload.model_name,
        system_mode=payload.system_mode,
        max_iterations=10, 
        objective=AgentOptimizationObjective.AUTO,
        resolved_objective=AgentOptimizationObjective.ACCURACY,
    )
    
    # Call the graph's parse node directly to get the LLM result
    parsed_state = await builder._node_parse(temp_state)
    
    previews = []
    for exp in parsed_state.experiments:
        previews.append(ExperimentPlanPreview(
            name=exp.name,
            plan_summary=exp.plan_summary or "",
            config_patch=exp.config_patch,
            # Fallback values if LLM doesn't generate them
            estimated_minutes=0, 
            estimated_gpu_vram_gb=0.0,
        ))

    # Persist the draft (write snapshot to DB, set status to pending_review)
    history_service = AgentOptimizationHistoryService(session)
    
    job = await history_service.create_job(
        task_id=f"draft-{uuid.uuid4().hex[:8]}",
        goal=payload.goal,
        job_name=payload.job_name,
        model_name=payload.model_name,
        system_mode=payload.system_mode,
        max_iterations=len(previews) or 1,
        status="pending_review",
        snapshot={
            "goal": payload.goal,
            "job_name": payload.job_name,
            "status": "pending_review",
            "experiments": [],
            "draft_experiments": [p.model_dump() for p in previews]
        }
    )

    return AgentPlanPreviewResponse(
        optimization_job_id=job.id,
        goal=payload.goal,
        experiments=previews,
        system_mode=payload.system_mode
    )

@agent_router.post(
    "/optimization-jobs/{job_id}/revise",
    response_model=AgentPlanPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Revise a pending plan draft using natural language",
)
async def revise_plan_preview(
    job_id: int,
    payload: AgentPlanReviseRequest,
    session: AsyncSessionDep,
) -> AgentPlanPreviewResponse:
    history_service = AgentOptimizationHistoryService(session)
    job = await history_service.get_job_or_raise(job_id)
    
    current_status = job.status.value if hasattr(job.status, "value") else str(job.status)
    if current_status != "pending_review":
        raise HTTPException(status_code=400, detail="Only pending_review jobs can be revised.")

    snapshot = job.snapshot_json
    current_experiments = snapshot.get("draft_experiments", [])
    if not current_experiments:
        raise HTTPException(status_code=400, detail="No draft experiments found to revise.")

    instructions = (
        "You are an AI assistant managing a Federated Learning experiment plan. "
        "The user wants to modify the current experiment configuration based on their feedback. "
        "Update the JSON configuration to reflect their request. "
        "Respond ONLY with a valid JSON array of experiment objects, matching the original schema. "
        "Do not include any other text."
    )
    
    input_text = (
        f"Current Plan (JSON):\n{json.dumps(current_experiments, indent=2)}\n\n"
        f"User Modification Request:\n{payload.instruction}\n\n"
        "Please output the updated JSON array of experiments:"
    )

    llm_service = LLMService()
    model_name = snapshot.get("model_name") 
    text, _ = await llm_service.generate_text(
        model=model_name if model_name else None,
        instructions=instructions,
        input_text=input_text,
    )

    clean_text = text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        updated_experiments = json.loads(clean_text)
        if not isinstance(updated_experiments, list):
            raise ValueError("LLM did not return a list.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM response: {str(e)}")

    snapshot["draft_experiments"] = updated_experiments
    await history_service.update_job_snapshot(task_id=job.task_id, snapshot=snapshot)

    return AgentPlanPreviewResponse(
        optimization_job_id=job.id,
        goal=snapshot.get("goal", ""),
        experiments=[ExperimentPlanPreview.model_validate(exp) for exp in updated_experiments],
        system_mode=snapshot.get("system_mode", "simulation")
    )