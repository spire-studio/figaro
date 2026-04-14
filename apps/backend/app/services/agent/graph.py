"""
LangGraph workflow for bench-mode FL experiment execution.

Pipeline: parse(N configs) -> launch_batch(N runs) -> collect_all -> report
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
import uuid
from collections.abc import Awaitable, Callable

from langgraph.graph import END, StateGraph  # type: ignore[import-untyped]
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import exceptions
from app.core.db import AsyncSessionLocal
from app.models.agent import AgentExperimentRunStatus
from app.services.agent.experiment_service import AgentExperimentRunService, AgentExperimentService
from app.services.llm import LLMService

from .capabilities import get_platform_capabilities
from .planning import build_initial_config, select_llm_model
from .prompts import build_plan_prompt, build_plan_system_instructions
from .state import AgentState, ExperimentPlan, ExperimentRecord
from .summary import build_results_table, build_summary_text, get_last_global_accuracy


class FederatedAgentGraphBuilder:
    """
    Build a LangGraph pipeline that parses a natural-language experiment
    request into N configs, runs them all, and produces a comparison report.
    """

    def __init__(
        self,
        *,
        llm_service: LLMService,
        session: AsyncSession,
        progress_callback: Callable[[AgentState], Awaitable[None] | None] | None = None,
    ) -> None:
        self._llm = llm_service
        self._session = session
        self._progress_callback = progress_callback

    def build(self) -> StateGraph:
        """Build and return a StateGraph operating on AgentState."""
        graph = StateGraph(AgentState)
        graph.add_node("parse", self._node_parse)
        graph.add_node("run_sequential", self._node_run_sequential)
        graph.add_node("report", self._node_report)

        graph.set_entry_point("parse")
        graph.add_edge("parse", "run_sequential")
        graph.add_edge("run_sequential", "report")
        graph.add_edge("report", END)
        return graph

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def _node_parse(self, state: AgentState) -> AgentState:
        """
        Call the LLM to parse the user's natural-language request into
        a list of experiment configurations.
        """
        state.phase = "parsing"
        await self._publish_progress(state)

        experiment_service = AgentExperimentService(self._session)
        schema = experiment_service.get_config_schema()
        capabilities = get_platform_capabilities()
        base_config = build_initial_config(schema)

        instructions = build_plan_system_instructions(state.goal)
        prompt = build_plan_prompt(
            state=state,
            capabilities=capabilities,
            base_config=base_config,
        )

        text, _ = await self._llm.generate_text(
            model=select_llm_model(state.model_name),
            instructions=instructions,
            input_text=prompt,
        )

        # Parse LLM response into experiment plans
        experiments: list[ExperimentPlan] = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                plan_summary = parsed.get("plan_summary", "")
                raw_experiments = parsed.get("experiments", [])
                if isinstance(raw_experiments, list):
                    for idx, exp in enumerate(raw_experiments):
                        if not isinstance(exp, dict):
                            continue
                        name = exp.get("name", f"exp-{idx + 1}")
                        config = exp.get("config", {})
                        if not isinstance(config, dict):
                            config = {}
                        # Merge with base config and normalize
                        merged = self._deep_merge_dicts(base_config, config)
                        try:
                            normalized = experiment_service.normalize_simulation_config(merged)
                        except exceptions.BadRequestError:
                            normalized = experiment_service.normalize_simulation_config(
                                copy.deepcopy(base_config)
                            )
                        experiments.append(
                            ExperimentPlan(
                                iteration=idx + 1,
                                iteration_goal=f"Run experiment: {name}",
                                name=name,
                                plan_summary=plan_summary,
                                config_patch=config,
                            )
                        )
        except (json.JSONDecodeError, Exception):
            pass

        # Fallback: if no experiments were parsed, run a single default
        if not experiments:
            normalized_base = experiment_service.normalize_simulation_config(
                copy.deepcopy(base_config)
            )
            experiments.append(
                ExperimentPlan(
                    iteration=1,
                    iteration_goal="Run default experiment (LLM parse failed)",
                    name="default",
                    plan_summary="Fallback: running single default configuration.",
                    config_patch={},
                )
            )

        state.experiments = experiments
        state.max_iterations = len(experiments)  # sync progress total with actual count
        state.phase = "parsed"
        await self._publish_progress(state)
        return state

    async def _node_run_sequential(self, state: AgentState) -> AgentState:
        """
        Run experiments one by one: launch → wait → record → next.
        Publishes progress after each experiment completes so the
        frontend can show results in real time.
        """
        experiment_service = AgentExperimentService(self._session)
        run_service = AgentExperimentRunService(self._session)
        schema = experiment_service.get_config_schema()
        base_config = build_initial_config(schema)

        max_wait_seconds = 3600
        poll_interval = 2
        total = len(state.experiments)

        for idx, plan in enumerate(state.experiments):
            state.current_plan = plan
            state.iteration = idx + 1

            # -- Build config --
            merged = self._deep_merge_dicts(base_config, plan.config_patch)
            try:
                config = experiment_service.normalize_simulation_config(merged)
            except exceptions.BadRequestError:
                config = experiment_service.normalize_simulation_config(
                    copy.deepcopy(base_config)
                )
            state.current_config = config

            # -- Launch --
            ts = int(time.time())
            exp_name = plan.name or f"exp-{idx + 1}"
            base_name = (
                f"{state.job_name.strip()}-{exp_name}"
                if state.job_name and state.job_name.strip()
                else f"bench-{exp_name}-{ts}"
            )
            description = f"Bench experiment '{exp_name}' for: {state.goal}"

            try:
                experiment = await experiment_service.create_experiment(base_name, description)
            except exceptions.JobAlreadyExists:
                unique_name = f"{base_name}-{uuid.uuid4().hex[:8]}"
                experiment = await experiment_service.create_experiment(unique_name, description)

            experiment = await experiment_service.update_experiment_config(experiment.id, config)
            run = await run_service.start_run(experiment.id)

            run_id = run.id
            state.current_job_id = experiment.id
            state.current_job_name = experiment.name
            state.current_run_id = run_id
            state.current_run_status = str(run.status)
            state.phase = self._truncate_phase(f"running {idx + 1}/{total}: {exp_name}")
            await self._publish_progress(state)

            # -- Wait for completion --
            waited = 0
            while waited < max_wait_seconds:
                async with AsyncSessionLocal() as session:
                    poll_run_service = AgentExperimentRunService(session)
                    run = await poll_run_service.get_run(run_id)
                    state.current_run_status = str(run.status)
                    if run.status not in (
                        AgentExperimentRunStatus.QUEUED,
                        AgentExperimentRunStatus.RUNNING,
                    ):
                        break
                await asyncio.sleep(poll_interval)
                waited += poll_interval

            # -- Collect metrics --
            async with AsyncSessionLocal() as session:
                metrics_run_service = AgentExperimentRunService(session)
                metrics = await metrics_run_service.get_run_metrics(run_id)

            record = ExperimentRecord(
                iteration=idx + 1,
                run_id=run_id,
                job_id=experiment.id,
                config=copy.deepcopy(config),
                metrics=metrics,
                name=plan.name,
                iteration_goal=plan.iteration_goal,
                plan_summary=plan.plan_summary,
                score=get_last_global_accuracy(metrics),
                decision="recorded",
            )
            state.experiment_results.append(record)
            state.history.append(record)

            state.phase = f"completed {idx + 1}/{total}"
            await self._publish_progress(state)

        state.phase = "all_done"
        await self._publish_progress(state)
        return state

    async def _node_report(self, state: AgentState) -> AgentState:
        """
        Call the LLM to generate a comparison report from all experiment results.
        """
        state.phase = "reporting"
        await self._publish_progress(state)

        # Build results context for the LLM
        results_table = build_results_table(state)
        results_detail = []
        for record in state.experiment_results:
            detail = {
                "name": record.name,
                "config": record.config,
                "final_accuracy": record.score,
                "metrics_summary": {
                    k: v
                    for k, v in record.metrics.items()
                    if k in ("global_results",)
                },
            }
            results_detail.append(detail)

        instructions = (
            "You are a federated learning experiment analyst. "
            "Given the results of multiple FL experiments, generate a SHORT "
            "comparison report in markdown. ALWAYS respond in English. "
            "Keep it under 300 words. Include:\n"
            "1. Brief summary (1-2 sentences)\n"
            "2. Results comparison table (markdown table)\n"
            "3. Key findings (2-3 bullet points)\n"
            "Do NOT include recommendations, debugging steps, or verbose explanations. "
            "Focus only on the data and key takeaways."
        )
        input_text = (
            f"Original experiment request: {state.goal}\n\n"
            f"Results table:\n{results_table}\n\n"
            f"Detailed results:\n{json.dumps(results_detail, indent=2, default=str)}\n\n"
            "Generate a comprehensive comparison report."
        )

        try:
            report_text, _ = await self._llm.generate_text(
                model=select_llm_model(state.model_name),
                instructions=instructions,
                input_text=input_text,
            )
            state.summary = report_text
        except Exception:
            # Fall back to the basic table summary
            state.summary = build_summary_text(state=state)

        state.terminated = True
        state.phase = "completed"
        await self._publish_progress(state)
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _publish_progress(self, state: AgentState) -> None:
        """Emit the latest graph state to the optional runtime progress sink."""
        if self._progress_callback is None:
            return
        await self._progress_callback(state)

    # agent_optimization_jobs.current_phase is VARCHAR(255) in Postgres; never
    # serialize anything longer or the whole agent task crashes on snapshot persist.
    _PHASE_MAX_LEN = 255

    @classmethod
    def _truncate_phase(cls, value: str) -> str:
        if len(value) <= cls._PHASE_MAX_LEN:
            return value
        return value[: cls._PHASE_MAX_LEN - 1] + "…"

    @staticmethod
    def _deep_merge_dicts(base: dict, override: dict) -> dict:
        """Deep-merge override values into base without mutating either input."""
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key.startswith("_"):
                continue  # skip internal metadata keys
            base_value = merged.get(key)
            if isinstance(base_value, dict) and isinstance(value, dict):
                merged[key] = FederatedAgentGraphBuilder._deep_merge_dicts(
                    base_value, value
                )
            else:
                merged[key] = copy.deepcopy(value)
        return merged
