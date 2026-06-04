import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import {
  agentApi,
  type AgentExperimentResponse,
  type AgentHistoryFilters,
  type AgentOptimizationObjective,
  type AgentOptimizationJobSummary,
  type AgentOptimizeProgressResponse,
  type AgentOptimizeResponse,
  type AgentRunResponse,
} from "../../api/agent";
import type { AgentPageProps, AgentWorkflowStep, AgentPlanDraft } from "../../pages/types";
import { latestLlmLoss } from "../simulation/llm-metrics";
import { toErrorMessage } from "../simulation/utils";
import {
  buildAgentConfig,
  applyConfigConstraints,
  inferSftSettingsForDatasetPath,
  materializeAgentConfigConstraints,
  missingLlmResourceMessages,
  removeValueByPath,
  setConfigValue,
} from "./schema";

const DEFAULT_GOAL = "Compare non-IID alpha=0.1, 0.3, 0.5 on the selected dataset";
const POLL_INTERVAL_MS = 1500;

function makeDefaultJobName(): string {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, "0");
  return `agent-opt-${now.getUTCFullYear()}${pad(now.getUTCMonth() + 1)}${pad(now.getUTCDate())}-${pad(now.getUTCHours())}${pad(now.getUTCMinutes())}${pad(now.getUTCSeconds())}`;
}

function toFinalResult(progress: AgentOptimizeProgressResponse): AgentOptimizeResponse {
  return {
    goal: progress.goal,
    job_name: progress.job_name,
    max_iterations: progress.max_iterations,
    objective: progress.objective,
    resolved_objective: progress.resolved_objective,
    iterations_executed: progress.completed_iterations,
    best_config: progress.best_config,
    best_metrics: progress.best_metrics,
    experiments: progress.experiments,
    summary_text: progress.summary_text,
  };
}

function toHistorySummary(progress: AgentOptimizeProgressResponse): AgentOptimizationJobSummary | null {
  if (progress.optimization_job_id === null) {
    return null;
  }
  const llmLoss = latestLlmLoss(progress.best_metrics);
  const accuracy = progress.best_metrics?.global_results?.global_accuracy ?? [];
  const lastAccuracy = accuracy.length > 0 && typeof accuracy[accuracy.length - 1] === "number"
    ? accuracy[accuracy.length - 1]
    : null;
  const bestScore = llmLoss !== null ? -llmLoss : lastAccuracy;
  return {
    optimization_job_id: progress.optimization_job_id,
    task_id: progress.task_id,
    job_name: progress.job_name,
    status: progress.status,
    goal: progress.goal,
    model_name: progress.model_name,
    objective: progress.objective,
    resolved_objective: progress.resolved_objective,
    current_phase: progress.current_phase,
    max_iterations: progress.max_iterations,
    current_iteration: progress.current_iteration,
    completed_iterations: progress.completed_iterations,
    best_score: bestScore,
    created_at: progress.created_at ?? new Date().toISOString(),
    updated_at: progress.updated_at ?? new Date().toISOString(),
    finished_at: progress.finished_at,
  };
}

export function useAgentController(): AgentPageProps {
  const [workflowStep, setWorkflowStep] = useState<AgentWorkflowStep>("home");
  const [draftPlan, setDraftPlan] = useState<AgentPlanDraft | null>(null);
  const [configSchema, setConfigSchema] = useState<Record<string, unknown> | null>(null);
  const [configConstraints, setConfigConstraints] = useState<Record<string, unknown>>({});
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [goal, setGoal] = useState(DEFAULT_GOAL);
  const [jobName, setJobName] = useState(() => makeDefaultJobName());
  const [maxIterations, setMaxIterations] = useState(2);
  const [modelName, setModelName] = useState("");
  const [objective, setObjective] = useState<AgentOptimizationObjective>("auto");
  const [defaultModelName, setDefaultModelName] = useState("");
  const [modelOptions, setModelOptions] = useState<string[]>([]);
  const [experiments, setExperiments] = useState<AgentExperimentResponse[]>([]);
  const [experimentRuns, setExperimentRuns] = useState<AgentRunResponse[]>([]);
  const [selectedExperimentId, setSelectedExperimentId] = useState<number | null>(null);
  const [historyJobs, setHistoryJobs] = useState<AgentOptimizationJobSummary[]>([]);
  const [historyFilters, setHistoryFilters] = useState<AgentHistoryFilters>({ status: "all", objective: "all" });
  const [progress, setProgress] = useState<AgentOptimizeProgressResponse | null>(null);
  const [result, setResult] = useState<AgentOptimizeResponse | null>(null);
  const [selectedHistory, setSelectedHistory] = useState<AgentOptimizeProgressResponse | null>(null);
  const [selectedHistoryJobId, setSelectedHistoryJobId] = useState<number | null>(null);
  const [lastSubmittedGoal, setLastSubmittedGoal] = useState<string | null>(null);

  function notifyError(error: unknown, id?: string): void {
    toast.error(toErrorMessage(error), { id });
  }

  function upsertHistoryJob(summary: AgentOptimizationJobSummary | null): void {
    if (summary === null) {
      return;
    }
    setHistoryJobs((current) => {
      const next = current.filter((item) => item.optimization_job_id !== summary.optimization_job_id);
      next.unshift(summary);
      next.sort((left, right) => right.updated_at.localeCompare(left.updated_at));
      return next;
    });
  }

  async function refreshHistoryJobs(filters: AgentHistoryFilters = historyFilters): Promise<void> {
    const jobs = await agentApi.listOptimizationJobs(filters);
    setHistoryJobs(jobs);
    if (jobs.length > 0 && selectedHistoryJobId === null && !busy && !progress) {
      const first = jobs[0];
      setSelectedHistoryJobId(first.optimization_job_id);
      try {
        const detail = await agentApi.getOptimizationJob(first.optimization_job_id);
        setSelectedHistory(detail);
      } catch (error) {
        notifyError(error, "agent-history-detail-default");
      }
    }
  }

  async function refreshExperiments(): Promise<void> {
    const list = await agentApi.listExperiments();
    setExperiments(list);
  }

  async function refreshLlmResources(): Promise<void> {
    const [models, schema] = await Promise.all([
      agentApi.listModels(),
      agentApi.getConfigSchema(),
    ]);

    setModelOptions(models.models);
    setDefaultModelName(models.default_model);
    setConfigSchema(schema);
  }

  async function selectExperiment(experimentId: number): Promise<void> {
    setSelectedExperimentId(experimentId);
    try {
      const runs = await agentApi.listExperimentRuns(experimentId);
      setExperimentRuns(runs);
    } catch (error) {
      notifyError(error, "agent-experiment-runs");
      setExperimentRuns([]);
    }
  }

  async function selectHistoryJob(optimizationJobId: number): Promise<void> {
    setSelectedHistoryJobId(optimizationJobId);
    try {
      const detail = await agentApi.getOptimizationJob(optimizationJobId);
      setSelectedHistory(detail);

      if (detail.status === "pending_review" && detail.draft_experiments?.length) {
        setDraftPlan({
          job_id: detail.optimization_job_id!,
          goal: detail.goal,
          experiments: detail.draft_experiments,
          config_constraints: detail.config_constraints ?? {},
        });
        setConfigConstraints(detail.config_constraints ?? {});
        setWorkflowStep("preview");
      } else if (detail.status === "completed") {
        setResult(toFinalResult(detail));
      }
    } catch (error) {
      notifyError(error, "agent-history-detail");
    }
  }
  
  async function handleGeneratePlan(): Promise<void> {
    const trimmedGoal = goal.trim();
    if (!trimmedGoal) {
      toast.warning("Please describe your experiment.");
      return;
    }
    const trimmedJobName = jobName.trim();
    if (!trimmedJobName) {
      toast.warning("Please provide a job name.");
      return;
    }
    const submissionConstraints = materializeAgentConfigConstraints(configSchema, configConstraints);
    const missingResources = missingLlmResourceMessages(
      configSchema,
      buildAgentConfig(configSchema, submissionConstraints),
    );
    if (missingResources.length > 0) {
      toast.error(missingResources.join(", "));
      return;
    }
  
    setBusy(true);
    try {
      const data = await agentApi.generatePlan({
        goal: trimmedGoal,
        job_name: trimmedJobName,
        model_name: modelName.trim().length > 0 ? modelName.trim() : null,
        system_mode: "simulation",
        config_constraints: submissionConstraints,
      });
  
      setDraftPlan({
        job_id: data.optimization_job_id!,
        goal: data.goal,
        experiments: data.experiments,
        config_constraints: data.config_constraints ?? submissionConstraints,
      });
      
      setWorkflowStep("preview");
      void refreshHistoryJobs(); // Refresh sidebar to show PENDING_REVIEW job
    } catch (error) {
      notifyError(error, "agent-plan");
    } finally {
      setBusy(false);
    }
  }
  
  async function handleExecutePlan(editedExperiments: any[]): Promise<void> {
    if (!draftPlan) return;
    const submissionConstraints = materializeAgentConfigConstraints(
      configSchema,
      draftPlan.config_constraints ?? configConstraints,
    );
    const disabledResources = editedExperiments.flatMap((exp) =>
      missingLlmResourceMessages(
        configSchema,
        buildAgentConfig(configSchema, applyConfigConstraints(exp.config_patch ?? {}, submissionConstraints)),
      ),
    );
    if (disabledResources.length > 0) {
      toast.error(Array.from(new Set(disabledResources)).join(", "));
      return;
    }
    
    setBusy(true);
    setResult(null);
    setProgress(null);
    
    try {
      const data = await agentApi.startOptimizeFromDraft({
        goal: draftPlan.goal,
        job_name: jobName,
        max_iterations: editedExperiments.length,
        system_mode: "simulation",
        model_name: modelName.trim().length > 0 ? modelName.trim() : null,
        objective: objective,
        planned_experiments: editedExperiments, // Bypass LLM parse in backend
        config_constraints: submissionConstraints,
      });
      
      setProgress(data);
      setActiveTaskId(data.task_id);
      setSelectedHistory(data);
      if (data.optimization_job_id !== null) {
        setSelectedHistoryJobId(data.optimization_job_id);
      }
      upsertHistoryJob(toHistorySummary(data));
      setWorkflowStep("running");
    } catch (error) {
      notifyError(error, "agent-execute");
      setBusy(false);
    }
  }

  async function handleOptimize(): Promise<void> {
    const trimmedGoal = goal.trim();
    if (!trimmedGoal) {
      toast.warning("Please describe your experiment.");
      return;
    }
    const trimmedJobName = jobName.trim();
    if (!trimmedJobName) {
      toast.warning("Please provide a job name for this experiment.");
      return;
    }
    const submissionConstraints = materializeAgentConfigConstraints(configSchema, configConstraints);
    const missingResources = missingLlmResourceMessages(
      configSchema,
      buildAgentConfig(configSchema, submissionConstraints),
    );
    if (missingResources.length > 0) {
      toast.error(missingResources.join(", "));
      return;
    }

    setBusy(true);
    setResult(null);
    setProgress(null);
    setLastSubmittedGoal(trimmedGoal);
    try {
      const data = await agentApi.startOptimize({
        goal: trimmedGoal,
        max_iterations: Math.max(1, Math.min(50, Math.floor(maxIterations))),
        system_mode: "simulation",
        model_name: modelName.trim().length > 0 ? modelName.trim() : null,
        job_name: trimmedJobName,
        objective,
        config_constraints: submissionConstraints,
      });
      setProgress(data);
      setActiveTaskId(data.task_id);
      setSelectedHistory(data);
      if (data.optimization_job_id !== null) {
        setSelectedHistoryJobId(data.optimization_job_id);
      }
      upsertHistoryJob(toHistorySummary(data));
      setJobName(makeDefaultJobName());
      setWorkflowStep("running");
    } catch (error) {
      notifyError(error, "agent-optimize");
      setBusy(false);
      setActiveTaskId(null);
    }
  }

  function clearResult(): void {
    setResult(null);
    setProgress(null);
    setActiveTaskId(null);
  }

  const presets = useMemo(
    () => [
      "Compare non-IID alpha=0.1, 0.3, 0.5 on the selected dataset",
      "Compare 10 clients vs 20 clients with FedAvg",
      "Test training rounds 10, 20, 50 on accuracy",
      "Run LLM PEFT SFT with LoRA rank 8 and rank 16 on a JSONL prompt/completion dataset",
    ],
    [],
  );

  useEffect(() => {
    let cancelled = false;
    void refreshLlmResources().catch((error: unknown) => {
      if (!cancelled) {
        toast.error(toErrorMessage(error), { id: "agent-llm-resources" });
      }
    });
    void refreshHistoryJobs().catch((error: unknown) => {
      if (!cancelled) {
        notifyError(error, "agent-history-jobs");
      }
    });
    void refreshExperiments().catch((error: unknown) => {
      if (!cancelled) {
        notifyError(error, "agent-experiments");
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!activeTaskId) {
      return;
    }

    let cancelled = false;

    const pollProgress = async () => {
      while (!cancelled) {
        try {
          const nextProgress = await agentApi.getOptimizeProgress(activeTaskId);
          if (cancelled) {
            return;
          }

          setProgress(nextProgress);
          if (selectedHistoryJobId !== null && nextProgress.optimization_job_id === selectedHistoryJobId) {
            setSelectedHistory(nextProgress);
          }
          upsertHistoryJob(toHistorySummary(nextProgress));

          if (nextProgress.status === "completed") {
            setResult(toFinalResult(nextProgress));
            setBusy(false);
            setActiveTaskId(null);
            setWorkflowStep("results");
            void refreshHistoryJobs().catch((error: unknown) => notifyError(error, "agent-history-refresh"));
            void refreshExperiments().catch((error: unknown) => notifyError(error, "agent-experiments-refresh"));
            toast.success("Experiment run finished.");
            return;
          }

          if (nextProgress.status === "failed") {
            setBusy(false);
            setActiveTaskId(null);
            void refreshHistoryJobs().catch((error: unknown) => notifyError(error, "agent-history-refresh"));
            void refreshExperiments().catch((error: unknown) => notifyError(error, "agent-experiments-refresh"));
            toast.error(nextProgress.error_message ?? "Experiment run failed.", { id: "agent-optimize" });
            return;
          }
        } catch (error) {
          if (cancelled) {
            return;
          }
          notifyError(error, "agent-progress");
          setBusy(false);
          setActiveTaskId(null);
          return;
        }

        await new Promise((resolve) => window.setTimeout(resolve, POLL_INTERVAL_MS));
      }
    };

    void pollProgress();

    return () => {
      cancelled = true;
    };
  }, [activeTaskId]);

  function setConfigConstraint(path: string, value: unknown): void {
    setConfigConstraints((current) => {
      let next = setConfigValue(current, path, value);
      if (path === "sft.dataset_path") {
        for (const [inferredPath, inferredValue] of Object.entries(inferSftSettingsForDatasetPath(value))) {
          next = setConfigValue(next, inferredPath, inferredValue);
        }
      }
      return next;
    });
  }

  function clearConfigConstraint(path: string): void {
    setConfigConstraints((current) => removeValueByPath(current, path));
  }

  return {
    activeTaskId,
    busy,
    clearConfigConstraint,
    clearResult,
    configConstraints,
    configSchema,
    defaultModelName,
    experiments,
    experimentRuns,
    goal,
    handleOptimize,
    historyFilters,
    historyJobs,
    jobName,
    lastSubmittedGoal,
    maxIterations,
    modelName,
    objective,
    modelOptions,
    notifyError,
    presets,
    progress,
    result,
    selectedExperimentId,
    selectedHistory,
    selectedHistoryJobId,
    selectExperiment,
    selectHistoryJob,
    refreshHistoryJobs,
    refreshLlmResources,
    setConfigConstraint,
    setGoal,
    setJobName,
    setMaxIterations,
    setModelName,
    setObjective,
    setHistoryFilters,
    workflowStep,
    setWorkflowStep,
    draftPlan,
    setDraftPlan,
    handleGeneratePlan,
    handleExecutePlan,
  };
}
