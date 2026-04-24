import { baseUrl } from "./client";
import { getHttpErrorMessage } from "./error";
import type { RunMetrics } from "./runs";

export type AgentOptimizationObjective = "auto" | "accuracy";

export type AgentOptimizeRequest = {
  goal: string;
  max_iterations: number;
  system_mode: "simulation";
  model_name?: string | null;
  job_name?: string | null;
  objective: AgentOptimizationObjective;
};

export type AgentConfigChange = {
  path: string;
  change_type: string;
  old_value: unknown;
  new_value: unknown;
};

export type AgentCurrentPlan = {
  iteration: number;
  iteration_goal: string;
  plan_summary: string | null;
  hypothesis: string | null;
  rationale: string[];
  config_patch: Record<string, unknown>;
  config_diff: AgentConfigChange[];
};

export type AgentExperimentSummary = {
  iteration: number;
  run_id: string;
  job_id: number;
  iteration_goal: string | null;
  plan_summary: string | null;
  hypothesis: string | null;
  rationale: string[];
  config_patch: Record<string, unknown>;
  config_diff: AgentConfigChange[];
  config: Record<string, unknown>;
  metrics: RunMetrics;
  score: number | null;
  result_summary: string | null;
  decision: string | null;
  lessons_learned: string[];
};

export type AgentOptimizeResponse = {
  goal: string;
  job_name: string | null;
  max_iterations: number;
  objective: AgentOptimizationObjective;
  resolved_objective: AgentOptimizationObjective;
  iterations_executed: number;
  best_config: Record<string, unknown> | null;
  best_metrics: RunMetrics | null;
  experiments: AgentExperimentSummary[];
  summary_text: string | null;
};

export type AgentCurrentExperiment = {
  iteration: number;
  phase: string | null;
  job_id: number | null;
  job_name: string | null;
  run_id: string | null;
  run_status: string | null;
  config: Record<string, unknown> | null;
  metrics: RunMetrics | null;
};

export type AgentOptimizeProgressResponse = {
  optimization_job_id: number | null;
  task_id: string;
  status: string;
  goal: string;
  job_name: string | null;
  max_iterations: number;
  model_name: string | null;
  objective: AgentOptimizationObjective;
  resolved_objective: AgentOptimizationObjective;
  current_phase: string | null;
  current_iteration: number;
  completed_iterations: number;
  current_plan: AgentCurrentPlan | null;
  current_experiment: AgentCurrentExperiment | null;
  best_config: Record<string, unknown> | null;
  best_metrics: RunMetrics | null;
  experiments: AgentExperimentSummary[];
  draft_experiments?: any[];
  summary_text: string | null;
  error_message: string | null;
  created_at: string | null;
  updated_at: string | null;
  finished_at: string | null;
};

export type AgentModelsResponse = {
  models: string[];
  default_model: string;
};

export type AgentOptimizationJobSummary = {
  optimization_job_id: number;
  task_id: string;
  job_name: string | null;
  status: string;
  goal: string;
  model_name: string | null;
  objective: AgentOptimizationObjective;
  resolved_objective: AgentOptimizationObjective;
  current_phase: string | null;
  max_iterations: number;
  current_iteration: number;
  completed_iterations: number;
  best_score: number | null;
  created_at: string;
  updated_at: string;
  finished_at: string | null;
};

// -----------------------------------------------------------------------
//  Agent Experiment / Run types (DB-backed tables)
// -----------------------------------------------------------------------

export type AgentExperimentResponse = {
  id: number;
  name: string;
  description: string | null;
  status: string;
  config_json: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

export type AgentRunResponse = {
  id: string;
  experiment_id: number;
  status: string;
  config_json: Record<string, unknown>;
  metrics_json: Record<string, unknown>;
  started_at: string | null;
  ended_at: string | null;
  created_at: string;
};

export type AgentRunLogResponse = {
  id: number;
  level: string;
  message: string;
  created_at: string;
};

export interface ExperimentPlanPreview {
  name: string;
  plan_summary: string;
  config_patch: Record<string, unknown>;
  estimated_minutes?: number;
  estimated_gpu_vram_gb?: number;
  risk_warnings?: string[];
};

export interface AgentPlanPreviewRequest {
  goal: string;
  job_name: string;
  model_name: string | null;
  system_mode: string;
};

export interface AgentPlanPreviewResponse {
  optimization_job_id: number;
  goal: string;
  experiments: ExperimentPlanPreview[];
  system_mode: string;
};

export interface AgentPlanReviseRequest {
  instruction: string;
};

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await getHttpErrorMessage(response));
  }
  return (await response.json()) as T;
}

export const agentApi = {
  async listModels(): Promise<AgentModelsResponse> {
    const response = await fetch(`${baseUrl}/api/v1/agent/models`);
    return readJson<AgentModelsResponse>(response);
  },

  async optimize(payload: AgentOptimizeRequest): Promise<AgentOptimizeResponse> {
    const response = await fetch(`${baseUrl}/api/v1/agent/optimize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return readJson<AgentOptimizeResponse>(response);
  },

  async startOptimize(payload: AgentOptimizeRequest): Promise<AgentOptimizeProgressResponse> {
    const response = await fetch(`${baseUrl}/api/v1/agent/optimize/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return readJson<AgentOptimizeProgressResponse>(response);
  },

  async getOptimizeProgress(taskId: string): Promise<AgentOptimizeProgressResponse> {
    const response = await fetch(`${baseUrl}/api/v1/agent/optimize/${encodeURIComponent(taskId)}`);
    return readJson<AgentOptimizeProgressResponse>(response);
  },

  async listOptimizationJobs(): Promise<AgentOptimizationJobSummary[]> {
    const response = await fetch(`${baseUrl}/api/v1/agent/optimization-jobs`);
    return readJson<AgentOptimizationJobSummary[]>(response);
  },

  async getOptimizationJob(optimizationJobId: number): Promise<AgentOptimizeProgressResponse> {
    const response = await fetch(`${baseUrl}/api/v1/agent/optimization-jobs/${encodeURIComponent(optimizationJobId)}`);
    return readJson<AgentOptimizeProgressResponse>(response);
  },

  // -- Agent Experiment / Run endpoints (DB-backed tables) ----------------

  async listExperiments(): Promise<AgentExperimentResponse[]> {
    const response = await fetch(`${baseUrl}/api/v1/agent/experiments`);
    return readJson<AgentExperimentResponse[]>(response);
  },

  async getExperiment(experimentId: number): Promise<AgentExperimentResponse> {
    const response = await fetch(`${baseUrl}/api/v1/agent/experiments/${encodeURIComponent(experimentId)}`);
    return readJson<AgentExperimentResponse>(response);
  },

  async listExperimentRuns(experimentId: number): Promise<AgentRunResponse[]> {
    const response = await fetch(`${baseUrl}/api/v1/agent/experiments/${encodeURIComponent(experimentId)}/runs`);
    return readJson<AgentRunResponse[]>(response);
  },

  async listRuns(): Promise<AgentRunResponse[]> {
    const response = await fetch(`${baseUrl}/api/v1/agent/runs`);
    return readJson<AgentRunResponse[]>(response);
  },

  async getRun(runId: string): Promise<AgentRunResponse> {
    const response = await fetch(`${baseUrl}/api/v1/agent/runs/${encodeURIComponent(runId)}`);
    return readJson<AgentRunResponse>(response);
  },

  async getRunMetrics(runId: string): Promise<{ run_id: string; metrics: Record<string, unknown> }> {
    const response = await fetch(`${baseUrl}/api/v1/agent/runs/${encodeURIComponent(runId)}/metrics`);
    return readJson<{ run_id: string; metrics: Record<string, unknown> }>(response);
  },

  async getRunLogs(runId: string): Promise<AgentRunLogResponse[]> {
    const response = await fetch(`${baseUrl}/api/v1/agent/runs/${encodeURIComponent(runId)}/logs`);
    return readJson<AgentRunLogResponse[]>(response);
  },

  async generatePlan(request: AgentPlanPreviewRequest): Promise<AgentPlanPreviewResponse> {
    const response = await fetch(`${baseUrl}/api/v1/agent/optimize/plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    return readJson<AgentPlanPreviewResponse>(response);
  },

  async startOptimizeFromDraft(request: any): Promise<any> {
    const response = await fetch(`${baseUrl}/api/v1/agent/optimize/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    return readJson<any>(response);
  },

  async revisePlan(jobId: number, request: AgentPlanReviseRequest): Promise<AgentPlanPreviewResponse> {
    const response = await fetch(`${baseUrl}/api/v1/agent/optimization-jobs/${encodeURIComponent(jobId)}/revise`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || "Failed to revise plan");
    }
    return response.json();
  },
};