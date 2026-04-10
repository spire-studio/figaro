import { api, baseUrl } from "./client";
import { extractErrorMessage, getHttpErrorMessage } from "./error";
import type { components } from "./openapi";

type JobUpdateRequest = components["schemas"]["JobUpdateRequest"];
type JobCopyRequest = components["schemas"]["JobCopyRequest"];
type ConfigSchema = Record<string, unknown>;
type JobCreatePayload = {
  name: string;
  description?: string | null;
};
type JobConfigPayload = {
  job_id: number;
  config_json: Record<string, unknown>;
  updated_at: string;
};

async function unwrap<T>(promise: Promise<{ data?: T; error?: unknown }>): Promise<T> {
  const { data, error } = await promise;
  if (error || data === undefined) {
    throw new Error(extractErrorMessage(error) ?? "API request failed");
  }
  return data;
}

async function removeJob(jobId: number): Promise<{ message: string; detail?: string | null }> {
  const response = await fetch(`${baseUrl}/api/v1/jobs/${jobId}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(await getHttpErrorMessage(response));
  }
  return (await response.json()) as { message: string; detail?: string | null };
}

async function rerunRun(runId: string): Promise<components["schemas"]["RunResponse"]> {
  const response = await fetch(`${baseUrl}/api/v1/runs/${encodeURIComponent(runId)}/rerun`, { method: "POST" });
  if (!response.ok) {
    throw new Error(await getHttpErrorMessage(response));
  }
  return (await response.json()) as components["schemas"]["RunResponse"];
}

async function deleteRun(runId: string): Promise<{ message: string; detail?: string | null }> {
  const response = await fetch(`${baseUrl}/api/v1/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(await getHttpErrorMessage(response));
  }
  return (await response.json()) as { message: string; detail?: string | null };
}

async function getConfigSchema(): Promise<ConfigSchema> {
  const response = await fetch(`${baseUrl}/api/v1/jobs/config/schema`);
  if (!response.ok) {
    throw new Error(await getHttpErrorMessage(response));
  }
  return (await response.json()) as ConfigSchema;
}

async function createJob(payload: JobCreatePayload): Promise<components["schemas"]["JobResponse"]> {
  const response = await fetch(`${baseUrl}/api/v1/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await getHttpErrorMessage(response));
  }
  return (await response.json()) as components["schemas"]["JobResponse"];
}

async function getJobConfig(jobId: number): Promise<JobConfigPayload> {
  const response = await fetch(`${baseUrl}/api/v1/jobs/${jobId}/config`);
  if (!response.ok) {
    throw new Error(await getHttpErrorMessage(response));
  }
  return (await response.json()) as JobConfigPayload;
}

async function updateJobConfig(jobId: number, config: Record<string, unknown>): Promise<JobConfigPayload> {
  const response = await fetch(`${baseUrl}/api/v1/jobs/${jobId}/config`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  });
  if (!response.ok) {
    throw new Error(await getHttpErrorMessage(response));
  }
  return (await response.json()) as JobConfigPayload;
}

export const jobsApi = {
  health: () => unwrap(api.GET("/api/v1/health/fastapi")),
  getConfigSchema: () => getConfigSchema(),
  list: () => unwrap(api.GET("/api/v1/jobs")),
  get: (jobId: number) => unwrap(api.GET("/api/v1/jobs/{job_id}", { params: { path: { job_id: jobId } } })),
  create: (payload: JobCreatePayload) => createJob(payload),
  remove: (jobId: number) => removeJob(jobId),
  update: (jobId: number, payload: JobUpdateRequest) =>
    unwrap(api.PATCH("/api/v1/jobs/{job_id}", { params: { path: { job_id: jobId } }, body: payload })),
  copy: (jobId: number, payload: JobCopyRequest) =>
    unwrap(api.POST("/api/v1/jobs/{job_id}/copy", { params: { path: { job_id: jobId } }, body: payload })),
  getConfig: (jobId: number) => getJobConfig(jobId),
  updateConfig: (jobId: number, config: Record<string, unknown>) => updateJobConfig(jobId, config),
  run: (jobId: number) => unwrap(api.POST("/api/v1/jobs/{job_id}/runs", { params: { path: { job_id: jobId } } })),
  rerunRun: (runId: string) => rerunRun(runId),
  deleteRun: (runId: string) => deleteRun(runId),
  getRun: (runId: string) => unwrap(api.GET("/api/v1/runs/{run_id}", { params: { path: { run_id: runId } } })),
  stopRun: (runId: string) => unwrap(api.POST("/api/v1/runs/{run_id}/stop", { params: { path: { run_id: runId } } })),
  listRunLogs: (runId: string, limit = 500) =>
    unwrap(api.GET("/api/v1/runs/{run_id}/logs", { params: { path: { run_id: runId }, query: { limit } } })),
};
