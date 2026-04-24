import { baseUrl as defaultBaseUrl } from "./client";
import { getHttpErrorMessage } from "./error";

export type DistributedJob = {
  id: number;
  name: string;
  description: string | null;
  expected_clients: number;
  config_json: Record<string, unknown>;
  status: "draft" | "waiting_clients" | "ready" | "running" | "finished";
  created_at: string;
  updated_at: string;
};

export type DistributedSession = {
  id: string;
  job_id: number;
  server_ip: string;
  server_port: number;
  status: "waiting_clients" | "ready_to_start" | "running" | "finished" | "cancelled";
  started_at: string | null;
  ended_at: string | null;
  created_at: string;
  updated_at: string;
};

export type DistributedClient = {
  id: string;
  session_id: string;
  participant_name: string;
  assigned_participant_id: number | null;
  status: "pending_approval" | "approved" | "ready" | "running" | "rejected" | "cancelled" | "disconnected";
  metadata_json: Record<string, unknown>;
  requested_at: string;
  approved_at: string | null;
  ready_at: string | null;
  last_seen_at: string;
  created_at: string;
  updated_at: string;
};

export type DistributedClientRuntimeConfig = {
  participant_id: string;
  session_id: string;
  assigned_participant_id: number;
  server_ip: string;
  server_port: number;
  config_json: Record<string, unknown>;
};

export type DistributedRuntimeStatus = {
  role: string;
  runtime_id: string;
  status: string;
  pid: number | null;
  started_at: string | null;
  ended_at: string | null;
  exit_code: number | null;
  config_path: string;
  log_path: string | null;
  results_path: string | null;
  total_rounds: number;
};

export type DistributedSessionProgress = {
  session_id: string;
  status: string;
  pid: number | null;
  started_at: string | null;
  ended_at: string | null;
  exit_code: number | null;
  total_rounds: number;
  last_round: number;
  latest_global_loss: number | null;
  latest_global_accuracy: number | null;
  metrics_json: Record<string, unknown>;
  log_path: string | null;
};

type DistributedJobCreatePayload = {
  name: string;
  description?: string | null;
};

type DistributedJobConfigUpdatePayload = {
  config: Record<string, unknown>;
};

type DistributedSessionCreatePayload = {
  server_ip: string;
  server_port: number;
};

type DistributedConnectPayload = {
  participant_name: string;
  metadata_json?: Record<string, unknown>;
};

type DistributedLocalClientStartPayload = {
  participant_id: string;
  config_json: Record<string, unknown>;
};

function normalizeBaseUrl(raw?: string): string {
  const candidate = (raw ?? defaultBaseUrl).trim();
  if (candidate.length === 0) return defaultBaseUrl;
  return candidate.replace(/\/+$/, "");
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await getHttpErrorMessage(response));
  }
  return (await response.json()) as T;
}

export const distributedApi = {
  async listJobs(base?: string): Promise<DistributedJob[]> {
    const response = await fetch(`${normalizeBaseUrl(base)}/api/v1/distributed/jobs`);
    return readJson<DistributedJob[]>(response);
  },

  async getJob(jobId: number, base?: string): Promise<DistributedJob> {
    const response = await fetch(`${normalizeBaseUrl(base)}/api/v1/distributed/jobs/${jobId}`);
    return readJson<DistributedJob>(response);
  },

  async getActiveSession(jobId: number, base?: string): Promise<DistributedSession | null> {
    const response = await fetch(`${normalizeBaseUrl(base)}/api/v1/distributed/jobs/${jobId}/session`);
    if (response.status === 404) {
      return null;
    }
    return readJson<DistributedSession>(response);
  },

  async createJob(payload: DistributedJobCreatePayload, base?: string): Promise<DistributedJob> {
    const response = await fetch(`${normalizeBaseUrl(base)}/api/v1/distributed/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return readJson<DistributedJob>(response);
  },

  async updateJobConfig(jobId: number, payload: DistributedJobConfigUpdatePayload, base?: string): Promise<DistributedJob> {
    const response = await fetch(`${normalizeBaseUrl(base)}/api/v1/distributed/jobs/${jobId}/config`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return readJson<DistributedJob>(response);
  },

  async createOrGetSession(jobId: number, payload: DistributedSessionCreatePayload, base?: string): Promise<DistributedSession> {
    const response = await fetch(`${normalizeBaseUrl(base)}/api/v1/distributed/jobs/${jobId}/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return readJson<DistributedSession>(response);
  },

  async getSession(sessionId: string, base?: string): Promise<DistributedSession> {
    const response = await fetch(`${normalizeBaseUrl(base)}/api/v1/distributed/sessions/${encodeURIComponent(sessionId)}`);
    return readJson<DistributedSession>(response);
  },

  async listSessionClients(sessionId: string, base?: string): Promise<DistributedClient[]> {
    const response = await fetch(
      `${normalizeBaseUrl(base)}/api/v1/distributed/sessions/${encodeURIComponent(sessionId)}/participants`,
    );
    return readJson<DistributedClient[]>(response);
  },

  async requestConnect(sessionId: string, payload: DistributedConnectPayload, base?: string): Promise<DistributedClient> {
    const response = await fetch(
      `${normalizeBaseUrl(base)}/api/v1/distributed/sessions/${encodeURIComponent(sessionId)}/connect`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
    );
    return readJson<DistributedClient>(response);
  },

  async getClient(clientId: string, base?: string): Promise<DistributedClient> {
    const response = await fetch(`${normalizeBaseUrl(base)}/api/v1/distributed/participants/${encodeURIComponent(clientId)}`);
    return readJson<DistributedClient>(response);
  },

  async getClientRuntimeConfig(clientId: string, base?: string): Promise<DistributedClientRuntimeConfig> {
    const response = await fetch(
      `${normalizeBaseUrl(base)}/api/v1/distributed/participants/${encodeURIComponent(clientId)}/runtime-config`,
    );
    return readJson<DistributedClientRuntimeConfig>(response);
  },

  async approveClient(clientId: string, base?: string): Promise<DistributedClient> {
    const response = await fetch(
      `${normalizeBaseUrl(base)}/api/v1/distributed/participants/${encodeURIComponent(clientId)}/approve`,
      { method: "POST" },
    );
    return readJson<DistributedClient>(response);
  },

  async rejectClient(clientId: string, base?: string): Promise<DistributedClient> {
    const response = await fetch(
      `${normalizeBaseUrl(base)}/api/v1/distributed/participants/${encodeURIComponent(clientId)}/reject`,
      { method: "POST" },
    );
    return readJson<DistributedClient>(response);
  },

  async readyClient(clientId: string, base?: string): Promise<DistributedClient> {
    const response = await fetch(
      `${normalizeBaseUrl(base)}/api/v1/distributed/participants/${encodeURIComponent(clientId)}/ready`,
      { method: "POST" },
    );
    return readJson<DistributedClient>(response);
  },

  async cancelClient(clientId: string, base?: string): Promise<DistributedClient> {
    const response = await fetch(
      `${normalizeBaseUrl(base)}/api/v1/distributed/participants/${encodeURIComponent(clientId)}/cancel`,
      { method: "POST" },
    );
    return readJson<DistributedClient>(response);
  },

  async startSession(sessionId: string, base?: string): Promise<DistributedSession> {
    const response = await fetch(
      `${normalizeBaseUrl(base)}/api/v1/distributed/sessions/${encodeURIComponent(sessionId)}/start`,
      { method: "POST" },
    );
    return readJson<DistributedSession>(response);
  },

  async startLocalClientRuntime(payload: DistributedLocalClientStartPayload, base?: string): Promise<DistributedRuntimeStatus> {
    const response = await fetch(
      `${normalizeBaseUrl(base)}/api/v1/distributed/runtime/participant/start`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
    );
    return readJson<DistributedRuntimeStatus>(response);
  },

  async getSessionProgress(sessionId: string, base?: string): Promise<DistributedSessionProgress> {
    const response = await fetch(
      `${normalizeBaseUrl(base)}/api/v1/distributed/sessions/${encodeURIComponent(sessionId)}/progress`,
    );
    return readJson<DistributedSessionProgress>(response);
  },
};
