import type { components } from "./openapi";
import { getHttpErrorMessage } from "./error";

type Run = components["schemas"]["RunResponse"];

export type RunClientMetricSeries = {
  train_loss: number[];
  train_acc: number[];
  test_loss: number[];
  test_acc: number[];
};

export type RunMetrics = {
  experiment_info: {
    basic: Record<string, unknown>;
    federated: Record<string, unknown>;
    attack: Record<string, unknown>;
    defense: Record<string, unknown>;
    security: Record<string, unknown>;
  };
  global_results: {
    rounds: number[];
    global_loss: number[];
    global_accuracy: number[];
  };
  client_results: Record<string, RunClientMetricSeries>;
};

const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await getHttpErrorMessage(response));
  }
  return (await response.json()) as T;
}

export const runsApi = {
  async list(limit = 200): Promise<Run[]> {
    const response = await fetch(`${baseUrl}/api/v1/runs?limit=${encodeURIComponent(String(limit))}`);
    return readJson<Run[]>(response);
  },
  async getMetrics(runId: string): Promise<RunMetrics> {
    const response = await fetch(`${baseUrl}/api/v1/runs/${encodeURIComponent(runId)}/metrics`);
    return readJson<RunMetrics>(response);
  },
};
