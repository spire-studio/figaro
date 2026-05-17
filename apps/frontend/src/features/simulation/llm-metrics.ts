import type { LineSeries } from "../../pages/types";
import type { RunMetrics } from "../../api/runs";

const LLM_COLORS = {
  trainLoss: "#dc2626",
  validationLoss: "#f59e0b",
  perplexity: "#7c3aed",
  throughput: "#0891b2",
  adapterSize: "#16a34a",
};

function numericList(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => Number(item)).filter((item) => Number.isFinite(item));
}

function recordOf(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

export function isLlmRunMetrics(metrics: RunMetrics | null | undefined): boolean {
  if (!metrics) return false;
  const basic = recordOf(metrics.experiment_info?.basic);
  const llmResults = recordOf(metrics.llm_results);
  const llmEvaluation = recordOf(metrics.llm_evaluation);
  const llmRuntime = recordOf(metrics.llm_runtime);
  return (
    basic.task_type === "llm_peft_sft" ||
    numericList(llmResults.rounds).length > 0 ||
    Object.keys(llmEvaluation).length > 0 ||
    Object.keys(llmRuntime).length > 0 ||
    (metrics.llm_artifacts?.length ?? 0) > 0
  );
}

export function llmRounds(metrics: RunMetrics | null | undefined): number[] {
  if (!metrics) return [];
  const explicitRounds = numericList(metrics.llm_results?.rounds);
  if (explicitRounds.length > 0) return explicitRounds;
  return numericList(metrics.global_results?.rounds);
}

export function llmMetricSeries(
  metrics: RunMetrics | null | undefined,
  key: keyof NonNullable<RunMetrics["llm_results"]>,
  label: string,
  color: string,
): LineSeries[] {
  if (key === "rounds") return [];
  return [
    {
      key: `llm-${String(key)}`,
      label,
      color,
      values: numericList(metrics?.llm_results?.[key]),
    },
  ];
}

export function llmTrainLossSeries(metrics: RunMetrics | null | undefined): LineSeries[] {
  return llmMetricSeries(metrics, "train_loss", "Train Loss", LLM_COLORS.trainLoss);
}

export function llmValidationLossSeries(metrics: RunMetrics | null | undefined): LineSeries[] {
  return llmMetricSeries(metrics, "validation_loss", "Validation Loss", LLM_COLORS.validationLoss);
}

export function llmPerplexitySeries(metrics: RunMetrics | null | undefined): LineSeries[] {
  return llmMetricSeries(metrics, "perplexity", "Perplexity", LLM_COLORS.perplexity);
}

export function llmThroughputSeries(metrics: RunMetrics | null | undefined): LineSeries[] {
  return llmMetricSeries(metrics, "token_throughput", "Tokens/s", LLM_COLORS.throughput);
}

export function llmAdapterSizeSeries(metrics: RunMetrics | null | undefined): LineSeries[] {
  return llmMetricSeries(metrics, "adapter_size_bytes", "Adapter Size", LLM_COLORS.adapterSize);
}

export function formatBytes(value: unknown): string {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "-";
  if (bytes < 1024) return `${bytes.toFixed(0)} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let scaled = bytes / 1024;
  let unitIndex = 0;
  while (scaled >= 1024 && unitIndex < units.length - 1) {
    scaled /= 1024;
    unitIndex += 1;
  }
  return `${scaled.toFixed(scaled >= 10 ? 1 : 2)} ${units[unitIndex]}`;
}

export function shortHash(value: unknown): string {
  if (typeof value !== "string" || value.length === 0) return "-";
  return value.length <= 12 ? value : `${value.slice(0, 12)}...`;
}

export function selectedClientsText(value: unknown): string {
  if (!Array.isArray(value)) return "-";
  return value.map((item) => String(item)).join(", ");
}
