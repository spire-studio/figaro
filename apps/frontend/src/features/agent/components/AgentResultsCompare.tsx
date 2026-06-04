import { useEffect, useMemo, useState } from "react";
import { History, Bot, Activity, Trophy, ArrowRight, Target, Calendar, FileJson, Settings2, Search, RotateCcw, GitCompare } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../../../components/ui/card";
import { Badge } from "../../../components/ui/badge";
import { Button } from "../../../components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../../components/ui/select";
import { Separator } from "../../../components/ui/separator";
import { MiniLineChart } from "../../simulation/components/MiniLineChart";
import { getValueByPath, isRecord, toClientSeries } from "../../simulation/utils";
import {
  formatBytes,
  isLlmRunMetrics,
  llmAdapterSizeSeries,
  llmPerplexitySeries,
  llmRounds,
  llmThroughputSeries,
  llmTrainLossSeries,
  llmValidationLossSeries,
  latestLlmLoss,
  selectedClientsText,
  shortHash,
} from "../../simulation/llm-metrics";
import { fmt } from "../../../lib/time";
import type { AgentPageProps } from "../../../pages/types";
import { baseUrl } from "../../../api/client";
import { agentApi, type AgentConfigChange, type AgentConfigVersion, type AgentHistoryFilters } from "../../../api/agent";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { collectAgentSchemaFields, formatFieldValue, optionDisabled, optionLabel, type AgentSchemaField } from "../schema";

const CHART_COLORS = [
  "hsl(var(--primary))", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"
];

const FALLBACK_CONFIG_FIELDS = [
  { path: "dataset.name", label: "Dataset" },
  { path: "dataset.distribution", label: "Distribution" },
  { path: "dataset.alpha", label: "Alpha" },
  { path: "model.name", label: "Model" },
  { path: "federated.aggregation", label: "Aggregation" },
  { path: "federated.num_clients", label: "Clients" },
  { path: "federated.num_rounds", label: "Rounds" },
  { path: "federated.local_epochs", label: "Local Epochs" },
  { path: "federated.learning_rate", label: "Learning Rate" },
];

type ConfigSummaryItem = {
  path: string;
  label: string;
  value: string;
};

const HISTORY_STATUS_OPTIONS = [
  { value: "all", label: "All status" },
  { value: "completed", label: "Completed" },
  { value: "running", label: "Running" },
  { value: "pending_review", label: "Pending review" },
  { value: "failed", label: "Failed" },
  { value: "queued", label: "Queued" },
];

const OBJECTIVE_OPTIONS = [
  { value: "all", label: "All objectives" },
  { value: "auto", label: "Auto" },
  { value: "accuracy", label: "Accuracy" },
];

type SelectOption = {
  value: string;
  label: string;
};

function asConfigRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function summarizeSchemaField(field: AgentSchemaField, config: Record<string, unknown>): ConfigSummaryItem | null {
  const value = getValueByPath(config, field.path);
  if (value === undefined) return null;
  return {
    path: field.path,
    label: field.label,
    value: field.type === "select" ? optionLabel(field.definition, value) : formatFieldValue(value),
  };
}

function summarizeFallbackField(
  field: { path: string; label: string },
  config: Record<string, unknown>,
): ConfigSummaryItem | null {
  const value = getValueByPath(config, field.path);
  if (value === undefined) return null;
  return {
    path: field.path,
    label: field.label,
    value: formatFieldValue(value),
  };
}

function buildConfigSummary(
  config: Record<string, unknown> | null,
  schema: Record<string, unknown> | null,
): ConfigSummaryItem[] {
  if (!config) return [];
  const schemaItems = collectAgentSchemaFields(schema, { featuredOnly: true })
    .map((field) => summarizeSchemaField(field, config))
    .filter((item): item is ConfigSummaryItem => item !== null);

  if (schemaItems.length > 0) {
    return schemaItems;
  }

  return FALLBACK_CONFIG_FIELDS
    .map((field) => summarizeFallbackField(field, config))
    .filter((item): item is ConfigSummaryItem => item !== null);
}

function formatDiffValue(value: unknown): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function experimentRankScore(experiment: any): number {
  const llmLoss = latestLlmLoss(experiment?.metrics);
  if (llmLoss !== null) return -llmLoss;
  const score = Number(experiment?.score);
  return Number.isFinite(score) ? score : Number.NEGATIVE_INFINITY;
}

function diffBadgeClass(changeType: string): string {
  if (changeType === "added" || changeType === "initialize") return "border-emerald-500/70 text-emerald-600";
  if (changeType === "removed") return "border-red-500/70 text-red-600";
  return "border-amber-500/70 text-amber-600";
}

function schemaSelectOptions(schema: Record<string, unknown> | null, path: string): SelectOption[] {
  const field = collectAgentSchemaFields(schema, { featuredOnly: false }).find((item) => item.path === path);
  if (!field) return [];
  return field.options
    .filter((option) => !optionDisabled(field.definition, option))
    .map((option) => ({ value: String(option), label: optionLabel(field.definition, option) }));
}

function optionSelect(
  label: string,
  value: string | undefined,
  options: SelectOption[],
  onChange: (value: string) => void,
) {
  return (
    <div className="space-y-1">
      <div className="text-[10px] font-semibold uppercase text-muted-foreground">{label}</div>
      <Select value={value || "all"} onValueChange={onChange}>
        <SelectTrigger className="h-8 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">Any</SelectItem>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

export function AgentResultsCompare(props: AgentPageProps) {
  const {
    historyFilters,
    historyJobs,
    selectedHistory,
    selectHistoryJob,
    setHistoryFilters,
    refreshHistoryJobs,
    setWorkflowStep,
    configSchema,
    modelOptions,
    notifyError,
  } = props;

  const jobs = historyJobs || [];
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [configVersions, setConfigVersions] = useState<AgentConfigVersion[]>([]);
  const [fromVersionKey, setFromVersionKey] = useState("");
  const [toVersionKey, setToVersionKey] = useState("");
  const [configDiff, setConfigDiff] = useState<AgentConfigChange[]>([]);

  const experiments = selectedHistory?.experiments || [];
  const bestExp = experiments.reduce((prev: any, curr: any) =>
    experimentRankScore(curr) > experimentRankScore(prev) ? curr : prev, experiments[0] || null);
  const bestConfig = asConfigRecord(selectedHistory?.best_config) ?? asConfigRecord(bestExp?.config);
  const bestConfigSummary = useMemo(
    () => buildConfigSummary(bestConfig, configSchema),
    [bestConfig, configSchema],
  );
  const bestConfigJson = useMemo(
    () => (bestConfig ? JSON.stringify(bestConfig, null, 2) : ""),
    [bestConfig],
  );
  const datasetOptions = useMemo(() => schemaSelectOptions(configSchema, "dataset.name"), [configSchema]);
  const configModelOptions = useMemo(() => schemaSelectOptions(configSchema, "model.name"), [configSchema]);
  const aggregationOptions = useMemo(() => schemaSelectOptions(configSchema, "federated.aggregation"), [configSchema]);
  const llmModelOptions = useMemo(
    () => modelOptions.map((item) => ({ value: item, label: item })),
    [modelOptions],
  );

  useEffect(() => {
    if (bestExp) {
      setSelectedRunId(bestExp.run_id);
    } else {
      setSelectedRunId(null);
      setMetrics(null);
    }
  }, [selectedHistory]);

  useEffect(() => {
    if (!selectedRunId) {
      setMetrics(null);
      return;
    }
    fetch(`${baseUrl}/api/v1/agent/runs/${selectedRunId}/metrics`)
      .then(res => res.json())
      .then(data => {
        if (data.metrics) setMetrics(data.metrics);
      })
      .catch(() => {});
  }, [selectedRunId]);

  useEffect(() => {
    const optimizationJobId = selectedHistory?.optimization_job_id;
    if (!optimizationJobId) {
      setConfigVersions([]);
      setFromVersionKey("");
      setToVersionKey("");
      setConfigDiff([]);
      return;
    }

    let cancelled = false;
    agentApi.listConfigVersions(optimizationJobId)
      .then((versions) => {
        if (cancelled) return;
        const experimentVersions = versions.filter((version) => version.source === "experiment");
        const firstVersion = experimentVersions[0];
        const lastVersion = experimentVersions[experimentVersions.length - 1];
        setConfigVersions(experimentVersions);
        setFromVersionKey(firstVersion ? String(firstVersion.id) : "");
        setToVersionKey(lastVersion ? String(lastVersion.id) : "");
        setConfigDiff([]);
      })
      .catch((error) => {
        if (!cancelled) {
          setConfigVersions([]);
          setFromVersionKey("");
          setToVersionKey("");
          setConfigDiff([]);
          notifyError(error, "agent-config-versions");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedHistory?.optimization_job_id]);

  useEffect(() => {
    const optimizationJobId = selectedHistory?.optimization_job_id;
    const fromVersionId = Number(fromVersionKey);
    const toVersionId = Number(toVersionKey);
    if (
      !optimizationJobId ||
      !fromVersionKey ||
      !toVersionKey ||
      !Number.isFinite(fromVersionId) ||
      !Number.isFinite(toVersionId) ||
      fromVersionId === toVersionId
    ) {
      setConfigDiff([]);
      return;
    }

    let cancelled = false;
    agentApi.getConfigDiff(
      optimizationJobId,
      toVersionId,
      fromVersionId,
    )
      .then((payload) => {
        if (!cancelled) {
          setConfigDiff(payload.changes);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setConfigDiff([]);
          notifyError(error, "agent-config-diff");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedHistory?.optimization_job_id, fromVersionKey, toVersionKey]);

  function updateHistoryFilter<K extends keyof AgentHistoryFilters>(key: K, value: AgentHistoryFilters[K]): void {
    setHistoryFilters((current) => ({ ...current, [key]: value }));
  }

  async function applyHistoryFilters(): Promise<void> {
    const { q: _unusedTextFilter, ...selectableFilters } = historyFilters;
    await refreshHistoryJobs(selectableFilters);
  }

  function resetHistoryFilters(): void {
    const next: AgentHistoryFilters = { status: "all", objective: "all" };
    setHistoryFilters(next);
    void refreshHistoryJobs(next).catch((error: unknown) => notifyError(error, "agent-history-filter-reset"));
  }

  const isLlmRun = isLlmRunMetrics(metrics);
  const globalResults = metrics?.global_results || {};
  const actualDataLength = isLlmRun
    ? (metrics?.llm_results?.rounds?.length || metrics?.llm_results?.train_loss?.length || 0)
    : (globalResults.global_accuracy?.length || 0);
  const rounds = isLlmRun
    ? llmRounds(metrics)
    : globalResults.rounds ? globalResults.rounds.slice(0, actualDataLength) : Array.from({ length: actualDataLength }, (_, i) => i + 1);
  const safeSlice = (arr: any[]) => (arr || []).slice(0, actualDataLength);

  const globalAccSeries = [{ key: "g_acc", label: "Global Accuracy", color: CHART_COLORS[0], values: safeSlice(globalResults.global_accuracy) }];
  const globalLossSeries = [{ key: "g_loss", label: "Global Loss", color: CHART_COLORS[3], values: safeSlice(globalResults.global_loss) }];
  const clientTrainAccSeries = toClientSeries(metrics, "train_acc", rounds);
  const clientTrainLossSeries = toClientSeries(metrics, "train_loss", rounds);
  const clientTestAccSeries = toClientSeries(metrics, "test_acc", rounds);
  const clientTestLossSeries = toClientSeries(metrics, "test_loss", rounds);

  return (
    <div className="grid h-full gap-4 xl:grid-cols-[360px_1fr]">
      
      <Card className="flex flex-col min-h-0 bg-muted/10 border-r shadow-none rounded-none sm:rounded-xl">
        <CardHeader className="pb-3 px-4">
          <CardTitle className="text-sm font-bold flex items-center gap-2">
            <History className="h-4 w-4 text-primary" /> Job History
          </CardTitle>
          <CardDescription className="text-xs">Past agent optimizations</CardDescription>
        </CardHeader>
        <CardContent className="flex-1 overflow-auto space-y-3 px-3 pb-4">
          <div className="space-y-3 rounded-lg border bg-background/70 p-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
              <Search className="h-4 w-4 text-muted-foreground" />
              Filter history
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Select value={historyFilters.status ?? "all"} onValueChange={(value) => updateHistoryFilter("status", value)}>
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {HISTORY_STATUS_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={historyFilters.objective ?? "all"} onValueChange={(value) => updateHistoryFilter("objective", value as AgentHistoryFilters["objective"])}>
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {OBJECTIVE_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {optionSelect("Dataset", historyFilters.dataset, datasetOptions, (value) => updateHistoryFilter("dataset", value))}
              {optionSelect("Model", historyFilters.config_model, configModelOptions, (value) => updateHistoryFilter("config_model", value))}
              {optionSelect("Aggregation", historyFilters.aggregation, aggregationOptions, (value) => updateHistoryFilter("aggregation", value))}
              {optionSelect("LLM", historyFilters.model_name, llmModelOptions, (value) => updateHistoryFilter("model_name", value))}
            </div>
            <div className="flex gap-2">
              <Button type="button" size="sm" className="h-8 flex-1 text-xs" onClick={() => void applyHistoryFilters().catch((error: unknown) => notifyError(error, "agent-history-filter"))}>
                <Search className="mr-2 h-3.5 w-3.5" /> Apply
              </Button>
              <Button type="button" variant="outline" size="sm" className="h-8 px-2" onClick={resetHistoryFilters} title="Reset filters">
                <RotateCcw className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>
          {jobs.length === 0 && <div className="text-xs text-muted-foreground text-center py-8">No history yet.</div>}
          
          {jobs.map((job: any) => {
            const isSelected = selectedHistory?.optimization_job_id === job.optimization_job_id;
            const isCompleted = job.status === "completed";
            
            return (
              <div 
                key={job.optimization_job_id} 
                onClick={() => selectHistoryJob(job.optimization_job_id)}
                className={`flex flex-col p-3 rounded-lg border text-xs cursor-pointer transition-all hover:shadow-sm
                  ${isSelected ? 'bg-primary/5 border-primary/40 ring-1 ring-primary/20' : 'bg-card border-border hover:border-primary/30'}`}
              >
                <div className="flex justify-between items-start mb-2">
                  <span className={`font-bold ${isSelected ? 'text-primary' : 'text-foreground'}`}>
                    {job.job_name || `Job #${job.optimization_job_id}`}
                  </span>
                  <Badge variant={isCompleted ? "outline" : "secondary"} className={`text-[9px] px-1.5 py-0 h-4 ${isCompleted ? 'border-emerald-500 text-emerald-600' : ''}`}>
                    {job.status}
                  </Badge>
                </div>
                <div className="flex items-start gap-1.5 text-muted-foreground mb-2">
                  <Target className="h-3.5 w-3.5 shrink-0 mt-0.5 opacity-70" />
                  <span className="line-clamp-2 leading-relaxed">{job.goal}</span>
                </div>
                <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground/70 mt-auto pt-2 border-t">
                  <Calendar className="h-3 w-3" />
                  <span>{fmt(job.created_at)}</span>
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      <div className="flex flex-col min-h-0 overflow-auto pr-2 gap-4 pb-6">
        {!selectedHistory ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground opacity-60">
            <History className="h-16 w-16 mb-4 opacity-20" />
            <p>Select a historical job from the sidebar to view its insights and charts.</p>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold tracking-tight">{selectedHistory.job_name || `Optimization Job #${selectedHistory.optimization_job_id}`}</h2>
                <p className="text-muted-foreground text-sm mt-1 flex items-center gap-2">
                  <Target className="h-4 w-4" /> {selectedHistory.goal}
                </p>
              </div>
              <Button onClick={() => setWorkflowStep("home")} size="sm">
                Start New Agent Job <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <Card className="border-emerald-200 bg-emerald-50/30 dark:border-emerald-900/40 dark:bg-emerald-950/20 shadow-sm">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-emerald-100 dark:bg-emerald-900/60 rounded-full">
                      <Trophy className="h-5 w-5 text-emerald-600" />
                    </div>
                    <p className="font-bold text-emerald-800 dark:text-emerald-300">Optimal Configuration</p>
                  </div>
                  <div className="text-3xl font-black text-emerald-700 dark:text-emerald-400 mb-2">
                    {isLlmRunMetrics(bestExp?.metrics)
                      ? latestLlmLoss(bestExp?.metrics) !== null
                        ? latestLlmLoss(bestExp?.metrics)?.toFixed(4)
                        : "N/A"
                      : bestExp?.score != null
                        ? `${(bestExp.score * 100).toFixed(2)}%`
                        : "N/A"}
                  </div>
                  <p className="text-xs text-emerald-700/70 dark:text-emerald-400/70">
                    Experiment <strong>"{bestExp?.name}"</strong> yielded the {isLlmRunMetrics(bestExp?.metrics) ? "lowest visible LLM loss" : "highest accuracy"}.
                  </p>
                  {bestConfig ? (
                    <>
                      <Separator className="my-4 bg-emerald-200/80 dark:bg-emerald-900/70" />
                      <div className="space-y-3">
                        <div className="flex items-center gap-2 text-xs font-bold uppercase text-emerald-800/80 dark:text-emerald-300/80">
                          <Settings2 className="h-3.5 w-3.5" />
                          Configuration
                        </div>
                        {bestConfigSummary.length > 0 && (
                          <div className="grid gap-2 sm:grid-cols-2">
                            {bestConfigSummary.map((item) => (
                              <div
                                key={item.path}
                                className="rounded-md border border-emerald-200/70 bg-background/75 px-3 py-2 dark:border-emerald-900/60 dark:bg-background/40"
                              >
                                <div className="text-[10px] font-semibold uppercase text-muted-foreground">
                                  {item.label}
                                </div>
                                <div className="mt-1 truncate text-xs font-semibold text-foreground">
                                  {item.value}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                        <div className="rounded-md border border-emerald-200/70 bg-background/80 dark:border-emerald-900/60 dark:bg-background/40">
                          <div className="flex items-center gap-2 border-b px-3 py-2 text-xs font-semibold text-emerald-800 dark:text-emerald-300">
                            <FileJson className="h-3.5 w-3.5" />
                            Full config
                          </div>
                          <pre className="max-h-64 overflow-auto p-3 text-[11px] leading-relaxed text-muted-foreground">{bestConfigJson}</pre>
                        </div>
                      </div>
                    </>
                  ) : (
                    <p className="mt-4 text-xs text-emerald-700/70 dark:text-emerald-400/70">
                      Configuration is not available for this historical result.
                    </p>
                  )}
                </CardContent>
              </Card>

              <Card className="shadow-sm border-primary/20 bg-primary/[0.02]">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2 text-primary font-bold">
                    <Bot className="h-4 w-4" /> Agent Summary Report
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-sm leading-relaxed text-foreground/90">
                  {selectedHistory.summary_text ? (
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      components={{
                        h3: ({node, ...props}) => <h3 className="text-md font-bold mt-4 mb-2 text-foreground" {...props} />,
                        p: ({node, ...props}) => <p className="mb-3 last:mb-0" {...props} />,
                        ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-3 space-y-1 marker:text-primary/70" {...props} />,
                        strong: ({node, ...props}) => <strong className="font-semibold text-foreground" {...props} />,
                        table: ({node, ...props}) => (
                          <div className="my-4 w-full overflow-y-auto rounded-md border">
                            <table className="w-full text-left border-collapse text-sm" {...props} />
                          </div>
                        ),
                        th: ({node, ...props}) => <th className="border-b bg-muted/50 px-4 py-2 font-semibold text-foreground" {...props} />,
                        td: ({node, ...props}) => <td className="border-b px-4 py-2 text-muted-foreground" {...props} />,
                      }}
                    >
                      {selectedHistory.summary_text}
                    </ReactMarkdown>
                  ) : (
                    <span className="italic text-muted-foreground">No final report was generated by the agent for this job.</span>
                  )}
                </CardContent>
              </Card>
            </div>

            <Card className="shadow-sm">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <GitCompare className="h-4 w-4 text-primary" /> Configuration Versions
                </CardTitle>
                <CardDescription className="text-xs">Compare captured experiment configs</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {configVersions.length === 0 ? (
                  <div className="rounded-md border border-dashed py-6 text-center text-xs text-muted-foreground">
                    No experiment configs for this job.
                  </div>
                ) : (
                  <>
                    <div className="grid gap-2 md:grid-cols-[1fr_1fr]">
                      <Select value={fromVersionKey} onValueChange={setFromVersionKey}>
                        <SelectTrigger className="h-9 text-xs">
                          <SelectValue placeholder="From experiment" />
                        </SelectTrigger>
                        <SelectContent>
                          {configVersions.map((version) => (
                            <SelectItem key={version.id} value={String(version.id)}>
                              {version.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Select value={toVersionKey} onValueChange={setToVersionKey}>
                        <SelectTrigger className="h-9 text-xs">
                          <SelectValue placeholder="To experiment" />
                        </SelectTrigger>
                        <SelectContent>
                          {configVersions.map((version) => (
                            <SelectItem key={version.id} value={String(version.id)}>
                              {version.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid gap-2 md:grid-cols-3">
                      {configVersions.slice(-3).map((version) => (
                        <div key={version.id} className="rounded-md border bg-muted/20 px-3 py-2">
                          <div className="flex items-center justify-between gap-2">
                            <span className="truncate text-xs font-semibold">{version.label}</span>
                            <Badge variant="outline" className="shrink-0 text-[10px]">{version.source}</Badge>
                          </div>
                          <div className="mt-1 text-[10px] text-muted-foreground">
                            Iteration {version.iteration || "-"} - {fmt(version.created_at)}
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="max-h-72 overflow-auto rounded-md border">
                      {configDiff.length === 0 ? (
                        <div className="p-4 text-center text-xs text-muted-foreground">No config changes.</div>
                      ) : (
                        <div className="divide-y">
                          {configDiff.map((change, index) => (
                            <div key={`${change.path}-${index}`} className="grid gap-2 p-3 text-xs md:grid-cols-[180px_90px_1fr]">
                              <div className="break-all font-semibold text-foreground">{change.path}</div>
                              <Badge variant="outline" className={`h-5 w-fit text-[10px] ${diffBadgeClass(change.change_type)}`}>
                                {change.change_type}
                              </Badge>
                              <div className="grid gap-1 text-muted-foreground sm:grid-cols-2">
                                <div className="min-w-0 rounded bg-muted/30 px-2 py-1">
                                  <span className="font-semibold text-foreground/70">Before </span>
                                  <span className="break-all">{formatDiffValue(change.old_value)}</span>
                                </div>
                                <div className="min-w-0 rounded bg-muted/30 px-2 py-1">
                                  <span className="font-semibold text-foreground/70">After </span>
                                  <span className="break-all">{formatDiffValue(change.new_value)}</span>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </CardContent>
            </Card>

            <div className="mt-4">
              <h3 className="text-sm font-bold tracking-tight flex items-center gap-2 mb-3">
                <Activity className="h-4 w-4 text-primary" /> 
                Explore Experiment Metrics
              </h3>
              <div className="flex flex-wrap gap-2">
                {experiments.map((exp: any, i: number) => {
                  const isBest = exp.run_id === bestExp?.run_id;
                  const isActive = selectedRunId === exp.run_id;
                  
                  return (
                    <Badge
                      key={exp.run_id}
                      variant={isActive ? "default" : "outline"}
                      className={`cursor-pointer px-3 py-1.5 text-xs transition-colors hover:bg-primary/80 hover:text-primary-foreground
                        ${isActive && isBest ? 'bg-emerald-600 hover:bg-emerald-700 text-white border-emerald-600' : ''}
                        ${!isActive && isBest ? 'border-emerald-500 text-emerald-600 bg-emerald-50 dark:bg-emerald-950' : ''}
                      `}
                      onClick={() => setSelectedRunId(exp.run_id)}
                    >
                      {exp.name || `Exp ${i + 1}`}
                      {isBest && <Trophy className="w-3 h-3 ml-1.5 inline-block" />}
                    </Badge>
                  )
                })}
              </div>
            </div>

            {selectedRunId ? (
              <>
                {!isLlmRun && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                    <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Global Accuracy" xValues={rounds} series={globalAccSeries} formatter={(v: number) => `${(v * 100).toFixed(2)}%`} /></CardContent></Card>
                    <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Global Loss" xValues={rounds} series={globalLossSeries} formatter={(v: number) => v.toFixed(4)} /></CardContent></Card>
                    <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Client Train Accuracy" xValues={rounds} series={clientTrainAccSeries} formatter={(v: number) => `${(v * 100).toFixed(2)}%`} /></CardContent></Card>
                    <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Client Test Accuracy" xValues={rounds} series={clientTestAccSeries} formatter={(v: number) => `${(v * 100).toFixed(2)}%`} /></CardContent></Card>
                    <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Client Train Loss" xValues={rounds} series={clientTrainLossSeries} formatter={(v: number) => v.toFixed(4)} /></CardContent></Card>
                    <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Client Test Loss" xValues={rounds} series={clientTestLossSeries} formatter={(v: number) => v.toFixed(4)} /></CardContent></Card>
                  </div>
                )}
                {isLlmRun && (
                  <div className="space-y-4 mt-2">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="LLM Train Loss" xValues={rounds} series={llmTrainLossSeries(metrics)} formatter={(v: number) => v.toFixed(4)} /></CardContent></Card>
                      <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="LLM Validation Loss" xValues={rounds} series={llmValidationLossSeries(metrics)} formatter={(v: number) => v.toFixed(4)} emptyMessage="No validation data available." /></CardContent></Card>
                      <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Perplexity" xValues={rounds} series={llmPerplexitySeries(metrics)} formatter={(v: number) => v.toFixed(2)} /></CardContent></Card>
                      <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Token Throughput" xValues={rounds} series={llmThroughputSeries(metrics)} formatter={(v: number) => `${v.toFixed(1)} tok/s`} /></CardContent></Card>
                      <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Adapter Size" xValues={rounds} series={llmAdapterSizeSeries(metrics)} formatter={(v: number) => formatBytes(v)} /></CardContent></Card>
                      <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Client Train Loss" xValues={rounds} series={clientTrainLossSeries} formatter={(v: number) => v.toFixed(4)} /></CardContent></Card>
                    </div>
                    <Card className="shadow-sm">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Adapter Lineage</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="overflow-auto rounded-md border">
                          <table className="w-full min-w-[680px] text-left text-xs">
                            <thead className="border-b bg-muted/40 text-muted-foreground">
                              <tr>
                                <th className="px-3 py-2 font-medium">Round</th>
                                <th className="px-3 py-2 font-medium">Clients</th>
                                <th className="px-3 py-2 font-medium">Size</th>
                                <th className="px-3 py-2 font-medium">SHA-256</th>
                                <th className="px-3 py-2 font-medium">Parent</th>
                              </tr>
                            </thead>
                            <tbody>
                              {(metrics?.llm_artifacts ?? []).map((artifact: Record<string, unknown>, index: number) => (
                                <tr key={`${artifact.path ?? index}`} className="border-b last:border-none">
                                  <td className="px-3 py-2 font-mono">{formatFieldValue(artifact.round)}</td>
                                  <td className="px-3 py-2 font-mono">{selectedClientsText(artifact.selected_clients)}</td>
                                  <td className="px-3 py-2 font-mono">{formatBytes(artifact.size_bytes)}</td>
                                  <td className="px-3 py-2 font-mono" title={String(artifact.sha256 ?? "")}>{shortHash(artifact.sha256)}</td>
                                  <td className="px-3 py-2 font-mono" title={String(artifact.parent_sha256 ?? "")}>{shortHash(artifact.parent_sha256)}</td>
                                </tr>
                              ))}
                              {(metrics?.llm_artifacts ?? []).length === 0 && (
                                <tr>
                                  <td colSpan={5} className="px-3 py-6 text-center text-muted-foreground">No adapter artifacts recorded yet.</td>
                                </tr>
                              )}
                            </tbody>
                          </table>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </>
            ) : (
               <div className="text-center py-10 text-sm text-muted-foreground border border-dashed rounded-lg">
                 Select an experiment above to load its charts.
               </div>
            )}
          </>
        )}
      </div>
      
    </div>
  );
}
