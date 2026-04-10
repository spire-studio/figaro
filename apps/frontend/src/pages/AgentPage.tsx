import { useState } from "react";
import { BarChart3, Bot, ChevronLeft, FlaskConical, Loader2, Play, RotateCcw, Sparkles } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { AgentOptimizationObjective } from "../api/agent";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Textarea } from "../components/ui/textarea";
import type { AgentPageProps } from "./types";

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function getLastAccuracy(metrics: Record<string, unknown> | null): number | null {
  const globalResults = asRecord(metrics?.global_results);
  const accuracy = globalResults?.global_accuracy;
  if (!Array.isArray(accuracy)) return null;
  const nums = accuracy.filter((item): item is number => typeof item === "number");
  return nums.length > 0 ? nums[nums.length - 1] : null;
}

function getRounds(metrics: Record<string, unknown> | null): number {
  const globalResults = asRecord(metrics?.global_results);
  const rounds = globalResults?.rounds;
  return Array.isArray(rounds) ? rounds.length : 0;
}

function fmt(value: number | null): string {
  return value === null ? "-" : value.toFixed(4);
}

function getConfigLabel(config: Record<string, unknown> | null): string {
  if (!config) return "-";
  const dataset = asRecord(config.dataset);
  const federated = asRecord(config.federated);
  const parts: string[] = [];
  if (dataset) {
    const name = dataset.name ?? "";
    const dist = dataset.distribution ?? "";
    const alpha = dataset.alpha;
    parts.push(`${name} ${dist}${alpha !== undefined ? ` α=${alpha}` : ""}`);
  }
  if (federated) {
    parts.push(`${federated.num_clients ?? "?"}c/${federated.num_rounds ?? "?"}r`);
  }
  return parts.join(" · ") || "-";
}

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

export function AgentPage(props: AgentPageProps) {
  const {
    busy,
    clearResult,
    defaultModelName,
    experiments,
    experimentRuns,
    goal,
    handleOptimize,
    historyJobs,
    lastSubmittedGoal,
    modelName,
    modelOptions,
    presets,
    progress,
    result,
    selectedExperimentId,
    selectedHistory,
    selectedHistoryJobId,
    selectExperiment,
    selectHistoryJob,
    setGoal,
    setModelName,
  } = props;

  const [topTab, setTopTab] = useState<"experiment" | "runs">("experiment");
  const [selectedRunJobId, setSelectedRunJobId] = useState<number | null>(null);

  // Current experiment data
  const summaryData = busy ? (progress ?? selectedHistory) : (selectedHistory ?? progress);
  const currentExperiments = [...(summaryData?.experiments ?? result?.experiments ?? [])];
  const summaryText = summaryData?.summary_text ?? result?.summary_text ?? null;
  const completedCount = summaryData?.completed_iterations ?? result?.iterations_executed ?? 0;
  const rawTotal = summaryData?.max_iterations ?? result?.max_iterations ?? 0;
  const totalCount = Math.max(rawTotal, completedCount, currentExperiments.length);
  const showProgress = busy;
  const defaultLabel = defaultModelName ? `Use backend default (${defaultModelName})` : "Use backend default";
  const hasResults = currentExperiments.length > 0 || result;

  // Selected run from history (for Runs tab detail view)
  const selectedRunJob = historyJobs.find((j) => j.optimization_job_id === selectedRunJobId) ?? null;
  const selectedRunDetail = selectedRunJobId !== null && selectedHistoryJobId === selectedRunJobId ? selectedHistory : null;
  const selectedRunExperiments = [...(selectedRunDetail?.experiments ?? [])];
  const selectedRunSummary = selectedRunDetail?.summary_text ?? null;

  return (
    <div className="space-y-0">
      {/* ── Top tab bar ── */}
      <div className="flex items-center justify-end rounded-t-lg border border-b-0 border-slate-200/80 bg-slate-50/70 px-4 py-3 dark:border-slate-700/60 dark:bg-slate-900/40">
        <Tabs value={topTab} onValueChange={(v) => setTopTab(v as "experiment" | "runs")}>
          <TabsList>
            <TabsTrigger value="experiment">Experiment</TabsTrigger>
            <TabsTrigger value="runs">Runs</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      <div className="rounded-b-lg border bg-card/30 p-4">
        {/* ════════════════════════════════════════════ */}
        {/* EXPERIMENT TAB: Input + Current Results      */}
        {/* ════════════════════════════════════════════ */}
        {topTab === "experiment" && (
          <div className="grid gap-4 xl:grid-cols-[380px_minmax(0,1fr)]">
            {/* Left: Controls */}
            <div className="space-y-4">
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center gap-2">
                    <Bot className="h-5 w-5" />
                    <CardTitle>Intelligent Experiment Runner</CardTitle>
                  </div>
                  <CardDescription>
                    Describe your FL experiments in natural language. AI parses, runs, and compares results automatically.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Textarea
                    value={goal}
                    onChange={(event) => setGoal(event.target.value)}
                    className="min-h-[120px]"
                    placeholder="e.g. Compare CIFAR-10 non-IID alpha=0.1, 0.3, 0.5"
                  />
                  <div className="flex flex-wrap gap-1.5">
                    {presets.map((preset) => (
                      <Button key={preset} variant="outline" size="sm" className="h-auto w-full justify-start py-1.5 text-xs" title={preset} onClick={() => setGoal(preset)}>
                        <Sparkles className="mr-1 h-3 w-3 shrink-0" />
                        <span className="text-left">{preset}</span>
                      </Button>
                    ))}
                  </div>
                  <div className="space-y-2">
                    <p className="text-xs text-muted-foreground">LLM Model</p>
                    <Select value={modelName || "__default__"} onValueChange={(value) => setModelName(value === "__default__" ? "" : value)}>
                      <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="Use backend default" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__default__">{defaultLabel}</SelectItem>
                        {modelOptions.map((m) => <SelectItem key={m} value={m}>{m}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex gap-2 pt-1">
                    <Button onClick={() => handleOptimize().catch(() => undefined)} disabled={busy || goal.trim().length === 0} className="flex-1" size="sm">
                      {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                      Run Experiments
                    </Button>
                    <Button variant="outline" size="sm" onClick={clearResult} disabled={busy || (!result && !progress)}>
                      <RotateCcw className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Progress */}
              {showProgress && (
                <Card>
                  <CardContent className="py-3">
                    <div className="flex items-center gap-3">
                      <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                      <div className="flex-1">
                        <div className="mb-1 flex justify-between text-xs">
                          <span className="truncate">{summaryData?.current_phase ?? "Running..."}</span>
                          <span className="text-muted-foreground">{completedCount}/{totalCount}</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-muted">
                          <div className="h-1.5 rounded-full bg-primary transition-all" style={{ width: `${totalCount > 0 ? (completedCount / totalCount) * 100 : 0}%` }} />
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Right: Current Results */}
            <div className="space-y-4">
              {!busy && !hasResults && (
                <Card className="flex min-h-[400px] items-center justify-center">
                  <CardContent className="text-center">
                    <FlaskConical className="mx-auto h-10 w-10 text-muted-foreground/30" />
                    <p className="mt-3 text-sm text-muted-foreground">Describe your experiment and click Run Experiments to start</p>
                  </CardContent>
                </Card>
              )}

              {currentExperiments.length > 0 && (
                <Card>
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4" />
                      <CardTitle className="text-sm">Results</CardTitle>
                    </div>
                    {lastSubmittedGoal && <CardDescription className="line-clamp-1 text-xs">{lastSubmittedGoal}</CardDescription>}
                  </CardHeader>
                  <CardContent>
                    <ResultsTable experiments={currentExperiments} />
                  </CardContent>
                </Card>
              )}

              {summaryText && (
                <Card>
                  <CardHeader className="pb-2"><CardTitle className="text-sm">Key Findings</CardTitle></CardHeader>
                  <CardContent>
                    <div className="markdown-report max-h-[450px] overflow-y-auto rounded-lg border bg-muted/20 p-4">
                      <Markdown remarkPlugins={[remarkGfm]}>{summaryText}</Markdown>
                    </div>
                  </CardContent>
                </Card>
              )}

              {currentExperiments.length > 0 && (
                <Card>
                  <CardContent className="py-3">
                    <details className="rounded-lg">
                      <summary className="cursor-pointer text-xs font-medium">View detailed JSON</summary>
                      <pre className="mt-3 max-h-[300px] overflow-auto rounded-lg bg-muted/30 p-3 text-[11px]">
                        {JSON.stringify(currentExperiments.map((e) => ({ name: e.iteration_goal, config: e.config, metrics: e.metrics, score: e.score })), null, 2)}
                      </pre>
                    </details>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        )}

        {/* ════════════════════════════════════════════ */}
        {/* RUNS TAB: History of all runs                */}
        {/* ════════════════════════════════════════════ */}
        {topTab === "runs" && (
          <div className="grid gap-4 xl:grid-cols-[380px_minmax(0,1fr)]">
            {/* Left: Runs list */}
            <div className="space-y-1.5">
              <p className="mb-2 text-xs font-medium text-muted-foreground">All Runs (newest first)</p>
              {historyJobs.length === 0 && (
                <p className="py-8 text-center text-xs text-muted-foreground">No runs yet. Go to Experiment tab to start.</p>
              )}
              {historyJobs.map((job) => (
                <button
                  key={job.optimization_job_id}
                  type="button"
                  className={`flex w-full items-start justify-between rounded-lg border p-3 text-left transition-colors ${
                    selectedRunJobId === job.optimization_job_id ? "bg-muted" : "hover:bg-muted/60"
                  }`}
                  onClick={() => {
                    setSelectedRunJobId(job.optimization_job_id);
                    selectHistoryJob(job.optimization_job_id).catch(() => undefined);
                  }}
                >
                  <div className="min-w-0">
                    <p className="truncate text-xs font-semibold">{job.job_name ?? `Run #${job.optimization_job_id}`}</p>
                    <p className="mt-0.5 line-clamp-2 text-[11px] text-muted-foreground">{job.goal}</p>
                    <p className="mt-0.5 text-[10px] text-muted-foreground">
                      {job.completed_iterations} experiments{job.best_score !== null ? ` · best=${job.best_score.toFixed(4)}` : ""}
                    </p>
                  </div>
                  <Badge variant="outline" className="ml-2 shrink-0 text-[10px]">{job.status}</Badge>
                </button>
              ))}
            </div>

            {/* Right: Selected run detail */}
            <div className="space-y-4">
              {selectedRunJobId === null && (
                <Card className="flex min-h-[300px] items-center justify-center">
                  <CardContent className="text-center">
                    <p className="text-sm text-muted-foreground">Select a run from the list to view details</p>
                  </CardContent>
                </Card>
              )}

              {selectedRunJob && (
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">{selectedRunJob.job_name ?? `Run #${selectedRunJob.optimization_job_id}`}</CardTitle>
                    <CardDescription className="text-xs">{selectedRunJob.goal}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex flex-wrap gap-2 text-xs">
                      <Badge variant="outline">{selectedRunJob.status}</Badge>
                      <Badge variant="outline">{selectedRunJob.completed_iterations} experiments</Badge>
                      {selectedRunJob.best_score !== null && <Badge variant="success">best={selectedRunJob.best_score.toFixed(4)}</Badge>}
                    </div>

                    {/* Sub-experiments table */}
                    {selectedRunExperiments.length > 0 && (
                      <ResultsTable experiments={selectedRunExperiments} />
                    )}

                    {/* Summary */}
                    {selectedRunSummary && (
                      <div className="markdown-report max-h-[350px] overflow-y-auto rounded-lg border bg-muted/20 p-4">
                        <Markdown remarkPlugins={[remarkGfm]}>{selectedRunSummary}</Markdown>
                      </div>
                    )}

                    {selectedRunExperiments.length === 0 && !selectedRunSummary && (
                      <p className="py-4 text-center text-xs text-muted-foreground">
                        {selectedRunJob.status === "running" ? "Experiments are still running..." : "No experiment data available"}
                      </p>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Shared Results Table                                               */
/* ------------------------------------------------------------------ */

function ResultsTable({ experiments }: { experiments: Array<{ iteration_goal?: string | null; plan_summary?: string | null; config?: Record<string, unknown> | null; metrics?: unknown; score?: number | null; decision?: string | null; run_id?: string }> }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            <th className="pb-2 pr-3">#</th>
            <th className="pb-2 pr-3">Experiment</th>
            <th className="pb-2 pr-3">Config</th>
            <th className="pb-2 pr-3 text-right">Accuracy</th>
            <th className="pb-2 pr-3 text-right">Rounds</th>
            <th className="pb-2 text-right">Status</th>
          </tr>
        </thead>
        <tbody>
          {experiments.map((exp, idx) => {
            const metrics = exp.metrics as Record<string, unknown> | null;
            const config = exp.config ?? null;
            const acc = getLastAccuracy(metrics);
            const rounds = getRounds(metrics);
            const isRunning = exp.decision === null || exp.decision === "pending";
            const allAccs = experiments.map((e) => getLastAccuracy(e.metrics as Record<string, unknown> | null)).filter((v): v is number => v !== null);
            const bestAcc = allAccs.length > 0 ? Math.max(...allAccs) : null;
            const isBest = acc !== null && bestAcc !== null && acc === bestAcc && experiments.length > 1;

            return (
              <tr key={exp.run_id ?? idx} className="border-b last:border-0">
                <td className="py-2 pr-3 text-xs text-muted-foreground">{idx + 1}</td>
                <td className="py-2 pr-3 text-xs font-medium">{exp.iteration_goal?.replace("Run experiment: ", "") ?? `Exp ${idx + 1}`}</td>
                <td className="py-2 pr-3 text-[11px] text-muted-foreground">{getConfigLabel(config)}</td>
                <td className="py-2 pr-3 text-right font-mono text-xs">
                  {isBest ? <span className="font-semibold text-emerald-600">{fmt(acc)}</span> : fmt(acc)}
                </td>
                <td className="py-2 pr-3 text-right text-xs">{rounds || "-"}</td>
                <td className="py-2 text-right">
                  {isRunning ? <Badge variant="warning" className="text-[10px]">running</Badge>
                    : isBest ? <Badge variant="success" className="text-[10px]">best</Badge>
                    : <Badge variant="outline" className="text-[10px]">done</Badge>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
