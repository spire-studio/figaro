import { BarChart3, FileText, GitBranch, Play, RefreshCcw, Square, TerminalSquare, Trash2 } from "lucide-react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "../../components/ui/alert-dialog";
import { Badge } from "../../components/ui/badge";
import { Button } from "../../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card";
import { Separator } from "../../components/ui/separator";
import {
  formatBytes,
  isLlmRunMetrics,
  llmAdapterSizeSeries,
  llmPerplexitySeries,
  llmRounds,
  llmThroughputSeries,
  llmTrainLossSeries,
  llmValidationLossSeries,
  selectedClientsText,
  shortHash,
} from "../../features/simulation/llm-metrics";
import { fmt } from "../../lib/time";
import type { Run, RunLog, SimulationPageProps } from "../types";

export function SimulationRunsTab(props: SimulationPageProps) {
  const {
    MiniLineChart,
    busy,
    clientTestAccSeries,
    clientTestLossSeries,
    clientTrainAccSeries,
    clientTrainLossSeries,
    compressionInfo,
    encryptionInfo,
    experimentBasic,
    experimentFederated,
    globalAccuracySeries,
    globalLossSeries,
    handleDeleteRun,
    handleRerun,
    handleStopRun,
    isRecord,
    loadRunBundle,
    loadRunMetrics,
    notifyError,
    runLogs,
    runLogsRef,
    runMetrics,
    runRounds,
    runStatusVariant,
    runs,
    selectedRun,
    selectedRunId,
    setSelectedRunId,
    shortRunId,
    toDisplayText,
    toNumber,
  } = props;
  const isLlmRun = isLlmRunMetrics(runMetrics);
  const llmMetricRounds = llmRounds(runMetrics);
  const llmDataset = runMetrics.llm_dataset ?? {};
  const llmEvaluation = runMetrics.llm_evaluation ?? {};
  const llmRuntime = runMetrics.llm_runtime ?? {};
  const llmArtifacts = runMetrics.llm_artifacts ?? [];

  return (
    <div className="grid min-h-[calc(100vh-180px)] gap-4 xl:grid-cols-[330px_minmax(0,1fr)]">
      <Card className="min-h-0">
        <CardHeader>
          <CardTitle>Runs</CardTitle>
          <CardDescription>All runs across all statuses.</CardDescription>
        </CardHeader>
        <CardContent className="flex h-[calc(100%-96px)] min-h-0 flex-col gap-3">
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground">Run List</p>
            <Button variant="ghost" size="sm" onClick={() => props.refreshRuns().catch((err: unknown) => notifyError(err))}>
              <RefreshCcw className="h-3.5 w-3.5" />
            </Button>
          </div>

          <div className="min-h-0 flex-1 space-y-2 overflow-auto">
            {runs.map((run: Run) => (
              <button
                key={run.id}
                className={`w-full rounded-md border px-3 py-2 text-left transition ${
                  selectedRunId === run.id ? "bg-muted" : "hover:bg-muted/60"
                }`}
                onClick={() => setSelectedRunId(run.id)}
              >
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="truncate text-xs font-semibold" title={`${run.job_name ?? `Job #${run.job_id}`} #${run.id}`}>
                    {(run.job_name ?? `Job #${run.job_id}`)} #{shortRunId(run.id)}
                  </span>
                  <Badge variant={runStatusVariant[run.status] ?? "secondary"}>{run.status}</Badge>
                </div>
                <div className="text-xs text-muted-foreground">{fmt(run.created_at)}</div>
              </button>
            ))}
            {runs.length === 0 && <p className="text-xs text-muted-foreground">No runs yet.</p>}
          </div>
        </CardContent>
      </Card>

      <Card className="min-h-0">
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="space-y-1">
            <CardTitle>Selected Run</CardTitle>
            <CardDescription>Details, logs, and artifacts for the selected run.</CardDescription>
          </div>
          {selectedRun && (
            <div className="flex w-full flex-wrap gap-2 sm:w-auto sm:justify-end">
              <Button
                className="bg-amber-500 text-white hover:bg-amber-400"
                size="sm"
                onClick={() => handleRerun().catch((err: unknown) => notifyError(err))}
                disabled={busy}
              >
                <Play className="mr-2 h-4 w-4" />
                Rerun
              </Button>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button
                    variant="destructive"
                    size="sm"
                    disabled={busy || selectedRun.status === "running" || selectedRun.status === "queued"}
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    Delete Run
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Delete this run?</AlertDialogTitle>
                    <AlertDialogDescription>This removes run details, logs, and stored run artifacts.</AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      disabled={busy || selectedRun.status === "running" || selectedRun.status === "queued"}
                      onClick={() => {
                        void handleDeleteRun();
                      }}
                    >
                      Delete Run
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="destructive" size="sm" disabled={busy || selectedRun.status !== "running"}>
                    <Square className="mr-2 h-4 w-4" />
                    Stop Run
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Stop this running job?</AlertDialogTitle>
                    <AlertDialogDescription>The run will be terminated and marked as cancelled.</AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Keep Running</AlertDialogCancel>
                    <AlertDialogAction
                      disabled={busy || selectedRun.status !== "running"}
                      onClick={() => {
                        void handleStopRun();
                      }}
                    >
                      Stop Run
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  Promise.all([loadRunBundle(selectedRun.id), loadRunMetrics(selectedRun.id)]).catch((err: unknown) =>
                    notifyError(err),
                  )
                }
              >
                <RefreshCcw className="mr-2 h-4 w-4" />
                Refresh Details
              </Button>
            </div>
          )}
        </CardHeader>
        <CardContent className="space-y-4">
          {!selectedRun && <p className="text-sm text-muted-foreground">Select a run from the left list.</p>}

          {selectedRun && (
            <>
              <div className="grid gap-3 rounded-lg border p-3 md:grid-cols-2 xl:grid-cols-3">
                <div>
                  <p className="text-xs text-muted-foreground">Run ID</p>
                  <p className="truncate font-mono text-sm" title={selectedRun.id}>
                    {shortRunId(selectedRun.id)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Status</p>
                  <Badge variant={runStatusVariant[selectedRun.status] ?? "secondary"}>{selectedRun.status}</Badge>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Job ID</p>
                  <p className="text-sm">{selectedRun.job_id}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Job Name</p>
                  <p className="truncate text-sm">{selectedRun.job_name ?? "-"}</p>
                </div>
                <div className="md:col-span-2 xl:col-span-3">
                  <p className="text-xs text-muted-foreground">Job Description</p>
                  <p className="line-clamp-2 text-sm">{selectedRun.job_description ?? "No description"}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Started</p>
                  <p className="text-sm">{fmt(selectedRun.started_at)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Ended</p>
                  <p className="text-sm">{fmt(selectedRun.ended_at)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Exit Code</p>
                  <p className="text-sm">{selectedRun.exit_code ?? "-"}</p>
                </div>
              </div>

              <div className="space-y-2">
                <p className="flex items-center gap-2 text-xs text-muted-foreground">
                  <FileText className="h-4 w-4" />
                  Experiment Context
                </p>
                <div className="grid gap-3 xl:grid-cols-2">
                  {!isLlmRun && (
                    <Card className="border-blue-200/60 bg-blue-50/40 dark:border-blue-900/40 dark:bg-blue-950/20">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Basic Context</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div>
                          <p className="text-xs text-muted-foreground">Dataset</p>
                          <p>{toDisplayText(experimentBasic.dataset_name)}</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Distribution</p>
                          <p>
                            {toDisplayText(experimentBasic.distribution)}
                            {toNumber(experimentBasic.alpha) !== null ? ` (alpha=${toNumber(experimentBasic.alpha)})` : ""}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Model</p>
                          <p>{toDisplayText(experimentBasic.model_name)}</p>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {isLlmRun && (
                    <Card className="border-blue-200/60 bg-blue-50/40 dark:border-blue-900/40 dark:bg-blue-950/20">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">LLM Context</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div>
                          <p className="text-xs text-muted-foreground">Base Model</p>
                          <p className="break-all">{toDisplayText(experimentBasic.base_model)}</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">SFT / PEFT</p>
                          <p>
                            {toDisplayText(experimentBasic.sft_format)} / {toDisplayText(experimentBasic.peft_method)}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Runtime</p>
                          <p>{toDisplayText(llmRuntime.status)}</p>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  <Card className="border-indigo-200/60 bg-indigo-50/40 dark:border-indigo-900/40 dark:bg-indigo-950/20">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm">Federated Config</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <div>
                        <p className="text-xs text-muted-foreground">Rounds / Clients</p>
                        <p>
                          {toDisplayText(experimentFederated.num_rounds)} rounds / {toDisplayText(experimentFederated.num_clients)} clients
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Per Round</p>
                        <p>{toDisplayText(experimentFederated.clients_per_round)} clients selected</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Local Training</p>
                        <p>
                          epochs={toDisplayText(experimentFederated.local_epochs)}, lr={toDisplayText(experimentFederated.learning_rate)}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Aggregation</p>
                        <p>{toDisplayText(experimentFederated.aggregation)}</p>
                      </div>
                    </CardContent>
                  </Card>

                  {!isLlmRun && (
                    <Card className="border-emerald-200/60 bg-emerald-50/40 dark:border-emerald-900/40 dark:bg-emerald-950/20">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm">Security</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3 text-sm">
                      <div className="grid grid-cols-2 gap-2">
                        <div className="rounded-md border bg-background/70 p-2">
                          <p className="text-[11px] text-muted-foreground">Encryption</p>
                          <p className="text-sm">{Boolean(encryptionInfo.enabled) ? "Enabled" : "Disabled"}</p>
                          <p className="text-xs text-muted-foreground">{toDisplayText(encryptionInfo.type, "-")}</p>
                        </div>
                        <div className="rounded-md border bg-background/70 p-2">
                          <p className="text-[11px] text-muted-foreground">Compression</p>
                          <p className="text-sm">{Boolean(compressionInfo.enabled) ? "Enabled" : "Disabled"}</p>
                          <p className="text-xs text-muted-foreground">{toDisplayText(compressionInfo.type, "-")}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  )}

                  {isLlmRun && (
                    <Card className="border-cyan-200/60 bg-cyan-50/40 dark:border-cyan-900/40 dark:bg-cyan-950/20">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">SFT Dataset</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div>
                          <p className="text-xs text-muted-foreground">Records</p>
                          <p>{toDisplayText(llmDataset.num_records)}</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Client Split</p>
                          <p className="break-all">
                            {Array.isArray(llmDataset.client_record_counts) ? llmDataset.client_record_counts.join(", ") : "-"}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Prompt Template</p>
                          <p>{toDisplayText(llmDataset.prompt_template)}</p>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {isLlmRun && (
                    <Card className="border-amber-200/60 bg-amber-50/40 dark:border-amber-900/40 dark:bg-amber-950/20">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">Evaluation</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div>
                          <p className="text-xs text-muted-foreground">Status</p>
                          <p>{Boolean(llmEvaluation.enabled) ? "Enabled" : "Disabled"}</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Validation Records</p>
                          <p>{toDisplayText(llmEvaluation.num_records)}</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Last Validation Loss</p>
                          <p>{toDisplayText(llmEvaluation.last_validation_loss)}</p>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </div>

              <Separator />

              <div className="space-y-2">
                <p className="flex items-center gap-2 text-xs text-muted-foreground">
                  <TerminalSquare className="h-4 w-4" />
                  Logs
                </p>
                <div ref={runLogsRef} className="max-h-[280px] overflow-auto rounded-md border bg-muted/20 p-2 font-mono text-[11px]">
                  {runLogs.map((log: RunLog) => (
                    <div key={log.id} className="grid grid-cols-[136px_56px_1fr] gap-2 border-b py-1 last:border-none">
                      <span className="text-muted-foreground">{fmt(log.created_at)}</span>
                      <span className={log.level === "ERROR" ? "text-red-500 dark:text-red-300" : "text-cyan-500 dark:text-cyan-300"}>
                        {log.level}
                      </span>
                      <span className="break-all">{log.message}</span>
                    </div>
                  ))}
                  {runLogs.length === 0 && <p className="text-muted-foreground">No logs yet.</p>}
                </div>
              </div>

              <div className="space-y-2">
                <p className="flex items-center gap-2 text-xs text-muted-foreground">
                  <BarChart3 className="h-4 w-4" />
                  {isLlmRun ? "LLM Training Curves" : "Training Curves"}
                </p>
                {!isLlmRun && (
                  <div className="grid gap-3 xl:grid-cols-2">
                    <MiniLineChart
                      title="Global Accuracy"
                      xValues={runRounds}
                      series={globalAccuracySeries}
                      formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
                    />
                    <MiniLineChart
                      title="Global Loss"
                      xValues={runRounds}
                      series={globalLossSeries}
                      formatter={(value: number) => value.toFixed(4)}
                    />
                    <MiniLineChart
                      title="Client Train Acc"
                      xValues={runRounds}
                      series={clientTrainAccSeries}
                      formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
                    />
                    <MiniLineChart
                      title="Client Train Loss"
                      xValues={runRounds}
                      series={clientTrainLossSeries}
                      formatter={(value: number) => value.toFixed(4)}
                    />
                    <MiniLineChart
                      title="Client Test Acc"
                      xValues={runRounds}
                      series={clientTestAccSeries}
                      formatter={(value: number) => `${(value * 100).toFixed(2)}%`}
                    />
                    <MiniLineChart
                      title="Client Test Loss"
                      xValues={runRounds}
                      series={clientTestLossSeries}
                      formatter={(value: number) => value.toFixed(4)}
                    />
                  </div>
                )}
                {isLlmRun && (
                  <div className="grid gap-3 xl:grid-cols-2">
                    <MiniLineChart
                      title="LLM Train Loss"
                      xValues={llmMetricRounds}
                      series={llmTrainLossSeries(runMetrics)}
                      formatter={(value: number) => value.toFixed(4)}
                    />
                    <MiniLineChart
                      title="LLM Validation Loss"
                      xValues={llmMetricRounds}
                      series={llmValidationLossSeries(runMetrics)}
                      formatter={(value: number) => value.toFixed(4)}
                      emptyMessage="No validation data available."
                    />
                    <MiniLineChart
                      title="Perplexity"
                      xValues={llmMetricRounds}
                      series={llmPerplexitySeries(runMetrics)}
                      formatter={(value: number) => value.toFixed(2)}
                    />
                    <MiniLineChart
                      title="Token Throughput"
                      xValues={llmMetricRounds}
                      series={llmThroughputSeries(runMetrics)}
                      formatter={(value: number) => `${value.toFixed(1)} tok/s`}
                    />
                    <MiniLineChart
                      title="Adapter Size"
                      xValues={llmMetricRounds}
                      series={llmAdapterSizeSeries(runMetrics)}
                      formatter={(value: number) => formatBytes(value)}
                    />
                    <MiniLineChart
                      title="Client Train Loss"
                      xValues={llmMetricRounds}
                      series={clientTrainLossSeries}
                      formatter={(value: number) => value.toFixed(4)}
                    />
                  </div>
                )}
              </div>

              {isLlmRun && (
                <div className="space-y-2">
                  <p className="flex items-center gap-2 text-xs text-muted-foreground">
                    <GitBranch className="h-4 w-4" />
                    Adapter Lineage
                  </p>
                  <div className="overflow-auto rounded-md border">
                    <table className="w-full min-w-[720px] text-left text-xs">
                      <thead className="border-b bg-muted/40 text-muted-foreground">
                        <tr>
                          <th className="px-3 py-2 font-medium">Round</th>
                          <th className="px-3 py-2 font-medium">Selected Clients</th>
                          <th className="px-3 py-2 font-medium">Size</th>
                          <th className="px-3 py-2 font-medium">SHA-256</th>
                          <th className="px-3 py-2 font-medium">Parent</th>
                          <th className="px-3 py-2 font-medium">Path</th>
                        </tr>
                      </thead>
                      <tbody>
                        {llmArtifacts.map((artifact, index) => (
                          <tr key={`${artifact.path ?? index}`} className="border-b last:border-none">
                            <td className="px-3 py-2 font-mono">{toDisplayText(artifact.round)}</td>
                            <td className="px-3 py-2 font-mono">{selectedClientsText(artifact.selected_clients)}</td>
                            <td className="px-3 py-2 font-mono">{formatBytes(artifact.size_bytes)}</td>
                            <td className="px-3 py-2 font-mono" title={String(artifact.sha256 ?? "")}>
                              {shortHash(artifact.sha256)}
                            </td>
                            <td className="px-3 py-2 font-mono" title={String(artifact.parent_sha256 ?? "")}>
                              {shortHash(artifact.parent_sha256)}
                            </td>
                            <td className="max-w-[260px] truncate px-3 py-2 font-mono" title={String(artifact.path ?? "")}>
                              {toDisplayText(artifact.path)}
                            </td>
                          </tr>
                        ))}
                        {llmArtifacts.length === 0 && (
                          <tr>
                            <td colSpan={6} className="px-3 py-6 text-center text-muted-foreground">
                              No adapter artifacts recorded yet.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
