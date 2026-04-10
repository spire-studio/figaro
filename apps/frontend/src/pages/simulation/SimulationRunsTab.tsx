import { BarChart3, FileText, Play, RefreshCcw, Square, TerminalSquare, Trash2 } from "lucide-react";

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
import { fmt } from "../../lib/time";
import type { Run, RunLog, SimulationPageProps } from "../types";

export function SimulationRunsTab(props: SimulationPageProps) {
  const {
    MiniLineChart,
    attackEnabled,
    busy,
    clientTestAccSeries,
    clientTestLossSeries,
    clientTrainAccSeries,
    clientTrainLossSeries,
    compressionInfo,
    defenseParams,
    encryptionInfo,
    experimentAttack,
    experimentBasic,
    experimentDefense,
    experimentFederated,
    globalAccuracySeries,
    globalLossSeries,
    handleDeleteRun,
    handleRerun,
    handleStopRun,
    isRecord,
    loadRunBundle,
    loadRunMetrics,
    maliciousClients,
    notifyError,
    runLogs,
    runLogsRef,
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

                  <Card className="border-red-200/60 bg-red-50/40 dark:border-red-900/40 dark:bg-red-950/20">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between gap-2">
                        <CardTitle className="text-sm">Attack Details</CardTitle>
                        <Badge variant={attackEnabled ? "danger" : "outline"}>{attackEnabled ? "Enabled" : "Disabled"}</Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <div>
                        <p className="text-xs text-muted-foreground">Strategy</p>
                        <p>{toDisplayText(experimentAttack.global_type, "None")}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Malicious Clients</p>
                        <p>{maliciousClients.length > 0 ? maliciousClients.map((clientId: string) => `#${clientId}`).join(", ") : "None"}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Per-client Rules</p>
                        <p>{isRecord(experimentAttack.per_client_settings) ? Object.keys(experimentAttack.per_client_settings).length : 0}</p>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="border-emerald-200/60 bg-emerald-50/40 dark:border-emerald-900/40 dark:bg-emerald-950/20">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm">Security &amp; Defense</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3 text-sm">
                      <div>
                        <p className="text-xs text-muted-foreground">Defense</p>
                        <p>
                          {toDisplayText(experimentDefense.strategy, "None")}
                          {Boolean(experimentDefense.enabled) ? " (enabled)" : " (disabled)"}
                        </p>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {Object.entries(defenseParams)
                            .slice(0, 6)
                            .map(([key, value]) => (
                              <span
                                key={`defense-${key}`}
                                className="rounded border bg-background/80 px-1.5 py-0.5 text-[11px] text-muted-foreground"
                              >
                                {key}: {String(value)}
                              </span>
                            ))}
                          {Object.keys(defenseParams).length === 0 && <span className="text-xs text-muted-foreground">No defense params</span>}
                        </div>
                      </div>
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
                  Training Curves
                </p>
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
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
