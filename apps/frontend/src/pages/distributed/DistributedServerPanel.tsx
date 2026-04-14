import { Loader2, MoreHorizontal, Play, Plus, RefreshCcw } from "lucide-react";

import { Badge } from "../../components/ui/badge";
import { Button } from "../../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card";
import { Input } from "../../components/ui/input";
import { NumberStepper } from "../../components/ui/number-stepper";
import { Textarea } from "../../components/ui/textarea";
import { SchemaConfigDialog } from "../../features/config/SchemaConfigDialog";
import type { DistributedJob, DistributedPageProps } from "../types";

export function DistributedServerPanel(props: DistributedPageProps) {
  const {
    allDistributedClientsReady,
    approvedDistributedSlots,
    busy,
    configSchema,
    distributedClientStatusVariant,
    distributedClients,
    distributedConfigClients,
    distributedConfigDialogOpen,
    distributedConfigDraft,
    distributedJobDescription,
    distributedJobName,
    distributedJobStatusVariant,
    distributedJobs,
    distributedServerIp,
    distributedServerPort,
    distributedSession,
    distributedSessionProgress,
    distributedSessionStatusVariant,
    handleApproveDistributedClient,
    handleCreateDistributedJob,
    handleCreateOrAttachDistributedSession,
    handleRejectDistributedClient,
    handleSaveDistributedJobConfig,
    handleStartDistributedTraining,
    notifyError,
    openDistributedConfigDialog,
    readyDistributedSlots,
    refreshDistributedJobs,
    refreshDistributedSessionBundle,
    refreshDistributedSessionProgress,
    renderSchemaSection,
    selectedDistributedJob,
    selectedDistributedJobId,
    setDistributedConfigClients,
    setDistributedConfigDialogOpen,
    setDistributedConfigProperty,
    setDistributedJobDescription,
    setDistributedJobName,
    setDistributedServerIp,
    setDistributedServerPort,
    setSelectedDistributedJobId,
  } = props;

  return (
    <>
      <div className="grid min-h-[calc(100vh-260px)] gap-4 xl:grid-cols-[360px_minmax(0,1fr)]">
        <Card className="min-h-0">
          <CardHeader>
            <CardTitle>Distributed Jobs</CardTitle>
            <CardDescription>Create a job first, then use the menu on each row to configure it.</CardDescription>
          </CardHeader>
          <CardContent className="flex h-[calc(100%-96px)] min-h-0 flex-col gap-3">
            <div className="space-y-2 rounded-md border p-2">
              <Input
                placeholder="Distributed job name"
                value={distributedJobName}
                onChange={(event) => setDistributedJobName(event.target.value)}
              />
              <Textarea
                placeholder="Description"
                className="min-h-[68px]"
                value={distributedJobDescription}
                onChange={(event) => setDistributedJobDescription(event.target.value)}
              />
              <Button className="w-full" onClick={() => handleCreateDistributedJob().catch((err: unknown) => notifyError(err))} disabled={busy}>
                {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Plus className="mr-2 h-4 w-4" />}
                Create Distributed Job
              </Button>
            </div>

            <div className="flex items-center justify-between">
              <p className="text-xs text-muted-foreground">Job List</p>
              <Button variant="ghost" size="sm" onClick={() => refreshDistributedJobs().catch((err: unknown) => notifyError(err))}>
                <RefreshCcw className="h-3.5 w-3.5" />
              </Button>
            </div>

            <div className="min-h-0 flex-1 space-y-2 overflow-auto">
              {distributedJobs.map((job: DistributedJob) => (
                <div
                  key={job.id}
                  className={`w-full rounded-md border px-3 py-2 text-left transition ${
                    selectedDistributedJobId === job.id ? "bg-muted" : "hover:bg-muted/60"
                  }`}
                >
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <button className="min-w-0 flex-1 text-left" onClick={() => setSelectedDistributedJobId(job.id)}>
                      <span className="truncate text-sm font-semibold">{job.name}</span>
                    </button>
                    <div className="flex items-center gap-2">
                      <Badge variant={distributedJobStatusVariant[job.status] ?? "secondary"}>{job.status}</Badge>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 p-0"
                        onClick={(event) => {
                          event.stopPropagation();
                          openDistributedConfigDialog(job);
                        }}
                        disabled={configSchema === null}
                      >
                        <span className="sr-only">Open job config</span>
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">expected_clients={job.expected_clients}</p>
                </div>
              ))}
              {distributedJobs.length === 0 && <p className="text-xs text-muted-foreground">No distributed jobs yet.</p>}
            </div>
          </CardContent>
        </Card>

        <Card className="min-h-0">
          <CardHeader>
            <CardTitle>Server Session</CardTitle>
            <CardDescription>Select a distributed job first, then create a waiting session.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {!selectedDistributedJob && <p className="text-sm text-muted-foreground">Select a distributed job from the left panel.</p>}

            {selectedDistributedJob && (
              <>
                <div className="grid gap-2 md:grid-cols-2">
                  <div>
                    <p className="text-xs text-muted-foreground">Selected Job</p>
                    <p className="text-sm font-medium">{selectedDistributedJob.name}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Expected Clients</p>
                    <p className="text-sm">{selectedDistributedJob.expected_clients}</p>
                  </div>
                </div>

                <div className="space-y-2 rounded-md border p-3">
                  <p className="text-xs text-muted-foreground">Server Endpoint (shown after selecting a job)</p>
                  <div className="grid gap-2 md:grid-cols-[1fr_140px_auto]">
                    <Input value={distributedServerIp} onChange={(event) => setDistributedServerIp(event.target.value)} placeholder="Server IP" />
                    <Input
                      value={distributedServerPort}
                      onChange={(event) => setDistributedServerPort(event.target.value)}
                      placeholder="Port"
                      type="number"
                    />
                    <Button
                      onClick={() => handleCreateOrAttachDistributedSession().catch((err: unknown) => notifyError(err))}
                      disabled={busy}
                    >
                      Wait Clients
                    </Button>
                  </div>
                  {distributedSession && (
                    <div className="rounded-md border bg-muted/20 p-2 text-sm">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-muted-foreground">Session:</span>
                        <span className="font-mono">{distributedSession.id}</span>
                        <Badge variant={distributedSessionStatusVariant[distributedSession.status] ?? "secondary"}>
                          {distributedSession.status}
                        </Badge>
                      </div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {distributedSession.server_ip}:{distributedSession.server_port}
                      </div>
                    </div>
                  )}
                </div>

                {distributedSession && (
                  <>
                    <div className="flex items-center justify-between">
                      <p className="text-xs text-muted-foreground">Connection Requests</p>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          Promise.all([
                            refreshDistributedSessionBundle(distributedSession.id),
                            refreshDistributedSessionProgress(distributedSession.id),
                          ]).catch((err: unknown) => notifyError(err))
                        }
                      >
                        <RefreshCcw className="h-3.5 w-3.5" />
                      </Button>
                    </div>

                    <div className="max-h-[320px] space-y-2 overflow-auto rounded-md border p-2">
                      {distributedClients.map((item) => (
                        <div key={item.id} className="rounded border bg-muted/20 p-2 text-sm">
                          <div className="mb-1 flex items-center justify-between gap-2">
                            <span className="font-medium">{item.participant_name}</span>
                            <Badge variant={distributedClientStatusVariant[item.status] ?? "secondary"}>{item.status}</Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            participant_id={item.assigned_participant_id ?? "-"}
                          </div>
                          {item.status === "pending_approval" && (
                            <div className="mt-2 flex gap-2">
                              <Button
                                size="sm"
                                variant="secondary"
                                onClick={() => handleApproveDistributedClient(item.id).catch((err: unknown) => notifyError(err))}
                                disabled={busy}
                              >
                                Approve
                              </Button>
                              <Button
                                size="sm"
                                variant="destructive"
                                onClick={() => handleRejectDistributedClient(item.id).catch((err: unknown) => notifyError(err))}
                                disabled={busy}
                              >
                                Reject
                              </Button>
                            </div>
                          )}
                        </div>
                      ))}
                      {distributedClients.length === 0 && (
                        <p className="text-xs text-muted-foreground">No client connection requests yet.</p>
                      )}
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border p-3">
                      <div className="text-sm">
                        <p>
                          Approved Slots: {approvedDistributedSlots.size}/{selectedDistributedJob.expected_clients}
                        </p>
                        <p>
                          Ready Slots: {readyDistributedSlots.size}/{selectedDistributedJob.expected_clients}
                        </p>
                      </div>
                      <Button
                        onClick={() => handleStartDistributedTraining().catch((err: unknown) => notifyError(err))}
                        disabled={busy || !allDistributedClientsReady || distributedSession.status === "running"}
                      >
                        <Play className="mr-2 h-4 w-4" />
                        Start Training
                      </Button>
                    </div>

                    <div className="rounded-md border p-3 text-sm">
                      <p className="text-xs text-muted-foreground">Training Progress</p>
                      {!distributedSessionProgress && <p className="mt-1 text-muted-foreground">No runtime progress yet.</p>}
                      {distributedSessionProgress && (
                        <div className="mt-1 space-y-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <Badge variant={distributedSessionProgress.status === "running" ? "warning" : "secondary"}>
                              {distributedSessionProgress.status}
                            </Badge>
                            {distributedSessionProgress.pid !== null && (
                              <span className="text-xs text-muted-foreground">pid={distributedSessionProgress.pid}</span>
                            )}
                          </div>
                          <p>
                            Round: {distributedSessionProgress.last_round}/{distributedSessionProgress.total_rounds}
                          </p>
                          <p>Global Loss: {distributedSessionProgress.latest_global_loss ?? "-"}</p>
                          <p>Global Accuracy: {distributedSessionProgress.latest_global_accuracy ?? "-"}</p>
                          {distributedSessionProgress.log_path && (
                            <p className="text-xs text-muted-foreground">log: {distributedSessionProgress.log_path}</p>
                          )}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <SchemaConfigDialog
        open={distributedConfigDialogOpen}
        onOpenChange={setDistributedConfigDialogOpen}
        title="Distributed Job Config"
        description="Edit server-side config fields."
        busy={busy}
        schema={configSchema}
        role="server"
        properties={distributedConfigDraft}
        onChangeProperty={setDistributedConfigProperty}
        renderSchemaSection={renderSchemaSection}
        onSave={() => {
          void handleSaveDistributedJobConfig().catch((err: unknown) => notifyError(err));
        }}
        beforeSchema={
          <div className="rounded-md border border-border/70 bg-muted/25 p-3">
            <p className="mb-1 text-xs text-muted-foreground">Expected Clients</p>
            <NumberStepper
              value={distributedConfigClients}
              min={1}
              step={1}
              onValueChange={(next) => setDistributedConfigClients(Math.max(1, Math.floor(next)))}
            />
          </div>
        }
      />
    </>
  );
}
