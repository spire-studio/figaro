import { useMemo, useState } from "react";
import {
  Copy,
  FileText,
  Loader2,
  MoreHorizontal,
  Play,
  Plus,
  RefreshCcw,
  Server,
  Trash2,
  UserRound,
} from "lucide-react";

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
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "../../components/ui/dropdown-menu";
import { Input } from "../../components/ui/input";
import { NumberStepper } from "../../components/ui/number-stepper";
import { Separator } from "../../components/ui/separator";
import { Textarea } from "../../components/ui/textarea";
import { SchemaConfigDialog } from "../../features/config/SchemaConfigDialog";
import { getValueByPath, normalizeClientId, setValueByPath } from "../../features/simulation/utils";
import type { Job, SimulationPageProps, TopologyNode } from "../types";

const NODE_SIZE = { width: 160, height: 74 };
type TopologyConfigDraft = {
  nodeId: string;
  role: TopologyNode["role"];
  properties: Record<string, unknown>;
};
type TopologyNodeContextMenu = {
  nodeId: string;
  x: number;
  y: number;
};

export function SimulationJobsTab(props: SimulationPageProps) {
  const {
    applyTopologyNodeConfig,
    busy,
    canvasHighlighted,
    canvasRef,
    clientNodes,
    cloneTopologyNode,
    configJsonError,
    configSchema,
    deleteJobTarget,
    deleteTopologyNode,
    editingJobDescription,
    editingJobName,
    handleCreateJob,
    handleDeleteJob,
    handleDuplicateJob,
    handleSaveJobMeta,
    handleStartRun,
    jobConfigText,
    jobMetaDialogOpen,
    jobStatusVariant,
    jobs,
    newJobDescription,
    newJobName,
    nodeCenter,
    notifyError,
    onCanvasDragLeave,
    onCanvasDragOver,
    onCanvasDrop,
    onToolDragStart,
    onTopologyNodeMouseDown,
    openJobMetadataDialog,
    rectEdgePoint,
    refreshJobs,
    renderSchemaSection,
    requestDeleteJob,
    selectedJob,
    selectedJobId,
    selectedTopologyNodeId,
    serverNode,
    setDeleteJobTarget,
    setEditingJobDescription,
    setEditingJobMetaId,
    setEditingJobName,
    setJobConfigText,
    setJobMetaDialogOpen,
    setNewJobDescription,
    setNewJobName,
    setSelectedJobId,
    setSelectedTopologyNodeId,
    topologyCanvasMinHeight,
    topologyNodes,
  } = props;
  const [topologyConfigDialogOpen, setTopologyConfigDialogOpen] = useState(false);
  const [topologyConfigDraft, setTopologyConfigDraft] = useState<TopologyConfigDraft | null>(null);
  const [topologyNodeContextMenu, setTopologyNodeContextMenu] = useState<TopologyNodeContextMenu | null>(null);
  const topologyContextNode = useMemo(() => {
    if (!topologyNodeContextMenu) return null;
    return topologyNodes.find((node) => node.id === topologyNodeContextMenu.nodeId) ?? null;
  }, [topologyNodeContextMenu, topologyNodes]);

  function openTopologyConfigDialog(node: TopologyNode): void {
    setSelectedTopologyNodeId(node.id);
    setTopologyConfigDraft({
      nodeId: node.id,
      role: node.role,
      properties: structuredClone(node.properties),
    });
    setTopologyConfigDialogOpen(true);
  }

  function setTopologyConfigProperty(path: string, value: unknown): void {
    setTopologyConfigDraft((previous) => {
      if (!previous) return previous;
      const nextProperties = structuredClone(previous.properties);
      setValueByPath(nextProperties, path, value);
      return { ...previous, properties: nextProperties };
    });
  }

  function handleSaveTopologyConfig(): void {
    if (!topologyConfigDraft) return;
    applyTopologyNodeConfig(topologyConfigDraft.nodeId, topologyConfigDraft.properties);
    setTopologyConfigDialogOpen(false);
    setTopologyConfigDraft(null);
  }

  function openTopologyNodeContextMenu(event: React.MouseEvent<HTMLButtonElement>, nodeId: string): void {
    event.preventDefault();
    setSelectedTopologyNodeId(nodeId);
    setTopologyNodeContextMenu({
      nodeId,
      x: event.clientX,
      y: event.clientY,
    });
  }

  return (
    <div className="grid min-h-[calc(100vh-180px)] gap-4 xl:grid-cols-[330px_minmax(0,1fr)]">
      <Card className="min-h-0">
        <CardHeader>
          <CardTitle>Jobs</CardTitle>
          <CardDescription>Select a job to edit on the right.</CardDescription>
        </CardHeader>
        <CardContent className="flex h-[calc(100%-96px)] min-h-0 flex-col gap-3">
          <div className="space-y-2 rounded-md border p-2">
            <Input placeholder="New job name" value={newJobName} onChange={(event) => setNewJobName(event.target.value)} />
            <Textarea
              placeholder="New job description"
              className="min-h-[68px]"
              value={newJobDescription}
              onChange={(event) => setNewJobDescription(event.target.value)}
            />
            <Button
              className="w-full"
              onClick={() => handleCreateJob().catch((err: unknown) => notifyError(err))}
              disabled={busy || newJobName.trim().length === 0 || newJobDescription.trim().length === 0}
            >
              {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Plus className="mr-2 h-4 w-4" />}
              Create Job
            </Button>
          </div>

          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground">Job List</p>
            <Button variant="ghost" size="sm" onClick={() => refreshJobs().catch((err: unknown) => notifyError(err))}>
              <RefreshCcw className="h-3.5 w-3.5" />
            </Button>
          </div>

          <div className="min-h-0 flex-1 space-y-2 overflow-auto">
            {jobs.map((job: Job) => (
              <div
                key={job.id}
                className={`rounded-md border px-3 py-2 transition ${
                  selectedJobId === job.id ? "bg-muted" : "hover:bg-muted/60"
                }`}
              >
                <div className="flex items-start gap-2">
                  <button type="button" className="min-w-0 flex-1 text-left" onClick={() => setSelectedJobId(job.id)}>
                    <div className="mb-1 flex items-center justify-between gap-2">
                      <span className="truncate text-sm font-semibold">{job.name}</span>
                      <Badge variant={jobStatusVariant[job.status] ?? "secondary"}>{job.status}</Badge>
                    </div>
                    <p className="line-clamp-2 text-xs text-muted-foreground">{job.description ?? "No description"}</p>
                  </button>

                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        type="button"
                        className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-transparent text-muted-foreground transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        aria-label="Open job actions"
                      >
                        <MoreHorizontal className="h-4 w-4" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onSelect={(event) => {
                          event.preventDefault();
                          openJobMetadataDialog(job);
                        }}
                      >
                        Edit Metadata
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onSelect={(event) => {
                          event.preventDefault();
                          void handleDuplicateJob(job);
                        }}
                      >
                        Duplicate Job
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="text-destructive focus:text-destructive"
                        onSelect={(event) => {
                          event.preventDefault();
                          requestDeleteJob(job);
                        }}
                      >
                        Delete Job
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
            ))}
            {jobs.length === 0 && <p className="text-xs text-muted-foreground">No jobs yet.</p>}
          </div>
        </CardContent>
      </Card>

      {!selectedJob && (
        <Card className="min-h-0">
          <CardHeader>
            <CardTitle>Create a Job First</CardTitle>
            <CardDescription>Fill in both job name and job description on the left, then create a job.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p>Selected Job and Topology Builder are available only after creating and selecting a job.</p>
          </CardContent>
        </Card>
      )}

      {selectedJob && (
        <div className="grid gap-4">
          <Card className="order-0">
            <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div className="space-y-1">
                <CardTitle>Selected Job</CardTitle>
                <CardDescription>Job details and current config for the selected job.</CardDescription>
              </div>
              <div className="flex w-full flex-wrap gap-2 sm:w-auto sm:justify-end">
                <Button
                  className="bg-emerald-600 text-white hover:bg-emerald-500"
                  size="sm"
                  onClick={() => handleStartRun().catch((err: unknown) => notifyError(err))}
                  disabled={busy}
                >
                  <Play className="mr-2 h-4 w-4" />
                  Start Run
                </Button>
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button variant="destructive" size="sm" disabled={busy}>
                      <Trash2 className="mr-2 h-4 w-4" />
                      Delete Job
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>Delete this job?</AlertDialogTitle>
                      <AlertDialogDescription>
                        This will remove the job, its config, and related runs/logs/results.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Cancel</AlertDialogCancel>
                      <AlertDialogAction
                        disabled={busy}
                        onClick={() => {
                          void handleDeleteJob(selectedJob);
                        }}
                      >
                        Delete Job
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleDuplicateJob(selectedJob).catch((err: unknown) => notifyError(err))}
                  disabled={busy}
                >
                  <Copy className="mr-2 h-4 w-4" />
                  Duplicate Job
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-2 md:grid-cols-2">
                <div>
                  <p className="mb-1 text-xs text-muted-foreground">Job ID</p>
                  <p className="font-mono text-sm">{selectedJob.id}</p>
                </div>
                <div>
                  <p className="mb-1 text-xs text-muted-foreground">Status</p>
                  <Badge variant={jobStatusVariant[selectedJob.status] ?? "secondary"}>{selectedJob.status}</Badge>
                </div>
                <div>
                  <p className="mb-1 text-xs text-muted-foreground">Job Name</p>
                  <p className="text-sm">{selectedJob.name}</p>
                </div>
                <div>
                  <p className="mb-1 text-xs text-muted-foreground">Description</p>
                  <p className="line-clamp-2 text-sm">{selectedJob.description ?? "No description"}</p>
                </div>
              </div>

              <Separator />

              <div className="space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                    <Server className="h-3.5 w-3.5" />
                    Topology Builder
                  </p>
                  <p className="text-xs text-muted-foreground">Drag nodes and adjust node properties</p>
                </div>
                <div className="grid min-h-0 gap-3 lg:grid-cols-[150px_minmax(0,1fr)]">
                  <div className="space-y-2">
                    <Button
                      className="w-full justify-start border-blue-300/70 bg-blue-50/80 text-blue-950 hover:bg-blue-100 dark:border-blue-500/40 dark:bg-blue-500/10 dark:text-blue-100 dark:hover:bg-blue-500/20"
                      variant="outline"
                      draggable
                      onDragStart={(event) => onToolDragStart(event, "server")}
                    >
                      <Server className="mr-2 h-4 w-4" />
                      Server
                    </Button>
                    <Button
                      className="w-full justify-start border-emerald-300/70 bg-emerald-50/80 text-emerald-950 hover:bg-emerald-100 dark:border-emerald-500/40 dark:bg-emerald-500/10 dark:text-emerald-100 dark:hover:bg-emerald-500/20"
                      variant="outline"
                      draggable
                      onDragStart={(event) => onToolDragStart(event, "client")}
                    >
                      <UserRound className="mr-2 h-4 w-4" />
                      Client
                    </Button>

                    <Separator />
                  </div>

                  <div className="min-w-0 space-y-3">
                    <div
                      ref={canvasRef}
                      onDrop={onCanvasDrop}
                      onDragOver={onCanvasDragOver}
                      onDragLeave={onCanvasDragLeave}
                      style={{ minHeight: `${topologyCanvasMinHeight}px` }}
                      className={`relative min-h-[360px] overflow-hidden rounded-md border bg-muted/20 ${
                        canvasHighlighted ? "ring-2 ring-primary" : ""
                      }`}
                    >
                      <svg className="pointer-events-none absolute inset-0 h-full w-full">
                        {serverNode &&
                          clientNodes.map((client: TopologyNode) => {
                            const serverCenter = nodeCenter(serverNode);
                            const clientCenter = nodeCenter(client);
                            const serverEdge = rectEdgePoint(serverCenter, clientCenter);
                            const clientEdge = rectEdgePoint(clientCenter, serverCenter);

                            return (
                              <line
                                key={`${serverNode.id}-${client.id}`}
                                x1={clientEdge.x}
                                y1={clientEdge.y}
                                x2={serverEdge.x}
                                y2={serverEdge.y}
                                stroke="#64748b"
                                strokeWidth={2}
                                strokeOpacity={0.9}
                              />
                            );
                          })}
                      </svg>

                      {topologyNodes.map((node: TopologyNode) => (
                        <div
                          key={node.id}
                          style={{ left: node.x, top: node.y, width: NODE_SIZE.width, height: NODE_SIZE.height }}
                          className={`absolute rounded-md border px-2 py-1 text-left shadow-sm ${
                            node.role === "server"
                              ? "border-blue-300/70 bg-blue-50/80 text-blue-950 dark:border-blue-500/40 dark:bg-blue-500/10 dark:text-blue-100"
                              : "border-emerald-300/70 bg-emerald-50/80 text-emerald-950 dark:border-emerald-500/40 dark:bg-emerald-500/10 dark:text-emerald-100"
                          } ${node.id === selectedTopologyNodeId ? "ring-2 ring-primary" : ""}`}
                        >
                          <button
                            type="button"
                            onMouseDown={(event) => onTopologyNodeMouseDown(event, node)}
                            onClick={() => setSelectedTopologyNodeId(node.id)}
                            onContextMenu={(event) => openTopologyNodeContextMenu(event, node.id)}
                            className="h-full w-full text-left"
                          >
                            <div className="mb-1 flex items-center justify-between pr-7 text-[11px] text-muted-foreground">
                              <span>{node.role}</span>
                            </div>
                            <div className="truncate pr-7 text-sm font-medium">{node.label}</div>
                            {node.role === "client" && (
                              <div className="font-mono text-[11px] text-muted-foreground">id: {node.clientId}</div>
                            )}
                          </button>
                        </div>
                      ))}
                    </div>

                    <DropdownMenu
                      open={topologyNodeContextMenu !== null}
                      onOpenChange={(open) => {
                        if (!open) {
                          setTopologyNodeContextMenu(null);
                        }
                      }}
                    >
                      <DropdownMenuTrigger asChild>
                        <button
                          type="button"
                          aria-hidden
                          tabIndex={-1}
                          style={{
                            position: "fixed",
                            left: topologyNodeContextMenu?.x ?? -9999,
                            top: topologyNodeContextMenu?.y ?? -9999,
                            width: 1,
                            height: 1,
                            opacity: 0,
                            pointerEvents: "none",
                          }}
                        />
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="start" sideOffset={4}>
                        <DropdownMenuItem
                          disabled={!topologyContextNode}
                          onSelect={(event) => {
                            event.preventDefault();
                            if (!topologyContextNode) return;
                            openTopologyConfigDialog(topologyContextNode);
                          }}
                        >
                          Edit Config
                        </DropdownMenuItem>
                        {topologyContextNode?.role === "client" && (
                          <DropdownMenuItem
                            onSelect={(event) => {
                              event.preventDefault();
                              cloneTopologyNode(topologyContextNode.id);
                            }}
                          >
                            Duplicate
                          </DropdownMenuItem>
                        )}
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          className="text-destructive focus:text-destructive"
                          disabled={!topologyContextNode}
                          onSelect={(event) => {
                            event.preventDefault();
                            if (!topologyContextNode) return;
                            deleteTopologyNode(topologyContextNode.id);
                          }}
                        >
                          Delete Node
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
              </div>

              <Separator />

              <div className="space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                    <FileText className="h-3.5 w-3.5" />
                    Current Config (JSON)
                  </p>
                  <p className="text-xs text-muted-foreground">Auto sync enabled</p>
                </div>
                <Textarea
                  className="min-h-[500px] font-mono text-[12px]"
                  value={jobConfigText}
                  onChange={(event) => setJobConfigText(event.target.value)}
                />
                {configJsonError && <p className="text-xs text-red-500">{configJsonError}</p>}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <SchemaConfigDialog
        open={topologyConfigDialogOpen}
        onOpenChange={(open) => {
          setTopologyConfigDialogOpen(open);
          if (!open) {
            setTopologyConfigDraft(null);
          }
        }}
        title="Node Config"
        description="Edit selected node config fields."
        schema={configSchema}
        role={topologyConfigDraft?.role ?? "client"}
        properties={topologyConfigDraft?.properties ?? null}
        onChangeProperty={setTopologyConfigProperty}
        renderSchemaSection={renderSchemaSection}
        onSave={handleSaveTopologyConfig}
        beforeSchema={
          topologyConfigDraft?.role === "client" ? (
            <div className="rounded-md border border-border/70 bg-muted/25 p-3">
              <p className="mb-1 text-xs text-muted-foreground">Client ID</p>
              <NumberStepper
                value={normalizeClientId(getValueByPath(topologyConfigDraft.properties, "distributed.client_id"))}
                min={0}
                step={1}
                onValueChange={(next) => setTopologyConfigProperty("distributed.client_id", normalizeClientId(next))}
              />
            </div>
          ) : undefined
        }
      />

      <AlertDialog
        open={jobMetaDialogOpen}
        onOpenChange={(open) => {
          setJobMetaDialogOpen(open);
          if (!open) {
            setEditingJobMetaId(null);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Edit Job Metadata</AlertDialogTitle>
            <AlertDialogDescription>Update the job name and description.</AlertDialogDescription>
          </AlertDialogHeader>
          <div className="space-y-2">
            <Input value={editingJobName} onChange={(event) => setEditingJobName(event.target.value)} placeholder="Job name" />
            <Textarea
              className="min-h-[96px]"
              value={editingJobDescription}
              onChange={(event) => setEditingJobDescription(event.target.value)}
              placeholder="Job description"
            />
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>Cancel</AlertDialogCancel>
            <Button onClick={() => handleSaveJobMeta().catch((err: unknown) => notifyError(err))} disabled={busy}>
              Save Metadata
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog
        open={deleteJobTarget !== null}
        onOpenChange={(open) => {
          if (!open) {
            setDeleteJobTarget(null);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this job?</AlertDialogTitle>
            <AlertDialogDescription>
              {deleteJobTarget
                ? `This will remove "${deleteJobTarget.name}", its config, and related runs/logs/results.`
                : "This will remove the selected job and all related data."}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={busy || deleteJobTarget === null}
              onClick={() => {
                if (!deleteJobTarget) return;
                void handleDeleteJob(deleteJobTarget);
              }}
            >
              Delete Job
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
