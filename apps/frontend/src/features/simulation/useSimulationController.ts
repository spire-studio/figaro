import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { toast } from "sonner";

import { jobsApi } from "../../api/jobs";
import { runsApi } from "../../api/runs";
import { renderSchemaSection } from "../config/render-schema-section";
import { MiniLineChart } from "./components/MiniLineChart";
import {
  EMPTY_RUN_METRICS,
  NODE_SIZE,
  buildConfigFromTopology,
  getTopologyCanvasMinHeight,
  isRecord,
  jobStatusVariant,
  makeInitialTopologyNodes,
  nextClientId,
  nodeCenter,
  normalizeClientId,
  parseJsonObject,
  rectEdgePoint,
  runStatusVariant,
  setValueByPath,
  shortRunId,
  stableStringify,
  toClientSeries,
  toDisplayText,
  toErrorMessage,
  toNumber,
  topologyFromConfig,
  uid,
  getValueByPath,
  createNodeProperties,
} from "./utils";
import type { RunMetrics } from "../../api/runs";
import type { SimulationPageProps, SystemMode, TopologyNode } from "../../pages/types";

type DraggingNode = {
  id: string;
  offsetX: number;
  offsetY: number;
};

export function useSimulationController(): SimulationPageProps {
  const [activeTab, setActiveTab] = useState<"jobs" | "runs">("jobs");
  const [busy, setBusy] = useState(false);
  const [configSchema, setConfigSchema] = useState<Record<string, unknown> | null>(null);

  const [jobs, setJobs] = useState<SimulationPageProps["jobs"]>([]);
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);
  const [jobConfigText, setJobConfigText] = useState("");
  const [configJsonError, setConfigJsonError] = useState<string | null>(null);
  const [jobMetaDialogOpen, setJobMetaDialogOpen] = useState(false);
  const [editingJobMetaId, setEditingJobMetaId] = useState<number | null>(null);
  const [editingJobName, setEditingJobName] = useState("");
  const [editingJobDescription, setEditingJobDescription] = useState("");
  const [deleteJobTarget, setDeleteJobTarget] = useState<SimulationPageProps["deleteJobTarget"]>(null);

  const [newJobName, setNewJobName] = useState("");
  const [newJobDescription, setNewJobDescription] = useState("");

  const [topologyNodes, setTopologyNodes] = useState<TopologyNode[]>(() => makeInitialTopologyNodes(null, "simulation"));
  const [selectedTopologyNodeId, setSelectedTopologyNodeId] = useState<string | null>(null);
  const [canvasHighlighted, setCanvasHighlighted] = useState(false);
  const [systemMode, setSystemMode] = useState<SystemMode>("simulation");

  const [runs, setRuns] = useState<SimulationPageProps["runs"]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runDetails, setRunDetails] = useState<SimulationPageProps["selectedRun"]>(null);
  const [runLogs, setRunLogs] = useState<SimulationPageProps["runLogs"]>([]);
  const [runMetrics, setRunMetrics] = useState<RunMetrics>(EMPTY_RUN_METRICS);

  const canvasRef = useRef<HTMLDivElement | null>(null);
  const draggingNodeRef = useRef<DraggingNode | null>(null);
  const runLogsRef = useRef<HTMLDivElement | null>(null);
  const configSyncSourceRef = useRef<"topology" | "json" | null>(null);
  const configAutoSaveTimerRef = useRef<number | null>(null);
  const configSaveSeqRef = useRef(0);
  const lastSavedConfigRef = useRef<string | null>(null);
  const isHydratingJobRef = useRef(false);
  const jobHydrationSeqRef = useRef(0);

  const selectedJob = useMemo(() => jobs.find((job) => job.id === selectedJobId) ?? null, [jobs, selectedJobId]);

  const selectedRun = useMemo(() => {
    if (!selectedRunId) return null;
    if (runDetails && runDetails.id === selectedRunId) return runDetails;
    return runs.find((run) => run.id === selectedRunId) ?? null;
  }, [runs, runDetails, selectedRunId]);

  const selectedTopologyNode = useMemo(
    () => topologyNodes.find((node) => node.id === selectedTopologyNodeId) ?? null,
    [topologyNodes, selectedTopologyNodeId],
  );

  const serverNode = useMemo(() => topologyNodes.find((node) => node.role === "server") ?? null, [topologyNodes]);
  const clientNodes = useMemo(() => topologyNodes.filter((node) => node.role === "client"), [topologyNodes]);
  const topologyCanvasMinHeight = useMemo(() => getTopologyCanvasMinHeight(clientNodes.length), [clientNodes.length]);
  const runRounds = useMemo(() => runMetrics.global_results.rounds, [runMetrics]);
  const globalAccuracySeries = useMemo(
    () => [{ key: "global-accuracy", label: "Global", color: "#2563eb", values: runMetrics.global_results.global_accuracy }],
    [runMetrics],
  );
  const globalLossSeries = useMemo(
    () => [{ key: "global-loss", label: "Global", color: "#f59e0b", values: runMetrics.global_results.global_loss }],
    [runMetrics],
  );
  const clientTrainAccSeries = useMemo(() => toClientSeries(runMetrics, "train_acc"), [runMetrics]);
  const clientTrainLossSeries = useMemo(() => toClientSeries(runMetrics, "train_loss"), [runMetrics]);
  const clientTestAccSeries = useMemo(() => toClientSeries(runMetrics, "test_acc"), [runMetrics]);
  const clientTestLossSeries = useMemo(() => toClientSeries(runMetrics, "test_loss"), [runMetrics]);
  const experimentBasic = useMemo(
    () => (isRecord(runMetrics.experiment_info.basic) ? runMetrics.experiment_info.basic : {}),
    [runMetrics],
  );
  const experimentFederated = useMemo(
    () => (isRecord(runMetrics.experiment_info.federated) ? runMetrics.experiment_info.federated : {}),
    [runMetrics],
  );
  const experimentSecurity = useMemo(
    () => (isRecord(runMetrics.experiment_info.security) ? runMetrics.experiment_info.security : {}),
    [runMetrics],
  );
  const encryptionInfo = isRecord(experimentSecurity.encryption) ? experimentSecurity.encryption : {};
  const compressionInfo = isRecord(experimentSecurity.compression) ? experimentSecurity.compression : {};

  function notifyError(error: unknown, id?: string): void {
    const message = toErrorMessage(error);
    toast.error(message, { id });
  }

  async function persistSelectedJobConfig(config: Record<string, unknown>, stableConfig: string): Promise<void> {
    if (!selectedJobId) return;
    const saveSeq = ++configSaveSeqRef.current;
    await jobsApi.updateConfig(selectedJobId, config);
    if (saveSeq !== configSaveSeqRef.current) return;
    lastSavedConfigRef.current = stableConfig;
  }

  async function refreshJobs(): Promise<void> {
    const data = await jobsApi.list();
    setJobs(data);
    setSelectedJobId((current) => {
      if (current !== null && data.some((job) => job.id === current)) return current;
      return data.length > 0 ? data[0].id : null;
    });
  }

  async function refreshRuns(): Promise<void> {
    const data = await runsApi.list(500);
    setRuns(data);
    setSelectedRunId((current) => {
      if (current && data.some((run) => run.id === current)) return current;
      return data.length > 0 ? data[0].id : null;
    });
  }

  async function loadJobEditor(jobId: number): Promise<void> {
    const config = await jobsApi.getConfig(jobId);
    const configText = JSON.stringify(config.config_json, null, 2);
    setJobConfigText(configText);

    try {
      const parsed = parseJsonObject(configText);
      lastSavedConfigRef.current = stableStringify(parsed);
      const topology = topologyFromConfig(parsed, configSchema);
      setTopologyNodes(topology.nodes);
      setSystemMode(topology.mode);
      setSelectedTopologyNodeId(null);
    } catch {
      lastSavedConfigRef.current = null;
      setTopologyNodes(makeInitialTopologyNodes(configSchema, "simulation"));
      setSystemMode("simulation");
    }
  }

  async function loadRunBundle(runId: string): Promise<void> {
    const [run, logs] = await Promise.all([jobsApi.getRun(runId), jobsApi.listRunLogs(runId, 500)]);
    setRunDetails(run);
    setRunLogs(logs);
    setRuns((previous) => {
      const index = previous.findIndex((item) => item.id === run.id);
      if (index < 0) return [run, ...previous];
      const next = [...previous];
      next[index] = run;
      return next;
    });
  }

  async function loadRunMetrics(runId: string): Promise<void> {
    const metrics = await runsApi.getMetrics(runId);
    setRunMetrics(metrics);
  }

  useEffect(() => {
    const initialTopology = makeInitialTopologyNodes(null, "simulation");
    const initialConfig = buildConfigFromTopology(initialTopology, "simulation", null);
    setJobConfigText(JSON.stringify(initialConfig, null, 2));
  }, []);

  useEffect(() => {
    Promise.all([
      refreshJobs(),
      refreshRuns(),
      jobsApi.getConfigSchema().then((schema) => setConfigSchema(schema)),
    ]).catch((err: unknown) => notifyError(err, "bootstrap-error"));
  }, []);

  useEffect(() => {
    if (selectedJobId === null) {
      jobHydrationSeqRef.current += 1;
      isHydratingJobRef.current = false;
      if (configAutoSaveTimerRef.current !== null) {
        window.clearTimeout(configAutoSaveTimerRef.current);
        configAutoSaveTimerRef.current = null;
      }
      lastSavedConfigRef.current = null;
      const initial = makeInitialTopologyNodes(configSchema, "simulation");
      const nextConfig = buildConfigFromTopology(initial, "simulation", configSchema);
      setJobConfigText(JSON.stringify(nextConfig, null, 2));
      setConfigJsonError(null);
      setTopologyNodes(initial);
      setSystemMode("simulation");
      return;
    }

    const hydrationSeq = ++jobHydrationSeqRef.current;
    isHydratingJobRef.current = true;

    loadJobEditor(selectedJobId)
      .catch((err: unknown) => notifyError(err, "job-editor-error"))
      .finally(() => {
        if (hydrationSeq === jobHydrationSeqRef.current) {
          isHydratingJobRef.current = false;
        }
      });
  }, [selectedJobId, configSchema]);

  useEffect(() => {
    if (!selectedJob) return;
    if (isHydratingJobRef.current) return;
    if (configSyncSourceRef.current === "json") {
      configSyncSourceRef.current = null;
      return;
    }

    const nextConfig = buildConfigFromTopology(topologyNodes, systemMode, configSchema);
    const nextText = JSON.stringify(nextConfig, null, 2);
    if (nextText !== jobConfigText) {
      configSyncSourceRef.current = "topology";
      setJobConfigText(nextText);
      setConfigJsonError(null);
    }
  }, [topologyNodes, systemMode, configSchema, selectedJob, jobConfigText]);

  useEffect(() => {
    if (!selectedJob) return;
    if (isHydratingJobRef.current) return;
    if (configSyncSourceRef.current === "topology") {
      configSyncSourceRef.current = null;
      return;
    }

    try {
      const parsed = parseJsonObject(jobConfigText);
      const currentConfig = buildConfigFromTopology(topologyNodes, systemMode, configSchema);
      if (stableStringify(parsed) === stableStringify(currentConfig)) {
        setConfigJsonError(null);
        return;
      }

      const topology = topologyFromConfig(parsed, configSchema);
      configSyncSourceRef.current = "json";
      setTopologyNodes(topology.nodes);
      setSystemMode(topology.mode);
      setSelectedTopologyNodeId(null);
      setConfigJsonError(null);
    } catch {
      setConfigJsonError("Invalid JSON. Topology updates when JSON becomes valid.");
    }
  }, [jobConfigText, selectedJob, configSchema]);

  useEffect(() => {
    if (!selectedJobId) return;
    if (isHydratingJobRef.current) return;

    let parsed: Record<string, unknown>;
    try {
      parsed = parseJsonObject(jobConfigText);
    } catch {
      return;
    }

    const stableConfig = stableStringify(parsed);
    if (stableConfig === lastSavedConfigRef.current) return;

    if (configAutoSaveTimerRef.current !== null) {
      window.clearTimeout(configAutoSaveTimerRef.current);
    }

    configAutoSaveTimerRef.current = window.setTimeout(() => {
      void persistSelectedJobConfig(parsed, stableConfig).catch((err: unknown) =>
        notifyError(err, "config-autosave-error"),
      );
    }, 600);

    return () => {
      if (configAutoSaveTimerRef.current !== null) {
        window.clearTimeout(configAutoSaveTimerRef.current);
        configAutoSaveTimerRef.current = null;
      }
    };
  }, [jobConfigText, selectedJobId]);

  useEffect(() => {
    if (!selectedRunId) {
      setRunDetails(null);
      setRunLogs([]);
      setRunMetrics(EMPTY_RUN_METRICS);
      return;
    }

    Promise.all([loadRunBundle(selectedRunId), loadRunMetrics(selectedRunId)]).catch((err: unknown) =>
      notifyError(err, "run-load-error"),
    );
  }, [selectedRunId]);

  useEffect(() => {
    if (!selectedRunId || !selectedRun) return;
    if (!["queued", "running"].includes(selectedRun.status)) return;

    const timer = window.setInterval(() => {
      Promise.all([refreshRuns(), loadRunBundle(selectedRunId), loadRunMetrics(selectedRunId)]).catch((err: unknown) =>
        notifyError(err, "run-refresh-error"),
      );
    }, 2000);

    return () => window.clearInterval(timer);
  }, [selectedRunId, selectedRun]);

  useEffect(() => {
    if (activeTab !== "runs") return;
    if (!runLogsRef.current) return;

    const raf = window.requestAnimationFrame(() => {
      if (!runLogsRef.current) return;
      runLogsRef.current.scrollTop = runLogsRef.current.scrollHeight;
    });

    return () => window.cancelAnimationFrame(raf);
  }, [runLogs, selectedRunId, activeTab]);

  useEffect(() => {
    function onMouseMove(event: MouseEvent): void {
      if (!draggingNodeRef.current || !canvasRef.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left - draggingNodeRef.current.offsetX;
      const y = event.clientY - rect.top - draggingNodeRef.current.offsetY;

      const clampedX = Math.max(0, Math.min(rect.width - NODE_SIZE.width, x));
      const clampedY = Math.max(0, Math.min(rect.height - NODE_SIZE.height, y));

      setTopologyNodes((previous) =>
        previous.map((node) => (node.id === draggingNodeRef.current?.id ? { ...node, x: clampedX, y: clampedY } : node)),
      );
    }

    function onMouseUp(): void {
      draggingNodeRef.current = null;
    }

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, []);

  function patchSelectedTopologyNode(patch: Partial<TopologyNode>): void {
    if (!selectedTopologyNode) return;
    patchTopologyNode(selectedTopologyNode.id, patch);
  }

  function setSelectedNodeProperty(path: string, value: unknown): void {
    if (!selectedTopologyNode) return;
    setTopologyNodeProperty(selectedTopologyNode.id, path, value);
  }

  function normalizeTopologyNode(node: TopologyNode): TopologyNode {
    const nextProperties = structuredClone(node.properties);
    if (node.role !== "client") {
      setValueByPath(nextProperties, "system.node_role", "server");
      return { ...node, properties: nextProperties, clientId: null, label: "Server" };
    }

    const nextId = normalizeClientId(node.clientId ?? getValueByPath(nextProperties, "distributed.client_id"));
    setValueByPath(nextProperties, "distributed.client_id", nextId);
    setValueByPath(nextProperties, "system.node_role", "client");
    return { ...node, properties: nextProperties, clientId: nextId, label: `Client-${nextId}` };
  }

  function patchTopologyNode(nodeId: string, patch: Partial<TopologyNode>): void {
    setTopologyNodes((previous) =>
      previous.map((node) => (node.id === nodeId ? normalizeTopologyNode({ ...node, ...patch }) : node)),
    );
  }

  function setTopologyNodeProperty(nodeId: string, path: string, value: unknown): void {
    setTopologyNodes((previous) =>
      previous.map((node) => {
        if (node.id !== nodeId) return node;
        const nextProperties = structuredClone(node.properties);
        setValueByPath(nextProperties, path, value);
        return normalizeTopologyNode({ ...node, properties: nextProperties });
      }),
    );
  }

  function applyTopologyNodeConfig(nodeId: string, properties: Record<string, unknown>): void {
    patchTopologyNode(nodeId, { properties: structuredClone(properties) });
  }

  function onToolDragStart(event: DragEvent<HTMLButtonElement>, role: "server" | "client"): void {
    event.dataTransfer.setData("application/x-topology-role", role);
    event.dataTransfer.effectAllowed = "copy";
  }

  function onCanvasDragOver(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    setCanvasHighlighted(true);
  }

  function onCanvasDragLeave(): void {
    setCanvasHighlighted(false);
  }

  function onCanvasDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    setCanvasHighlighted(false);

    const role = event.dataTransfer.getData("application/x-topology-role");
    if (role !== "server" && role !== "client") return;
    if (!canvasRef.current) return;

    if (role === "server" && topologyNodes.some((node) => node.role === "server")) {
      toast.warning("Only one server node is allowed.");
      return;
    }

    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width - NODE_SIZE.width, event.clientX - rect.left - NODE_SIZE.width / 2));
    const y = Math.max(0, Math.min(rect.height - NODE_SIZE.height, event.clientY - rect.top - NODE_SIZE.height / 2));

    if (role === "server") {
      const node: TopologyNode = {
        id: uid("server"),
        role: "server",
        label: "Server",
        x,
        y,
        clientId: null,
        properties: createNodeProperties(configSchema, "server", systemMode, null),
      };
      setTopologyNodes((previous) => [...previous, node]);
      setSelectedTopologyNodeId(node.id);
      return;
    }

    const nextId = nextClientId(topologyNodes);
    const node: TopologyNode = {
      id: uid("client"),
      role: "client",
      label: `Client-${nextId}`,
      x,
      y,
      clientId: nextId,
      properties: createNodeProperties(configSchema, "client", systemMode, nextId),
    };
    setTopologyNodes((previous) => [...previous, node]);
    setSelectedTopologyNodeId(node.id);
  }

  function onTopologyNodeMouseDown(event: ReactMouseEvent<HTMLButtonElement>, node: TopologyNode): void {
    if (event.button !== 0) return;
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    draggingNodeRef.current = {
      id: node.id,
      offsetX: event.clientX - rect.left - node.x,
      offsetY: event.clientY - rect.top - node.y,
    };
    setSelectedTopologyNodeId(node.id);
  }

  function deleteSelectedTopologyNode(): void {
    if (!selectedTopologyNode) return;
    deleteTopologyNode(selectedTopologyNode.id);
  }

  function cloneSelectedTopologyNode(): void {
    if (!selectedTopologyNode) return;
    cloneTopologyNode(selectedTopologyNode.id);
  }

  function deleteTopologyNode(nodeId: string): void {
    const target = topologyNodes.find((node) => node.id === nodeId);
    if (!target) return;
    if (target.role === "server") {
      toast.warning("Server node cannot be deleted.");
      return;
    }

    setTopologyNodes((previous) => previous.filter((node) => node.id !== nodeId));
    setSelectedTopologyNodeId((current) => (current === nodeId ? null : current));
  }

  function cloneTopologyNode(nodeId: string): void {
    const source = topologyNodes.find((node) => node.id === nodeId);
    if (!source || source.role !== "client") return;

    const nextId = nextClientId(topologyNodes);
    const node: TopologyNode = {
      ...source,
      id: uid("client"),
      label: `Client-${nextId}`,
      clientId: nextId,
      x: source.x + 20,
      y: source.y + 20,
      properties: (() => {
        const nextProperties = structuredClone(source.properties);
        setValueByPath(nextProperties, "distributed.client_id", nextId);
        setValueByPath(nextProperties, "system.node_role", "client");
        return nextProperties;
      })(),
    };

    setTopologyNodes((previous) => [...previous, node]);
    setSelectedTopologyNodeId(node.id);
  }

  async function handleCreateJob(): Promise<void> {
    const name = newJobName.trim();
    const description = newJobDescription.trim();
    if (!name || !description) {
      toast.warning("Job name and description are required.");
      return;
    }

    setBusy(true);
    try {
      const created = await jobsApi.create({ name, description });
      await refreshJobs();
      setSelectedJobId(created.id);
      setNewJobName("");
      setNewJobDescription("");
      toast.success(`Job "${created.name}" created.`);
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  function openJobMetadataDialog(job: SimulationPageProps["jobs"][number]): void {
    setEditingJobMetaId(job.id);
    setEditingJobName(job.name);
    setEditingJobDescription(job.description ?? "");
    setJobMetaDialogOpen(true);
  }

  async function handleSaveJobMeta(): Promise<void> {
    if (editingJobMetaId === null) return;

    const name = editingJobName.trim();
    if (!name) {
      toast.warning("Job name is required.");
      return;
    }

    setBusy(true);
    try {
      await jobsApi.update(editingJobMetaId, { name, description: editingJobDescription.trim() || null });
      await refreshJobs();
      setJobMetaDialogOpen(false);
      setEditingJobMetaId(null);
      toast.success("Job metadata saved.");
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleDuplicateJob(job: SimulationPageProps["jobs"][number]): Promise<void> {
    setBusy(true);
    try {
      const created = await jobsApi.copy(job.id, { name: `${job.name}-copy` });
      await refreshJobs();
      setSelectedJobId(created.id);
      toast.success(`Job copied as "${created.name}".`);
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  function requestDeleteJob(job: SimulationPageProps["jobs"][number]): void {
    setDeleteJobTarget(job);
  }

  async function handleDeleteJob(target: SimulationPageProps["jobs"][number]): Promise<void> {
    setBusy(true);
    try {
      await jobsApi.remove(target.id);
      await Promise.all([refreshJobs(), refreshRuns()]);
      setSelectedTopologyNodeId(null);
      setDeleteJobTarget(null);
      toast.success(`Job "${target.name}" deleted.`);
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleStartRun(): Promise<void> {
    if (!selectedJobId) return;

    setBusy(true);
    try {
      let parsedConfig: Record<string, unknown>;
      try {
        parsedConfig = parseJsonObject(jobConfigText);
      } catch {
        toast.warning("Config JSON is invalid. Fix it before starting a run.");
        return;
      }

      const stableConfig = stableStringify(parsedConfig);
      if (stableConfig !== lastSavedConfigRef.current) {
        if (configAutoSaveTimerRef.current !== null) {
          window.clearTimeout(configAutoSaveTimerRef.current);
          configAutoSaveTimerRef.current = null;
        }
        await persistSelectedJobConfig(parsedConfig, stableConfig);
      }

      const started = await jobsApi.run(selectedJobId);
      await refreshRuns();
      setSelectedRunId(started.id);
      setActiveTab("runs");
      await Promise.all([loadRunBundle(started.id), loadRunMetrics(started.id)]);
      toast.success(`Run #${shortRunId(started.id)} started.`);
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleStopRun(): Promise<void> {
    if (!selectedRunId) return;

    setBusy(true);
    try {
      await jobsApi.stopRun(selectedRunId);
      await Promise.all([refreshRuns(), loadRunBundle(selectedRunId), loadRunMetrics(selectedRunId)]);
      toast.warning(`Run #${shortRunId(selectedRunId)} stop requested.`);
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleRerun(): Promise<void> {
    if (!selectedRunId) return;

    setBusy(true);
    try {
      const created = await jobsApi.rerunRun(selectedRunId);
      await refreshRuns();
      setSelectedRunId(created.id);
      await Promise.all([loadRunBundle(created.id), loadRunMetrics(created.id)]);
      toast.success(`Rerun created: #${shortRunId(created.id)}.`);
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleDeleteRun(): Promise<void> {
    if (!selectedRunId) return;

    setBusy(true);
    try {
      await jobsApi.deleteRun(selectedRunId);
      await refreshRuns();
      setSelectedRunId((current) => (current === selectedRunId ? null : current));
      setRunDetails(null);
      setRunLogs([]);
      setRunMetrics(EMPTY_RUN_METRICS);
      toast.success(`Run #${shortRunId(selectedRunId)} deleted.`);
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  return {
    MiniLineChart,
    activeTab,
    busy,
    canvasHighlighted,
    canvasRef,
    applyTopologyNodeConfig,
    clientNodes,
    cloneTopologyNode,
    clientTestAccSeries,
    clientTestLossSeries,
    clientTrainAccSeries,
    clientTrainLossSeries,
    cloneSelectedTopologyNode,
    compressionInfo,
    configJsonError,
    configSchema,
    deleteJobTarget,
    deleteSelectedTopologyNode,
    deleteTopologyNode,
    editingJobDescription,
    editingJobName,
    encryptionInfo,
    experimentBasic,
    experimentFederated,
    globalAccuracySeries,
    globalLossSeries,
    handleCreateJob,
    handleDeleteJob,
    handleDeleteRun,
    handleDuplicateJob,
    handleRerun,
    handleSaveJobMeta,
    handleStartRun,
    handleStopRun,
    isRecord,
    jobConfigText,
    jobMetaDialogOpen,
    jobStatusVariant,
    jobs,
    loadRunBundle,
    loadRunMetrics,
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
    patchSelectedTopologyNode,
    rectEdgePoint,
    refreshJobs,
    refreshRuns,
    renderSchemaSection,
    requestDeleteJob,
    runLogs,
    runLogsRef,
    runRounds,
    runStatusVariant,
    runs,
    selectedJob,
    selectedJobId,
    selectedRun,
    selectedRunId,
    selectedTopologyNode,
    selectedTopologyNodeId,
    serverNode,
    setActiveTab,
    setDeleteJobTarget,
    setEditingJobDescription,
    setEditingJobMetaId,
    setEditingJobName,
    setJobConfigText,
    setJobMetaDialogOpen,
    setNewJobDescription,
    setNewJobName,
    setSelectedJobId,
    setSelectedNodeProperty,
    setTopologyNodeProperty,
    setSelectedRunId,
    setSelectedTopologyNodeId,
    shortRunId,
    toDisplayText,
    toNumber,
    topologyCanvasMinHeight,
    topologyNodes,
  };
}
