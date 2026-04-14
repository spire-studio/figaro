import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { distributedApi, type DistributedJob } from "../../api/distributed";
import { renderSchemaSection } from "../config/render-schema-section";
import {
  DEFAULT_DISTRIBUTED_GRPC_HOST,
  DEFAULT_DISTRIBUTED_GRPC_PORT,
  DEFAULT_DISTRIBUTED_SERVER_API_BASE,
  buildConfigFromSchema,
  createNodeProperties,
  distributedClientStatusVariant,
  distributedJobStatusVariant,
  distributedSessionStatusVariant,
  getValueByPath,
  isRecord,
  setValueByPath,
  toErrorMessage,
  toNumber,
} from "../simulation/utils";
import type { DistributedPageProps, PageMode } from "../../pages/types";

type UseDistributedControllerArgs = {
  pageMode: PageMode;
  configSchema: Record<string, unknown> | null;
};

export function useDistributedController({
  pageMode,
  configSchema,
}: UseDistributedControllerArgs): DistributedPageProps {
  const [busy, setBusy] = useState(false);
  const [distributedRole, setDistributedRole] = useState<DistributedPageProps["distributedRole"]>("server");
  const [distributedJobs, setDistributedJobs] = useState<DistributedPageProps["distributedJobs"]>([]);
  const [selectedDistributedJobId, setSelectedDistributedJobId] = useState<number | null>(null);
  const [distributedSession, setDistributedSession] = useState<DistributedPageProps["distributedSession"]>(null);
  const [distributedClients, setDistributedClients] = useState<DistributedPageProps["distributedClients"]>([]);
  const [distributedJobName, setDistributedJobName] = useState("");
  const [distributedJobDescription, setDistributedJobDescription] = useState("");
  const [distributedConfigDialogOpen, setDistributedConfigDialogOpen] = useState(false);
  const [distributedConfigJobId, setDistributedConfigJobId] = useState<number | null>(null);
  const [distributedConfigDraft, setDistributedConfigDraft] = useState<Record<string, unknown> | null>(null);
  const [distributedConfigClients, setDistributedConfigClients] = useState(1);
  const [distributedServerIp, setDistributedServerIp] = useState(DEFAULT_DISTRIBUTED_GRPC_HOST);
  const [distributedServerPort, setDistributedServerPort] = useState(DEFAULT_DISTRIBUTED_GRPC_PORT);
  const [distributedSessionProgress, setDistributedSessionProgress] = useState<DistributedPageProps["distributedSessionProgress"]>(null);

  const [clientServerApiBase, setClientServerApiBase] = useState(DEFAULT_DISTRIBUTED_SERVER_API_BASE);
  const [clientSessionId, setClientSessionId] = useState("");
  const [clientName, setClientName] = useState("");
  const [clientConnecting, setClientConnecting] = useState(false);
  const [connectedClientId, setConnectedClientId] = useState<string | null>(null);
  const [connectedClientInfo, setConnectedClientInfo] = useState<DistributedPageProps["connectedClientInfo"]>(null);
  const [clientConnectionMessage, setClientConnectionMessage] = useState("Idle");

  const clientConnectCancelledRef = useRef(false);
  const clientConnectTimerRef = useRef<number | null>(null);
  const clientStatusPollRef = useRef<number | null>(null);

  const selectedDistributedJob = useMemo(
    () => distributedJobs.find((job) => job.id === selectedDistributedJobId) ?? null,
    [distributedJobs, selectedDistributedJobId],
  );

  const approvedDistributedSlots = useMemo(() => {
    return new Set(
      distributedClients
        .filter((item) => ["approved", "ready", "running"].includes(item.status))
        .map((item) => item.assigned_participant_id)
        .filter((item): item is number => item !== null),
    );
  }, [distributedClients]);
  const readyDistributedSlots = useMemo(() => {
    return new Set(
      distributedClients
        .filter((item) => ["ready", "running"].includes(item.status))
        .map((item) => item.assigned_participant_id)
        .filter((item): item is number => item !== null),
    );
  }, [distributedClients]);
  const allDistributedClientsReady = selectedDistributedJob
    ? readyDistributedSlots.size >= selectedDistributedJob.expected_clients
    : false;

  function notifyError(error: unknown, id?: string): void {
    toast.error(toErrorMessage(error), { id });
  }

  function clearClientConnectTimer(): void {
    if (clientConnectTimerRef.current !== null) {
      window.clearTimeout(clientConnectTimerRef.current);
      clientConnectTimerRef.current = null;
    }
  }

  function clearClientStatusPoll(): void {
    if (clientStatusPollRef.current !== null) {
      window.clearInterval(clientStatusPollRef.current);
      clientStatusPollRef.current = null;
    }
  }

  function resetClientConnectionState(message = "Idle"): void {
    clearClientConnectTimer();
    clearClientStatusPoll();
    setClientConnecting(false);
    setConnectedClientId(null);
    setConnectedClientInfo(null);
    setClientConnectionMessage(message);
  }

  async function refreshDistributedJobs(): Promise<void> {
    const data = await distributedApi.listJobs();
    setDistributedJobs(data);
    setSelectedDistributedJobId((current) => {
      if (current !== null && data.some((job) => job.id === current)) return current;
      return data.length > 0 ? data[0].id : null;
    });
  }

  async function refreshDistributedSessionBundle(sessionId: string): Promise<void> {
    const [sessionData, clientsData] = await Promise.all([
      distributedApi.getSession(sessionId),
      distributedApi.listSessionClients(sessionId),
    ]);
    setDistributedSession(sessionData);
    setDistributedClients(clientsData);
  }

  async function refreshDistributedSessionProgress(sessionId: string): Promise<void> {
    const progress = await distributedApi.getSessionProgress(sessionId);
    setDistributedSessionProgress(progress);
  }

  function resolveDistributedConfigClientCount(config: Record<string, unknown>, fallback = 1): number {
    const fromFederated = toNumber(getValueByPath(config, "federated.num_clients"));
    const fromDataset = toNumber(getValueByPath(config, "dataset.num_clients"));
    const raw = fromFederated ?? fromDataset ?? fallback;
    const normalized = Number.isFinite(raw) ? Math.floor(Number(raw)) : fallback;
    return Math.max(1, normalized);
  }

  function openDistributedConfigDialog(job: DistributedJob): void {
    if (!configSchema) {
      toast.warning("Config schema is not loaded yet.");
      return;
    }
    const defaults = createNodeProperties(configSchema, "server", "distributed", null);
    const merged = deepMergeConfig(
      defaults,
      isRecord(job.config_json) ? (job.config_json as Record<string, unknown>) : {},
    );
    setValueByPath(merged, "system.mode", "distributed");
    setValueByPath(merged, "system.node_role", "server");
    const nextClients = resolveDistributedConfigClientCount(merged, job.expected_clients);
    setValueByPath(merged, "federated.num_clients", nextClients);
    setValueByPath(merged, "dataset.num_clients", nextClients);

    setDistributedConfigJobId(job.id);
    setDistributedConfigDraft(merged);
    setDistributedConfigClients(nextClients);
    setDistributedConfigDialogOpen(true);
  }

  function setDistributedConfigProperty(path: string, value: unknown): void {
    setDistributedConfigDraft((previous) => {
      if (!previous) return previous;
      const next = structuredClone(previous);
      setValueByPath(next, path, value);
      return next;
    });
  }

  async function handleSaveDistributedJobConfig(): Promise<void> {
    if (!distributedConfigJobId || !distributedConfigDraft || !configSchema) return;
    const clientCount = Math.max(1, Math.floor(distributedConfigClients));
    setBusy(true);
    try {
      const nextConfig = buildConfigFromSchema(configSchema, distributedConfigDraft);
      setValueByPath(nextConfig, "system.mode", "distributed");
      setValueByPath(nextConfig, "system.node_role", "server");
      setValueByPath(nextConfig, "federated.num_clients", clientCount);
      setValueByPath(nextConfig, "dataset.num_clients", clientCount);
      const updated = await distributedApi.updateJobConfig(distributedConfigJobId, { config: nextConfig });
      await refreshDistributedJobs();
      setSelectedDistributedJobId(updated.id);
      setDistributedServerIp(String(getValueByPath(nextConfig, "distributed.server_ip") ?? distributedServerIp));
      setDistributedServerPort(String(getValueByPath(nextConfig, "distributed.port") ?? distributedServerPort));
      setDistributedConfigDialogOpen(false);
      toast.success(`Job "${updated.name}" config updated.`);
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleCreateDistributedJob(): Promise<void> {
    const name = distributedJobName.trim();
    if (name.length === 0) {
      toast.warning("Distributed job name is required.");
      return;
    }

    setBusy(true);
    try {
      const created = await distributedApi.createJob({
        name,
        description: distributedJobDescription.trim() || null,
      });
      await refreshDistributedJobs();
      setSelectedDistributedJobId(created.id);
      setDistributedSession(null);
      setDistributedClients([]);
      setDistributedJobName("");
      setDistributedJobDescription("");
      toast.success(`Distributed job "${created.name}" created.`);
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleCreateOrAttachDistributedSession(): Promise<void> {
    if (!selectedDistributedJobId) return;
    const serverIp = distributedServerIp.trim() || DEFAULT_DISTRIBUTED_GRPC_HOST;
    const parsedPort = Number(distributedServerPort);
    if (!Number.isFinite(parsedPort) || parsedPort <= 0) {
      toast.warning("Server port must be a positive number.");
      return;
    }

    setBusy(true);
    try {
      const sessionData = await distributedApi.createOrGetSession(selectedDistributedJobId, {
        server_ip: serverIp,
        server_port: Math.floor(parsedPort),
      });
      setDistributedSession(sessionData);
      await refreshDistributedSessionBundle(sessionData.id);
      await refreshDistributedSessionProgress(sessionData.id);
      toast.success(`Session #${sessionData.id} is waiting for clients.`);
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleApproveDistributedClient(clientId: string): Promise<void> {
    setBusy(true);
    try {
      await distributedApi.approveClient(clientId);
      if (distributedSession) {
        await refreshDistributedSessionBundle(distributedSession.id);
      }
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleRejectDistributedClient(clientId: string): Promise<void> {
    setBusy(true);
    try {
      await distributedApi.rejectClient(clientId);
      if (distributedSession) {
        await refreshDistributedSessionBundle(distributedSession.id);
      }
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleStartDistributedTraining(): Promise<void> {
    if (!distributedSession) return;
    setBusy(true);
    try {
      const started = await distributedApi.startSession(distributedSession.id);
      setDistributedSession(started);
      await refreshDistributedSessionBundle(started.id);
      await refreshDistributedSessionProgress(started.id);
      toast.success("Distributed session started.");
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  function scheduleClientConnectAttempt(): void {
    clearClientConnectTimer();
    clientConnectTimerRef.current = window.setTimeout(() => {
      void runClientConnectAttempt();
    }, 2000);
  }

  async function runClientConnectAttempt(): Promise<void> {
    if (clientConnectCancelledRef.current) return;
    const base = clientServerApiBase.trim();
    const sessionId = clientSessionId.trim();
    const name = clientName.trim();
    if (!base || !sessionId || !name) {
      setClientConnecting(false);
      setClientConnectionMessage("Server API / Session ID / Client Name are required.");
      return;
    }

    try {
      const connected = await distributedApi.requestConnect(
        sessionId,
        { participant_name: name, metadata_json: { source: "distributed-client-ui" } },
        base,
      );
      setClientConnecting(false);
      setConnectedClientId(connected.id);
      setConnectedClientInfo(connected);
      setClientConnectionMessage("Connected. Waiting for server approval...");
      clearClientStatusPoll();
      clientStatusPollRef.current = window.setInterval(() => {
        void distributedApi
          .getClient(connected.id, base)
          .then((latest) => {
            setConnectedClientInfo(latest);
            if (latest.status === "approved") {
              setClientConnectionMessage("Approved by server. Configure local client and click Ready.");
            } else if (latest.status === "ready") {
              setClientConnectionMessage("Ready. Waiting for server to start training.");
            } else if (latest.status === "running") {
              setClientConnectionMessage("Server started training.");
            } else if (latest.status === "rejected") {
              setClientConnectionMessage("Rejected by server.");
            } else if (latest.status === "cancelled") {
              setClientConnectionMessage("Connection cancelled.");
            } else {
              setClientConnectionMessage("Connected. Waiting for server approval...");
            }
          })
          .catch((err: unknown) => {
            setClientConnectionMessage(`Polling failed: ${toErrorMessage(err)}`);
          });
      }, 2000);
    } catch (err: unknown) {
      if (clientConnectCancelledRef.current) return;
      setClientConnectionMessage(`Connect failed (${toErrorMessage(err)}). Retrying in 2s...`);
      scheduleClientConnectAttempt();
    }
  }

  async function handleClientConnect(): Promise<void> {
    clientConnectCancelledRef.current = false;
    clearClientStatusPoll();
    setConnectedClientId(null);
    setConnectedClientInfo(null);
    setClientConnecting(true);
    setClientConnectionMessage("Connecting to server...");
    await runClientConnectAttempt();
  }

  async function handleClientCancelConnect(): Promise<void> {
    clientConnectCancelledRef.current = true;
    clearClientConnectTimer();
    const targetClientId = connectedClientId;
    const targetBase = clientServerApiBase.trim();
    resetClientConnectionState("Connection cancelled.");
    if (targetClientId) {
      try {
        await distributedApi.cancelClient(targetClientId, targetBase);
      } catch {
        // ignore
      }
    }
  }

  async function handleClientReady(): Promise<void> {
    if (!connectedClientId) return;
    const serverBase = clientServerApiBase.trim();
    if (!serverBase) {
      toast.warning("Server API Base URL is required.");
      return;
    }
    setBusy(true);
    try {
      const runtimeConfig = await distributedApi.getClientRuntimeConfig(connectedClientId, serverBase);
      await distributedApi.startLocalClientRuntime({
        participant_id: connectedClientId,
        config_json: runtimeConfig.config_json,
      });

      const readyClient = await distributedApi.readyClient(connectedClientId, serverBase);
      setConnectedClientInfo(readyClient);
      setClientConnectionMessage("Ready. Local client runtime started, waiting for server to start training.");
      toast.success("Client runtime started and marked ready.");
    } catch (err: unknown) {
      notifyError(err);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (pageMode !== "distributed" || distributedRole !== "server") return;
    refreshDistributedJobs().catch((err: unknown) => notifyError(err, "distributed-jobs-load-error"));
  }, [pageMode, distributedRole]);

  useEffect(() => {
    let cancelled = false;
    setDistributedSession(null);
    setDistributedClients([]);
    if (!selectedDistributedJob) return;
    const jobConfig = isRecord(selectedDistributedJob.config_json)
      ? (selectedDistributedJob.config_json as Record<string, unknown>)
      : {};
    const configuredIp = String(getValueByPath(jobConfig, "distributed.server_ip") ?? "").trim();
    const nextIp =
      configuredIp.length > 0 && configuredIp !== "127.0.0.1" && configuredIp !== "localhost"
        ? configuredIp
        : DEFAULT_DISTRIBUTED_GRPC_HOST;
    const configuredPort = String(getValueByPath(jobConfig, "distributed.port") ?? "").trim();
    const nextPort = configuredPort.length > 0 ? configuredPort : DEFAULT_DISTRIBUTED_GRPC_PORT;
    setDistributedServerIp(nextIp);
    setDistributedServerPort(nextPort);
    setDistributedSessionProgress(null);

    if (pageMode !== "distributed" || distributedRole !== "server") {
      return () => {
        cancelled = true;
      };
    }

    void (async () => {
      try {
        const activeSession = await distributedApi.getActiveSession(selectedDistributedJob.id);
        if (!activeSession || cancelled) return;
        setDistributedSession(activeSession);
        await Promise.all([
          refreshDistributedSessionBundle(activeSession.id),
          refreshDistributedSessionProgress(activeSession.id),
        ]);
      } catch (err: unknown) {
        if (cancelled) return;
        notifyError(err, "distributed-session-restore-error");
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedDistributedJobId, pageMode, distributedRole]);

  useEffect(() => {
    if (pageMode !== "distributed" || distributedRole !== "server" || !distributedSession) return;
    const timer = window.setInterval(() => {
      void Promise.all([
        refreshDistributedSessionBundle(distributedSession.id),
        refreshDistributedSessionProgress(distributedSession.id),
      ]).catch((err: unknown) => notifyError(err, "distributed-session-refresh-error"));
    }, 2000);
    return () => window.clearInterval(timer);
  }, [pageMode, distributedRole, distributedSession]);

  useEffect(() => {
    if (distributedRole === "client") {
      setSelectedDistributedJobId(null);
      setDistributedSession(null);
      setDistributedClients([]);
      setDistributedSessionProgress(null);
    }
  }, [distributedRole]);

  useEffect(() => {
    if (pageMode !== "distributed") {
      clientConnectCancelledRef.current = true;
      resetClientConnectionState("Idle");
    }
  }, [pageMode]);

  useEffect(() => {
    return () => {
      clearClientConnectTimer();
      clearClientStatusPoll();
    };
  }, []);

  return {
    allDistributedClientsReady,
    approvedDistributedSlots,
    busy,
    clientConnecting,
    clientConnectionMessage,
    clientName,
    clientServerApiBase,
    clientSessionId,
    configSchema,
    connectedClientId,
    connectedClientInfo,
    distributedClientStatusVariant,
    distributedClients,
    distributedConfigClients,
    distributedConfigDialogOpen,
    distributedConfigDraft,
    distributedJobDescription,
    distributedJobName,
    distributedJobStatusVariant,
    distributedJobs,
    distributedRole,
    distributedServerIp,
    distributedServerPort,
    distributedSession,
    distributedSessionProgress,
    distributedSessionStatusVariant,
    handleApproveDistributedClient,
    handleClientCancelConnect,
    handleClientConnect,
    handleClientReady,
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
    setClientName,
    setClientServerApiBase,
    setClientSessionId,
    setDistributedConfigClients,
    setDistributedConfigDialogOpen,
    setDistributedConfigProperty,
    setDistributedJobDescription,
    setDistributedJobName,
    setDistributedRole,
    setDistributedServerIp,
    setDistributedServerPort,
    setSelectedDistributedJobId,
  };
}

function deepMergeConfig(base: Record<string, unknown>, override: Record<string, unknown>): Record<string, unknown> {
  const output = structuredClone(base);
  for (const [key, value] of Object.entries(override)) {
    if (isRecord(value) && isRecord(output[key])) {
      output[key] = deepMergeConfig(output[key] as Record<string, unknown>, value);
      continue;
    }
    output[key] = value;
  }
  return output;
}
