import type {
  Dispatch,
  DragEvent as ReactDragEvent,
  MouseEvent as ReactMouseEvent,
  ReactNode,
  RefObject,
  SetStateAction,
} from "react";

import type { DistributedClient, DistributedJob, DistributedSession, DistributedSessionProgress } from "../api/distributed";
import type {
  AgentExperimentResponse,
  AgentOptimizationJobSummary,
  AgentOptimizeProgressResponse,
  AgentOptimizeResponse,
  AgentOptimizationObjective,
  AgentRunResponse,
} from "../api/agent";
import type { components } from "../api/openapi";
import type { RunMetrics } from "../api/runs";

export type Job = components["schemas"]["JobResponse"];
export type Run = components["schemas"]["RunResponse"];
export type RunLog = components["schemas"]["RunLogResponse"];

export type TabKey = "jobs" | "runs";
export type DistributedRole = "server" | "client";
export type PageMode = "simulation" | "distributed" | "agent";
export type NodeRole = "server" | "client";
export type SystemMode = "simulation" | "distributed";
export type BadgeVariant = "default" | "secondary" | "outline" | "success" | "warning" | "danger";

export type AgentWorkflowStep = "home" | "preview" | "running" | "results";

export type AgentPlanDraft = {
  job_id: number;
  goal: string;
  experiments: any[];
  config_constraints?: Record<string, unknown>;
};

export type Point = {
  x: number;
  y: number;
};

export type TopologyNode = {
  id: string;
  role: NodeRole;
  label: string;
  x: number;
  y: number;
  clientId: number | null;
  properties: Record<string, unknown>;
};

export type LineSeries = {
  key: string;
  label: string;
  color: string;
  values: number[];
};

export type MiniLineChartProps = {
  title: string;
  xValues: number[];
  series: LineSeries[];
  formatter?: (value: number) => string;
};

export type SimulationPageProps = {
  MiniLineChart: (props: MiniLineChartProps) => ReactNode;
  activeTab: TabKey;
  applyTopologyNodeConfig: (nodeId: string, properties: Record<string, unknown>) => void;
  busy: boolean;
  canvasHighlighted: boolean;
  canvasRef: RefObject<HTMLDivElement | null>;
  clientNodes: TopologyNode[];
  cloneTopologyNode: (nodeId: string) => void;
  clientTestAccSeries: LineSeries[];
  clientTestLossSeries: LineSeries[];
  clientTrainAccSeries: LineSeries[];
  clientTrainLossSeries: LineSeries[];
  cloneSelectedTopologyNode: () => void;
  compressionInfo: Record<string, unknown>;
  configJsonError: string | null;
  configSchema: Record<string, unknown> | null;
  deleteJobTarget: Job | null;
  deleteSelectedTopologyNode: () => void;
  deleteTopologyNode: (nodeId: string) => void;
  editingJobDescription: string;
  editingJobName: string;
  encryptionInfo: Record<string, unknown>;
  experimentBasic: Record<string, unknown>;
  experimentFederated: Record<string, unknown>;
  globalAccuracySeries: LineSeries[];
  globalLossSeries: LineSeries[];
  handleCreateJob: () => Promise<void>;
  handleDeleteJob: (target: Job) => Promise<void>;
  handleDeleteRun: () => Promise<void>;
  handleDuplicateJob: (job: Job) => Promise<void>;
  handleRerun: () => Promise<void>;
  handleSaveJobMeta: () => Promise<void>;
  handleStartRun: () => Promise<void>;
  handleStopRun: () => Promise<void>;
  isRecord: (value: unknown) => value is Record<string, unknown>;
  jobConfigText: string;
  jobMetaDialogOpen: boolean;
  jobStatusVariant: Record<string, BadgeVariant>;
  jobs: Job[];
  loadRunBundle: (runId: string) => Promise<void>;
  loadRunMetrics: (runId: string) => Promise<void>;
  newJobDescription: string;
  newJobName: string;
  nodeCenter: (node: TopologyNode) => Point;
  notifyError: (error: unknown, id?: string) => void;
  onCanvasDragLeave: () => void;
  onCanvasDragOver: (event: ReactDragEvent<HTMLDivElement>) => void;
  onCanvasDrop: (event: ReactDragEvent<HTMLDivElement>) => void;
  onToolDragStart: (event: ReactDragEvent<HTMLButtonElement>, role: NodeRole) => void;
  onTopologyNodeMouseDown: (event: ReactMouseEvent<HTMLButtonElement>, node: TopologyNode) => void;
  openJobMetadataDialog: (job: Job) => void;
  patchSelectedTopologyNode: (patch: Partial<TopologyNode>) => void;
  rectEdgePoint: (from: Point, to: Point) => Point;
  refreshJobs: () => Promise<void>;
  refreshRuns: () => Promise<void>;
  renderSchemaSection: (
    schemaSection: Record<string, unknown>,
    pathPrefix: string,
    role: NodeRole,
    properties: Record<string, unknown>,
    onChangeProperty: (path: string, value: unknown) => void,
  ) => ReactNode[];
  requestDeleteJob: (job: Job) => void;
  runLogs: RunLog[];
  runLogsRef: RefObject<HTMLDivElement | null>;
  runRounds: number[];
  runStatusVariant: Record<string, BadgeVariant>;
  runs: Run[];
  selectedJob: Job | null;
  selectedJobId: number | null;
  selectedRun: Run | null;
  selectedRunId: string | null;
  selectedTopologyNode: TopologyNode | null;
  selectedTopologyNodeId: string | null;
  serverNode: TopologyNode | null;
  setActiveTab: Dispatch<SetStateAction<TabKey>>;
  setDeleteJobTarget: Dispatch<SetStateAction<Job | null>>;
  setEditingJobDescription: Dispatch<SetStateAction<string>>;
  setEditingJobMetaId: Dispatch<SetStateAction<number | null>>;
  setEditingJobName: Dispatch<SetStateAction<string>>;
  setJobConfigText: Dispatch<SetStateAction<string>>;
  setJobMetaDialogOpen: Dispatch<SetStateAction<boolean>>;
  setNewJobDescription: Dispatch<SetStateAction<string>>;
  setNewJobName: Dispatch<SetStateAction<string>>;
  setSelectedJobId: Dispatch<SetStateAction<number | null>>;
  setSelectedNodeProperty: (path: string, value: unknown) => void;
  setSelectedRunId: Dispatch<SetStateAction<string | null>>;
  setSelectedTopologyNodeId: Dispatch<SetStateAction<string | null>>;
  setTopologyNodeProperty: (nodeId: string, path: string, value: unknown) => void;
  shortRunId: (runId: string) => string;
  toDisplayText: (value: unknown, fallback?: string) => string;
  toNumber: (value: unknown) => number | null;
  topologyCanvasMinHeight: number;
  topologyNodes: TopologyNode[];
};

export type DistributedPageProps = {
  allDistributedClientsReady: boolean;
  approvedDistributedSlots: Set<number>;
  busy: boolean;
  clientConnecting: boolean;
  clientConnectionMessage: string;
  clientName: string;
  clientServerApiBase: string;
  clientSessionId: string;
  configSchema: Record<string, unknown> | null;
  connectedClientId: string | null;
  connectedClientInfo: DistributedClient | null;
  distributedClientStatusVariant: Record<string, BadgeVariant>;
  distributedClients: DistributedClient[];
  distributedConfigClients: number;
  distributedConfigDialogOpen: boolean;
  distributedConfigDraft: Record<string, unknown> | null;
  distributedJobDescription: string;
  distributedJobName: string;
  distributedJobStatusVariant: Record<string, BadgeVariant>;
  distributedJobs: DistributedJob[];
  distributedRole: DistributedRole;
  distributedServerIp: string;
  distributedServerPort: string;
  distributedSession: DistributedSession | null;
  distributedSessionProgress: DistributedSessionProgress | null;
  distributedSessionStatusVariant: Record<string, BadgeVariant>;
  handleApproveDistributedClient: (clientId: string) => Promise<void>;
  handleClientCancelConnect: () => Promise<void>;
  handleClientConnect: () => Promise<void>;
  handleClientReady: () => Promise<void>;
  handleCreateDistributedJob: () => Promise<void>;
  handleCreateOrAttachDistributedSession: () => Promise<void>;
  handleRejectDistributedClient: (clientId: string) => Promise<void>;
  handleSaveDistributedJobConfig: () => Promise<void>;
  handleStartDistributedTraining: () => Promise<void>;
  notifyError: (error: unknown, id?: string) => void;
  openDistributedConfigDialog: (job: DistributedJob) => void;
  readyDistributedSlots: Set<number>;
  refreshDistributedJobs: () => Promise<void>;
  refreshDistributedSessionBundle: (sessionId: string) => Promise<void>;
  refreshDistributedSessionProgress: (sessionId: string) => Promise<void>;
  renderSchemaSection: (
    schemaSection: Record<string, unknown>,
    pathPrefix: string,
    role: NodeRole,
    properties: Record<string, unknown>,
    onChangeProperty: (path: string, value: unknown) => void,
  ) => ReactNode[];
  selectedDistributedJob: DistributedJob | null;
  selectedDistributedJobId: number | null;
  setClientName: Dispatch<SetStateAction<string>>;
  setClientServerApiBase: Dispatch<SetStateAction<string>>;
  setClientSessionId: Dispatch<SetStateAction<string>>;
  setDistributedConfigClients: Dispatch<SetStateAction<number>>;
  setDistributedConfigDialogOpen: Dispatch<SetStateAction<boolean>>;
  setDistributedConfigProperty: (path: string, value: unknown) => void;
  setDistributedJobDescription: Dispatch<SetStateAction<string>>;
  setDistributedJobName: Dispatch<SetStateAction<string>>;
  setDistributedRole: Dispatch<SetStateAction<DistributedRole>>;
  setDistributedServerIp: Dispatch<SetStateAction<string>>;
  setDistributedServerPort: Dispatch<SetStateAction<string>>;
  setSelectedDistributedJobId: Dispatch<SetStateAction<number | null>>;
};

export type AgentPageProps = {
  activeTaskId: string | null;
  busy: boolean;
  clearResult: () => void;
  configConstraints: Record<string, unknown>;
  configSchema: Record<string, unknown> | null;
  defaultModelName: string;
  experiments: AgentExperimentResponse[];
  experimentRuns: AgentRunResponse[];
  goal: string;
  handleOptimize: () => Promise<void>;
  historyJobs: AgentOptimizationJobSummary[];
  jobName: string;
  lastSubmittedGoal: string | null;
  maxIterations: number;
  modelName: string;
  objective: AgentOptimizationObjective;
  modelOptions: string[];
  notifyError: (error: unknown, id?: string) => void;
  presets: string[];
  progress: AgentOptimizeProgressResponse | null;
  result: AgentOptimizeResponse | null;
  selectedExperimentId: number | null;
  selectedHistory: AgentOptimizeProgressResponse | null;
  selectedHistoryJobId: number | null;
  selectExperiment: (experimentId: number) => Promise<void>;
  selectHistoryJob: (optimizationJobId: number) => Promise<void>;
  setConfigConstraint: (path: string, value: unknown) => void;
  clearConfigConstraint: (path: string) => void;
  setGoal: Dispatch<SetStateAction<string>>;
  setJobName: Dispatch<SetStateAction<string>>;
  setMaxIterations: Dispatch<SetStateAction<number>>;
  setModelName: Dispatch<SetStateAction<string>>;
  setObjective: Dispatch<SetStateAction<AgentOptimizationObjective>>;
  workflowStep: AgentWorkflowStep;
  setWorkflowStep: Dispatch<SetStateAction<AgentWorkflowStep>>;
  draftPlan: AgentPlanDraft | null;
  setDraftPlan: Dispatch<SetStateAction<AgentPlanDraft | null>>;
  handleGeneratePlan: () => Promise<void>;
  handleExecutePlan: (editedExperiments: any[]) => Promise<void>;
  optimizationJobs?: any[];
};

export type { RunMetrics };
export type { DistributedClient, DistributedJob, DistributedSession, DistributedSessionProgress };
