import { baseUrl } from "../../api/client";
import type { RunMetrics } from "../../api/runs";
import type { BadgeVariant, LineSeries, NodeRole, Point, SystemMode, TopologyNode } from "../../pages/types";

export const NODE_SIZE = { width: 160, height: 74 };
const TOPOLOGY_LAYOUT = {
  minX: 40,
  maxX: 440,
  serverY: 50,
  clientY: 250,
};
const CLIENT_LAYOUT = {
  minGapX: 16,
  rowGap: 28,
};

export const DEFAULT_JOB_CONFIG = {
  task: { type: "classic_fl" },
  system: { mode: "simulation" as SystemMode },
  dataset: { name: "CIFAR-10", data_dir: "./datasets", distribution: "non_iid", alpha: 0.5 },
  federated: { num_clients: 3, num_rounds: 10, clients_per_round: 3, local_epochs: 5, learning_rate: 0.01 },
};

export type SchemaFieldType = "text" | "number" | "bool" | "select" | "list_int";

export type SchemaNode = {
  type?: SchemaFieldType;
  default?: unknown;
  options?: unknown[];
  role?: string;
  depends_on?: string;
  hidden?: boolean;
  ui?: Record<string, unknown>;
  [key: string]: unknown;
};

export const jobStatusVariant: Record<string, BadgeVariant> = {
  draft: "secondary",
  ready: "success",
  archived: "outline",
  done: "success",
};

export const runStatusVariant: Record<string, BadgeVariant> = {
  queued: "secondary",
  running: "warning",
  succeeded: "success",
  failed: "danger",
  cancelled: "outline",
};

export const distributedJobStatusVariant: Record<string, BadgeVariant> = {
  draft: "secondary",
  waiting_clients: "warning",
  ready: "success",
  running: "warning",
  finished: "outline",
};

export const distributedSessionStatusVariant: Record<string, BadgeVariant> = {
  waiting_clients: "secondary",
  ready_to_start: "success",
  running: "warning",
  finished: "outline",
  cancelled: "danger",
};

export const distributedClientStatusVariant: Record<string, BadgeVariant> = {
  pending_approval: "secondary",
  approved: "outline",
  ready: "success",
  running: "warning",
  rejected: "danger",
  cancelled: "outline",
  disconnected: "danger",
};

export const EMPTY_RUN_METRICS: RunMetrics = {
  experiment_info: {
    basic: {},
    federated: {},
    security: {},
  },
  global_results: {
    rounds: [],
    global_loss: [],
    global_accuracy: [],
  },
  llm_results: {
    rounds: [],
    train_loss: [],
    validation_loss: [],
    perplexity: [],
    token_throughput: [],
    adapter_size_bytes: [],
  },
  llm_dataset: {},
  llm_evaluation: {},
  llm_runtime: {},
  llm_artifacts: [],
  client_results: {},
};

const CLIENT_SERIES_COLORS = [
  "#2563eb",
  "#16a34a",
  "#ea580c",
  "#dc2626",
  "#7c3aed",
  "#0891b2",
  "#ca8a04",
];

export const DEFAULT_DISTRIBUTED_SERVER_API_BASE = import.meta.env.VITE_DISTRIBUTED_SERVER_API_BASE ?? baseUrl;
export const DEFAULT_DISTRIBUTED_GRPC_HOST = import.meta.env.VITE_DISTRIBUTED_GRPC_HOST ?? "localhost";
export const DEFAULT_DISTRIBUTED_GRPC_PORT = import.meta.env.VITE_DISTRIBUTED_GRPC_PORT ?? "50052";

export function nodeCenter(node: TopologyNode): Point {
  return {
    x: node.x + NODE_SIZE.width / 2,
    y: node.y + NODE_SIZE.height / 2,
  };
}

export function rectEdgePoint(from: Point, to: Point): Point {
  const halfWidth = NODE_SIZE.width / 2;
  const halfHeight = NODE_SIZE.height / 2;
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  if (dx === 0 && dy === 0) {
    return from;
  }

  const scale = 1 / Math.max(Math.abs(dx) / halfWidth, Math.abs(dy) / halfHeight);
  return {
    x: from.x + dx * scale,
    y: from.y + dy * scale,
  };
}

export function toDisplayText(value: unknown, fallback = "-"): string {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string" && value.trim().length === 0) return fallback;
  return String(value);
}

export function toNumber(value: unknown): number | null {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return null;
  return parsed;
}

export function uid(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

export function shortRunId(runId: string): string {
  return runId.slice(0, 8);
}

export function toErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message;
  }
  return "Request failed";
}

export function parseJsonObject(text: string): Record<string, unknown> {
  const parsed = JSON.parse(text);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("Config JSON must be an object.");
  }
  return parsed as Record<string, unknown>;
}

export function stableStringify(value: unknown): string {
  const normalize = (input: unknown): unknown => {
    if (Array.isArray(input)) {
      return input.map((item) => normalize(item));
    }
    if (isRecord(input)) {
      const output: Record<string, unknown> = {};
      for (const key of Object.keys(input).sort()) {
        output[key] = normalize(input[key]);
      }
      return output;
    }
    return input;
  };

  return JSON.stringify(normalize(value));
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function getValueByPath(obj: Record<string, unknown>, path: string): unknown {
  return path
    .split(".")
    .filter(Boolean)
    .reduce<unknown>((current, key) => (isRecord(current) ? current[key] : undefined), obj);
}

export function setValueByPath(obj: Record<string, unknown>, path: string, value: unknown): void {
  const segments = path.split(".").filter(Boolean);
  if (segments.length === 0) return;

  let current: Record<string, unknown> = obj;
  for (let i = 0; i < segments.length - 1; i += 1) {
    const key = segments[i];
    const next = current[key];
    if (!isRecord(next)) {
      current[key] = {};
    }
    current = current[key] as Record<string, unknown>;
  }
  current[segments[segments.length - 1]] = value;
}

export function normalizeClientId(value: unknown): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 0;
  return Math.max(0, Math.floor(parsed));
}

export function normalizeListInt(value: unknown): number[] {
  if (Array.isArray(value)) {
    return value.map((item) => Number(item)).filter((item) => Number.isFinite(item)).map((item) => Math.floor(item));
  }
  if (typeof value === "string") {
    return value
      .split(",")
      .map((part) => Number(part.trim()))
      .filter((item) => Number.isFinite(item))
      .map((item) => Math.floor(item));
  }
  return [];
}

export function isFieldDefinition(value: unknown): value is SchemaNode {
  return isRecord(value) && typeof value.type === "string";
}

export function normalizeSchemaDefault(node: SchemaNode): unknown {
  if (node.default !== undefined) return node.default;
  if (node.type === "bool") return false;
  if (node.type === "number") return 0;
  if (node.type === "list_int") return [];
  if (node.type === "select" && Array.isArray(node.options) && node.options.length > 0) return node.options[0];
  return "";
}

export function initDefaultsFromSchema(schema: Record<string, unknown>): Record<string, unknown> {
  const output: Record<string, unknown> = {};
  for (const [key, definition] of Object.entries(schema)) {
    if (key === "role" || key === "depends_on" || key === "hidden" || key === "ui") continue;
    if (!isRecord(definition)) continue;

    if (isFieldDefinition(definition)) {
      output[key] = normalizeSchemaDefault(definition);
      continue;
    }

    output[key] = initDefaultsFromSchema(definition);
  }
  return output;
}

export function deepMerge(base: Record<string, unknown>, override: Record<string, unknown>): Record<string, unknown> {
  const output = structuredClone(base);
  for (const [key, value] of Object.entries(override)) {
    if (isRecord(value) && isRecord(output[key])) {
      output[key] = deepMerge(output[key] as Record<string, unknown>, value);
      continue;
    }
    output[key] = value;
  }
  return output;
}

export function buildConfigFromSchema(schema: Record<string, unknown>, properties: Record<string, unknown>): Record<string, unknown> {
  const output: Record<string, unknown> = {};
  for (const [key, definition] of Object.entries(schema)) {
    if (key === "role" || key === "depends_on" || key === "hidden" || key === "ui") continue;
    if (!isRecord(definition)) continue;

    if (isFieldDefinition(definition)) {
      const value = properties[key] ?? normalizeSchemaDefault(definition);
      if (definition.type === "number") {
        const parsed = Number(value);
        output[key] = Number.isFinite(parsed) ? parsed : Number(normalizeSchemaDefault(definition) ?? 0);
      } else if (definition.type === "bool") {
        output[key] = Boolean(value);
      } else if (definition.type === "list_int") {
        output[key] = normalizeListInt(value);
      } else {
        output[key] = value;
      }
      continue;
    }

    const childProps = isRecord(properties[key]) ? (properties[key] as Record<string, unknown>) : {};
    output[key] = buildConfigFromSchema(definition, childProps);
  }
  return output;
}

export function createNodeProperties(
  schema: Record<string, unknown> | null,
  role: NodeRole,
  mode: SystemMode,
  clientId: number | null,
): Record<string, unknown> {
  const defaults = schema ? initDefaultsFromSchema(schema) : structuredClone(DEFAULT_JOB_CONFIG as Record<string, unknown>);
  const properties = structuredClone(defaults);

  setValueByPath(properties, "system.mode", mode);
  setValueByPath(properties, "system.node_role", role);

  if (role === "client") {
    const safeId = normalizeClientId(clientId);
    setValueByPath(properties, "distributed.client_id", safeId);
  }

  return properties;
}

export function makeInitialTopologyNodes(schema: Record<string, unknown> | null, mode: SystemMode): TopologyNode[] {
  const serverX = (TOPOLOGY_LAYOUT.minX + TOPOLOGY_LAYOUT.maxX) / 2;
  return [
    {
      id: uid("server"),
      role: "server",
      label: "Server",
      x: serverX,
      y: TOPOLOGY_LAYOUT.serverY,
      clientId: null,
      properties: createNodeProperties(schema, "server", mode, null),
    },
    {
      id: uid("client"),
      role: "client",
      label: "Client-0",
      x: TOPOLOGY_LAYOUT.minX,
      y: TOPOLOGY_LAYOUT.clientY,
      clientId: 0,
      properties: createNodeProperties(schema, "client", mode, 0),
    },
    {
      id: uid("client"),
      role: "client",
      label: "Client-1",
      x: (TOPOLOGY_LAYOUT.minX + TOPOLOGY_LAYOUT.maxX) / 2,
      y: TOPOLOGY_LAYOUT.clientY,
      clientId: 1,
      properties: createNodeProperties(schema, "client", mode, 1),
    },
    {
      id: uid("client"),
      role: "client",
      label: "Client-2",
      x: TOPOLOGY_LAYOUT.maxX,
      y: TOPOLOGY_LAYOUT.clientY,
      clientId: 2,
      properties: createNodeProperties(schema, "client", mode, 2),
    },
  ];
}

export function nextClientId(nodes: TopologyNode[]): number {
  const maxId = nodes
    .filter((node) => node.role === "client")
    .reduce((max, node) => Math.max(max, node.clientId ?? -1), -1);
  return maxId + 1;
}

function getClientColumnCount(clientCount: number): number {
  if (clientCount <= 1) return 1;
  if (clientCount <= 3) return clientCount;

  const availableWidth = TOPOLOGY_LAYOUT.maxX - TOPOLOGY_LAYOUT.minX;
  const maxColumnsNoOverlap = Math.max(
    2,
    Math.floor(availableWidth / (NODE_SIZE.width + CLIENT_LAYOUT.minGapX)) + 1,
  );
  const desiredColumns = Math.ceil(Math.sqrt(clientCount));
  return Math.max(2, Math.min(maxColumnsNoOverlap, desiredColumns));
}

function getClientLayoutPoint(index: number, totalClients: number): Point {
  const safeTotal = Math.max(1, totalClients);
  const columns = getClientColumnCount(safeTotal);
  const rows = Math.ceil(safeTotal / columns);
  const row = Math.floor(index / columns);
  const col = index % columns;
  const itemsBeforeRow = row * columns;
  const itemsInRow = row === rows - 1 ? safeTotal - itemsBeforeRow : columns;

  const availableWidth = TOPOLOGY_LAYOUT.maxX - TOPOLOGY_LAYOUT.minX;
  const spacing = itemsInRow <= 1 ? 0 : availableWidth / (itemsInRow - 1);
  const rowSpan = spacing * Math.max(0, itemsInRow - 1);
  const centeredOffset = (availableWidth - rowSpan) / 2;

  const x = TOPOLOGY_LAYOUT.minX + centeredOffset + spacing * col;
  const y = TOPOLOGY_LAYOUT.clientY + row * (NODE_SIZE.height + CLIENT_LAYOUT.rowGap);
  return { x, y };
}

export function getTopologyCanvasMinHeight(clientCount: number): number {
  const baseMinHeight = 360;
  const safeCount = Math.max(1, clientCount);
  const columns = getClientColumnCount(safeCount);
  const rows = Math.ceil(safeCount / columns);
  const neededHeight =
    TOPOLOGY_LAYOUT.clientY + (rows - 1) * (NODE_SIZE.height + CLIENT_LAYOUT.rowGap) + NODE_SIZE.height + 24;
  return Math.max(baseMinHeight, neededHeight);
}

export function buildConfigFromTopology(
  nodes: TopologyNode[],
  mode: SystemMode,
  schema: Record<string, unknown> | null,
): Record<string, unknown> {
  const serverNode = nodes.find((node) => node.role === "server");
  const serverProps = serverNode?.properties ?? createNodeProperties(schema, "server", mode, null);
  const config = schema
    ? buildConfigFromSchema(schema, serverProps)
    : deepMerge(structuredClone(DEFAULT_JOB_CONFIG as Record<string, unknown>), serverProps);

  const clients = nodes.filter((node) => node.role === "client");

  setValueByPath(config, "system.mode", mode);
  setValueByPath(config, "dataset.num_clients", clients.length);
  setValueByPath(config, "federated.num_clients", clients.length);

  return config;
}

export function topologyFromConfig(
  config: Record<string, unknown>,
  schema: Record<string, unknown> | null,
): { nodes: TopologyNode[]; mode: SystemMode } {
  const federated = (config.federated ?? {}) as Record<string, unknown>;
  const system = (config.system ?? {}) as Record<string, unknown>;

  const mode = system.mode === "distributed" ? "distributed" : "simulation";
  const rawNum = Number(federated.num_clients ?? 3);
  const numClients = Number.isFinite(rawNum) && rawNum > 0 ? Math.floor(rawNum) : 3;

  const baseProperties = schema
    ? deepMerge(initDefaultsFromSchema(schema), config)
    : deepMerge(structuredClone(DEFAULT_JOB_CONFIG as Record<string, unknown>), config);

  const clients: TopologyNode[] = Array.from({ length: numClients }, (_, index) => {
    const position = getClientLayoutPoint(index, numClients);
    return {
      id: uid("client"),
      role: "client",
      label: `Client-${index}`,
      x: position.x,
      y: position.y,
      clientId: index,
      properties: (() => {
        const clientProps = structuredClone(baseProperties);
        setValueByPath(clientProps, "system.mode", mode);
        setValueByPath(clientProps, "system.node_role", "client");
        setValueByPath(clientProps, "distributed.client_id", index);
        return clientProps;
      })(),
    };
  });

  return {
    mode,
    nodes: [
      {
        id: uid("server"),
        role: "server",
        label: "Server",
        x: (TOPOLOGY_LAYOUT.minX + TOPOLOGY_LAYOUT.maxX) / 2,
        y: TOPOLOGY_LAYOUT.serverY,
        clientId: null,
        properties: (() => {
          const serverProps = structuredClone(baseProperties);
          setValueByPath(serverProps, "system.mode", mode);
          setValueByPath(serverProps, "system.node_role", "server");
          return serverProps;
        })(),
      },
      ...clients,
    ],
  };
}

export function getLineSeriesBounds(series: LineSeries[]): { min: number; max: number } {
  const values = series.flatMap((item) => item.values).filter((value) => Number.isFinite(value));
  if (values.length === 0) {
    return { min: 0, max: 1 };
  }

  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const padding = (max - min) * 0.08;
  return { min: min - padding, max: max + padding };
}

export function polylinePoints(values: number[], pointsCount: number, yMin: number, yMax: number, width: number, height: number): string {
  if (values.length === 0 || pointsCount <= 1) {
    return "";
  }

  const xStep = width / (pointsCount - 1);
  const ySpan = yMax - yMin || 1;

  return values
    .map((value, index) => {
      const x = xStep * index;
      const normalized = (value - yMin) / ySpan;
      const y = height - normalized * height;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function parseClientOrder(clientName: string): number {
  const match = clientName.match(/(\d+)$/);
  if (!match) return Number.MAX_SAFE_INTEGER;
  return Number(match[1]);
}

export function toClientSeries(metrics: RunMetrics, key: keyof RunMetrics["client_results"][string]): LineSeries[] {
  return Object.entries(metrics.client_results)
    .sort(([a], [b]) => parseClientOrder(a) - parseClientOrder(b) || a.localeCompare(b))
    .map(([clientName, values], index) => ({
      key: `${key}-${clientName}`,
      label: clientName,
      color: CLIENT_SERIES_COLORS[index % CLIENT_SERIES_COLORS.length],
      values: values[key],
    }));
}
