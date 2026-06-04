import {
  deepMerge,
  getValueByPath,
  initDefaultsFromSchema,
  isFieldDefinition,
  isRecord,
  normalizeListInt,
  normalizeSchemaDefault,
  setValueByPath,
  type SchemaNode,
} from "../simulation/utils";
import {
  dynamicBooleanState,
  dynamicOptionState,
  fieldHint,
  optionLabel as compatibilityOptionLabel,
  optionMeta as compatibilityOptionMeta,
  staticOptionDisabled,
} from "../config/compatibility";

const META_KEYS = new Set(["role", "depends_on", "hidden", "ui"]);
const AGENT_SECTIONS = new Set([
  "task",
  "dataset",
  "model",
  "llm",
  "sft",
  "peft",
  "evaluation",
  "federated",
  "compression",
  "privacy",
]);
const REQUIRED_LLM_RESOURCE_FIELDS = new Set(["llm.base_model", "sft.dataset_path", "evaluation.dataset_path"]);

export type AgentSchemaField = {
  path: string;
  key: string;
  section: string;
  label: string;
  type: SchemaNode["type"];
  definition: SchemaNode;
  defaultValue: unknown;
  featured: boolean;
  group: string;
  options: unknown[];
  dependsOn?: string;
};

function labelize(key: string): string {
  if (key === "enable") return "Enabled";
  return key
    .split("_")
    .map((part) => (part.length > 0 ? `${part[0].toUpperCase()}${part.slice(1)}` : part))
    .join(" ");
}

export function fieldUi(definition: SchemaNode): Record<string, unknown> {
  return isRecord(definition.ui) ? definition.ui : {};
}

export function fieldLabel(key: string, definition: SchemaNode): string {
  const ui = fieldUi(definition);
  return typeof ui.label === "string" && ui.label.trim() ? ui.label : labelize(key);
}

export function optionMeta(definition: SchemaNode, option: unknown): Record<string, unknown> {
  return compatibilityOptionMeta(definition, option);
}

export function optionLabel(definition: SchemaNode, option: unknown): string {
  return compatibilityOptionLabel(definition, option);
}

export function optionDisabled(definition: SchemaNode, option: unknown): boolean {
  return staticOptionDisabled(definition, option);
}

export function optionDisabledForConfig(
  definition: SchemaNode,
  option: unknown,
  path: string,
  config: Record<string, unknown>,
): boolean {
  return optionDisabled(definition, option) || dynamicOptionState(path, option, config).disabled;
}

export function optionDisableReasonForConfig(
  definition: SchemaNode,
  option: unknown,
  path: string,
  config: Record<string, unknown>,
): string | null {
  const meta = optionMeta(definition, option);
  if (optionDisabled(definition, option)) {
    return typeof meta.description === "string" ? meta.description : "Not executable yet.";
  }
  return dynamicOptionState(path, option, config).reason ?? null;
}

export function booleanDisabledForConfig(path: string, value: unknown, config: Record<string, unknown>): boolean {
  return dynamicBooleanState(path, value, config).disabled;
}

export function booleanDisableReasonForConfig(path: string, value: unknown, config: Record<string, unknown>): string | null {
  return dynamicBooleanState(path, value, config).reason ?? null;
}

export function fieldCompatibilityHint(path: string, config: Record<string, unknown>): string | null {
  return fieldHint(path, config);
}

export function fieldEmptyMessage(field: AgentSchemaField): string | null {
  const ui = fieldUi(field.definition);
  if (typeof ui.empty_message === "string" && ui.empty_message.trim().length > 0) {
    return ui.empty_message;
  }
  if (field.path === "llm.base_model") return "No local models found";
  if (field.path === "sft.dataset_path") return "No training datasets found";
  if (field.path === "evaluation.dataset_path") return "No evaluation datasets found";
  return null;
}

function isAgentVisible(definition: SchemaNode): boolean {
  const ui = fieldUi(definition);
  return definition.hidden !== true || ui.agent_visible === true;
}

export function checkDependency(properties: Record<string, unknown>, dependsOn?: string): boolean {
  if (!dependsOn) return true;
  return dependsOn.split("&&").every((condition) => {
    const match = condition.trim().match(/(.+)\s*==\s*(.+)/);
    if (!match) return true;
    const actual = getValueByPath(properties, match[1].trim());
    const rawTarget = match[2].trim().replace(/^['"]|['"]$/g, "");
    if (rawTarget === "true") return Boolean(actual) === true;
    if (rawTarget === "false") return Boolean(actual) === false;
    const numeric = Number(rawTarget);
    if (Number.isFinite(numeric)) return Number(actual) === numeric;
    return String(actual) === rawTarget;
  });
}

function combineDependencies(parent?: string, child?: string): string | undefined {
  const dependencies = [parent, child].filter((value): value is string => typeof value === "string" && value.trim().length > 0);
  return dependencies.length > 0 ? dependencies.join(" && ") : undefined;
}

export function agentFieldVisibleForConfig(field: AgentSchemaField, config: Record<string, unknown>): boolean {
  return checkDependency(config, field.dependsOn ?? field.definition.depends_on);
}

export function collectAgentSchemaFields(
  schema: Record<string, unknown> | null,
  options: { featuredOnly?: boolean } = {},
): AgentSchemaField[] {
  if (!schema) return [];
  const fields: AgentSchemaField[] = [];

  function walk(node: Record<string, unknown>, prefix: string, section: string, inheritedDependsOn?: string): void {
    for (const [key, rawDefinition] of Object.entries(node)) {
      if (META_KEYS.has(key) || !isRecord(rawDefinition)) continue;
      const definition = rawDefinition as SchemaNode;
      const path = prefix ? `${prefix}.${key}` : key;
      const currentSection = prefix ? section : key;
      if (!AGENT_SECTIONS.has(currentSection)) continue;
      const dependsOn = combineDependencies(inheritedDependsOn, definition.depends_on);

      if (isFieldDefinition(definition)) {
        const ui = fieldUi(definition);
        const featured = ui.featured === true;
        if (options.featuredOnly && !featured) continue;
        if (!isAgentVisible(definition)) continue;
        fields.push({
          path,
          key,
          section: currentSection,
          label: fieldLabel(key, definition),
          type: definition.type,
          definition,
          defaultValue: normalizeSchemaDefault(definition),
          featured,
          group: typeof ui.group === "string" ? ui.group : "Advanced",
          options: Array.isArray(definition.options) ? definition.options : [],
          dependsOn,
        });
        continue;
      }

      walk(definition, path, currentSection, dependsOn);
    }
  }

  walk(schema, "", "");
  return fields;
}

export function buildAgentConfig(schema: Record<string, unknown> | null, patch: Record<string, unknown>): Record<string, unknown> {
  const defaults = schema ? initDefaultsFromSchema(schema) : {};
  const config = deepMerge(defaults, patch);
  setValueByPath(config, "system.mode", "simulation");
  setValueByPath(config, "system.node_role", "server");
  return config;
}

export function applyConfigConstraints(
  config: Record<string, unknown>,
  constraints: Record<string, unknown> | null | undefined,
): Record<string, unknown> {
  return deepMerge(config, constraints ?? {});
}

export function materializeAgentConfigConstraints(
  schema: Record<string, unknown> | null,
  constraints: Record<string, unknown>,
): Record<string, unknown> {
  const config = buildAgentConfig(schema, constraints);
  if (getValueByPath(config, "task.type") !== "llm_peft_sft") {
    return constraints;
  }

  let next = structuredClone(constraints);
  for (const path of ["task.type", "llm.base_model"]) {
    const value = getValueByPath(config, path);
    if (typeof value === "string" && value.trim().length > 0) {
      setValueByPath(next, path, value);
    }
  }

  const datasetPath = getValueByPath(config, "sft.dataset_path");
  if (typeof datasetPath === "string" && datasetPath.trim().length > 0) {
    setValueByPath(next, "sft.dataset_path", datasetPath);
    for (const [inferredPath, inferredValue] of Object.entries(inferSftSettingsForDatasetPath(datasetPath))) {
      setValueByPath(next, inferredPath, inferredValue);
    }
  }

  const evaluationEnabled = getValueByPath(config, "evaluation.enable");
  if (typeof evaluationEnabled === "boolean") {
    setValueByPath(next, "evaluation.enable", evaluationEnabled);
  }
  const evaluationPath = getValueByPath(config, "evaluation.dataset_path");
  if (evaluationEnabled === true && typeof evaluationPath === "string" && evaluationPath.trim().length > 0) {
    setValueByPath(next, "evaluation.dataset_path", evaluationPath);
  }

  return next;
}

export function missingLlmResourceMessages(
  schema: Record<string, unknown> | null,
  config: Record<string, unknown>,
): string[] {
  const messages: string[] = [];
  for (const field of collectAgentSchemaFields(schema, { featuredOnly: false })) {
    if (!REQUIRED_LLM_RESOURCE_FIELDS.has(field.path)) continue;
    if (!agentFieldVisibleForConfig(field, config)) continue;
    if (field.options.length === 0) {
      messages.push(fieldEmptyMessage(field) ?? (field.path === "llm.base_model" ? "No local models found" : "No training datasets found"));
      continue;
    }
    const value = getValueByPath(config, field.path);
    if (typeof value !== "string" || value.trim().length === 0) {
      messages.push(`${field.label} is required`);
    }
  }
  return Array.from(new Set(messages));
}

export function removeValueByPath(obj: Record<string, unknown>, path: string): Record<string, unknown> {
  const next = structuredClone(obj);
  const segments = path.split(".").filter(Boolean);
  if (segments.length === 0) return next;
  let current: Record<string, unknown> = next;
  for (let i = 0; i < segments.length - 1; i += 1) {
    const child: unknown = current[segments[i]];
    if (!isRecord(child)) return next;
    current = child;
  }
  delete current[segments[segments.length - 1]];
  return pruneEmptyObjects(next);
}

function pruneEmptyObjects(value: Record<string, unknown>): Record<string, unknown> {
  for (const key of Object.keys(value)) {
    const child: unknown = value[key];
    if (isRecord(child)) {
      const pruned = pruneEmptyObjects(child);
      if (Object.keys(pruned).length === 0) {
        delete value[key];
      }
    }
  }
  return value;
}

export function coerceFieldValue(field: AgentSchemaField, value: unknown): unknown {
  if (field.type === "number") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : Number(field.defaultValue ?? 0);
  }
  if (field.type === "bool") return Boolean(value);
  if (field.type === "list_int") return normalizeListInt(value);
  return value;
}

export function setConfigValue(config: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> {
  const next = structuredClone(config);
  setValueByPath(next, path, value);
  return next;
}

export function inferSftSettingsForDatasetPath(value: unknown): Record<string, string> {
  const datasetPath = String(value ?? "").trim().toLowerCase();
  const fileFormat = datasetPath.endsWith(".jsonl")
    ? "jsonl"
    : datasetPath.endsWith(".parquet")
      ? "parquet"
      : "auto";
  const dataFormat = datasetPath.includes("alpaca")
    ? "alpaca"
    : datasetPath.includes("messages") || datasetPath.includes("chat")
      ? "messages"
      : "prompt_completion";

  return {
    "sft.file_format": fileFormat,
    "sft.format": dataFormat,
  };
}

export function validateDisabledOptions(config: Record<string, unknown>, schema: Record<string, unknown> | null): string[] {
  const errors: string[] = [];
  for (const field of collectAgentSchemaFields(schema, { featuredOnly: false })) {
    if (!agentFieldVisibleForConfig(field, config)) continue;
    if (field.type !== "select") continue;
    const value = getValueByPath(config, field.path);
    if (value !== undefined && optionDisabledForConfig(field.definition, value, field.path, config)) {
      const reason = optionDisableReasonForConfig(field.definition, value, field.path, config);
      const suffix = reason ? ` ${reason}` : " It is not compatible with the current configuration.";
      errors.push(`${field.label}: ${optionLabel(field.definition, value)} is unavailable.${suffix}`);
    }
  }
  return errors;
}

export function formatFieldValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "boolean") return value ? "Enabled" : "Disabled";
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
}

export function selectedConstraintChips(
  schema: Record<string, unknown> | null,
  constraints: Record<string, unknown>,
  config?: Record<string, unknown>,
): Array<{ path: string; label: string; value: string }> {
  return collectAgentSchemaFields(schema, { featuredOnly: true })
    .filter((field) => (config ? agentFieldVisibleForConfig(field, config) : true))
    .map((field) => {
      const value = getValueByPath(constraints, field.path);
      if (value === undefined) return null;
      return {
        path: field.path,
        label: field.label,
        value: formatFieldValue(value),
      };
    })
    .filter((item): item is { path: string; label: string; value: string } => item !== null);
}
