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

const META_KEYS = new Set(["role", "depends_on", "hidden", "ui"]);
const AGENT_SECTIONS = new Set(["dataset", "model", "federated", "compression", "privacy"]);

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
  const options = fieldUi(definition).options;
  if (!isRecord(options)) return {};
  const meta = options[String(option)];
  return isRecord(meta) ? meta : {};
}

export function optionLabel(definition: SchemaNode, option: unknown): string {
  const meta = optionMeta(definition, option);
  return typeof meta.label === "string" && meta.label.trim() ? meta.label : String(option);
}

export function optionDisabled(definition: SchemaNode, option: unknown): boolean {
  return optionMeta(definition, option).disabled === true;
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

export function collectAgentSchemaFields(
  schema: Record<string, unknown> | null,
  options: { featuredOnly?: boolean } = {},
): AgentSchemaField[] {
  if (!schema) return [];
  const fields: AgentSchemaField[] = [];

  function walk(node: Record<string, unknown>, prefix: string, section: string): void {
    for (const [key, rawDefinition] of Object.entries(node)) {
      if (META_KEYS.has(key) || !isRecord(rawDefinition)) continue;
      const definition = rawDefinition as SchemaNode;
      const path = prefix ? `${prefix}.${key}` : key;
      const currentSection = prefix ? section : key;
      if (!AGENT_SECTIONS.has(currentSection)) continue;

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
        });
        continue;
      }

      walk(definition, path, currentSection);
    }
  }

  walk(schema, "", "");
  return fields;
}

export function buildAgentConfig(schema: Record<string, unknown> | null, patch: Record<string, unknown>): Record<string, unknown> {
  const defaults = schema ? initDefaultsFromSchema(schema) : {};
  return deepMerge(defaults, patch);
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

export function validateDisabledOptions(config: Record<string, unknown>, schema: Record<string, unknown> | null): string[] {
  const errors: string[] = [];
  for (const field of collectAgentSchemaFields(schema, { featuredOnly: false })) {
    if (field.type !== "select") continue;
    const value = getValueByPath(config, field.path);
    if (value !== undefined && optionDisabled(field.definition, value)) {
      errors.push(`${field.label}: ${optionLabel(field.definition, value)} is experimental and not executable yet.`);
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
): Array<{ path: string; label: string; value: string }> {
  return collectAgentSchemaFields(schema, { featuredOnly: true })
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
