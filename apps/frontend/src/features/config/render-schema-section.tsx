import type { ReactNode } from "react";

import { Input } from "../../components/ui/input";
import { NumberStepper } from "../../components/ui/number-stepper";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../components/ui/select";
import { Switch } from "../../components/ui/switch";
import type { NodeRole } from "../../pages/types";
import { getValueByPath, isFieldDefinition, isRecord, normalizeListInt, type SchemaNode } from "../simulation/utils";
import {
  dynamicBooleanState,
  dynamicOptionState,
  fieldHint,
  isModelCompatibleWithDataset,
  optionLabel,
  optionMeta,
  staticOptionDisabled,
} from "./compatibility";

function checkDependency(properties: Record<string, unknown>, dependsOn?: string): boolean {
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

function labelize(key: string): string {
  if (key === "enable") {
    return "Enabled";
  }
  return key
    .split("_")
    .map((part) => (part.length > 0 ? `${part[0].toUpperCase()}${part.slice(1)}` : part))
    .join(" ");
}

function parseOptionalNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

export function renderSchemaSection(
  schemaSection: Record<string, unknown>,
  pathPrefix: string,
  role: NodeRole,
  properties: Record<string, unknown>,
  onChangeProperty: (path: string, value: unknown) => void,
): ReactNode[] {
  const blocks: ReactNode[] = [];

  for (const [key, rawDefinition] of Object.entries(schemaSection)) {
    if (key === "role" || key === "depends_on" || key === "hidden" || key === "ui") continue;
    if (!isRecord(rawDefinition)) continue;

    const definition = rawDefinition as SchemaNode;
    const fullPath = pathPrefix ? `${pathPrefix}.${key}` : key;
    const roleTag = typeof definition.role === "string" ? definition.role : null;
    const dependsOn = typeof definition.depends_on === "string" ? definition.depends_on : undefined;
    const hidden = definition.hidden === true;

    if (hidden) continue;
    if (roleTag && roleTag !== role && roleTag !== "all") continue;
    if (!checkDependency(properties, dependsOn)) continue;

    if (isFieldDefinition(definition)) {
      const value = getValueByPath(properties, fullPath);
      if (definition.type === "bool") {
        const label = labelize(key);
        const state = dynamicBooleanState(fullPath, value, properties);
        blocks.push(
          <div
            key={fullPath}
            className="space-y-1 rounded-md border border-border/70 bg-muted/25 px-3 py-2 text-sm"
          >
            <div className="flex items-center justify-between">
              <span>{label}</span>
              <Switch
                checked={Boolean(value)}
                disabled={state.disabled}
                onCheckedChange={(checked) => onChangeProperty(fullPath, checked)}
                aria-label={label}
              />
            </div>
            {state.reason && <p className="text-[11px] text-muted-foreground">{state.reason}</p>}
          </div>,
        );
        continue;
      }

      if (definition.type === "select") {
        const options = Array.isArray(definition.options) ? definition.options : [];
        const hint = fieldHint(fullPath, properties);
        blocks.push(
          <div key={fullPath} className="space-y-1 rounded-md border border-border/70 bg-muted/25 p-2">
            <p className="text-xs text-muted-foreground">{labelize(key)}</p>
            <Select
              value={String(value ?? "")}
              onValueChange={(nextValue) => {
                const matched = options.find((option) => String(option) === nextValue);
                const next = matched ?? nextValue;
                onChangeProperty(fullPath, next);
                if (fullPath === "dataset.name") {
                  const currentModel = getValueByPath(properties, "model.name");
                  if (!isModelCompatibleWithDataset(currentModel, String(next))) {
                    onChangeProperty("model.name", "Auto");
                  }
                }
              }}
            >
              <SelectTrigger className="font-mono">
                <SelectValue placeholder="Select option" />
              </SelectTrigger>
              <SelectContent>
                {options.map((option) => {
                  const staticDisabled = staticOptionDisabled(definition, option);
                  const dynamicState = dynamicOptionState(fullPath, option, properties);
                  const disabled = staticDisabled || dynamicState.disabled;
                  const meta = optionMeta(definition, option);
                  const reason = staticDisabled
                    ? typeof meta.description === "string"
                      ? meta.description
                      : "Not executable yet."
                    : dynamicState.reason;
                  return (
                    <SelectItem
                      key={`${fullPath}-${String(option)}`}
                      value={String(option)}
                      disabled={disabled}
                      className="font-mono"
                    >
                      <span className="flex flex-col gap-0.5">
                        <span className="flex items-center gap-2">
                          {optionLabel(definition, option)}
                          {typeof meta.badge === "string" && <span className="text-[10px] text-amber-400">{meta.badge}</span>}
                        </span>
                        {reason && <span className="text-[10px] text-muted-foreground">{reason}</span>}
                      </span>
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
            {hint && <p className="text-[11px] text-muted-foreground">{hint}</p>}
          </div>,
        );
        continue;
      }

      if (definition.type === "list_int") {
        const text = Array.isArray(value) ? value.join(", ") : "";
        blocks.push(
          <div key={fullPath} className="space-y-1 rounded-md border border-border/70 bg-muted/25 p-2">
            <p className="text-xs text-muted-foreground">{labelize(key)}</p>
            <Input
              value={text}
              onChange={(event) => onChangeProperty(fullPath, normalizeListInt(event.target.value))}
              placeholder="0, 1, 2"
            />
          </div>,
        );
        continue;
      }

      if (definition.type === "number") {
        const min = parseOptionalNumber(definition.min);
        const max = parseOptionalNumber(definition.max);
        const step = parseOptionalNumber(definition.step) ?? 1;
        const defaultValue = parseOptionalNumber(definition.default) ?? 0;
        const currentValue = parseOptionalNumber(value) ?? defaultValue;
        blocks.push(
          <div key={fullPath} className="space-y-1 rounded-md border border-border/70 bg-muted/25 p-2">
            <p className="text-xs text-muted-foreground">{labelize(key)}</p>
            <NumberStepper
              value={currentValue}
              min={min}
              max={max}
              step={step}
              onValueChange={(next) => onChangeProperty(fullPath, next)}
              inputClassName="font-mono"
            />
          </div>,
        );
        continue;
      }

      blocks.push(
        <div key={fullPath} className="space-y-1 rounded-md border border-border/70 bg-muted/25 p-2">
          <p className="text-xs text-muted-foreground">{labelize(key)}</p>
          <Input value={String(value ?? "")} onChange={(event) => onChangeProperty(fullPath, event.target.value)} />
        </div>,
      );
      continue;
    }

    const children = renderSchemaSection(definition, fullPath, role, properties, onChangeProperty);
    if (children.length === 0) continue;

    blocks.push(
      <div key={fullPath} className="space-y-2 rounded-lg border border-border/70 bg-muted/35 p-3 shadow-sm">
        <p className="text-xs font-semibold text-muted-foreground">{labelize(key)}</p>
        <div className="space-y-2">{children}</div>
      </div>,
    );
  }

  return blocks;
}
