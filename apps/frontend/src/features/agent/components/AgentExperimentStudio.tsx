import { ArrowRight, Loader2, SlidersHorizontal, Sparkles, X } from "lucide-react";
import { useMemo } from "react";

import { Badge } from "../../../components/ui/badge";
import { Button } from "../../../components/ui/button";
import { Card, CardContent } from "../../../components/ui/card";
import { Input } from "../../../components/ui/input";
import { NumberStepper } from "../../../components/ui/number-stepper";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../../components/ui/select";
import { Switch } from "../../../components/ui/switch";
import { Textarea } from "../../../components/ui/textarea";
import type { AgentPageProps } from "../../../pages/types";
import { getValueByPath } from "../../simulation/utils";
import {
  buildAgentConfig,
  checkDependency,
  coerceFieldValue,
  collectAgentSchemaFields,
  formatFieldValue,
  optionDisabled,
  optionLabel,
  optionMeta,
  selectedConstraintChips,
  type AgentSchemaField,
} from "../schema";

export function AgentExperimentStudio(props: AgentPageProps) {
  const {
    clearConfigConstraint,
    configConstraints,
    configSchema,
    goal,
    setConfigConstraint,
    setGoal,
    jobName,
    setJobName,
    presets,
    busy,
    handleGeneratePlan,
  } = props;

  const fields = useMemo(() => collectAgentSchemaFields(configSchema, { featuredOnly: true }), [configSchema]);
  const effectiveConfig = useMemo(
    () => buildAgentConfig(configSchema, configConstraints),
    [configConstraints, configSchema],
  );
  const chips = useMemo(
    () => selectedConstraintChips(configSchema, configConstraints),
    [configConstraints, configSchema],
  );
  const visibleFields = fields.filter((field) => checkDependency(effectiveConfig, field.definition.depends_on));

  return (
    <div className="flex h-full w-full flex-col items-center justify-center space-y-6 max-w-3xl mx-auto py-12">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Figaro Studio</h1>
        <p className="text-muted-foreground">Describe your federated learning goals. The agent will design the configuration.</p>
      </div>

      <Card className="w-full shadow-lg border-primary/20">
        <CardContent className="p-6 space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Job Name</label>
            <Input 
              value={jobName} 
              onChange={(e) => setJobName(e.target.value)} 
              placeholder="e.g., resnet-cifar-tuning"
            />
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Experiment Goal</label>
            <Textarea 
              className="min-h-[120px] text-base resize-none"
              placeholder="E.g., Compare FedAvg vs FedProx on CIFAR-10 with high data heterogeneity (alpha=0.1)..."
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
            />
          </div>

          <div className="flex flex-wrap gap-2 pt-2">
            {presets.map((preset, idx) => (
              <button
                key={idx}
                onClick={() => setGoal(preset)}
                className="text-xs px-3 py-1.5 rounded-full bg-muted hover:bg-muted/80 transition-colors text-muted-foreground"
              >
                {preset}
              </button>
            ))}
          </div>

          <div className="space-y-3 rounded-lg border border-border/70 bg-muted/20 p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <SlidersHorizontal className="h-4 w-4 text-primary" />
                <p className="text-sm font-semibold">Default Config</p>
              </div>
            </div>

            {visibleFields.length === 0 && (
              <p className="text-sm text-muted-foreground">Config schema is loading.</p>
            )}
            {visibleFields.length > 0 && (
              <div className="grid gap-3 md:grid-cols-2">
                {visibleFields.map((field) => (
                  <ConstraintField
                    key={field.path}
                    field={field}
                    value={getValueByPath(configConstraints, field.path) ?? field.defaultValue}
                    onChange={(value) => setConfigConstraint(field.path, coerceFieldValue(field, value))}
                  />
                ))}
              </div>
            )}

            {chips.length > 0 && (
              <div className="flex flex-wrap gap-2 border-t border-border/70 pt-3">
                {chips.map((chip) => (
                  <Badge key={chip.path} variant="outline" className="gap-1 bg-background/50">
                    {chip.label}: <span className="font-mono">{chip.value}</span>
                    <button
                      type="button"
                      className="ml-1 rounded-full p-0.5 hover:bg-muted"
                      onClick={() => clearConfigConstraint(chip.path)}
                      aria-label={`Clear ${chip.label}`}
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Button 
        size="lg" 
        className="w-full sm:w-auto px-8 py-6 text-lg rounded-full transition-all"
        disabled={busy || !goal.trim()}
        onClick={() => { void handleGeneratePlan(); }}
      >
        {busy ? (
          <>
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            Generating Plan... 
          </>
        ) : (
          <>
            <Sparkles className="mr-2 h-5 w-5" />
            Generate Plan <ArrowRight className="ml-2 h-5 w-5" />
          </>
        )}
      </Button>
    </div>
  );
}

function parseOptionalNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function ConstraintField({
  field,
  value,
  onChange,
}: {
  field: AgentSchemaField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const hint = field.definition.ui && typeof field.definition.ui.prompt_hint === "string"
    ? field.definition.ui.prompt_hint
    : null;

  return (
    <div className="space-y-1 rounded-md border border-border/70 bg-background/60 p-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-medium text-muted-foreground">{field.label}</p>
        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{field.section}</span>
      </div>
      {field.type === "select" && (
        <Select value={String(value ?? "")} onValueChange={onChange}>
          <SelectTrigger className="font-mono">
            <SelectValue placeholder="Select option" />
          </SelectTrigger>
          <SelectContent>
            {field.options.map((option) => {
              const meta = optionMeta(field.definition, option);
              return (
                <SelectItem
                  key={`${field.path}-${String(option)}`}
                  value={String(option)}
                  disabled={optionDisabled(field.definition, option)}
                  className="font-mono"
                >
                  <span className="flex items-center gap-2">
                    {optionLabel(field.definition, option)}
                    {typeof meta.badge === "string" && <span className="text-[10px] text-amber-400">{meta.badge}</span>}
                  </span>
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
      )}
      {field.type === "number" && (
        <NumberStepper
          value={Number(value ?? field.defaultValue ?? 0)}
          min={parseOptionalNumber(field.definition.min)}
          max={parseOptionalNumber(field.definition.max)}
          step={parseOptionalNumber(field.definition.step) ?? 1}
          onValueChange={onChange}
          inputClassName="font-mono"
        />
      )}
      {field.type === "bool" && (
        <div className="flex h-10 items-center justify-between rounded-md border border-input bg-background/40 px-3">
          <span className="text-sm">{formatFieldValue(value)}</span>
          <Switch checked={Boolean(value)} onCheckedChange={onChange} />
        </div>
      )}
      {field.type !== "select" && field.type !== "number" && field.type !== "bool" && (
        <Input className="font-mono" value={formatFieldValue(value)} onChange={(event) => onChange(event.target.value)} />
      )}
      {hint && <p className="text-[11px] text-muted-foreground">{hint}</p>}
    </div>
  );
}
