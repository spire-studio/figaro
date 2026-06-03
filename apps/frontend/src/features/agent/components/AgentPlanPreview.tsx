import { useEffect, useMemo, useState } from "react";
import { ClipboardList, MessageSquare, Play, Sparkles, Target } from "lucide-react";
import { toast } from "sonner";

import { agentApi } from "../../../api/agent";
import { Badge } from "../../../components/ui/badge";
import { Button } from "../../../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../../components/ui/card";
import { Input } from "../../../components/ui/input";
import { NumberStepper } from "../../../components/ui/number-stepper";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../../components/ui/select";
import { Switch } from "../../../components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../../components/ui/tabs";
import { Textarea } from "../../../components/ui/textarea";
import type { AgentPageProps } from "../../../pages/types";
import { getValueByPath, isRecord } from "../../simulation/utils";
import {
  booleanDisabledForConfig,
  booleanDisableReasonForConfig,
  agentFieldVisibleForConfig,
  buildAgentConfig,
  coerceFieldValue,
  collectAgentSchemaFields,
  fieldCompatibilityHint,
  formatFieldValue,
  inferSftSettingsForDatasetPath,
  optionDisableReasonForConfig,
  optionDisabledForConfig,
  optionLabel,
  optionMeta,
  setConfigValue,
  validateDisabledOptions,
  type AgentSchemaField,
} from "../schema";
import { isModelCompatibleWithDataset } from "../../config/compatibility";

type LocalExperiment = {
  name: string;
  plan_summary: string;
  config_patch: Record<string, unknown>;
  _rawJsonString?: string;
  _jsonError?: boolean;
  _activeTab?: "form" | "json";
};

export function AgentPlanPreview(props: AgentPageProps) {
  const {
    configSchema,
    draftPlan,
    busy: globalBusy,
    handleExecutePlan,
    setWorkflowStep,
    setDraftPlan,
  } = props;

  const [localExperiments, setLocalExperiments] = useState<LocalExperiment[]>(() =>
    draftPlan?.experiments ? JSON.parse(JSON.stringify(draftPlan.experiments)) : [],
  );
  const [instruction, setInstruction] = useState("");
  const [isRevising, setIsRevising] = useState(false);

  const schemaFields = useMemo(() => collectAgentSchemaFields(configSchema), [configSchema]);

  useEffect(() => {
    setLocalExperiments(draftPlan?.experiments ? JSON.parse(JSON.stringify(draftPlan.experiments)) : []);
  }, [draftPlan?.job_id]);

  if (!draftPlan) return null;
  const isBusy = globalBusy || isRevising;

  const handleRevise = async () => {
    if (!instruction.trim() || !draftPlan.job_id) return;
    setIsRevising(true);
    try {
      const response = await agentApi.revisePlan(draftPlan.job_id, { instruction });
      const nextExperiments = response.experiments as LocalExperiment[];
      setLocalExperiments(nextExperiments);
      setDraftPlan({
        ...draftPlan,
        experiments: response.experiments,
        config_constraints: response.config_constraints ?? draftPlan.config_constraints,
      });
      setInstruction("");
      toast.success("Plan updated successfully by Agent.");
    } catch (error: any) {
      toast.error(error.message || "Failed to update plan via Agent.");
    } finally {
      setIsRevising(false);
    }
  };

  function patchExperiment(index: number, patch: Partial<LocalExperiment>): void {
    setLocalExperiments((current) => current.map((exp, idx) => (idx === index ? { ...exp, ...patch } : exp)));
  }

  function handleFormConfigChange(index: number, path: string, value: unknown): void {
    setLocalExperiments((current) =>
      current.map((exp, idx) => {
        if (idx !== index) return exp;
        let nextConfig = setConfigValue(exp.config_patch ?? {}, path, value);
        if (path === "dataset.name") {
          const fullConfig = buildAgentConfig(configSchema, nextConfig);
          const currentModel = getValueByPath(fullConfig, "model.name");
          if (!isModelCompatibleWithDataset(currentModel, String(value))) {
            nextConfig = setConfigValue(nextConfig, "model.name", "Auto");
          }
        }
        if (path === "sft.dataset_path") {
          for (const [inferredPath, inferredValue] of Object.entries(inferSftSettingsForDatasetPath(value))) {
            nextConfig = setConfigValue(nextConfig, inferredPath, inferredValue);
          }
        }
        return {
          ...exp,
          config_patch: nextConfig,
          _rawJsonString: JSON.stringify(nextConfig, null, 2),
          _jsonError: false,
        };
      }),
    );
  }

  function handleJsonConfigChange(index: number, newJsonString: string): void {
    const updated = [...localExperiments];
    try {
      const parsed = JSON.parse(newJsonString);
      if (!isRecord(parsed)) {
        throw new Error("Config must be an object.");
      }
      updated[index].config_patch = parsed;
      updated[index]._jsonError = false;
    } catch {
      updated[index]._jsonError = true;
    }
    updated[index]._rawJsonString = newJsonString;
    setLocalExperiments(updated);
  }

  const onRunClick = () => {
    const hasJsonErrors = localExperiments.some((exp) => exp._jsonError);
    if (hasJsonErrors) {
      toast.error("Please fix invalid JSON formatting before running.");
      return;
    }

    const disabledErrors = localExperiments.flatMap((exp) =>
      validateDisabledOptions(buildAgentConfig(configSchema, exp.config_patch ?? {}), configSchema),
    );
    if (disabledErrors.length > 0) {
      toast.error(disabledErrors[0]);
      return;
    }

    const cleanExperiments = localExperiments.map((exp) => {
      const { _rawJsonString, _jsonError, _activeTab, ...rest } = exp;
      return rest;
    });
    void handleExecutePlan(cleanExperiments);
  };

  return (
    <div className="space-y-4 h-full overflow-auto pb-6 pr-2">
      <div className="flex items-center justify-between mb-2">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Review & Edit Plan</h2>
          <p className="text-muted-foreground">The agent proposed {localExperiments.length} configurations to achieve your goal.</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => setWorkflowStep("home")} disabled={isBusy}>
            Back
          </Button>
          <Button onClick={onRunClick} disabled={isBusy} className="bg-emerald-600 hover:bg-emerald-700 text-white">
            <Play className="mr-2 h-4 w-4" /> Execute Plan
          </Button>
        </div>
      </div>

      <div className="bg-muted/40 p-4 rounded-lg border flex items-start gap-3">
        <Target className="h-5 w-5 text-primary mt-0.5" />
        <div>
          <p className="text-sm font-bold text-foreground">Experiment Goal</p>
          <p className="text-sm text-muted-foreground mt-1">{draftPlan.goal}</p>
        </div>
      </div>

      <Card className="border-primary/40 bg-primary/5 shadow-sm">
        <CardContent className="p-4 flex gap-3 items-center">
          <div className="p-2 bg-primary/10 rounded-full text-primary shrink-0">
            <MessageSquare className="h-5 w-5" />
          </div>
          <Input
            placeholder="Ask the Agent to tweak the plan (e.g., 'Make learning rate smaller' or 'Add one more experiment')..."
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void handleRevise(); }}
            disabled={isBusy}
            className="flex-1 bg-background"
          />
          <Button onClick={() => { void handleRevise(); }} disabled={isBusy || !instruction.trim()} className="shrink-0">
            {isRevising ? <Sparkles className="mr-2 h-4 w-4 animate-pulse" /> : <Sparkles className="mr-2 h-4 w-4" />}
            Ask Agent
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <ClipboardList className="h-5 w-5 text-primary" />
            Proposed Execution Plan
          </CardTitle>
          <CardDescription>
            Edit each experiment with schema-aware controls or switch to JSON for advanced changes.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {localExperiments.map((exp, idx) => (
              <ExperimentCard
                key={`${exp.name}-${idx}`}
                disabled={isBusy}
                experiment={exp}
                index={idx}
                schema={configSchema}
                fields={schemaFields}
                onConfigChange={(path, value) => handleFormConfigChange(idx, path, value)}
                onJsonChange={(text) => handleJsonConfigChange(idx, text)}
                onTabChange={(tab) => patchExperiment(idx, { _activeTab: tab })}
              />
            ))}
          </div>
        </CardContent>
      </Card>
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

function ExperimentCard({
  disabled,
  experiment,
  fields,
  index,
  onConfigChange,
  onJsonChange,
  onTabChange,
  schema,
}: {
  disabled: boolean;
  experiment: LocalExperiment;
  fields: AgentSchemaField[];
  index: number;
  onConfigChange: (path: string, value: unknown) => void;
  onJsonChange: (text: string) => void;
  onTabChange: (tab: "form" | "json") => void;
  schema: Record<string, unknown> | null;
}) {
  const fullConfig = buildAgentConfig(schema, experiment.config_patch ?? {});
  const visibleFields = fields.filter((field) => agentFieldVisibleForConfig(field, fullConfig));
  const grouped = visibleFields.reduce<Record<string, AgentSchemaField[]>>((acc, field) => {
    const section = field.section;
    acc[section] = [...(acc[section] ?? []), field];
    return acc;
  }, {});
  const jsonString = experiment._rawJsonString ?? JSON.stringify(experiment.config_patch ?? {}, null, 2);
  const disabledErrors = validateDisabledOptions(fullConfig, schema);
  const summaryFields = fields
    .filter((field) => field.featured && agentFieldVisibleForConfig(field, fullConfig))
    .map((field) => ({ field, value: getValueByPath(fullConfig, field.path) }))
    .filter((item) => item.value !== undefined)
    .slice(0, 8);

  return (
    <div className={`rounded-lg border p-4 bg-card shadow-sm relative overflow-hidden ${experiment._jsonError || disabledErrors.length ? "border-red-500" : ""}`}>
      <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary/20" />

      <div className="flex flex-wrap items-center gap-2 mb-3 pl-2">
        <Badge className="bg-primary/10 text-primary hover:bg-primary/20">Step {index + 1}</Badge>
        <span className="font-semibold">{experiment.name}</span>
      </div>

      <div className="pl-2 space-y-4">
        <p className="text-sm text-foreground/90">{experiment.plan_summary}</p>

        <div className="flex flex-wrap gap-2">
          {summaryFields.map(({ field, value }) => (
            <Badge key={field.path} variant="outline" className="bg-muted/30">
              {field.label}: <span className="ml-1 font-mono">{formatFieldValue(value)}</span>
            </Badge>
          ))}
        </div>

        {disabledErrors.length > 0 && (
          <p className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-300">
            {disabledErrors[0]}
          </p>
        )}

        <Tabs value={experiment._activeTab ?? "form"} onValueChange={(value) => onTabChange(value as "form" | "json")}>
          <TabsList>
            <TabsTrigger value="form">Form</TabsTrigger>
            <TabsTrigger value="json">JSON</TabsTrigger>
          </TabsList>
          <TabsContent value="form">
            <div className="grid gap-3 lg:grid-cols-2">
              {Object.entries(grouped).map(([section, sectionFields]) => (
                <div key={section} className="space-y-2 rounded-md border border-border/70 bg-muted/20 p-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{section}</p>
                  <div className="space-y-2">
                    {sectionFields.map((field) => (
                      <SchemaFieldControl
                        key={field.path}
                        config={fullConfig}
                        disabled={disabled}
                        field={field}
                        value={getValueByPath(fullConfig, field.path) ?? field.defaultValue}
                        onChange={(value) => onConfigChange(field.path, coerceFieldValue(field, value))}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>
          <TabsContent value="json">
            <div className="space-y-1">
              <Textarea
                className={`font-mono text-xs min-h-[220px] bg-muted/50 ${experiment._jsonError ? "focus-visible:ring-red-500" : ""}`}
                value={jsonString}
                onChange={(e) => onJsonChange(e.target.value)}
                disabled={disabled}
              />
              {experiment._jsonError && <p className="text-[10px] text-red-500 font-medium mt-1">Invalid JSON format</p>}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

function SchemaFieldControl({
  config,
  disabled,
  field,
  value,
  onChange,
}: {
  config: Record<string, unknown>;
  disabled: boolean;
  field: AgentSchemaField;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const compatibilityHint = fieldCompatibilityHint(field.path, config);
  const boolDisabled = disabled || booleanDisabledForConfig(field.path, value, config);
  const boolReason = booleanDisableReasonForConfig(field.path, value, config);
  const textOptions = field.options.map((option) => String(option));
  const textValue = formatFieldValue(value);
  const selectedTextOption = textOptions.includes(textValue) ? textValue : "";
  const customTextOption = textValue !== "-" && selectedTextOption === "" ? textValue : null;

  return (
    <div className="space-y-1 rounded-md border border-border/70 bg-background/60 p-2">
      <p className="text-xs text-muted-foreground">{field.label}</p>
      {field.type === "select" && (
        <Select value={String(value ?? "")} onValueChange={onChange} disabled={disabled}>
          <SelectTrigger className="font-mono">
            <SelectValue placeholder="Select option" />
          </SelectTrigger>
          <SelectContent>
            {field.options.map((option) => {
              const meta = optionMeta(field.definition, option);
              const optionDisabled = optionDisabledForConfig(field.definition, option, field.path, config);
              const reason = optionDisableReasonForConfig(field.definition, option, field.path, config);
              return (
                <SelectItem
                  key={`${field.path}-${String(option)}`}
                  value={String(option)}
                  disabled={optionDisabled}
                  className="font-mono"
                >
                  <span className="flex flex-col gap-0.5">
                    <span className="flex items-center gap-2">
                      {optionLabel(field.definition, option)}
                      {typeof meta.badge === "string" && <span className="text-[10px] text-amber-400">{meta.badge}</span>}
                    </span>
                    {reason && <span className="text-[10px] text-muted-foreground">{reason}</span>}
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
          disabled={disabled}
          inputClassName="font-mono"
        />
      )}
      {field.type === "bool" && (
        <div className="flex h-10 items-center justify-between rounded-md border border-input bg-background/40 px-3">
          <span className="text-sm">{formatFieldValue(value)}</span>
          <Switch checked={Boolean(value)} onCheckedChange={onChange} disabled={boolDisabled} />
        </div>
      )}
      {field.type !== "select" && field.type !== "number" && field.type !== "bool" && (
        textOptions.length > 0 ? (
          <Select value={customTextOption ?? selectedTextOption} onValueChange={onChange} disabled={disabled}>
            <SelectTrigger
              className="font-mono overflow-hidden [&>span]:block [&>span]:truncate [&>span]:whitespace-nowrap"
              title={textValue}
            >
              <SelectValue placeholder="Choose option" />
            </SelectTrigger>
            <SelectContent className="max-w-[min(36rem,calc(100vw-2rem))]">
              {customTextOption && (
                <SelectItem
                  value={customTextOption}
                  className="font-mono"
                  title={customTextOption}
                >
                  <span className="block max-w-full truncate">{customTextOption}</span>
                </SelectItem>
              )}
              {field.options.map((option) => {
                const itemDisabled = optionDisabledForConfig(field.definition, option, field.path, config);
                const label = optionLabel(field.definition, option);
                return (
                  <SelectItem
                    key={`${field.path}-text-option-${String(option)}`}
                    value={String(option)}
                    disabled={itemDisabled}
                    className="font-mono"
                    title={String(option)}
                  >
                    <span className="block max-w-full truncate">{label}</span>
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
        ) : (
          <Input
            className="font-mono"
            disabled={disabled}
            value={textValue}
            onChange={(event) => onChange(event.target.value)}
          />
        )
      )}
      {compatibilityHint && <p className="text-[11px] text-muted-foreground">{compatibilityHint}</p>}
      {boolReason && <p className="text-[11px] text-muted-foreground">{boolReason}</p>}
    </div>
  );
}
