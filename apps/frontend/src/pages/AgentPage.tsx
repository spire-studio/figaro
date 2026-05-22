import { useState } from "react";
import { BarChart3, Bot, ChevronLeft, FlaskConical, Loader2, Play, RotateCcw, Sparkles } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { AgentOptimizationObjective } from "../api/agent";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Textarea } from "../components/ui/textarea";
import { AgentExperimentStudio } from "../features/agent/components/AgentExperimentStudio";
import { AgentPlanPreview } from "../features/agent/components/AgentPlanPreview";
import { AgentRunDashboard } from "../features/agent/components/AgentRunDashboard";
import { AgentResultsCompare } from "../features/agent/components/AgentResultsCompare";
import type { AgentPageProps } from "./types";

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function getLastAccuracy(metrics: Record<string, unknown> | null): number | null {
  const globalResults = asRecord(metrics?.global_results);
  const accuracy = globalResults?.global_accuracy;
  if (!Array.isArray(accuracy)) return null;
  const nums = accuracy.filter((item): item is number => typeof item === "number");
  return nums.length > 0 ? nums[nums.length - 1] : null;
}

function getRounds(metrics: Record<string, unknown> | null): number {
  const globalResults = asRecord(metrics?.global_results);
  const rounds = globalResults?.rounds;
  return Array.isArray(rounds) ? rounds.length : 0;
}

function fmt(value: number | null): string {
  return value === null ? "-" : value.toFixed(4);
}

function getConfigLabel(config: Record<string, unknown> | null): string {
  if (!config) return "-";
  const dataset = asRecord(config.dataset);
  const federated = asRecord(config.federated);
  const parts: string[] = [];
  if (dataset) {
    const name = dataset.name ?? "";
    const dist = dataset.distribution ?? "";
    const alpha = dataset.alpha;
    parts.push(`${name} ${dist}${alpha !== undefined ? ` α=${alpha}` : ""}`);
  }
  if (federated) {
    parts.push(`${federated.num_clients ?? "?"}c/${federated.num_rounds ?? "?"}r`);
  }
  return parts.join(" · ") || "-";
}

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

export function AgentPage(props: AgentPageProps) {
  // Select which view to render based on current workflow step
  return (
    <div className="flex h-[calc(100vh-140px)] gap-6">
      <main className="flex-1 min-w-0 h-full">
        {props.workflowStep === "home" && <AgentExperimentStudio {...props} />}
        {props.workflowStep === "preview" && <AgentPlanPreview {...props} />}
        {props.workflowStep === "running" && <AgentRunDashboard {...props} />}
        {props.workflowStep === "results" && <AgentResultsCompare {...props} />}
      </main>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Shared Results Table                                               */
/* ------------------------------------------------------------------ */

function ResultsTable({ experiments }: { experiments: Array<{ iteration_goal?: string | null; plan_summary?: string | null; config?: Record<string, unknown> | null; metrics?: unknown; score?: number | null; decision?: string | null; run_id?: string }> }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            <th className="pb-2 pr-3">#</th>
            <th className="pb-2 pr-3">Experiment</th>
            <th className="pb-2 pr-3">Config</th>
            <th className="pb-2 pr-3 text-right">Accuracy</th>
            <th className="pb-2 pr-3 text-right">Rounds</th>
            <th className="pb-2 text-right">Status</th>
          </tr>
        </thead>
        <tbody>
          {experiments.map((exp, idx) => {
            const metrics = exp.metrics as Record<string, unknown> | null;
            const config = exp.config ?? null;
            const acc = getLastAccuracy(metrics);
            const rounds = getRounds(metrics);
            const isRunning = exp.decision === null || exp.decision === "pending";
            const allAccs = experiments.map((e) => getLastAccuracy(e.metrics as Record<string, unknown> | null)).filter((v): v is number => v !== null);
            const bestAcc = allAccs.length > 0 ? Math.max(...allAccs) : null;
            const isBest = acc !== null && bestAcc !== null && acc === bestAcc && experiments.length > 1;

            return (
              <tr key={exp.run_id ?? idx} className="border-b last:border-0">
                <td className="py-2 pr-3 text-xs text-muted-foreground">{idx + 1}</td>
                <td className="py-2 pr-3 text-xs font-medium">{exp.iteration_goal?.replace("Run experiment: ", "") ?? `Exp ${idx + 1}`}</td>
                <td className="py-2 pr-3 text-[11px] text-muted-foreground">{getConfigLabel(config)}</td>
                <td className="py-2 pr-3 text-right font-mono text-xs">
                  {isBest ? <span className="font-semibold text-emerald-600">{fmt(acc)}</span> : fmt(acc)}
                </td>
                <td className="py-2 pr-3 text-right text-xs">{rounds || "-"}</td>
                <td className="py-2 text-right">
                  {isRunning ? <Badge variant="warning" className="text-[10px]">running</Badge>
                    : isBest ? <Badge variant="success" className="text-[10px]">best</Badge>
                    : <Badge variant="outline" className="text-[10px]">done</Badge>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
