import { useEffect, useState } from "react";
import { History, Bot, Activity, CheckCircle2, Trophy, ArrowRight, Target, Calendar } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../../../components/ui/card";
import { Badge } from "../../../components/ui/badge";
import { Button } from "../../../components/ui/button";
import { Separator } from "../../../components/ui/separator";
import { MiniLineChart } from "../../simulation/components/MiniLineChart";
import { fmt } from "../../../lib/time";
import type { AgentPageProps } from "../../../pages/types";
import { baseUrl } from "../../../../src/api/client";

const CHART_COLORS = [
  "hsl(var(--primary))", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"
];

export function AgentResultsCompare(props: AgentPageProps) {
  const { historyJobs, selectedHistory, selectHistoryJob, setWorkflowStep } = props;

  const jobs = historyJobs || [];
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<any>(null);

  const experiments = selectedHistory?.experiments || [];
  const bestExp = experiments.reduce((prev: any, curr: any) => 
    (curr.score || 0) > (prev?.score || 0) ? curr : prev, experiments[0] || null);

  useEffect(() => {
    if (bestExp) {
      setSelectedRunId(bestExp.run_id);
    } else {
      setSelectedRunId(null);
      setMetrics(null);
    }
  }, [selectedHistory]);

  useEffect(() => {
    if (!selectedRunId) {
      setMetrics(null);
      return;
    }
    fetch(`${baseUrl}/api/v1/agent/runs/${selectedRunId}/metrics`)
      .then(res => res.json())
      .then(data => {
        if (data.metrics) setMetrics(data.metrics);
      })
      .catch(() => {});
  }, [selectedRunId]);

  const globalResults = metrics?.global_results || {};
  const clientResults = metrics?.client_results || {};
  const actualDataLength = globalResults.global_accuracy?.length || 0;
  const rounds = globalResults.rounds ? globalResults.rounds.slice(0, actualDataLength) : Array.from({ length: actualDataLength }, (_, i) => i + 1);
  const safeSlice = (arr: any[]) => (arr || []).slice(0, actualDataLength);

  const globalAccSeries = [{ key: "g_acc", label: "Global Accuracy", color: CHART_COLORS[0], values: safeSlice(globalResults.global_accuracy) }];
  const globalLossSeries = [{ key: "g_loss", label: "Global Loss", color: CHART_COLORS[3], values: safeSlice(globalResults.global_loss) }];
  const clientIds = Object.keys(clientResults);
  const clientTrainAccSeries = clientIds.map((cId, idx) => ({ key: `${cId}_train_acc`, label: cId, color: CHART_COLORS[(idx + 1) % CHART_COLORS.length], values: safeSlice(clientResults[cId].train_acc) }));
  const clientTrainLossSeries = clientIds.map((cId, idx) => ({ key: `${cId}_train_loss`, label: cId, color: CHART_COLORS[(idx + 1) % CHART_COLORS.length], values: safeSlice(clientResults[cId].train_loss) }));
  const clientTestAccSeries = clientIds.map((cId, idx) => ({ key: `${cId}_test_acc`, label: cId, color: CHART_COLORS[(idx + 1) % CHART_COLORS.length], values: safeSlice(clientResults[cId].test_acc) }));
  const clientTestLossSeries = clientIds.map((cId, idx) => ({ key: `${cId}_test_loss`, label: cId, color: CHART_COLORS[(idx + 1) % CHART_COLORS.length], values: safeSlice(clientResults[cId].test_loss) }));

  return (
    <div className="grid h-full gap-4 xl:grid-cols-[300px_1fr]">
      
      <Card className="flex flex-col min-h-0 bg-muted/10 border-r shadow-none rounded-none sm:rounded-xl">
        <CardHeader className="pb-3 px-4">
          <CardTitle className="text-sm font-bold flex items-center gap-2">
            <History className="h-4 w-4 text-primary" /> Job History
          </CardTitle>
          <CardDescription className="text-xs">Past agent optimizations</CardDescription>
        </CardHeader>
        <CardContent className="flex-1 overflow-auto space-y-3 px-3 pb-4">
          {jobs.length === 0 && <div className="text-xs text-muted-foreground text-center py-8">No history yet.</div>}
          
          {jobs.map((job: any) => {
            const isSelected = selectedHistory?.optimization_job_id === job.optimization_job_id;
            const isCompleted = job.status === "completed";
            
            return (
              <div 
                key={job.optimization_job_id} 
                onClick={() => selectHistoryJob(job.optimization_job_id)}
                className={`flex flex-col p-3 rounded-lg border text-xs cursor-pointer transition-all hover:shadow-sm
                  ${isSelected ? 'bg-primary/5 border-primary/40 ring-1 ring-primary/20' : 'bg-card border-border hover:border-primary/30'}`}
              >
                <div className="flex justify-between items-start mb-2">
                  <span className={`font-bold ${isSelected ? 'text-primary' : 'text-foreground'}`}>
                    {job.job_name || `Job #${job.optimization_job_id}`}
                  </span>
                  <Badge variant={isCompleted ? "outline" : "secondary"} className={`text-[9px] px-1.5 py-0 h-4 ${isCompleted ? 'border-emerald-500 text-emerald-600' : ''}`}>
                    {job.status}
                  </Badge>
                </div>
                <div className="flex items-start gap-1.5 text-muted-foreground mb-2">
                  <Target className="h-3.5 w-3.5 shrink-0 mt-0.5 opacity-70" />
                  <span className="line-clamp-2 leading-relaxed">{job.goal}</span>
                </div>
                <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground/70 mt-auto pt-2 border-t">
                  <Calendar className="h-3 w-3" />
                  <span>{fmt(job.created_at)}</span>
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      <div className="flex flex-col min-h-0 overflow-auto pr-2 gap-4 pb-6">
        {!selectedHistory ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground opacity-60">
            <History className="h-16 w-16 mb-4 opacity-20" />
            <p>Select a historical job from the sidebar to view its insights and charts.</p>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold tracking-tight">{selectedHistory.job_name || `Optimization Job #${selectedHistory.optimization_job_id}`}</h2>
                <p className="text-muted-foreground text-sm mt-1 flex items-center gap-2">
                  <Target className="h-4 w-4" /> {selectedHistory.goal}
                </p>
              </div>
              <Button onClick={() => setWorkflowStep("home")} size="sm">
                Start New Agent Job <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <Card className="border-emerald-200 bg-emerald-50/30 dark:border-emerald-900/40 dark:bg-emerald-950/20 shadow-sm">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-emerald-100 dark:bg-emerald-900/60 rounded-full">
                      <Trophy className="h-5 w-5 text-emerald-600" />
                    </div>
                    <p className="font-bold text-emerald-800 dark:text-emerald-300">Optimal Configuration</p>
                  </div>
                  <div className="text-3xl font-black text-emerald-700 dark:text-emerald-400 mb-2">
                    {bestExp?.score != null ? `${(bestExp.score * 100).toFixed(2)}%` : "N/A"}
                  </div>
                  <p className="text-xs text-emerald-700/70 dark:text-emerald-400/70">
                    Experiment <strong>"{bestExp?.name}"</strong> yielded the highest accuracy.
                  </p>
                </CardContent>
              </Card>

              <Card className="shadow-sm border-primary/20 bg-primary/[0.02]">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2 text-primary font-bold">
                    <Bot className="h-4 w-4" /> Agent Summary Report
                  </CardTitle>
                </CardHeader>
                <CardContent className="text-sm leading-relaxed text-foreground/90 whitespace-pre-wrap">
                  {selectedHistory.summary_text || <span className="italic text-muted-foreground">No final report was generated by the agent for this job.</span>}
                </CardContent>
              </Card>
            </div>

            <div className="mt-4">
              <h3 className="text-sm font-bold tracking-tight flex items-center gap-2 mb-3">
                <Activity className="h-4 w-4 text-primary" /> 
                Explore Experiment Metrics
              </h3>
              <div className="flex flex-wrap gap-2">
                {experiments.map((exp: any, i: number) => {
                  const isBest = exp.run_id === bestExp?.run_id;
                  const isActive = selectedRunId === exp.run_id;
                  
                  return (
                    <Badge
                      key={exp.run_id}
                      variant={isActive ? "default" : "outline"}
                      className={`cursor-pointer px-3 py-1.5 text-xs transition-colors hover:bg-primary/80 hover:text-primary-foreground
                        ${isActive && isBest ? 'bg-emerald-600 hover:bg-emerald-700 text-white border-emerald-600' : ''}
                        ${!isActive && isBest ? 'border-emerald-500 text-emerald-600 bg-emerald-50 dark:bg-emerald-950' : ''}
                      `}
                      onClick={() => setSelectedRunId(exp.run_id)}
                    >
                      {exp.name || `Exp ${i + 1}`}
                      {isBest && <Trophy className="w-3 h-3 ml-1.5 inline-block" />}
                    </Badge>
                  )
                })}
              </div>
            </div>

            {selectedRunId ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Global Accuracy" xValues={rounds} series={globalAccSeries} formatter={(v: number) => `${(v * 100).toFixed(2)}%`} /></CardContent></Card>
                <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Global Loss" xValues={rounds} series={globalLossSeries} formatter={(v: number) => v.toFixed(4)} /></CardContent></Card>
                <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Client Train Accuracy" xValues={rounds} series={clientTrainAccSeries} formatter={(v: number) => `${(v * 100).toFixed(2)}%`} /></CardContent></Card>
                <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Client Test Accuracy" xValues={rounds} series={clientTestAccSeries} formatter={(v: number) => `${(v * 100).toFixed(2)}%`} /></CardContent></Card>
                <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Client Train Loss" xValues={rounds} series={clientTrainLossSeries} formatter={(v: number) => v.toFixed(4)} /></CardContent></Card>
                <Card className="shadow-sm"><CardContent className="p-4"><MiniLineChart title="Client Test Loss" xValues={rounds} series={clientTestLossSeries} formatter={(v: number) => v.toFixed(4)} /></CardContent></Card>
              </div>
            ) : (
               <div className="text-center py-10 text-sm text-muted-foreground border border-dashed rounded-lg">
                 Select an experiment above to load its charts.
               </div>
            )}
          </>
        )}
      </div>
      
    </div>
  );
}