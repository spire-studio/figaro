import { useEffect, useState } from "react";
import { Activity, Bot, TerminalSquare, CheckCircle2, CircleDashed, LayoutGrid } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../../../components/ui/card";
import { Separator } from "../../../components/ui/separator";
import { MiniLineChart } from "../../simulation/components/MiniLineChart";
import { fmt } from "../../../lib/time";
import type { AgentPageProps } from "../../../pages/types";
import { baseUrl } from "../../../../src/api/client";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const CHART_COLORS = [
  "hsl(var(--primary))", 
  "#10b981", // Emerald
  "#f59e0b", // Amber
  "#ef4444", // Red
  "#8b5cf6", // Violet
  "#ec4899", // Pink
  "#06b6d4"  // Cyan
];

export function AgentRunDashboard({ progress }: AgentPageProps) {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<any>(null);

  if (!progress) return null;

  const currentExp = progress.current_experiment;
  const currentPlan = progress.current_plan;
  const experiments = progress.experiments || [];
  const activeRunId = selectedRunId 
    || currentExp?.run_id 
    || (experiments.length > 0 ? experiments[experiments.length - 1].run_id : null);

  useEffect(() => {
    if (!activeRunId) {
      setLiveMetrics(null);
      return;
    }
    
    let isSubscribed = true;
    const fetchLiveMetrics = async () => {
      try {
        const res = await fetch(`${baseUrl}/api/v1/agent/runs/${activeRunId}/metrics`);
        if (res.ok && isSubscribed) {
          const data = await res.json();
          if (data.metrics) {
            setLiveMetrics(data.metrics);
          }
        }
      } catch (e) {
      }
    };

    fetchLiveMetrics();
    const intervalId = setInterval(fetchLiveMetrics, 2000);
    
    return () => {
      isSubscribed = false;
      clearInterval(intervalId);
    };
  }, [activeRunId]);

  const activeExpMetadata = experiments.find((e: any) => e.run_id === activeRunId) 
                         || (currentExp?.run_id === activeRunId ? currentExp : null);

  const metrics = liveMetrics || activeExpMetadata?.metrics || {};
  const globalResults = metrics.global_results || {};
  const clientResults = metrics.client_results || {};
  const actualDataLength = globalResults.global_accuracy?.length || 0;

  const rounds = globalResults.rounds 
    ? globalResults.rounds.slice(0, actualDataLength) 
    : Array.from({ length: actualDataLength }, (_, i) => i + 1);

  const safeSlice = (arr: any[]) => (arr || []).slice(0, actualDataLength);

  // 1. Global Accuracy
  const globalAccSeries = [{ 
    key: "g_acc", label: "Global Accuracy", color: CHART_COLORS[0], 
    values: safeSlice(globalResults.global_accuracy) 
  }];

  // 2. Global Loss
  const globalLossSeries = [{ 
    key: "g_loss", label: "Global Loss", color: CHART_COLORS[3], 
    values: safeSlice(globalResults.global_loss) 
  }];

  const clientIds = Object.keys(clientResults);
  
  // 3. Client Train Acc
  const clientTrainAccSeries = clientIds.map((cId, idx) => ({
    key: `${cId}_train_acc`, label: cId, color: CHART_COLORS[(idx + 1) % CHART_COLORS.length],
    values: safeSlice(clientResults[cId].train_acc)
  }));

  // 4. Client Train Loss
  const clientTrainLossSeries = clientIds.map((cId, idx) => ({
    key: `${cId}_train_loss`, label: cId, color: CHART_COLORS[(idx + 1) % CHART_COLORS.length],
    values: safeSlice(clientResults[cId].train_loss)
  }));

  // 5. Client Test Acc
  const clientTestAccSeries = clientIds.map((cId, idx) => ({
    key: `${cId}_test_acc`, label: cId, color: CHART_COLORS[(idx + 1) % CHART_COLORS.length],
    values: safeSlice(clientResults[cId].test_acc)
  }));

  // 6. Client Test Loss
  const clientTestLossSeries = clientIds.map((cId, idx) => ({
    key: `${cId}_test_loss`, label: cId, color: CHART_COLORS[(idx + 1) % CHART_COLORS.length],
    values: safeSlice(clientResults[cId].test_loss)
  }));

  return (
    <div className="grid h-full gap-4 xl:grid-cols-[240px_1fr_300px]">
      <Card className="flex flex-col min-h-0 bg-muted/10 border-r shadow-none rounded-none sm:rounded-xl">
        <CardHeader className="pb-3 px-4">
          <CardTitle className="text-sm font-bold flex items-center gap-2">
            <LayoutGrid className="h-4 w-4 text-primary" /> Execution Queue
          </CardTitle>
          <CardDescription className="text-xs">{progress.completed_iterations} of {progress.max_iterations} done</CardDescription>
        </CardHeader>
        <CardContent className="flex-1 overflow-auto space-y-2 px-3 pb-4">
          {experiments.map((exp: any, idx: number) => {
            const isSelected = activeRunId === exp.run_id;
            return (
              <div 
                key={idx} 
                onClick={() => setSelectedRunId(exp.run_id)}
                className={`flex items-center justify-between p-2.5 rounded-md border text-xs cursor-pointer transition-colors
                  ${isSelected ? 'bg-primary/10 border-primary/40 shadow-sm' : 'bg-card hover:bg-muted border-transparent'}`}
              >
                <div className="flex flex-col min-w-0">
                  <span className={`font-semibold truncate ${isSelected ? 'text-primary' : 'text-foreground'}`}>
                    {`Exp ${idx + 1}`}
                  </span>
                  {exp.score != null && <span className="text-[10px] text-muted-foreground mt-0.5">Acc: {(exp.score * 100).toFixed(2)}%</span>}
                </div>
                <CheckCircle2 className="h-4 w-4 text-emerald-500 shrink-0 ml-2" />
              </div>
            );
          })}

          {progress.status === "running" && 
           progress.completed_iterations < progress.max_iterations && 
           currentPlan && 
           currentExp?.run_id && 
           !experiments.some((e: any) => e.run_id === currentExp.run_id) && (
            <div 
              onClick={() => setSelectedRunId(currentExp.run_id)}
              className={`p-2.5 rounded-md border cursor-pointer transition-colors
                ${activeRunId === currentExp.run_id ? 'bg-primary/10 border-primary shadow-sm' : 'bg-card border-primary/40 hover:bg-muted'}`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-bold text-primary truncate">Running: </span>
                <CircleDashed className="h-3.5 w-3.5 text-primary animate-spin shrink-0 ml-2" />
              </div>
              <div className="text-[10px] text-muted-foreground truncate">Round {currentExp.iteration} ...</div>
            </div>
          )}

          {progress.status === "running" && 
           progress.completed_iterations >= progress.max_iterations && (
            <div className="p-2.5 rounded-md border bg-primary/5 border-primary/50 shadow-sm transition-colors cursor-default">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-bold text-primary truncate">Generating Agent Summary</span>
                <CircleDashed className="h-3.5 w-3.5 text-primary animate-spin shrink-0 ml-2" />
              </div>
              <div className="text-[10px] text-muted-foreground truncate flex items-center gap-1">
                <Bot className="h-3 w-3" /> Analyzing metrics & drafting report...
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex flex-col min-h-0 overflow-auto pr-2">
        <div className="mb-4">
          <h3 className="text-lg font-bold tracking-tight flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" /> 
            Live Metrics <span className="text-muted-foreground text-sm font-normal">({"Initializing..."})</span>
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pb-4">
          {/* Chart 1: Global Acc */}
          <Card className="shadow-sm"><CardContent className="p-4">
            <MiniLineChart title="Global Accuracy" xValues={rounds} series={globalAccSeries} formatter={(v: number) => `${(v * 100).toFixed(2)}%`} />
          </CardContent></Card>
          
          {/* Chart 2: Global Loss */}
          <Card className="shadow-sm"><CardContent className="p-4">
            <MiniLineChart title="Global Loss" xValues={rounds} series={globalLossSeries} formatter={(v: number) => v.toFixed(4)} />
          </CardContent></Card>

          {/* Chart 3: Client Train Acc */}
          <Card className="shadow-sm"><CardContent className="p-4">
            <MiniLineChart title="Client Train Accuracy" xValues={rounds} series={clientTrainAccSeries} formatter={(v: number) => `${(v * 100).toFixed(2)}%`} />
          </CardContent></Card>

          {/* Chart 4: Client Test Acc */}
          <Card className="shadow-sm"><CardContent className="p-4">
            <MiniLineChart title="Client Test Accuracy" xValues={rounds} series={clientTestAccSeries} formatter={(v: number) => `${(v * 100).toFixed(2)}%`} />
          </CardContent></Card>

          {/* Chart 5: Client Train Loss */}
          <Card className="shadow-sm"><CardContent className="p-4">
            <MiniLineChart title="Client Train Loss" xValues={rounds} series={clientTrainLossSeries} formatter={(v: number) => v.toFixed(4)} />
          </CardContent></Card>

          {/* Chart 6: Client Test Loss */}
          <Card className="shadow-sm"><CardContent className="p-4">
            <MiniLineChart title="Client Test Loss" xValues={rounds} series={clientTestLossSeries} formatter={(v: number) => v.toFixed(4)} />
          </CardContent></Card>
        </div>
      </div>

      <div className="flex flex-col gap-4 min-h-0 overflow-auto pb-4">
        <Card className="border-primary/20 bg-primary/[0.02]">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2 text-primary font-bold">
              <Bot className="h-5 w-5" /> Agent Insights
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-xs">
            {progress.status === "completed" ? (
              <div>
                <p className="font-bold mb-2 uppercase tracking-wider text-[10px] text-muted-foreground">Final Summary</p>
                <div className="text-foreground/90 text-sm leading-relaxed max-w-none">
                  {progress.summary_text ? (
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      components={{
                        // 自定义 Markdown 元素的 Tailwind 样式
                        p: ({node, ...props}) => <p className="mb-3 last:mb-0" {...props} />,
                        ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-3 space-y-1" {...props} />,
                        strong: ({node, ...props}) => <strong className="font-semibold text-foreground" {...props} />,
                        table: ({node, ...props}) => (
                          <div className="my-4 w-full overflow-y-auto">
                            <table className="w-full text-left border-collapse text-xs" {...props} />
                          </div>
                        ),
                        th: ({node, ...props}) => <th className="border-b bg-muted/50 px-3 py-2 font-medium" {...props} />,
                        td: ({node, ...props}) => <td className="border-b px-3 py-2 text-muted-foreground" {...props} />,
                      }}
                    >
                      {progress.summary_text}
                    </ReactMarkdown>
                  ) : (
                    "The agent is finalizing the comparison report..."
                  )}
                </div>
              </div>
            ) : (
              <>
                {currentPlan?.hypothesis && (
                  <div>
                    <p className="font-bold mb-1 uppercase tracking-wider text-[10px] text-muted-foreground">Hypothesis</p>
                    <p className="text-foreground/90 italic">"{currentPlan.hypothesis}"</p>
                  </div>
                )}
                <Separator className="opacity-50" />
                {currentPlan?.rationale && currentPlan.rationale.length > 0 && (
                  <div>
                    <p className="font-bold mb-2 uppercase tracking-wider text-[10px] text-muted-foreground">Reasoning</p>
                    <ul className="space-y-1.5">
                      {currentPlan.rationale.map((r: string, i: number) => (
                        <li key={i} className="flex gap-2 text-muted-foreground leading-relaxed">
                          <span className="text-primary mt-0.5">•</span><span>{r}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </div>
      
    </div>
  );
}