import { useState } from "react";
import { Play, Clock, Cpu, AlertTriangle, Sparkles, MessageSquare, ClipboardList, Target } from "lucide-react";
import { Button } from "../../../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../../../components/ui/card";
import { Badge } from "../../../components/ui/badge";
import { Input } from "../../../components/ui/input";
import { Textarea } from "../../../components/ui/textarea";
import { toast } from "sonner";
import { agentApi } from "../../../api/agent";
import type { AgentPageProps } from "../../../pages/types";

export function AgentPlanPreview(props: AgentPageProps) {
  const { draftPlan, busy: globalBusy, handleExecutePlan, setWorkflowStep, setDraftPlan } = props;

  const [localExperiments, setLocalExperiments] = useState(() => 
    draftPlan?.experiments ? JSON.parse(JSON.stringify(draftPlan.experiments)) : []
  );
  const [instruction, setInstruction] = useState("");
  const [isRevising, setIsRevising] = useState(false);

  if (!draftPlan) return null;
  const isBusy = globalBusy || isRevising;

  const handleRevise = async () => {
    if (!instruction.trim() || !draftPlan.job_id) return;
    setIsRevising(true);
    try {
      const response = await agentApi.revisePlan(draftPlan.job_id, { instruction });
      setLocalExperiments(response.experiments);
      setDraftPlan({
        ...draftPlan,
        experiments: response.experiments
      });
      setInstruction("");
      toast.success("Plan updated successfully by Agent!");
    } catch (error: any) {
      toast.error(error.message || "Failed to update plan via Agent.");
    } finally {
      setIsRevising(false);
    }
  };

  const handleConfigChange = (index: number, newJsonString: string) => {
    const updated = [...localExperiments];
    try {
      updated[index].config_patch = JSON.parse(newJsonString);
      updated[index]._jsonError = false;
    } catch (e) {
      updated[index]._jsonError = true;
    }
    updated[index]._rawJsonString = newJsonString;
    setLocalExperiments(updated);
  };

  const onRunClick = () => {
    const hasErrors = localExperiments.some((exp: any) => exp._jsonError);
    if (hasErrors) {
      toast.error("Please fix invalid JSON formatting before running.");
      return;
    }
    const cleanExperiments = localExperiments.map((exp: any) => {
      const { _rawJsonString, _jsonError, ...rest } = exp;
      return rest;
    });
    void handleExecutePlan(cleanExperiments);
  };

  const experiments = draftPlan.experiments || [];
  const totalMinutes = experiments.reduce((acc, curr) => acc + (curr.estimated_minutes || 0), 0);

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

      {/* Resource Estimation Summary */}
      {/* <div className="grid grid-cols-3 gap-4">
        <Card className="bg-muted/30">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="p-2 bg-blue-100 dark:bg-blue-900/40 rounded-full text-blue-600"><Clock className="h-5 w-5" /></div>
            <div>
              <p className="text-xs text-muted-foreground">Est. Total Time</p>
              <p className="font-semibold">{totalMinutes > 0 ? `~${totalMinutes} mins` : "Unknown"}</p>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-muted/30">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="p-2 bg-purple-100 dark:bg-purple-900/40 rounded-full text-purple-600"><Cpu className="h-5 w-5" /></div>
            <div>
              <p className="text-xs text-muted-foreground">Peak VRAM Required</p>
              <p className="font-semibold">
                {experiments.length > 0 && experiments[0].estimated_gpu_vram_gb 
                  ? `~${experiments[0].estimated_gpu_vram_gb} GB` : "Unknown"}
              </p>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-muted/30">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="p-2 bg-amber-100 dark:bg-amber-900/40 rounded-full text-amber-600"><AlertTriangle className="h-5 w-5" /></div>
            <div>
              <p className="text-xs text-muted-foreground">System Mode</p>
              <p className="font-semibold capitalize">Simulation</p>
            </div>
          </CardContent>
        </Card>
      </div> */}

      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <ClipboardList className="h-5 w-5 text-primary" />
            Proposed Execution Plan
          </CardTitle>
          <CardDescription>
            The agent has defined the following {localExperiments.length} steps to execute. Edit the config patch manually or use the AI input above.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {localExperiments.map((exp: any, idx: number) => {
              const jsonString = exp._rawJsonString ?? JSON.stringify(exp.config_patch, null, 2);
              return (
                <div key={idx} className={`rounded-lg border p-4 bg-card shadow-sm relative overflow-hidden ${exp._jsonError ? 'border-red-500' : ''}`}>
                  
                  <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary/20"></div>

                  <div className="flex items-center gap-2 mb-3 pl-2">
                    <Badge className="bg-primary/10 text-primary hover:bg-primary/20">Step {idx + 1}</Badge>
                    <span className="font-semibold">{exp.name}</span>
                  </div>
                  
                  <div className="pl-2">
                    <p className="text-sm text-foreground/90 mb-4">{exp.plan_summary}</p>
                    
                    <div className="space-y-1">
                      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Configuration Parameters</p>
                      <Textarea 
                        className={`font-mono text-xs min-h-[120px] bg-muted/50 ${exp._jsonError ? 'focus-visible:ring-red-500' : ''}`}
                        value={jsonString}
                        onChange={(e) => handleConfigChange(idx, e.target.value)}
                        disabled={isBusy}
                      />
                      {exp._jsonError && <p className="text-[10px] text-red-500 font-medium mt-1">Invalid JSON format</p>}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}