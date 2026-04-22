import { Sparkles, ArrowRight, Loader2 } from "lucide-react";
import { Button } from "../../../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../../../components/ui/card";
import { Textarea } from "../../../components/ui/textarea";
import { Input } from "../../../components/ui/input";
import type { AgentPageProps } from "../../../pages/types";

export function AgentExperimentStudio(props: AgentPageProps) {
  const { goal, setGoal, jobName, setJobName, presets, busy, handleGeneratePlan } = props;

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
            Generate Experiment Plan <ArrowRight className="ml-2 h-5 w-5" />
          </>
        )}
      </Button>
    </div>
  );
}