import { BriefcaseBusiness, Sparkles, LayoutDashboard, LineChart } from "lucide-react";
import { Card, CardContent } from "./components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "./components/ui/tabs";

import { useAgentController } from "./features/agent/useAgentController";
import { AgentExperimentStudio } from "./features/agent/components/AgentExperimentStudio";
import { AgentPlanPreview } from "./features/agent/components/AgentPlanPreview";
import { AgentRunDashboard } from "./features/agent/components/AgentRunDashboard";
import { AgentResultsCompare } from "./features/agent/components/AgentResultsCompare";

export default function App() {
  const agentProps = useAgentController();
  const { workflowStep, setWorkflowStep, draftPlan } = agentProps;

  const activeTab = 
    (workflowStep === "home" || workflowStep === "preview") ? "studio" :
    (workflowStep === "running") ? "dashboard" : 
    "results";

  const handleTabChange = (val: string) => {
    if (val === "studio") {
      setWorkflowStep(draftPlan ? "preview" : "home");
    } else if (val === "dashboard") {
      setWorkflowStep("running");
    } else if (val === "results") {
      setWorkflowStep("results");
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <div className="mx-auto w-full max-w-[1600px] space-y-4 p-4 md:p-6 flex-1 flex flex-col h-screen overflow-hidden">
        
        {/* 顶层全局导航栏 */}
        <Card className="shrink-0 border-primary/20 shadow-sm bg-card/50 backdrop-blur">
          <CardContent className="flex flex-wrap items-center justify-between gap-3 p-4">
            <div className="flex items-center gap-2">
              <div className="p-1.5 bg-primary/10 rounded-md">
                <BriefcaseBusiness className="h-5 w-5 text-primary" />
              </div>
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
                Figaro Agent Studio
              </h1>
            </div>
            
            <div className="flex items-center gap-2">
              <Tabs value={activeTab} onValueChange={handleTabChange}>
                <TabsList className="grid w-[500px] grid-cols-3">
                  <TabsTrigger value="studio" className="flex gap-2">
                    <Sparkles className="h-4 w-4" /> Agent Studio
                  </TabsTrigger>
                  <TabsTrigger value="dashboard" className="flex gap-2">
                    <LayoutDashboard className="h-4 w-4" /> Run Dashboard
                  </TabsTrigger>
                  <TabsTrigger value="results" className="flex gap-2">
                    <LineChart className="h-4 w-4" /> Results & History
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
          </CardContent>
        </Card>

        {/* 动态内容渲染区 */}
        <div className="flex-1 min-h-0 overflow-hidden">
          {activeTab === "studio" && workflowStep === "home" && <AgentExperimentStudio {...agentProps} />}
          {activeTab === "studio" && workflowStep === "preview" && <AgentPlanPreview {...agentProps} />}
          {activeTab === "dashboard" && <AgentRunDashboard {...agentProps} />}
          {activeTab === "results" && <AgentResultsCompare {...agentProps} />}
        </div>
      </div>
    </div>
  );
}