import { useState } from "react";
import { BriefcaseBusiness, Loader2 } from "lucide-react";

import { Card, CardContent } from "./components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "./components/ui/tabs";
import { useAgentController } from "./features/agent/useAgentController";
import { useDistributedController } from "./features/distributed/useDistributedController";
import { useSimulationController } from "./features/simulation/useSimulationController";
import { AgentPage } from "./pages/AgentPage";
import { DistributedPage } from "./pages/DistributedPage";
import { SimulationPage } from "./pages/SimulationPage";
import type { PageMode } from "./pages/types";

export default function App() {
  const [pageMode, setPageMode] = useState<PageMode>("agent");
  const simulationPageProps = useSimulationController();
  const agentPageProps = useAgentController();
  const distributedPageProps = useDistributedController({
    pageMode,
    configSchema: simulationPageProps.configSchema,
  });
  const busy = simulationPageProps.busy || distributedPageProps.busy || agentPageProps.busy;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto w-full max-w-[1600px] space-y-4 p-4 md:p-6">
        <Card>
          <CardContent className="flex flex-wrap items-center justify-between gap-3 p-4">
            <div className="flex items-center gap-2">
              <BriefcaseBusiness className="h-5 w-5" />
              <h1 className="text-lg font-semibold tracking-tight">Figaro Control Center</h1>
            </div>
            <div className="flex items-center gap-2">
              <Tabs value={pageMode} onValueChange={(value) => setPageMode(value as PageMode)}>
                <TabsList>
                  <TabsTrigger value="agent">Agent</TabsTrigger>
                  <TabsTrigger value="simulation">Simulation</TabsTrigger>
                  <TabsTrigger value="distributed">Distributed</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
          </CardContent>
        </Card>

        {pageMode === "simulation" && <SimulationPage {...simulationPageProps} busy={busy} />}
        {pageMode === "distributed" && <DistributedPage {...distributedPageProps} busy={busy} />}
        {pageMode === "agent" && <AgentPage {...agentPageProps} />}

        {busy && (
          <div className="fixed bottom-4 right-4 inline-flex items-center gap-2 rounded-md border bg-card px-3 py-2 text-xs text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Processing...
          </div>
        )}
      </div>
    </div>
  );
}
