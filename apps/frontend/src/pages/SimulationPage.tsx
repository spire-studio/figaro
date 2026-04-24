import { Tabs, TabsList, TabsTrigger } from "../components/ui/tabs";
import { SimulationJobsTab } from "./simulation/SimulationJobsTab";
import { SimulationRunsTab } from "./simulation/SimulationRunsTab";
import type { SimulationPageProps, TabKey } from "./types";

export function SimulationPage(props: SimulationPageProps) {
  const { activeTab, setActiveTab } = props;

  return (
    <div className="space-y-0">
      <div className="flex items-center justify-end rounded-t-lg border border-b-0 border-slate-200/80 bg-slate-50/70 px-4 py-3 dark:border-slate-700/60 dark:bg-slate-900/40">
        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as TabKey)}>
          <TabsList>
            <TabsTrigger value="jobs">Jobs</TabsTrigger>
            <TabsTrigger value="runs">Runs</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      <div className="rounded-b-lg border bg-card/30 p-4">
        {activeTab === "jobs" && <SimulationJobsTab {...props} />}
        {activeTab === "runs" && <SimulationRunsTab {...props} />}
      </div>
    </div>
  );
}
