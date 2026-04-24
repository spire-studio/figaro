import { Tabs, TabsList, TabsTrigger } from "../components/ui/tabs";
import { DistributedClientPanel } from "./distributed/DistributedClientPanel";
import { DistributedServerPanel } from "./distributed/DistributedServerPanel";
import type { DistributedPageProps, DistributedRole } from "./types";

export function DistributedPage(props: DistributedPageProps) {
  const { distributedRole, setDistributedRole } = props;

  return (
    <div className="space-y-0">
      <div className="flex items-center justify-end rounded-t-lg border border-b-0 border-slate-200/80 bg-slate-50/70 px-4 py-3 dark:border-slate-700/60 dark:bg-slate-900/40">
        <Tabs value={distributedRole} onValueChange={(value) => setDistributedRole(value as DistributedRole)}>
          <TabsList>
            <TabsTrigger value="server">Server</TabsTrigger>
            <TabsTrigger value="client">Client</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      <div className="rounded-b-lg border bg-card/30 p-4">
        {distributedRole === "server" && <DistributedServerPanel {...props} />}
        {distributedRole === "client" && <DistributedClientPanel {...props} />}
      </div>
    </div>
  );
}
