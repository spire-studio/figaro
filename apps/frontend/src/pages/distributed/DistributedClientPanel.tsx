import { Loader2, Play, Square } from "lucide-react";

import { Badge } from "../../components/ui/badge";
import { Button } from "../../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card";
import { Input } from "../../components/ui/input";
import type { DistributedPageProps } from "../types";

export function DistributedClientPanel(props: DistributedPageProps) {
  const {
    busy,
    clientConnecting,
    clientConnectionMessage,
    clientName,
    clientServerApiBase,
    clientSessionId,
    connectedClientId,
    connectedClientInfo,
    distributedClientStatusVariant,
    handleClientCancelConnect,
    handleClientConnect,
    handleClientReady,
    notifyError,
    setClientName,
    setClientServerApiBase,
    setClientSessionId,
  } = props;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Client Connection</CardTitle>
        <CardDescription>Connect to a server session, wait for approval, then mark this client as ready.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-2 md:grid-cols-2">
          <div className="md:col-span-2">
            <p className="mb-1 text-xs text-muted-foreground">Server API Base URL</p>
            <Input
              value={clientServerApiBase}
              onChange={(event) => setClientServerApiBase(event.target.value)}
              placeholder="http://localhost:8000"
            />
          </div>
          <div>
            <p className="mb-1 text-xs text-muted-foreground">Session ID</p>
            <Input value={clientSessionId} onChange={(event) => setClientSessionId(event.target.value)} placeholder="session id" />
          </div>
          <div>
            <p className="mb-1 text-xs text-muted-foreground">Client Name</p>
            <Input value={clientName} onChange={(event) => setClientName(event.target.value)} placeholder="client-host-01" />
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button onClick={() => handleClientConnect().catch((err: unknown) => notifyError(err))} disabled={clientConnecting || connectedClientId !== null}>
            {clientConnecting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
            Connect
          </Button>
          <Button
            variant="destructive"
            onClick={() => handleClientCancelConnect().catch((err: unknown) => notifyError(err))}
            disabled={!clientConnecting && connectedClientId === null}
          >
            <Square className="mr-2 h-4 w-4" />
            Cancel
          </Button>
        </div>

        <div className="rounded-md border bg-muted/20 p-3 text-sm">
          <p className="text-xs text-muted-foreground">Connection Status</p>
          <p>{clientConnectionMessage}</p>
          {connectedClientInfo && (
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <span className="font-mono text-xs">{connectedClientInfo.id}</span>
              <Badge variant={distributedClientStatusVariant[connectedClientInfo.status] ?? "secondary"}>
                {connectedClientInfo.status}
              </Badge>
              {connectedClientInfo.assigned_participant_id !== null && (
                <span className="text-xs text-muted-foreground">participant_id={connectedClientInfo.assigned_participant_id}</span>
              )}
            </div>
          )}
        </div>

        {connectedClientInfo?.status === "approved" && (
          <div className="rounded-md border p-3">
            <p className="text-xs text-muted-foreground">
              Server approved this client. Configure local runtime according to the assignment, then click Ready.
            </p>
            <Button className="mt-2" onClick={() => handleClientReady().catch((err: unknown) => notifyError(err))} disabled={busy}>
              Ready
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
