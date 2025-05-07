import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, XCircle, AlertCircle, Activity } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useMetrics } from "@/components/monitoring/MetricsProvider";

interface ServiceStatusProps {
  name: string;
  status: "healthy" | "degraded" | "down" | "unknown";
  latency?: number;
  lastChecked?: string;
}

function ServiceStatus({ name, status, latency, lastChecked }: ServiceStatusProps) {
  const getStatusIcon = () => {
    switch (status) {
      case "healthy":
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case "degraded":
        return <AlertCircle className="h-5 w-5 text-amber-500" />;
      case "down":
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Activity className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusBadge = () => {
    switch (status) {
      case "healthy":
        return <Badge className="bg-green-500/20 text-green-600 hover:bg-green-500/30 border-green-500">Healthy</Badge>;
      case "degraded":
        return <Badge className="bg-amber-500/20 text-amber-600 hover:bg-amber-500/30 border-amber-500">Degraded</Badge>;
      case "down":
        return <Badge className="bg-red-500/20 text-red-600 hover:bg-red-500/30 border-red-500">Down</Badge>;
      default:
        return <Badge variant="outline">Unknown</Badge>;
    }
  };

  return (
    <div className="flex items-center justify-between p-2 border-b last:border-0">
      <div className="flex items-center gap-2">
        {getStatusIcon()}
        <div>
          <p className="font-medium">{name}</p>
          {latency && <p className="text-xs text-muted-foreground">Latency: {latency}ms</p>}
        </div>
      </div>
      <div className="flex flex-col items-end gap-1">
        {getStatusBadge()}
        {lastChecked && <p className="text-xs text-muted-foreground">Last checked: {lastChecked}</p>}
      </div>
    </div>
  );
}

export function SystemStatus() {
  const { metrics, isLoading } = useMetrics();

  if (isLoading) {
    return (
      <Card className="w-full shadow-md animate-pulse">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl">System Status</CardTitle>
          <CardDescription>
            Loading status...
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="h-12 bg-accent/30 rounded"></div>
            <div className="h-12 bg-accent/30 rounded"></div>
            <div className="h-12 bg-accent/30 rounded"></div>
            <div className="h-12 bg-accent/30 rounded"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const overallStatus = metrics.systemStatus.services.every(s => s.status === "healthy") 
    ? "healthy" 
    : metrics.systemStatus.services.some(s => s.status === "down") 
      ? "down" 
      : "degraded";

  const statusText = {
    healthy: "All systems operational",
    degraded: "Some systems experiencing issues",
    down: "Critical systems down",
    unknown: "System status unknown",
  };

  return (
    <Card className="w-full shadow-md">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-xl">
          <span>System Status</span>
          {overallStatus === "healthy" && <CheckCircle className="text-green-500" />}
          {overallStatus === "degraded" && <AlertCircle className="text-amber-500" />}
          {overallStatus === "down" && <XCircle className="text-red-500" />}
        </CardTitle>
        <CardDescription>
          {statusText[overallStatus]}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-1">
          {metrics.systemStatus.services.map((service, index) => (
            <ServiceStatus
              key={index}
              name={service.name}
              status={service.status}
              latency={service.latency}
              lastChecked={service.lastChecked}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

