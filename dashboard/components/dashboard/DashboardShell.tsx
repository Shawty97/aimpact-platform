import React from "react";
import { MetricsGrid } from "./MetricsGrid";
import { SystemStatus } from "./SystemStatus";
import { AlertsPanel } from "@/components/monitoring/AlertsPanel";
import { MetricsProvider } from "@/components/monitoring/MetricsProvider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { RefreshCw, Download, BarChart3 } from "lucide-react";

export function DashboardShell() {
  const [isRefreshing, setIsRefreshing] = React.useState(false);

  const handleRefresh = () => {
    setIsRefreshing(true);
    // Simulating a refresh action
    setTimeout(() => {
      setIsRefreshing(false);
    }, 1000);
  };

  return (
    <MetricsProvider>
      <div className="flex flex-col gap-6">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">AImpact Dashboard</h1>
            <p className="text-muted-foreground">
              Monitor your AI platform performance and system health.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={isRefreshing}
            >
              <RefreshCw className={`mr-2 h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`} />
              Refresh
            </Button>
            <Button variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Export
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-3">
            <Tabs defaultValue="metrics" className="space-y-4">
              <TabsList>
                <TabsTrigger value="metrics">AI Metrics</TabsTrigger>
                <TabsTrigger value="performance">Performance</TabsTrigger>
                <TabsTrigger value="analytics">Analytics</TabsTrigger>
              </TabsList>
              <TabsContent value="metrics" className="space-y-4">
                <MetricsGrid />
              </TabsContent>
              <TabsContent value="performance">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <BarChart3 className="mr-2 h-5 w-5" />
                      Performance Analytics
                    </CardTitle>
                    <CardDescription>
                      Detailed performance metrics for your AImpact platform.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[400px] flex items-center justify-center border rounded-md">
                      <p className="text-muted-foreground">Performance charts will appear here</p>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              <TabsContent value="analytics">
                <Card>
                  <CardHeader>
                    <CardTitle>Usage Analytics</CardTitle>
                    <CardDescription>
                      Insights into how your platform is being used.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[400px] flex items-center justify-center border rounded-md">
                      <p className="text-muted-foreground">Analytics will appear here</p>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
          <div className="flex flex-col gap-6">
            <SystemStatus />
            <AlertsPanel />
          </div>
        </div>
      </div>
    </MetricsProvider>
  );
}

