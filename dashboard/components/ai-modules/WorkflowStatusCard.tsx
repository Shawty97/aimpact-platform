import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { GitBranch, Clock, CheckCircle, XCircle, Activity } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface WorkflowExecution {
  id: string;
  name: string;
  status: "running" | "completed" | "failed";
  duration: number;
  steps: number;
  stepsCompleted: number;
}

interface WorkflowStatusCardProps {
  activeWorkflows: number;
  completedToday: number;
  averageCompletionTime: number;
  successRate: number;
  recentExecutions: WorkflowExecution[];
}

export function WorkflowStatusCard({
  activeWorkflows = 0,
  completedToday = 0,
  averageCompletionTime = 0,
  successRate = 0,
  recentExecutions = [],
}: WorkflowStatusCardProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running":
        return <Activity className="h-4 w-4 text-blue-500" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return null;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "running":
        return <Badge variant="outline" className="text-blue-500 border-blue-500">Running</Badge>;
      case "completed":
        return <Badge variant="outline" className="text-green-500 border-green-500">Completed</Badge>;
      case "failed":
        return <Badge variant="outline" className="text-red-500 border-red-500">Failed</Badge>;
      default:
        return null;
    }
  };

  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <span>Workflows</span>
          <GitBranch className="text-purple-500" />
        </CardTitle>
        <CardDescription>
          {activeWorkflows} active workflows
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Today's Completed</p>
            <p className="text-lg font-medium">{completedToday}</p>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Avg Completion</p>
            <div className="flex items-center">
              <Clock className="mr-2 h-4 w-4 text-muted-foreground" />
              <p className="text-lg font-medium">{averageCompletionTime}s</p>
            </div>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Success Rate</p>
            <p className={`text-lg font-medium ${successRate < 90 ? 'text-red-500' : 'text-green-500'}`}>
              {successRate}%
            </p>
          </div>
        </div>

        <div className="space-y-2">
          <h4 className="text-sm font-medium">Recent Executions</h4>
          <div className="space-y-2">
            {recentExecutions.map((execution) => (
              <div key={execution.id} className="flex items-center justify-between rounded-lg border p-2 text-sm">
                <div className="flex items-center space-x-2">
                  {getStatusIcon(execution.status)}
                  <div>
                    <p className="font-medium">{execution.name}</p>
                    <p className="text-xs text-muted-foreground">
                      Steps: {execution.stepsCompleted}/{execution.steps} • {execution.duration}s
                    </p>
                  </div>
                </div>
                {getStatusBadge(execution.status)}
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

