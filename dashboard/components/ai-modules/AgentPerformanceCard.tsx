import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Bot, Clock, BarChart, AlertCircle } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface AgentStats {
  name: string;
  successRate: number;
  averageResponseTime: number;
  totalCalls: number;
}

interface AgentPerformanceCardProps {
  activeAgents: number;
  totalAgents: number;
  totalCalls: number;
  averageResponseTime: number;
  successRate: number;
  topAgents: AgentStats[];
}

export function AgentPerformanceCard({
  activeAgents = 0,
  totalAgents = 0,
  totalCalls = 0,
  averageResponseTime = 0,
  successRate = 0,
  topAgents = [],
}: AgentPerformanceCardProps) {
  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <span>Agent Performance</span>
          <Bot className="text-blue-500" />
        </CardTitle>
        <CardDescription>
          {activeAgents} of {totalAgents} agents active
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Total Calls</p>
            <div className="flex items-center">
              <BarChart className="mr-2 h-4 w-4 text-muted-foreground" />
              <p className="text-lg font-medium">{totalCalls}</p>
            </div>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Avg Response</p>
            <div className="flex items-center">
              <Clock className="mr-2 h-4 w-4 text-muted-foreground" />
              <p className="text-lg font-medium">{averageResponseTime}ms</p>
            </div>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Success Rate</p>
            <div className="flex items-center">
              <AlertCircle className={`mr-2 h-4 w-4 ${successRate < 90 ? 'text-red-500' : 'text-green-500'}`} />
              <p className={`text-lg font-medium ${successRate < 90 ? 'text-red-500' : 'text-green-500'}`}>
                {successRate}%
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <p className="text-sm font-medium">Top Performing Agents</p>
          {topAgents.map((agent, index) => (
            <div key={index} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span>{agent.name}</span>
                <span className="font-medium">{agent.successRate}%</span>
              </div>
              <Progress 
                value={agent.successRate} 
                className="h-2" 
                indicatorClassName={
                  agent.successRate > 95 ? 'bg-green-500' : 
                  agent.successRate > 85 ? 'bg-blue-500' : 
                  'bg-amber-500'
                }
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{agent.averageResponseTime}ms avg</span>
                <span>{agent.totalCalls} calls</span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

