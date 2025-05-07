import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowUp, ArrowDown, BarChart3, PieChart, Clock, Users } from "lucide-react";

interface MetricProps {
  title: string;
  value: string | number;
  change: number;
  icon: React.ReactNode;
}

function Metric({ title, value, change, icon }: MetricProps) {
  return (
    <div className="flex flex-col space-y-1">
      <p className="text-sm text-muted-foreground">{title}</p>
      <div className="flex items-center">
        <div className="mr-2 rounded-md bg-muted p-1">{icon}</div>
        <div>
          <p className="text-lg font-bold">{value}</p>
          <div className="flex items-center text-xs">
            {change > 0 ? (
              <ArrowUp className="mr-1 h-3 w-3 text-green-500" />
            ) : (
              <ArrowDown className="mr-1 h-3 w-3 text-red-500" />
            )}
            <span className={change > 0 ? "text-green-500" : "text-red-500"}>
              {Math.abs(change)}% from last period
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

interface AImpactMetricsCardProps {
  totalRequests: number;
  requestsChange: number;
  averageLatency: number;
  latencyChange: number;
  activeUsers: number;
  userChange: number;
  serviceUptime: number;
  uptimeChange: number;
}

export function AImpactMetricsCard({
  totalRequests = 0,
  requestsChange = 0,
  averageLatency = 0,
  latencyChange = 0,
  activeUsers = 0,
  userChange = 0,
  serviceUptime = 0,
  uptimeChange = 0,
}: AImpactMetricsCardProps) {
  

