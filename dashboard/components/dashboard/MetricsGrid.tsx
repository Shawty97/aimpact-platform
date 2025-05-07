import React from "react";
import { VoiceProcessingCard } from "@/components/ai-modules/VoiceProcessingCard";
import { AgentPerformanceCard } from "@/components/ai-modules/AgentPerformanceCard";
import { WorkflowStatusCard } from "@/components/ai-modules/WorkflowStatusCard";
import { AImpactMetricsCard } from "@/components/ai-modules/AImpactMetricsCard";
import { useMetrics } from "@/components/monitoring/MetricsProvider";

export function MetricsGrid() {
  const { metrics, isLoading } = useMetrics();

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-pulse">
        <div className="bg-accent/30 rounded-lg h-80"></div>
        <div className="bg-accent/30 rounded-lg h-80"></div>
        <div className="bg-accent/30 rounded-lg h-80"></div>
        <div className="bg-accent/30 rounded-lg h-80"></div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <VoiceProcessingCard
        processingStatus={metrics.voiceProcessing.status}
        confidence={metrics.voiceProcessing.confidence}
        speechToTextTime={metrics.voiceProcessing.speechToTextTime}
        emotionDetected={metrics.voiceProcessing.emotionDetected}
        processingCount={metrics.voiceProcessing.processingCount}
        errorRate={metrics.voiceProcessing.errorRate}
      />
      <AgentPerformanceCard
        activeAgents={metrics.agentPerformance.activeAgents}
        totalAgents={metrics.agentPerformance.totalAgents}
        totalCalls={metrics.agentPerformance.totalCalls}
        averageResponseTime={metrics.agentPerformance.averageResponseTime}
        successRate={metrics.agentPerformance.successRate}
        topAgents={metrics.agentPerformance.topAgents}
      />
      <WorkflowStatusCard
        activeWorkflows={metrics.workflowStatus.activeWorkflows}
        completedToday={metrics.workflowStatus.completedToday}
        averageCompletionTime={metrics.workflowStatus.averageCompletionTime}
        successRate={metrics.workflowStatus.successRate}
        recentExecutions={metrics.workflowStatus.recentExecutions}
      />
      <AImpactMetricsCard
        totalRequests={metrics.systemMetrics.totalRequests}
        requestsChange={metrics.systemMetrics.requestsChange}
        averageLatency={metrics.systemMetrics.averageLatency}
        latencyChange={metrics.systemMetrics.latencyChange}
        activeUsers={metrics.systemMetrics.activeUsers}
        userChange={metrics.systemMetrics.userChange}
        serviceUptime={metrics.systemMetrics.serviceUptime}
        uptimeChange={metrics.systemMetrics.uptimeChange}
      />
    </div>
  );
}

