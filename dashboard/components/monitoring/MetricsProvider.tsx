import React, { createContext, useContext, useState, useEffect } from "react";
import { MetricsSocket } from "./MetricsSocket";

// Define metric types
interface VoiceProcessingMetrics {
  status: "idle" | "recording" | "processing" | "error";
  confidence: number;
  speechToTextTime: number;
  emotionDetected: string;
  processingCount: number;
  errorRate: number;
}

interface AgentStats {
  name: string;
  successRate: number;
  averageResponseTime: number;
  totalCalls: number;
}

interface AgentPerformanceMetrics {
  activeAgents: number;
  totalAgents: number;
  totalCalls: number;
  averageResponseTime: number;
  successRate: number;
  topAgents: AgentStats[];
}

interface WorkflowExecution {
  id: string;
  name: string;
  status: "running" | "completed" | "failed";
  duration: number;
  steps: number;
  stepsCompleted: number;
}

interface WorkflowStatusMetrics {
  activeWorkflows: number;
  completedToday: number;
  averageCompletionTime: number;
  successRate: number;
  recentExecutions: WorkflowExecution[];
}

interface SystemMetricsData {
  totalRequests: number;
  requestsChange: number;
  averageLatency: number;
  latencyChange: number;
  activeUsers: number;
  userChange: number;
  serviceUptime: number;
  uptimeChange: number;
}

interface ServiceStatus {
  name: string;
  status: "healthy" | "degraded" | "down" | "unknown";
  latency?: number;
  lastChecked?: string;
}

interface SystemStatusData {
  services: ServiceStatus[];
}

interface Alert {
  id: string;
  severity: "critical" | "warning" | "info";
  message: string;
  source: string;
  timestamp: string;
  acknowledged: boolean;
}

interface MetricsData {
  voiceProcessing: VoiceProcessingMetrics;
  agentPerformance: AgentPerformanceMetrics;
  workflowStatus

