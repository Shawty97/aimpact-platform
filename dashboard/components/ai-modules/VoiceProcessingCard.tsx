import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Mic, Activity, Waveform, VolumeX } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface VoiceProcessingCardProps {
  processingStatus: "idle" | "recording" | "processing" | "error";
  confidence: number;
  speechToTextTime: number;
  emotionDetected?: string;
  processingCount: number;
  errorRate: number;
}

export function VoiceProcessingCard({
  processingStatus = "idle",
  confidence = 0,
  speechToTextTime = 0,
  emotionDetected = "neutral",
  processingCount = 0,
  errorRate = 0,
}: VoiceProcessingCardProps) {
  const statusColors = {
    idle: "text-gray-500",
    recording: "text-green-500 animate-pulse",
    processing: "text-blue-500",
    error: "text-red-500",
  };

  const statusMessages = {
    idle: "Ready for voice input",
    recording: "Recording audio...",
    processing: "Processing speech...",
    error: "Error processing voice",
  };

  const emotionColors = {
    happy: "text-yellow-500",
    sad: "text-blue-400",
    angry: "text-red-500",
    surprised: "text-purple-500",
    neutral: "text-gray-500",
  };

  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <span>Voice Processing</span>
          <span className={statusColors[processingStatus]}>
            {processingStatus === "recording" && <Mic className="animate-pulse" />}
            {processingStatus === "processing" && <Activity />}
            {processingStatus === "error" && <VolumeX />}
            {processingStatus === "idle" && <Waveform />}
          </span>
        </CardTitle>
        <CardDescription>
          {statusMessages[processingStatus]}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Recognition Confidence</span>
            <span className="font-medium">{confidence}%</span>
          </div>
          <Progress value={confidence} className="h-2" />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Speech to Text</p>
            <p className="text-lg font-medium">{speechToTextTime}ms</p>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Emotion</p>
            <p className={`text-lg font-medium ${emotionColors[emotionDetected] || 'text-gray-500'}`}>
              {emotionDetected || "Unknown"}
            </p>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Processed</p>
            <p className="text-lg font-medium">{processingCount}</p>
          </div>
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Error Rate</p>
            <p className={`text-lg font-medium ${errorRate > 5 ? 'text-red-500' : 'text-green-500'}`}>
              {errorRate}%
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

