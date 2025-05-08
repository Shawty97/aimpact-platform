import React, { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";
import { AlertCircle, Bot, Pencil, Plus, Save, Trash2, Workflow, X, ZoomIn, ZoomOut } from "lucide-react";

// Types
interface AgentNode {
  id: string;
  type: string;
  name: string;
  position: { x: number; y: number };
  properties: Record<string, any>;
}

interface Connection {
  id: string;
  sourceId: string;
  targetId: string;
}

interface AgentCanvasProps {
  initialAgents?: AgentNode[];
  initialConnections?: Connection[];
  onSave?: (agents: AgentNode[], connections: Connection[]) => void;
}

export function AgentCanvas({
  initialAgents = [],
  initialConnections = [],
  onSave,
}: AgentCanvasProps) {
  // State
  const [agents, setAgents] = useState<AgentNode[]>(initialAgents);
  const [connections, setConnections] = useState<Connection[]>(initialConnections);
  const [selectedAgent, setSelectedAgent] = useState<AgentNode | null>(null);
  const [draggedAgent, setDraggedAgent] = useState<{ id: string; startX: number; startY: number } | null>(null);
  const [newConnection, setNewConnection] = useState<{ sourceId: string; position: { x: number; y: number } } | null>(null);
  const [zoom, setZoom] = useState<number>(1);
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [panStart, setPanStart] = useState<{ x: number; y: number } | null>(null);
  const [availableAgentTypes, setAvailableAgentTypes] = useState([
    { id: 'language', name: 'Language Processor', icon: <Bot size={24} /> },
    { id: 'voice', name: 'Voice Agent', icon: <Bot size={24} /> },
    { id: 'workflow', name: 'Workflow', icon: <Workflow size={24} /> },
  ]);

  // Refs
  const canvasRef = useRef<HTMLDivElement>(null);

  // Generate unique ID
  const generateId = () => `agent-${Math.random().toString(36).substr(2, 9)}`;

  // Handle adding new agent
  const handleAddAgent = (type: string) => {
    const agentType = availableAgentTypes.find(t => t.id === type);
    if (!agentType) return;

    const newAgent: AgentNode = {
      id: generateId(),
      type,
      name: `${agentType.name} ${agents.length + 1}`,
      position: { x: 100, y: 100 },
      properties: { description: '', model: 'gpt-4', enabled: true },
    };

    setAgents([...agents, newAgent]);
    setSelectedAgent(newAgent);
  };

  // Handle agent selection
  const handleSelectAgent = (agent: AgentNode | null) => {
    setSelectedAgent(agent);
  };

  // Handle agent deletion
  const handleDeleteAgent = (id: string) => {
    setAgents(agents.filter(a => a.id !== id));
    setConnections(connections.filter(c => c.sourceId !== id && c.targetId !== id));
    if (selectedAgent?.id === id) setSelectedAgent(null);
  };

  // Update agent properties
  const handleUpdateAgentProperty = (key: string, value: any) => {
    if (!selectedAgent) return;

    const updatedAgent = {
      ...selectedAgent,
      [key]: value,
    };

    setAgents(agents.map(a => (a.id === selectedAgent.id ? updatedAgent : a)));
    setSelectedAgent(updatedAgent);
  };

  // Update agent properties nested
  const handleUpdateAgentNestedProperty = (key: string, value: any) => {
    if (!selectedAgent) return;

    const updatedAgent = {
      ...selectedAgent,
      properties: {
        ...selectedAgent.properties,
        [key]: value,
      },
    };

    setAgents(agents.map(a => (a.id === selectedAgent.id ? updatedAgent : a)));
    setSelectedAgent(updatedAgent);
  };

  // Handle mouse down on agent (start drag)
  const handleAgentMouseDown = (e: React.MouseEvent, agent: AgentNode) => {
    e.stopPropagation();
    setSelectedAgent(agent);
    setDraggedAgent({
      id: agent.id,
      startX: e.clientX,
      startY: e.clientY,
    });
  };

  // Handle mouse move (drag agent)
  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    if (draggedAgent) {
      const dx = (e.clientX - draggedAgent.startX) / zoom;
      const dy = (e.clientY - draggedAgent.startY) / zoom;

      setAgents(
        agents.map(a => {
          if (a.id === draggedAgent.id) {
            return {
              ...a,
              position: {
                x: a.position.x + dx,
                y: a.position.y + dy,
              },
            };
          }
          return a;
        })
      );

      setDraggedAgent({
        ...draggedAgent,
        startX: e.clientX,
        startY: e.clientY,
      });
    } else if (newConnection) {
      setNewConnection({
        ...newConnection,
        position: { x: e.clientX, y: e.clientY },
      });
    } else if (isPanning && panStart) {
      const dx = e.clientX - panStart.x;
      const dy = e.clientY - panStart.y;
      
      setPanOffset({
        x: panOffset.x + dx,
        y: panOffset.y + dy,
      });
      
      setPanStart({
        x: e.clientX,
        y: e.clientY,
      });
    }
  };

  // Handle mouse up (end drag)
  const handleCanvasMouseUp = () => {
    setDraggedAgent(null);
    setNewConnection(null);
    setIsPanning(false);
    setPanStart(null);
  };

  // Handle canvas mouse down (start connection or panning)
  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || e.button === 2 || (e.button === 0 && e.altKey)) {
      // Middle button or right button or Alt+left button
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
      e.preventDefault();
    } else if (e.target === canvasRef.current) {
      setSelectedAgent(null);
    }
  };

  // Handle connection creation from an agent
  const handleStartConnection = (e: React.MouseEvent, sourceId: string) => {
    e.stopPropagation();
    const sourceAgent = agents.find(a => a.id === sourceId);
    if (!sourceAgent) return;

    setNewConnection({
      sourceId,
      position: { x: e.clientX, y: e.clientY },
    });
  };

  // Handle dropping a connection on an agent
  const handleCompleteConnection = (targetId: string) => {
    if (!newConnection || newConnection.sourceId === targetId) return;

    const connectionId = `conn-${Math.random().toString(36).substr(2, 9)}`;
    const newConn: Connection = {
      id: connectionId,
      sourceId: newConnection.sourceId,
      targetId,
    };

    setConnections([...connections, newConn]);
    setNewConnection(null);
  };

  // Handle removing a connection
  const handleRemoveConnection = (id: string) => {
    setConnections(connections.filter(c => c.id !== id));
  };

  // Handle zoom
  const handleZoom = (factor: number) => {
    setZoom(prevZoom => {
      const newZoom = prevZoom * factor;
      return Math.min(Math.max(newZoom, 0.5), 2); // Limit zoom between 0.5 and 2
    });
  };

  // Handle save
  const handleSave = () => {
    if (onSave) {
      onSave(agents, connections);
    }
  };

  // Render connection line between agents
  const renderConnection = (connection: Connection) => {
    const source = agents.find(a => a.id === connection.sourceId);
    const target = agents.find(a => a.id === connection.targetId);

    if (!source || !target) return null;

    // Calculate midpoint of source agent
    const sourceX = source.position.x + 75;
    const sourceY = source.position.y + 50;

    // Calculate midpoint of target agent
    const targetX = target.position.x + 75;
    const targetY = target.position.y + 50;

    // Draw SVG path
    const path = `M ${sourceX} ${sourceY} C ${(sourceX + targetX) / 2} ${sourceY}, ${(sourceX + targetX) / 2} ${targetY}, ${targetX} ${targetY}`;

    return (
      <g key={connection.id} className="group">
        <path
          d={path}
          stroke="#4f46e5"
          strokeWidth="2"
          fill="none"
          className="transition-colors group-hover:stroke-blue-600"
        />
        <circle cx={targetX} cy={targetY} r="4" fill="#4f46e5" className="group-hover:fill-blue-600" />
        
        {/* Delete button */}
        <g 
          transform={`translate(${(sourceX + targetX) / 2 - 10}, ${(sourceY + targetY) / 2 - 10})`}
          className="opacity-0 group-hover:opacity-100 cursor-pointer transition-opacity"
          onClick={() => handleRemoveConnection(connection.id)}
        >
          <circle cx="10" cy="10" r="10" fill="white" />
          <X size={16} color="#ef4444" x="2" y="2" />
        </g>
      </g>
    );
  };

  // Render temporary connection line when creating new connection
  const renderNewConnection = () => {
    if (!newConnection) return null;

    const source = agents.find(a => a.id === newConnection.sourceId);
    if (!source) return null;

    // Calculate midpoint of source agent
    const sourceX = source.position.x + 75;
    const sourceY = source.position.y + 50;

    // Draw SVG path to mouse position
    const path = `M ${sourceX} ${sourceY} L ${(newConnection.position.x - panOffset.x) / zoom} ${(newConnection.position.y - panOffset.y) / zoom}`;

    return (
      <path
        d={path}
        stroke="#4f46e5"
        strokeWidth="2"
        strokeDasharray="5,5"
        fill="none"
      />
    );
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">Agent Canvas</h2>
        <div className="flex space-x-2">
          <Button size="sm" variant="outline" onClick={() => handleZoom(1.1)}>
            <ZoomIn className="h-4 w-4 mr-1" /> Zoom In
          </Button>
          <Button size="sm" variant="outline" onClick={() => handleZoom(0.9)}>
            <ZoomOut className="h-4 w-4 mr-1" /> Zoom Out
          </Button>
          <Button size="sm" variant="default" onClick={handleSave}>
            <Save className="h-4 w-4 mr-1" /> Save
          </Button>
        </div>
      </div>
      
      <div className="flex flex-1 gap-4 h-[calc(100vh-200px)]">
        {/* Agent Palette */}
        <Card className="w-48 shrink-0">
          <CardHeader className="py-3">
            <CardTitle className="text-sm">Agent Types</CardTitle>
          </CardHeader>
          <CardContent className="py-2">
            <div className="space-y-2">
              {availableAgentTypes.map(type => (
                <div 
                  key={type.id}
                  className="p-2 bg-accent/50 rounded-md flex items-center justify-between cursor-pointer hover:bg-accent/80 transition-colors"
                  onClick={() => handleAddAgent(type.id)}
                  draggable
                >
                  <div className="flex items-center gap-2">
                    {type.icon}
                    <span className="text-sm">{type.name}</span>
                  </div>
                  <Plus size={16} />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        
        {/* Canvas Area */}
        <Card className="flex-1 overflow-hidden">
          <CardContent className="p-0 h-full relative">
            <div
              ref={canvasRef}
              className="w-full h-full overflow-hidden bg-blue-50/50 relative"
              onMouseDown={handleCanvasMouseDown}
              onMouseMove={handleCanvasMouseMove}
              onMouseUp={handleCanvasMouseUp}
              onMouseLeave={handleCanvasMouseUp}
              style={{ cursor: isPanning ? 'grabbing' : 'default' }}
            >
              <div
                className="absolute top-0 left-0 w-full h-full"
                style={{
                  transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoom})`,
                  transformOrigin: '0 0',
                }}
              >
                {/* Grid lines */}
                <svg width="100%" height="

