import type { AgentType } from "@/data/types";

export interface Point {
  x: number;
  y: number;
}

export interface GridPos {
  gridX: number;
  gridY: number;
}

export interface Viewport {
  x: number;
  y: number;
  width: number;
  height: number;
  zoom: number;
}

export interface AgentVisual extends GridPos {
  id: string;
  name: string;
  agentType: AgentType;
  reputation: number;
  resources: number;
  totalPayoff: number;
  avgP: number;
  isFrozen: boolean;
  isQuarantined: boolean;
  scale: number;
  interactionsInitiated: number;
  interactionsReceived: number;
}

export interface InteractionArc {
  id: string;
  fromId: string;
  toId: string;
  p: number;
  accepted: boolean;
  progress: number; // 0-1 animation
  epoch: number;
}

export interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  size: number;
  color: string;
  alpha: number;
}

export interface OverlayState {
  interactions: boolean;
  metricsHud: boolean;
  threatZones: boolean;
  collusionLines: boolean;
  particles: boolean;
  minimap: boolean;
}

export type RenderEntity = {
  depth: number;
  render: (ctx: CanvasRenderingContext2D) => void;
};
