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
  walkOffsetX: number;   // screen-space px offset from grid center
  walkOffsetY: number;
  walkPhase: number;     // continuous radian, drives leg swing cycle
  facing: number;        // -1 = left, 1 = right
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
  digitalRain: boolean;
  tierraStrip: boolean;
}

export type RenderEntity = {
  depth: number;
  render: (ctx: CanvasRenderingContext2D) => void;
};

// --- Digital Rain ---

export interface RainColumn {
  x: number;
  speed: number;
  chars: string[];
  headY: number;
  charInterval: number;
  brightness: number;
  nextMutateTime: number;
}

export interface DigitalRainState {
  columns: RainColumn[];
  initialized: boolean;
}

// --- Code Trail ---

export interface CodeTrailParticle {
  x: number;
  y: number;
  char: string;
  color: string;
  alpha: number;
  life: number;
  maxLife: number;
  vy: number;
  vx: number;
  gravity: number;
  fontSize: number;
}

// --- Recompile Flash ---

export interface RecompileState {
  active: boolean;
  startTime: number;
  duration: number;
  scanlineY: number;
}
