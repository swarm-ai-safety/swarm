"use client";

import React, { createContext, useReducer, useCallback, useRef, useEffect } from "react";
import type { SimulationData, EpochSnapshot, AgentSnapshot } from "@/data/types";
import type { AgentVisual, InteractionArc, Viewport, OverlayState, Particle, DigitalRainState, CodeTrailParticle, RecompileState } from "@/engine/types";
import type { EnvironmentState } from "@/engine/systems/environment-system";
import { interpolateEnvironment } from "@/engine/systems/environment-system";
import { interpolateAgents, computeAgentPositions } from "@/engine/systems/animation-system";
import { InteractionSystem } from "@/engine/systems/interaction-system";
import { ParticleSystem } from "@/engine/systems/particle-system";
import { CodeTrailSystem } from "@/engine/systems/code-trail-system";
import { interpolateEpoch } from "@/data/interpolator";
import { groupAgentsByEpoch, getUniqueAgentIds } from "@/data/normalizer";
import { createViewport, fitBounds } from "@/engine/camera";
import { gridToScreen } from "@/engine/isometric";
import { createSnowflakes, createHazardParticles } from "@/engine/entities/effects";
import { ANIM, AGENT_GRID_SPACING } from "@/engine/constants";

// ─── State ────────────────────────────────────────────────────────

export interface SimState {
  data: SimulationData | null;
  currentEpoch: number;
  epochFraction: number; // 0-1 progress within epoch transition
  playing: boolean;
  speed: number; // 0.5, 1, 2, 4
  viewport: Viewport;
  hoveredAgent: string | null;
  selectedAgent: string | null;
  overlays: OverlayState;
  // Derived / computed per frame
  agents: AgentVisual[];
  arcs: InteractionArc[];
  environment: EnvironmentState;
  particles: Particle[];
  currentEpochSnap: EpochSnapshot | null;
  gridSize: number;
}

const defaultOverlays: OverlayState = {
  interactions: true,
  metricsHud: true,
  threatZones: true,
  collusionLines: true,
  particles: true,
  minimap: true,
  digitalRain: true,
  tierraStrip: true,
  networkWeb: false,
};

const initialState: SimState = {
  data: null,
  currentEpoch: 0,
  epochFraction: 0,
  playing: false,
  speed: 1,
  viewport: createViewport(800, 600),
  hoveredAgent: null,
  selectedAgent: null,
  overlays: defaultOverlays,
  agents: [],
  arcs: [],
  environment: { threatLevel: 0, toxicity: 0, giniCoefficient: 0, collusionRisk: 0, incoherence: 0, contagionDepth: 0, activeThreats: 0, reputationStd: 0, payoffStd: 0, avgSynergyScore: 0, avgCoordinationScore: 0, avgDegree: 0, avgClustering: 0 },
  particles: [],
  currentEpochSnap: null,
  gridSize: 10,
};

// ─── Actions ──────────────────────────────────────────────────────

type Action =
  | { type: "LOAD_DATA"; data: SimulationData }
  | { type: "SET_EPOCH"; epoch: number }
  | { type: "SET_PLAYING"; playing: boolean }
  | { type: "SET_SPEED"; speed: number }
  | { type: "SET_VIEWPORT"; viewport: Viewport }
  | { type: "RESIZE"; width: number; height: number }
  | { type: "SET_HOVERED"; agentId: string | null }
  | { type: "SET_SELECTED"; agentId: string | null }
  | { type: "TOGGLE_OVERLAY"; key: keyof OverlayState }
  | { type: "TICK"; agents: AgentVisual[]; arcs: InteractionArc[]; environment: EnvironmentState; particles: Particle[]; epoch: number; fraction: number; epochSnap: EpochSnapshot | null };

function reducer(state: SimState, action: Action): SimState {
  switch (action.type) {
    case "LOAD_DATA": {
      const agentIds = getUniqueAgentIds(action.data.agent_snapshots);
      const cols = Math.ceil(Math.sqrt(agentIds.length));
      const gridSize = Math.max((cols + 1) * AGENT_GRID_SPACING + 2, 10);
      return {
        ...state,
        data: action.data,
        currentEpoch: 0,
        epochFraction: 0,
        playing: false,
        gridSize,
        currentEpochSnap: action.data.epoch_snapshots[0] ?? null,
      };
    }
    case "SET_EPOCH":
      return { ...state, currentEpoch: action.epoch, epochFraction: 0 };
    case "SET_PLAYING":
      return { ...state, playing: action.playing };
    case "SET_SPEED":
      return { ...state, speed: action.speed };
    case "SET_VIEWPORT":
      return { ...state, viewport: action.viewport };
    case "RESIZE":
      if (state.viewport.width === action.width && state.viewport.height === action.height) return state;
      return { ...state, viewport: { ...state.viewport, width: action.width, height: action.height } };
    case "SET_HOVERED":
      return { ...state, hoveredAgent: action.agentId };
    case "SET_SELECTED":
      return { ...state, selectedAgent: action.agentId };
    case "TOGGLE_OVERLAY":
      return { ...state, overlays: { ...state.overlays, [action.key]: !state.overlays[action.key] } };
    case "TICK":
      return {
        ...state,
        agents: action.agents,
        arcs: action.arcs,
        environment: action.environment,
        particles: action.particles,
        currentEpoch: action.epoch,
        epochFraction: action.fraction,
        currentEpochSnap: action.epochSnap,
      };
    default:
      return state;
  }
}

// ─── Context ──────────────────────────────────────────────────────

export interface SimContextValue {
  state: SimState;
  dispatch: React.Dispatch<Action>;
  interactionSystem: React.RefObject<InteractionSystem>;
  particleSystem: React.RefObject<ParticleSystem>;
  codeTrailSystem: React.RefObject<CodeTrailSystem>;
  digitalRainRef: React.RefObject<DigitalRainState | null>;
  recompileStateRef: React.RefObject<RecompileState>;
  agentPositions: React.RefObject<Map<string, { gridX: number; gridY: number }>>;
  agentsByEpoch: React.RefObject<Map<number, Map<string, AgentSnapshot>>>;
}

export const SimContext = createContext<SimContextValue>(null!);

export function SimulationProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const interactionSystem = useRef(new InteractionSystem());
  const particleSystem = useRef(new ParticleSystem());
  const codeTrailSystem = useRef(new CodeTrailSystem());
  const digitalRainRef = useRef<DigitalRainState | null>(null);
  const recompileStateRef = useRef<RecompileState>({ active: false, startTime: 0, duration: 400, scanlineY: 0 });
  const agentPositions = useRef(new Map<string, { gridX: number; gridY: number }>());
  const agentsByEpoch = useRef(new Map<number, Map<string, AgentSnapshot>>());

  // When data loads, compute positions, group snapshots, and fit camera
  useEffect(() => {
    if (!state.data) return;
    const ids = getUniqueAgentIds(state.data.agent_snapshots);
    agentPositions.current = computeAgentPositions(ids);
    agentsByEpoch.current = groupAgentsByEpoch(state.data.agent_snapshots);
    interactionSystem.current.clear();
    particleSystem.current.clear();
    codeTrailSystem.current.clear();
    digitalRainRef.current = null;

    // Auto-fit camera to agent positions
    if (agentPositions.current.size > 0) {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const pos of agentPositions.current.values()) {
        const screen = gridToScreen(pos.gridX, pos.gridY);
        minX = Math.min(minX, screen.x);
        minY = Math.min(minY, screen.y - 120); // account for building height
        maxX = Math.max(maxX, screen.x);
        maxY = Math.max(maxY, screen.y + 30);
      }
      dispatch({ type: "SET_VIEWPORT", viewport: fitBounds(state.viewport, minX, minY, maxX, maxY) });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- viewport intentionally excluded to avoid infinite loop
  }, [state.data]);

  return (
    <SimContext.Provider
      value={{ state, dispatch, interactionSystem, particleSystem, codeTrailSystem, digitalRainRef, recompileStateRef, agentPositions, agentsByEpoch }}
    >
      {children}
    </SimContext.Provider>
  );
}
