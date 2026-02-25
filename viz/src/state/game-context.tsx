"use client";

import React, { createContext, useReducer, useCallback, useContext, useRef } from "react";
import { LiveEngine } from "@/engine/sim/live-engine";
import type { ScenarioConfig, GovernanceConfig, PayoffParams } from "@/engine/sim/types";
import type { AgentType, EpochSnapshot } from "@/data/types";
import type { ShockEvent } from "@/engine/sim/shocks";
import type { ChallengeLevel } from "@/engine/sim/campaign";
import { checkWinCondition, checkLoseCondition } from "@/engine/sim/campaign";

// ─── Types ───────────────────────────────────────────────────────

export type GameMode = "sandbox" | "campaign" | "redteam";
export type LevelStatus = "playing" | "won" | "lost" | "idle";

export interface GameState {
  mode: GameMode;
  isLive: boolean;
  tickRate: number; // ms between auto-ticks
  isPaused: boolean;
  // Campaign
  currentLevel: ChallengeLevel | null;
  levelStatus: LevelStatus;
  completedLevels: string[];
  hintIndex: number;
  // Red Team
  redTeamStartTime: number;
  redTeamBestScore: number;
  // Toast notifications
  toasts: ToastMessage[];
  toastCounter: number;
}

export interface ToastMessage {
  id: number;
  text: string;
  type: "info" | "success" | "warning" | "error";
  timestamp: number;
}

const initialState: GameState = {
  mode: "sandbox",
  isLive: false,
  tickRate: 200,
  isPaused: true,
  currentLevel: null,
  levelStatus: "idle",
  completedLevels: loadCompletedLevels(),
  hintIndex: 0,
  redTeamStartTime: 0,
  redTeamBestScore: loadRedTeamBest(),
  toasts: [],
  toastCounter: 0,
};

// ─── Persistence Helpers ─────────────────────────────────────────

function loadCompletedLevels(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const parsed = JSON.parse(localStorage.getItem("swarm-completed-levels") ?? "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((v: unknown) => typeof v === "string");
  } catch {
    return [];
  }
}

function saveCompletedLevels(levels: string[]): void {
  if (typeof window === "undefined") return;
  localStorage.setItem("swarm-completed-levels", JSON.stringify(levels));
}

function loadRedTeamBest(): number {
  if (typeof window === "undefined") return 0;
  try {
    const val = parseFloat(localStorage.getItem("swarm-redteam-best") ?? "0");
    return Number.isFinite(val) ? val : 0;
  } catch {
    return 0;
  }
}

function saveRedTeamBest(score: number): void {
  if (typeof window === "undefined") return;
  localStorage.setItem("swarm-redteam-best", String(score));
}

// ─── Actions ─────────────────────────────────────────────────────

type GameAction =
  | { type: "SET_MODE"; mode: GameMode }
  | { type: "START_LIVE" }
  | { type: "STOP_LIVE" }
  | { type: "SET_PAUSED"; paused: boolean }
  | { type: "TOGGLE_PAUSE" }
  | { type: "SET_TICK_RATE"; rate: number }
  | { type: "START_LEVEL"; level: ChallengeLevel }
  | { type: "LEVEL_WON" }
  | { type: "LEVEL_LOST" }
  | { type: "NEXT_HINT" }
  | { type: "START_REDTEAM" }
  | { type: "UPDATE_REDTEAM_SCORE"; score: number }
  | { type: "ADD_TOAST"; text: string; toastType: ToastMessage["type"] }
  | { type: "REMOVE_TOAST"; id: number };

function reducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "SET_MODE":
      return { ...state, mode: action.mode, levelStatus: "idle", currentLevel: null };
    case "START_LIVE":
      return { ...state, isLive: true, isPaused: false };
    case "STOP_LIVE":
      return { ...state, isLive: false, isPaused: true };
    case "SET_PAUSED":
      return { ...state, isPaused: action.paused };
    case "TOGGLE_PAUSE":
      return { ...state, isPaused: !state.isPaused };
    case "SET_TICK_RATE":
      return { ...state, tickRate: action.rate };
    case "START_LEVEL":
      return {
        ...state,
        currentLevel: action.level,
        levelStatus: "playing",
        isLive: true,
        isPaused: false,
        hintIndex: 0,
      };
    case "LEVEL_WON": {
      const completed = state.currentLevel
        ? [...new Set([...state.completedLevels, state.currentLevel.id])]
        : state.completedLevels;
      saveCompletedLevels(completed);
      return { ...state, levelStatus: "won", isPaused: true, completedLevels: completed };
    }
    case "LEVEL_LOST":
      return { ...state, levelStatus: "lost", isPaused: true };
    case "NEXT_HINT":
      return { ...state, hintIndex: state.hintIndex + 1 };
    case "START_REDTEAM":
      return { ...state, isLive: true, isPaused: false, redTeamStartTime: Date.now() };
    case "UPDATE_REDTEAM_SCORE": {
      const best = Math.max(state.redTeamBestScore, action.score);
      if (best > state.redTeamBestScore) saveRedTeamBest(best);
      return { ...state, redTeamBestScore: best };
    }
    case "ADD_TOAST":
      return {
        ...state,
        toasts: [
          ...state.toasts.slice(-4), // Keep max 5
          { id: state.toastCounter, text: action.text, type: action.toastType, timestamp: Date.now() },
        ],
        toastCounter: state.toastCounter + 1,
      };
    case "REMOVE_TOAST":
      return { ...state, toasts: state.toasts.filter((t) => t.id !== action.id) };
    default:
      return state;
  }
}

// ─── Context ─────────────────────────────────────────────────────

export interface GameContextValue {
  state: GameState;
  dispatch: React.Dispatch<GameAction>;
  engineRef: React.RefObject<LiveEngine | null>;

  // Convenience methods
  startSandbox: (config: ScenarioConfig) => void;
  startCampaignLevel: (level: ChallengeLevel) => void;
  startRedTeam: (config: ScenarioConfig) => void;
  stopGame: () => void;
  togglePause: () => void;
  setTickRate: (rate: number) => void;
  spawnAgent: (type: AgentType) => void;
  removeAgent: (id: string) => void;
  toggleFreeze: (id: string) => void;
  toggleQuarantine: (id: string) => void;
  updateGovernance: (gov: Partial<GovernanceConfig>) => void;
  updatePayoff: (params: Partial<PayoffParams>) => void;
  injectShock: (shock: ShockEvent) => void;
  checkConditions: (snapshots: EpochSnapshot[]) => void;
  addToast: (text: string, type?: ToastMessage["type"]) => void;
  saveGame: () => string | null;
  loadGame: (json: string) => void;
  computeRedTeamScore: () => number;
}

export const GameContext = createContext<GameContextValue>(null!);

export function GameProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const engineRef = useRef<LiveEngine | null>(null);

  const addToast = useCallback((text: string, type: ToastMessage["type"] = "info") => {
    dispatch({ type: "ADD_TOAST", text, toastType: type });
  }, []);

  const startSandbox = useCallback((config: ScenarioConfig) => {
    engineRef.current = new LiveEngine(config);
    dispatch({ type: "SET_MODE", mode: "sandbox" });
    dispatch({ type: "START_LIVE" });
    addToast("Sandbox started — use controls to intervene", "info");
  }, [addToast]);

  const startCampaignLevel = useCallback((level: ChallengeLevel) => {
    engineRef.current = new LiveEngine(level.config);
    dispatch({ type: "START_LEVEL", level });
    addToast(`Level: ${level.name}`, "info");
  }, [addToast]);

  const startRedTeam = useCallback((config: ScenarioConfig) => {
    engineRef.current = new LiveEngine(config);
    dispatch({ type: "SET_MODE", mode: "redteam" });
    dispatch({ type: "START_REDTEAM" });
    addToast("Red Team mode — cause maximum damage!", "warning");
  }, [addToast]);

  const stopGame = useCallback(() => {
    engineRef.current = null;
    dispatch({ type: "STOP_LIVE" });
  }, []);

  const togglePause = useCallback(() => {
    dispatch({ type: "TOGGLE_PAUSE" });
  }, []);

  const setTickRate = useCallback((rate: number) => {
    dispatch({ type: "SET_TICK_RATE", rate });
  }, []);

  const spawnAgent = useCallback((type: AgentType) => {
    if (!engineRef.current) return;
    try {
      const agent = engineRef.current.spawnAgent(type);
      addToast(`Spawned ${agent.name} (${type})`, "info");
    } catch (e) {
      addToast(e instanceof Error ? e.message : "Cannot spawn agent", "warning");
    }
  }, [addToast]);

  const removeAgent = useCallback((id: string) => {
    if (!engineRef.current) return;
    const agent = engineRef.current.agents.find((a) => a.id === id);
    engineRef.current.removeAgent(id);
    addToast(`Removed ${agent?.name ?? id}`, "warning");
  }, [addToast]);

  const toggleFreeze = useCallback((id: string) => {
    if (!engineRef.current) return;
    engineRef.current.toggleFreeze(id);
    const agent = engineRef.current.agents.find((a) => a.id === id);
    if (agent) {
      addToast(`${agent.name} ${agent.isFrozen ? "frozen" : "unfrozen"}`, agent.isFrozen ? "warning" : "info");
    }
  }, [addToast]);

  const toggleQuarantine = useCallback((id: string) => {
    if (!engineRef.current) return;
    engineRef.current.toggleQuarantine(id);
    const agent = engineRef.current.agents.find((a) => a.id === id);
    if (agent) {
      addToast(`${agent.name} ${agent.isQuarantined ? "quarantined" : "released"}`, agent.isQuarantined ? "error" : "info");
    }
  }, [addToast]);

  const updateGovernance = useCallback((gov: Partial<GovernanceConfig>) => {
    if (!engineRef.current) return;
    engineRef.current.updateGovernance(gov);
  }, []);

  const updatePayoff = useCallback((params: Partial<PayoffParams>) => {
    if (!engineRef.current) return;
    engineRef.current.updatePayoff(params);
  }, []);

  const injectShock = useCallback((shock: ShockEvent) => {
    if (!engineRef.current) return;
    engineRef.current.injectShock(shock);
    addToast(`Shock: ${shock.type.replace(/_/g, " ")}`, "warning");
  }, [addToast]);

  const checkConditions = useCallback((snapshots: EpochSnapshot[]) => {
    if (state.levelStatus !== "playing" || !state.currentLevel) return;

    const activeCount = engineRef.current
      ? engineRef.current.agents.filter((a) => !a.isFrozen).length
      : 0;

    if (checkWinCondition(state.currentLevel.winCondition, snapshots, activeCount)) {
      dispatch({ type: "LEVEL_WON" });
      addToast("Level Complete!", "success");
    } else if (checkLoseCondition(state.currentLevel.loseCondition, snapshots, activeCount)) {
      dispatch({ type: "LEVEL_LOST" });
      addToast("Level Failed", "error");
    }
  }, [state.levelStatus, state.currentLevel, addToast]);

  const computeRedTeamScore = useCallback((): number => {
    if (!engineRef.current) return 0;
    const snapshots = engineRef.current.getEpochSnapshots();
    if (snapshots.length === 0) return 0;

    const maxToxicity = Math.max(...snapshots.map((s) => s.toxicity_rate));
    const maxGini = Math.max(...snapshots.map((s) => s.gini_coefficient));
    const minWelfare = Math.min(...snapshots.map((s) => s.total_welfare));
    const maxFrozen = Math.max(...snapshots.map((s) => s.n_frozen));

    // Score: higher = more damage caused
    return (
      maxToxicity * 30 +
      maxGini * 20 +
      Math.max(0, -minWelfare) * 0.5 +
      maxFrozen * 5
    );
  }, []);

  const saveGame = useCallback((): string | null => {
    if (!engineRef.current) return null;
    return engineRef.current.serialize();
  }, []);

  const loadGame = useCallback((json: string) => {
    try {
      engineRef.current = LiveEngine.deserialize(json);
    } catch (e) {
      addToast(`Failed to load: ${e instanceof Error ? e.message : "invalid file"}`, "error");
      return;
    }
    dispatch({ type: "SET_MODE", mode: "sandbox" });
    dispatch({ type: "START_LIVE" });
    addToast("Game loaded", "info");
  }, [addToast]);

  return (
    <GameContext.Provider
      value={{
        state,
        dispatch,
        engineRef,
        startSandbox,
        startCampaignLevel,
        startRedTeam,
        stopGame,
        togglePause,
        setTickRate,
        spawnAgent,
        removeAgent,
        toggleFreeze,
        toggleQuarantine,
        updateGovernance,
        updatePayoff,
        injectShock,
        checkConditions,
        addToast,
        saveGame,
        loadGame,
        computeRedTeamScore,
      }}
    >
      {children}
    </GameContext.Provider>
  );
}

export function useGame() {
  return useContext(GameContext);
}
