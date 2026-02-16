import type { AgentType } from "@/data/types";

export const TILE_WIDTH = 64;
export const TILE_HEIGHT = 32;

export const MIN_ZOOM = 0.3;
export const MAX_ZOOM = 4;

export const COLORS = {
  bg: "#0D1117",
  panel: "#161B22",
  border: "#30363D",
  text: "#E6EDF3",
  muted: "#7D8590",
  accent: "#3ECFB4",
  secondary: "#F2994A",
  alert: "#EB5757",
  info: "#2F80ED",
  btn: "#21262D",
  btnHover: "#30363D",
} as const;

export const AGENT_COLORS: Record<AgentType, { primary: string; secondary: string; accent: string }> = {
  honest: { primary: "#1A6B5A", secondary: "#3ECFB4", accent: "#A8EFDF" },
  opportunistic: { primary: "#8B5E2F", secondary: "#F2994A", accent: "#FCDBB5" },
  deceptive: { primary: "#5B2D8B", secondary: "#BB6BD9", accent: "#E4BFF5" },
  adversarial: { primary: "#8B2D2D", secondary: "#EB5757", accent: "#F5BFBF" },
  rlm: { primary: "#1E4D8B", secondary: "#2F80ED", accent: "#A8CFF5" },
  crewai: { primary: "#2D6B2D", secondary: "#6FCF97", accent: "#C0F0D0" },
};

export const AGENT_LABELS: Record<AgentType, string> = {
  honest: "Paladin",
  opportunistic: "Merchant",
  deceptive: "Illusionist",
  adversarial: "Enforcer",
  rlm: "Technomancer",
  crewai: "Builder",
};

/** Grid spacing between agents */
export const AGENT_GRID_SPACING = 3;

/** Character pixel dimensions */
export const CHARACTER = {
  baseWidth: 32,
  baseHeight: 56,
} as const;

/** Animation durations in ms */
export const ANIM = {
  epochTransition: 2000,
  arcLifetime: 1500,
  particleLife: 1000,
} as const;

/** Per-type motion personality */
export interface AgentMotion {
  walkRate: number;
  bobAmplitude: number;
  bobAsymmetry: number;
  legSwingScale: number;
  armSwingScale: number;
  swayAmplitude: number;
  idleBob: number;
}

export const AGENT_MOTION: Record<AgentType, AgentMotion> = {
  honest:        { walkRate: 0.011, bobAmplitude: 1.2, bobAsymmetry: 0.1,  legSwingScale: 0.9, armSwingScale: 0.8, swayAmplitude: 0.3, idleBob: 0.4 },
  opportunistic: { walkRate: 0.014, bobAmplitude: 1.8, bobAsymmetry: 0.2,  legSwingScale: 1.1, armSwingScale: 1.2, swayAmplitude: 0.6, idleBob: 0.6 },
  deceptive:     { walkRate: 0.010, bobAmplitude: 0.8, bobAsymmetry: 0.3,  legSwingScale: 0.7, armSwingScale: 0.6, swayAmplitude: 0.8, idleBob: 0.3 },
  adversarial:   { walkRate: 0.016, bobAmplitude: 2.2, bobAsymmetry: 0.0,  legSwingScale: 1.3, armSwingScale: 1.4, swayAmplitude: 0.2, idleBob: 0.5 },
  rlm:           { walkRate: 0.012, bobAmplitude: 1.0, bobAsymmetry: 0.15, legSwingScale: 1.0, armSwingScale: 0.9, swayAmplitude: 0.4, idleBob: 0.35 },
  crewai:        { walkRate: 0.013, bobAmplitude: 1.5, bobAsymmetry: 0.1,  legSwingScale: 1.0, armSwingScale: 1.0, swayAmplitude: 0.5, idleBob: 0.5 },
};

/** Arc data stream config — ranges keyed by p-value */
export const ARC_STREAM = {
  charCountMin: 5,
  charCountMax: 14,
  charSpacingMin: 0.05,
  charSpacingMax: 0.12,
  mutateIntervalMin: 50,
  mutateIntervalMax: 120,
  glowRadiusMin: 5,
  glowRadiusMax: 12,
  gridSize: 3,
  gridSpacing: 6,
} as const;
