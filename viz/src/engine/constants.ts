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
