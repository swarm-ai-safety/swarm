/**
 * Shock injection system for the interactive sandbox game.
 *
 * Shocks are dramatic events that perturb the simulation, forcing players
 * to respond with governance adjustments. Each shock modifies the LiveEngine
 * state directly.
 */

import type { LiveEngine } from "./live-engine";
import type { AgentType } from "@/data/types";
import { AGENT_PROFILES } from "./agents";

/** Maximum agents spawned per shock (prevents runaway population) */
const MAX_SHOCK_SPAWN = 20;

/** Maximum resource value (prevents Infinity) */
const MAX_RESOURCES = 1e6;

export type ShockType =
  | "agent_wave"         // Inject N agents of a given type
  | "resource_crash"     // Halve all agent resources
  | "resource_boom"      // Double all agent resources
  | "info_asymmetry"     // Temporarily spike bad observables
  | "reputation_reset"   // Reset all reputations to 0.5
  | "mass_freeze"        // Freeze random 30% of agents
  | "mass_unfreeze"      // Unfreeze all agents
  | "reputation_shuffle" // Randomly reassign reputations
  | "toxin_injection";   // Force low-p interactions for N steps

export interface ShockEvent {
  type: ShockType;
  params: Record<string, number | string>;
}

/** Pre-defined shock templates for the UI */
export interface ShockTemplate {
  id: string;
  name: string;
  description: string;
  icon: string;
  shock: ShockEvent;
  color: string;
}

export const SHOCK_TEMPLATES: ShockTemplate[] = [
  {
    id: "adversarial-wave",
    name: "Adversarial Wave",
    description: "Spawn 3 adversarial agents into the system",
    icon: "\u26A0",
    shock: { type: "agent_wave", params: { agentType: "adversarial", count: 3 } },
    color: "#EB5757",
  },
  {
    id: "deceptive-infiltration",
    name: "Deceptive Infiltration",
    description: "Spawn 2 deceptive agents that look cooperative",
    icon: "\uD83C\uDFAD",
    shock: { type: "agent_wave", params: { agentType: "deceptive", count: 2 } },
    color: "#9B51E0",
  },
  {
    id: "resource-crash",
    name: "Resource Crash",
    description: "All agent resources halved instantly",
    icon: "\uD83D\uDCC9",
    shock: { type: "resource_crash", params: {} },
    color: "#F2994A",
  },
  {
    id: "resource-boom",
    name: "Resource Boom",
    description: "All agent resources doubled",
    icon: "\uD83D\uDCC8",
    shock: { type: "resource_boom", params: {} },
    color: "#6FCF97",
  },
  {
    id: "reputation-reset",
    name: "Reputation Reset",
    description: "Reset all reputations to 0.5",
    icon: "\uD83D\uDD04",
    shock: { type: "reputation_reset", params: {} },
    color: "#56CCF2",
  },
  {
    id: "mass-freeze",
    name: "Mass Freeze",
    description: "Freeze 30% of agents at random",
    icon: "\u2744\uFE0F",
    shock: { type: "mass_freeze", params: { fraction: 0.3 } },
    color: "#A8CFF5",
  },
  {
    id: "mass-unfreeze",
    name: "Mass Unfreeze",
    description: "Unfreeze all frozen agents",
    icon: "\u2600\uFE0F",
    shock: { type: "mass_unfreeze", params: {} },
    color: "#F2C94C",
  },
  {
    id: "reputation-shuffle",
    name: "Reputation Shuffle",
    description: "Randomly reassign all reputations",
    icon: "\uD83C\uDFB2",
    shock: { type: "reputation_shuffle", params: {} },
    color: "#BB6BD9",
  },
];

/** Apply a shock event to a live engine instance */
export function applyShock(engine: LiveEngine, shock: ShockEvent): void {
  const rng = engine.rng;

  switch (shock.type) {
    case "agent_wave": {
      const rawType = String(shock.params.agentType ?? "adversarial");
      const agentType: AgentType = (rawType in AGENT_PROFILES) ? rawType as AgentType : "adversarial";
      const rawCount = Number(shock.params.count) || 1;
      const count = Math.min(Math.max(1, Math.floor(rawCount)), MAX_SHOCK_SPAWN);
      for (let i = 0; i < count; i++) {
        try {
          engine.spawnAgent(agentType);
        } catch {
          break; // Population cap reached
        }
      }
      break;
    }

    case "resource_crash": {
      for (const agent of engine.agents) {
        agent.resources = Math.max(0, agent.resources * 0.5);
      }
      break;
    }

    case "resource_boom": {
      for (const agent of engine.agents) {
        agent.resources = Math.min(MAX_RESOURCES, agent.resources * 2);
      }
      break;
    }

    case "reputation_reset": {
      for (const agent of engine.agents) {
        agent.reputation = 0.5;
      }
      break;
    }

    case "mass_freeze": {
      const rawFraction = Number(shock.params.fraction) || 0.3;
      const fraction = Math.max(0, Math.min(1, rawFraction));
      const unfrozen = engine.agents.filter((a) => !a.isFrozen);
      const toFreeze = Math.ceil(unfrozen.length * fraction);
      // Fisher-Yates shuffle using seeded RNG for determinism
      const shuffled = [...unfrozen];
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      for (let i = 0; i < toFreeze && i < shuffled.length; i++) {
        shuffled[i].isFrozen = true;
      }
      break;
    }

    case "mass_unfreeze": {
      for (const agent of engine.agents) {
        agent.isFrozen = false;
      }
      break;
    }

    case "reputation_shuffle": {
      // Collect all reputations, Fisher-Yates shuffle with seeded RNG
      const reps = engine.agents.map((a) => a.reputation);
      for (let i = reps.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [reps[i], reps[j]] = [reps[j], reps[i]];
      }
      engine.agents.forEach((a, i) => {
        a.reputation = reps[i];
      });
      break;
    }

    case "info_asymmetry":
    case "toxin_injection":
      // Simulate asymmetric information by perturbing reputations
      // using seeded RNG for deterministic replay
      for (const agent of engine.agents) {
        const nudge = (rng() - 0.5) * 0.2;
        agent.reputation = Math.max(0, Math.min(1, agent.reputation + nudge));
      }
      break;
  }
}
