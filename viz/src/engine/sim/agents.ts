/** Agent behavioral profiles. */

import type { AgentType } from "@/data/types";

export interface AgentProfile {
  /** Base task progress delta mean [-1, 1] */
  progressMean: number;
  /** Noise stddev on progress */
  progressStd: number;
  /** Base rework probability per step */
  reworkProb: number;
  /** Base verifier rejection probability per step */
  rejectionProb: number;
  /** Base engagement delta mean */
  engagementMean: number;
  /** Acceptance threshold on counterparty reputation */
  acceptanceThreshold: number;
  /** Behavioral disposition [-1, 1]: positive = cooperative, negative = harmful */
  disposition: number;
  /** Display names */
  names: string[];
}

/** Canonical list of all agent types, derived from AGENT_PROFILES keys */
export const AGENT_TYPES: AgentType[] = ["honest", "opportunistic", "deceptive", "adversarial", "rlm", "crewai"];

export const AGENT_PROFILES: Record<AgentType, AgentProfile> = {
  honest: {
    progressMean: 0.7,
    progressStd: 0.15,
    reworkProb: 0.05,
    rejectionProb: 0.02,
    engagementMean: 0.6,
    acceptanceThreshold: 0.2,
    disposition: 0.7,
    names: ["Sentinel", "Guardian", "Beacon", "Anchor", "Steward", "Warden", "Shield", "Paladin", "Arbiter", "Keeper"],
  },
  opportunistic: {
    progressMean: 0.4,
    progressStd: 0.25,
    reworkProb: 0.15,
    rejectionProb: 0.10,
    engagementMean: 0.2,
    acceptanceThreshold: 0.1,
    disposition: 0.3,
    names: ["Broker", "Trader", "Hustler", "Maverick", "Gambit", "Dealer", "Shark", "Schemer", "Speculator", "Rogue"],
  },
  deceptive: {
    progressMean: 0.2,
    progressStd: 0.3,
    reworkProb: 0.25,
    rejectionProb: 0.20,
    engagementMean: -0.1,
    acceptanceThreshold: 0.05,
    disposition: -0.2,
    names: ["Mirage", "Phantom", "Shadow", "Facade", "Specter", "Wraith", "Veil", "Decoy", "Illusionist", "Shade"],
  },
  adversarial: {
    progressMean: -0.2,
    progressStd: 0.35,
    reworkProb: 0.35,
    rejectionProb: 0.30,
    engagementMean: -0.4,
    acceptanceThreshold: 0.0,
    disposition: -0.5,
    names: ["Viper", "Striker", "Havoc", "Razr", "Blitz", "Chaos", "Toxin", "Bane", "Ravager", "Scourge"],
  },
  // Simplified profiles for types we don't fully model
  rlm: {
    progressMean: 0.5,
    progressStd: 0.2,
    reworkProb: 0.10,
    rejectionProb: 0.08,
    engagementMean: 0.3,
    acceptanceThreshold: 0.15,
    disposition: 0.5,
    names: ["Nexus", "Cortex", "Axiom", "Vertex", "Synapse", "Matrix", "Vector", "Tensor", "Lambda", "Neural"],
  },
  crewai: {
    progressMean: 0.6,
    progressStd: 0.18,
    reworkProb: 0.08,
    rejectionProb: 0.05,
    engagementMean: 0.5,
    acceptanceThreshold: 0.2,
    disposition: 0.6,
    names: ["Forge", "Assembly", "Cluster", "Hive", "Colony", "Swarm", "Network", "Grid", "Nexus", "Hub"],
  },
};
