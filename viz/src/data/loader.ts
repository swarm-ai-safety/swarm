import type { SimulationData, InteractionEvent, SwarmEvent, AgentSnapshot, AgentType, EpochSnapshot, InteractionType } from "./types";
import { mulberry32 } from "@/engine/sim/rng";
import { AGENT_PROFILES, AGENT_TYPES } from "@/engine/sim/agents";

/** Load simulation data from a history.json file or File object */
export async function loadSimulationData(source: string | File): Promise<SimulationData> {
  let raw: string;
  if (typeof source === "string") {
    const resp = await fetch(source);
    if (!resp.ok) throw new Error(`Failed to load: ${resp.statusText}`);
    raw = await resp.text();
  } else {
    raw = await source.text();
  }

  const data = JSON.parse(raw) as SimulationData;
  validateData(data);

  // If agent_snapshots is empty, synthesize from epoch data
  if (data.agent_snapshots.length === 0 && data.epoch_snapshots.length > 0) {
    data.agent_snapshots = synthesizeAgentSnapshots(data);
  }

  // Normalize events: ensure it's an array if present, filter to interactions
  if (data.events && Array.isArray(data.events)) {
    data.events = data.events.filter(
      (evt) => !evt.event_type || evt.event_type === "interaction",
    );
  } else {
    data.events = undefined;
  }

  return data;
}

/** Parse JSONL events file */
export async function loadEvents(source: string | File): Promise<InteractionEvent[]> {
  let raw: string;
  if (typeof source === "string") {
    const resp = await fetch(source);
    if (!resp.ok) throw new Error(`Failed to load events: ${resp.statusText}`);
    raw = await resp.text();
  } else {
    raw = await source.text();
  }

  return raw
    .trim()
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as InteractionEvent)
    .filter((evt) => evt.event_type === "interaction");
}

function validateData(data: SimulationData) {
  if (!data.epoch_snapshots || !Array.isArray(data.epoch_snapshots)) {
    throw new Error("Missing epoch_snapshots array");
  }
  if (!Array.isArray(data.agent_snapshots)) {
    // Agent snapshots might be empty - that's ok, we'll derive from epochs
    data.agent_snapshots = data.agent_snapshots ?? [];
  }
  if (data.epoch_snapshots.length === 0) {
    throw new Error("No epoch snapshots found");
  }
}

/**
 * Synthesize per-agent-per-epoch snapshots from epoch-level data.
 * Uses seeded PRNG so the same history.json always produces the same agents.
 * The agent count comes from epoch_snapshots[0].n_agents.
 */
function synthesizeAgentSnapshots(data: SimulationData): AgentSnapshot[] {
  const rng = mulberry32(data.seed ?? 42);
  const nAgents = data.n_agents || data.epoch_snapshots[0].n_agents || 6;
  const snapshots: AgentSnapshot[] = [];

  // Assign types round-robin with variation
  const agents: { id: string; name: string; type: AgentType; disposition: number }[] = [];
  for (let i = 0; i < nAgents; i++) {
    const typeIdx = i % AGENT_TYPES.length;
    const agentType = AGENT_TYPES[typeIdx];
    const names = AGENT_PROFILES[agentType].names;
    const name = names[i % names.length] + (i >= AGENT_TYPES.length ? `-${Math.floor(i / AGENT_TYPES.length) + 1}` : "");

    const disposition = AGENT_PROFILES[agentType].disposition + (rng() - 0.5) * 0.3;

    agents.push({
      id: `agent-${i}`,
      name,
      type: agentType,
      disposition,
    });
  }

  // Track running state per agent
  const state = agents.map(() => ({
    reputation: 0.5 + (rng() - 0.5) * 0.2,
    resources: 10 + rng() * 5,
    totalPayoff: 0,
    initiated: 0,
    received: 0,
  }));

  for (const epoch of data.epoch_snapshots) {
    for (let i = 0; i < agents.length; i++) {
      const agent = agents[i];
      const s = state[i];
      const noise = rng() - 0.5;

      // Evolve reputation based on disposition + ecosystem state
      const ecosystemStress = (epoch.toxicity_rate ?? 0) * 0.5;
      const repDelta = agent.disposition * 0.05 + noise * 0.03 - ecosystemStress * 0.02;
      s.reputation = Math.max(0, Math.min(1, s.reputation + repDelta));

      // Resources evolve with payoff
      const payoffThisEpoch = (epoch.avg_payoff ?? 0) * (0.8 + agent.disposition * 0.4) + noise * 2;
      s.totalPayoff += payoffThisEpoch;
      s.resources = Math.max(0, s.resources + payoffThisEpoch * 0.1);

      // Interactions roughly equal share of total
      const shareOfInteractions = epoch.total_interactions / nAgents;
      const initDelta = Math.max(0, Math.round(shareOfInteractions * (0.4 + rng() * 0.3)));
      const recvDelta = Math.max(0, Math.round(shareOfInteractions * (0.3 + rng() * 0.3)));
      s.initiated += initDelta;
      s.received += recvDelta;

      // Avg p: based on disposition with epoch-level influence
      const avgP = Math.max(0, Math.min(1,
        0.5 + agent.disposition * 0.3 + noise * 0.1 - (epoch.toxicity_rate ?? 0) * 0.15
      ));

      // Frozen/quarantined: adversarial/deceptive agents may get flagged in high-threat epochs
      const threatLevel = epoch.ecosystem_threat_level ?? 0;
      const isFrozen = agent.type === "adversarial" && threatLevel > 0.5 && s.reputation < 0.3 && rng() > 0.4;
      const isQuarantined = agent.type === "deceptive" && threatLevel > 0.4 && s.reputation < 0.35 && rng() > 0.5;

      snapshots.push({
        agent_id: agent.id,
        epoch: epoch.epoch,
        name: agent.name,
        agent_type: agent.type,
        reputation: s.reputation,
        resources: s.resources,
        interactions_initiated: s.initiated,
        interactions_received: s.received,
        avg_p_initiated: avgP,
        avg_p_received: Math.max(0, Math.min(1, avgP + (rng() - 0.5) * 0.1)),
        total_payoff: s.totalPayoff,
        is_frozen: isFrozen,
        is_quarantined: isQuarantined,
      });
    }
  }

  return snapshots;
}
