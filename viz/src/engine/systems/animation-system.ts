import type { AgentVisual } from "../types";
import type { AgentSnapshot } from "@/data/types";
import { lerp, clamp, smootherstep } from "@/utils/math";
import { AGENT_GRID_SPACING } from "../constants";

/** Interpolate agent states between two epoch snapshots */
export function interpolateAgents(
  prevAgents: Map<string, AgentSnapshot>,
  nextAgents: Map<string, AgentSnapshot>,
  t: number,
  agentPositions: Map<string, { gridX: number; gridY: number }>,
): AgentVisual[] {
  const result: AgentVisual[] = [];
  const allIds = new Set([...prevAgents.keys(), ...nextAgents.keys()]);

  // Ease the interpolation parameter for smooth epoch transitions
  const st = smootherstep(0, 1, t);

  for (const id of allIds) {
    const prev = prevAgents.get(id);
    const next = nextAgents.get(id);
    const pos = agentPositions.get(id);
    if (!pos) continue;

    const source = prev ?? next!;
    const target = next ?? prev!;

    const reputation = lerp(source.reputation, target.reputation, st);
    const resources = lerp(source.resources, target.resources, st);
    const totalPayoff = lerp(source.total_payoff, target.total_payoff, st);
    const avgP = lerp(
      (source.avg_p_initiated + source.avg_p_received) / 2 || 0.5,
      (target.avg_p_initiated + target.avg_p_received) / 2 || 0.5,
      st,
    );

    // Boolean states snap at t=0.5
    const isFrozen = t < 0.5 ? source.is_frozen : target.is_frozen;
    const isQuarantined = t < 0.5 ? source.is_quarantined : target.is_quarantined;

    // Map reputation to scale (0.6-1.4)
    const scale = clamp(
      0.6 + (reputation + 1) * 0.4, // reputation ~[-1,1] -> 0.6-1.4
      0.6,
      1.4,
    );

    result.push({
      id,
      name: target.name ?? id.slice(0, 8),
      agentType: target.agent_type ?? "honest",
      reputation,
      resources,
      totalPayoff,
      avgP,
      isFrozen,
      isQuarantined,
      scale,
      gridX: pos.gridX,
      gridY: pos.gridY,
      interactionsInitiated: Math.round(lerp(source.interactions_initiated, target.interactions_initiated, st)),
      interactionsReceived: Math.round(lerp(source.interactions_received, target.interactions_received, st)),
      walkOffsetX: 0,
      walkOffsetY: 0,
      walkPhase: 0,
      facing: 1,
    });
  }

  return result;
}

/** Compute grid positions for agents (spiral layout) */
export function computeAgentPositions(agentIds: string[]): Map<string, { gridX: number; gridY: number }> {
  const positions = new Map<string, { gridX: number; gridY: number }>();
  const n = agentIds.length;
  const cols = Math.ceil(Math.sqrt(n));

  for (let i = 0; i < n; i++) {
    const row = Math.floor(i / cols);
    const col = i % cols;
    positions.set(agentIds[i], {
      gridX: col * AGENT_GRID_SPACING + 1,
      gridY: row * AGENT_GRID_SPACING + 1,
    });
  }

  return positions;
}
