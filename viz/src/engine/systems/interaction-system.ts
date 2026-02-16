import type { InteractionArc } from "../types";
import type { InteractionEvent } from "@/data/types";
import { ANIM } from "../constants";

export class InteractionSystem {
  arcs: InteractionArc[] = [];
  private arcCounter = 0;

  /** Add arcs from interaction events for an epoch */
  addFromEvents(events: InteractionEvent[], epoch: number) {
    for (const evt of events) {
      if (evt.epoch !== epoch) continue;
      this.arcs.push({
        id: `arc-${this.arcCounter++}`,
        fromId: evt.initiator,
        toId: evt.counterparty,
        p: evt.p,
        accepted: evt.accepted,
        progress: 0,
        epoch,
      });
    }
  }

  /** Add synthetic arcs when no event data (from agent snapshots) */
  addSyntheticArcs(
    agentIds: string[],
    epoch: number,
    epochInteractions: number,
    avgP: number,
    acceptRate: number,
  ) {
    const count = Math.min(epochInteractions, agentIds.length * 2);
    const rng = mulberry32(epoch * 1000 + 7);

    for (let i = 0; i < count; i++) {
      const fromIdx = Math.floor(rng() * agentIds.length);
      let toIdx = Math.floor(rng() * agentIds.length);
      if (toIdx === fromIdx) toIdx = (toIdx + 1) % agentIds.length;

      this.arcs.push({
        id: `arc-${this.arcCounter++}`,
        fromId: agentIds[fromIdx],
        toId: agentIds[toIdx],
        p: avgP + (rng() - 0.5) * 0.3,
        accepted: rng() < acceptRate,
        progress: rng() * 0.3, // stagger start
        epoch,
      });
    }
  }

  /** Update all arc animations — confident arcs travel faster */
  update(dt: number, currentEpoch: number) {
    for (let i = this.arcs.length - 1; i >= 0; i--) {
      const arc = this.arcs[i];
      const pClamped = Math.max(0, Math.min(1, arc.p));
      const pSpeedFactor = 0.8 + pClamped * 0.4; // high-p = faster
      arc.progress += (dt / ANIM.arcLifetime) * pSpeedFactor;
      // Remove completed arcs from non-current epochs
      if (arc.progress >= 1 && arc.epoch !== currentEpoch) {
        this.arcs.splice(i, 1);
      } else if (arc.progress >= 1) {
        arc.progress = 1;
      }
    }
  }

  /** Clear all arcs */
  clear() {
    this.arcs = [];
    this.arcCounter = 0;
  }

  /** Get active arcs */
  getActive(): InteractionArc[] {
    return this.arcs.filter((a) => a.progress < 1);
  }
}

function mulberry32(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
