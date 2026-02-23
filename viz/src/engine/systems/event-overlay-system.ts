import type { SwarmEvent } from "@/data/types";

/** A text particle that floats above an agent for a short duration */
export interface EventOverlay {
  id: string;
  agentId: string;
  text: string;
  color: string;
  life: number;
  maxLife: number;
  offsetY: number; // starts at 0, drifts upward
}

const OVERLAY_COLORS: Record<string, string> = {
  reputation_updated: "#3ECFB4",
  governance_cost_applied: "#F2994A",
  contract_signing: "#BB6BD9",
};

let overlayCounter = 0;

export class EventOverlaySystem {
  overlays: EventOverlay[] = [];

  /** Ingest SwarmEvents and create visual overlays */
  addFromEvents(events: SwarmEvent[]) {
    for (const evt of events) {
      if (evt.event_type === "reputation_updated" && evt.agent_id) {
        const delta = evt.payload.delta as number | undefined;
        if (delta == null) continue;
        const sign = delta >= 0 ? "+" : "";
        this.overlays.push({
          id: `ov-${overlayCounter++}`,
          agentId: evt.agent_id,
          text: `${sign}${delta.toFixed(2)} rep`,
          color: delta >= 0 ? "#3ECFB4" : "#EB5757",
          life: 1200,
          maxLife: 1200,
          offsetY: 0,
        });
      }

      if (evt.event_type === "governance_cost_applied" && evt.agent_id) {
        const cost = evt.payload.cost as number | undefined;
        this.overlays.push({
          id: `ov-${overlayCounter++}`,
          agentId: evt.agent_id,
          text: cost != null ? `-${cost.toFixed(2)} gov` : "gov cost",
          color: OVERLAY_COLORS.governance_cost_applied,
          life: 1000,
          maxLife: 1000,
          offsetY: 0,
        });
      }

      if (evt.event_type === "contract_signing" && evt.agent_id) {
        this.overlays.push({
          id: `ov-${overlayCounter++}`,
          agentId: evt.agent_id,
          text: "contract",
          color: OVERLAY_COLORS.contract_signing,
          life: 1500,
          maxLife: 1500,
          offsetY: 0,
        });
      }
    }
  }

  /** Advance overlay lifetimes, remove expired */
  update(dt: number) {
    for (let i = this.overlays.length - 1; i >= 0; i--) {
      const ov = this.overlays[i];
      ov.life -= dt;
      ov.offsetY -= dt * 0.03; // drift upward
      if (ov.life <= 0) {
        this.overlays.splice(i, 1);
      }
    }
  }

  /** Get overlays for a specific agent */
  getForAgent(agentId: string): EventOverlay[] {
    return this.overlays.filter((o) => o.agentId === agentId);
  }

  /** Alpha based on remaining life */
  getAlpha(overlay: EventOverlay): number {
    const t = overlay.life / overlay.maxLife;
    // Fade in quickly, fade out over last 30%
    if (t > 0.9) return (1 - t) * 10;
    if (t < 0.3) return t / 0.3;
    return 1;
  }

  clear() {
    this.overlays = [];
  }
}
