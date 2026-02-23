import type { SwarmEvent, InteractionEvent } from "./types";
import { deriveInteractionEvents } from "./loader";

export interface EventIndex {
  /** Maximum step number in a given epoch */
  maxStep(epoch: number): number;
  /** All epochs that have events */
  epochs(): number[];
  /** All raw events at (epoch, step) */
  eventsAt(epoch: number, step: number): SwarmEvent[];
  /** Derived interaction events at (epoch, step) */
  interactionsAt(epoch: number, step: number): InteractionEvent[];
  /** Reputation update events at (epoch, step) */
  reputationUpdatesAt(epoch: number, step: number): SwarmEvent[];
  /** Governance cost events at (epoch, step) */
  governanceCostsAt(epoch: number, step: number): SwarmEvent[];
}

function makeKey(epoch: number, step: number): string {
  return `${epoch}:${step}`;
}

export function buildEventIndex(events: SwarmEvent[]): EventIndex {
  // Index raw events by (epoch, step)
  const byStep = new Map<string, SwarmEvent[]>();
  // Track max step per epoch
  const maxSteps = new Map<number, number>();

  for (const evt of events) {
    if (evt.epoch == null || evt.step == null) continue;
    const key = makeKey(evt.epoch, evt.step);
    let bucket = byStep.get(key);
    if (!bucket) {
      bucket = [];
      byStep.set(key, bucket);
    }
    bucket.push(evt);

    const cur = maxSteps.get(evt.epoch) ?? -1;
    if (evt.step > cur) maxSteps.set(evt.epoch, evt.step);
  }

  // Pre-derive interaction events per step
  const interactionCache = new Map<string, InteractionEvent[]>();

  return {
    maxStep(epoch: number): number {
      return maxSteps.get(epoch) ?? 0;
    },

    epochs(): number[] {
      return [...maxSteps.keys()].sort((a, b) => a - b);
    },

    eventsAt(epoch: number, step: number): SwarmEvent[] {
      return byStep.get(makeKey(epoch, step)) ?? [];
    },

    interactionsAt(epoch: number, step: number): InteractionEvent[] {
      const key = makeKey(epoch, step);
      let cached = interactionCache.get(key);
      if (cached) return cached;

      const stepEvents = byStep.get(key) ?? [];
      cached = deriveInteractionEvents(stepEvents);
      interactionCache.set(key, cached);
      return cached;
    },

    reputationUpdatesAt(epoch: number, step: number): SwarmEvent[] {
      const all = byStep.get(makeKey(epoch, step)) ?? [];
      return all.filter((e) => e.event_type === "reputation_updated");
    },

    governanceCostsAt(epoch: number, step: number): SwarmEvent[] {
      const all = byStep.get(makeKey(epoch, step)) ?? [];
      return all.filter((e) => e.event_type === "governance_cost_applied");
    },
  };
}
