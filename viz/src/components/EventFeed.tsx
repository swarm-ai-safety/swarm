"use client";

import React, { useState, useMemo } from "react";
import { useSimulation } from "@/state/use-simulation";
import type { SwarmEvent, SwarmEventType } from "@/data/types";

const EVENT_COLORS: Record<SwarmEventType, string> = {
  simulation_started: "#7D8590",
  simulation_ended: "#7D8590",
  epoch_completed: "#7D8590",
  agent_created: "#2F80ED",
  contract_signing: "#BB6BD9",
  interaction_proposed: "#3ECFB4",
  interaction_completed: "#3ECFB4",
  governance_cost_applied: "#F2994A",
  reputation_updated: "#6FCF97",
  payoff_computed: "#E6EDF3",
  contract_metrics: "#BB6BD9",
};

const EVENT_LABELS: Partial<Record<SwarmEventType, string>> = {
  interaction_proposed: "PROPOSE",
  interaction_completed: "COMPLETE",
  reputation_updated: "REP",
  governance_cost_applied: "GOV",
  payoff_computed: "PAYOFF",
  agent_created: "AGENT",
  contract_signing: "CONTRACT",
  epoch_completed: "EPOCH",
};

function formatPayload(evt: SwarmEvent): string {
  switch (evt.event_type) {
    case "interaction_proposed":
      return `${evt.initiator_id} -> ${evt.counterparty_id} (${evt.payload.interaction_type})`;
    case "interaction_completed":
      return `${evt.interaction_id?.slice(0, 8)}... ${evt.payload.accepted ? "accepted" : "rejected"}`;
    case "reputation_updated": {
      const delta = evt.payload.delta as number;
      const sign = delta >= 0 ? "+" : "";
      return `${evt.agent_id} ${sign}${delta.toFixed(3)}`;
    }
    case "governance_cost_applied":
      return `${evt.agent_id} cost=${(evt.payload.cost as number)?.toFixed(2) ?? "?"}`;
    case "payoff_computed":
      return `${evt.initiator_id} -> ${evt.counterparty_id}`;
    default:
      return evt.agent_id ?? "";
  }
}

export function EventFeed() {
  const { data, eventIndex, currentEpoch, currentStep, stepPlayback } = useSimulation();
  const [visible, setVisible] = useState(false);

  // Get events for current position
  const events = useMemo((): SwarmEvent[] => {
    if (!eventIndex || !stepPlayback) return [];
    return eventIndex.eventsAt(currentEpoch, currentStep);
  }, [eventIndex, stepPlayback, currentEpoch, currentStep]);

  // Don't render toggle if no event data
  if (!data?.rawEvents || data.rawEvents.length === 0) return null;

  return (
    <>
      {/* Toggle button */}
      <button
        onClick={() => setVisible((v) => !v)}
        className={`absolute top-2 right-2 z-[10001] px-2 py-1 text-xs rounded transition-colors ${
          visible ? "bg-secondary text-bg font-bold" : "bg-btn hover:bg-btn-hover text-muted"
        }`}
      >
        Events
      </button>

      {/* Feed panel */}
      {visible && (
        <div className="absolute top-10 right-2 z-[10001] w-80 max-h-[50vh] bg-panel/95 border border-border rounded-lg overflow-hidden flex flex-col backdrop-blur-sm">
          <div className="px-3 py-2 border-b border-border text-xs text-muted font-mono">
            Epoch {currentEpoch} / Step {currentStep} — {events.length} events
          </div>
          <div className="overflow-y-auto flex-1 p-1">
            {events.length === 0 ? (
              <div className="text-xs text-muted/50 text-center py-4">No events at this step</div>
            ) : (
              events.map((evt, i) => (
                <div key={evt.event_id || i} className="flex items-start gap-2 px-2 py-1 hover:bg-btn/50 rounded text-xs">
                  <span
                    className="shrink-0 px-1 py-0.5 rounded font-mono text-[10px] leading-tight"
                    style={{ backgroundColor: EVENT_COLORS[evt.event_type] + "22", color: EVENT_COLORS[evt.event_type] }}
                  >
                    {EVENT_LABELS[evt.event_type] ?? evt.event_type.toUpperCase()}
                  </span>
                  <span className="text-muted truncate font-mono">{formatPayload(evt)}</span>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </>
  );
}
