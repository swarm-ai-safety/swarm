"use client";

import React from "react";
import { useSimulation } from "@/state/use-simulation";
import { AGENT_COLORS, AGENT_LABELS } from "@/engine/constants";
import { formatNumber, formatPercent } from "@/utils/format";

export function InfoPanel() {
  const { selectedAgent, agents, setSelected } = useSimulation();

  const agent = agents.find((a) => a.id === selectedAgent);
  if (!agent) return null;

  const colors = AGENT_COLORS[agent.agentType];

  return (
    <div className="absolute top-4 left-4 w-72 bg-panel border border-border rounded-lg shadow-lg z-20 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: colors.secondary }}
          />
          <span className="font-semibold text-sm">{agent.name}</span>
        </div>
        <button
          onClick={() => setSelected(null)}
          className="text-muted hover:text-text transition-colors text-lg leading-none"
        >
          x
        </button>
      </div>

      {/* Type */}
      <div className="px-4 py-2 border-b border-border">
        <span className="text-xs text-muted">Type: </span>
        <span
          className="text-xs font-medium"
          style={{ color: colors.secondary }}
        >
          {AGENT_LABELS[agent.agentType]} ({agent.agentType})
        </span>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-px bg-border">
        <Stat label="Reputation" value={formatNumber(agent.reputation, 3)} />
        <Stat label="Resources" value={formatNumber(agent.resources, 1)} />
        <Stat label="Total Payoff" value={formatNumber(agent.totalPayoff, 2)} />
        <Stat label="Avg P" value={formatPercent(agent.avgP)} />
        <Stat label="Initiated" value={String(agent.interactionsInitiated)} />
        <Stat label="Received" value={String(agent.interactionsReceived)} />
        <Stat label="Floors" value={String(agent.floors)} />
        <Stat
          label="Status"
          value={
            agent.isFrozen
              ? "FROZEN"
              : agent.isQuarantined
                ? "QUARANTINED"
                : "Active"
          }
          color={
            agent.isFrozen
              ? "#A8CFF5"
              : agent.isQuarantined
                ? "#EB5757"
                : "#6FCF97"
          }
        />
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="bg-panel px-3 py-2">
      <div className="text-[10px] text-muted uppercase tracking-wider">
        {label}
      </div>
      <div
        className="text-sm font-mono mt-0.5"
        style={color ? { color } : undefined}
      >
        {value}
      </div>
    </div>
  );
}
