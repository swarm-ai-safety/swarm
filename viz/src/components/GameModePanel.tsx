"use client";

import React, { useState, useCallback } from "react";
import { useGame } from "@/state/game-context";
import type { GameMode } from "@/state/game-context";
import { CampaignPanel } from "./CampaignPanel";
import { RedTeamPanel } from "./RedTeamPanel";
import { DEFAULT_CONFIG, type ScenarioConfig } from "@/engine/sim/types";

const MODE_OPTIONS: { key: GameMode; label: string; description: string }[] = [
  { key: "sandbox", label: "Sandbox", description: "Free-form simulation with live controls" },
  { key: "campaign", label: "Campaign", description: "Challenge levels with win/lose conditions" },
  { key: "redteam", label: "Red Team", description: "Cause maximum system failure" },
];

export function GameModePanel() {
  const { startSandbox, loadGame } = useGame();
  const [mode, setMode] = useState<GameMode>("sandbox");
  const [agentCounts, setAgentCounts] = useState({ honest: 3, opportunistic: 1, deceptive: 1, adversarial: 0 });

  const handleStartSandbox = useCallback(() => {
    const config: ScenarioConfig = {
      ...DEFAULT_CONFIG,
      agents: [
        { type: "honest", count: agentCounts.honest },
        { type: "opportunistic", count: agentCounts.opportunistic },
        { type: "deceptive", count: agentCounts.deceptive },
        { type: "adversarial", count: agentCounts.adversarial },
      ],
    };
    const totalAgents = Object.values(agentCounts).reduce((s, c) => s + c, 0);
    if (totalAgents < 2) {
      alert("Need at least 2 agents to start.");
      return;
    }
    startSandbox(config);
  }, [agentCounts, startSandbox]);

  const handleLoadSave = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const text = await file.text();
      loadGame(text);
    };
    input.click();
  }, [loadGame]);

  return (
    <div className="space-y-4">
      {/* Mode selector */}
      <div className="grid grid-cols-3 gap-1.5">
        {MODE_OPTIONS.map((opt) => (
          <button
            key={opt.key}
            onClick={() => setMode(opt.key)}
            className={`text-center px-2 py-2 rounded text-xs transition-colors ${
              mode === opt.key
                ? "bg-accent text-bg font-bold"
                : "bg-btn hover:bg-btn-hover"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Mode-specific content */}
      {mode === "sandbox" && (
        <div className="space-y-3">
          <p className="text-xs text-muted">
            Free sandbox — spawn agents, adjust governance, inject shocks in real-time.
          </p>

          {/* Quick agent config */}
          <div className="space-y-2">
            <div className="text-xs font-bold uppercase tracking-wider">Starting Agents</div>
            {(["honest", "opportunistic", "deceptive", "adversarial"] as const).map((type) => (
              <div key={type} className="flex items-center gap-2">
                <span className="text-xs text-muted w-28 shrink-0 capitalize">{type}</span>
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={1}
                  value={agentCounts[type]}
                  onChange={(e) => setAgentCounts((prev) => ({ ...prev, [type]: parseInt(e.target.value) }))}
                  className="flex-1 h-1 accent-accent"
                />
                <span className="text-xs font-mono w-6 text-right">{agentCounts[type]}</span>
              </div>
            ))}
            <p className="text-xs text-muted">
              Total: {Object.values(agentCounts).reduce((s, c) => s + c, 0)} agents
            </p>
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleStartSandbox}
              className="flex-1 py-2 rounded font-bold text-sm bg-accent text-bg hover:opacity-90 transition-opacity"
            >
              Start Sandbox
            </button>
            <button
              onClick={handleLoadSave}
              className="px-3 py-2 rounded text-sm bg-btn hover:bg-btn-hover transition-colors"
              title="Load a saved game"
            >
              Load
            </button>
          </div>
        </div>
      )}

      {mode === "campaign" && <CampaignPanel />}
      {mode === "redteam" && <RedTeamPanel />}
    </div>
  );
}
