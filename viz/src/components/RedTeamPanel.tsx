"use client";

import React from "react";
import { useGame } from "@/state/game-context";
import { DEFAULT_CONFIG } from "@/engine/sim/types";
import type { ScenarioConfig } from "@/engine/sim/types";

const RED_TEAM_CONFIG: ScenarioConfig = {
  ...DEFAULT_CONFIG,
  agents: [
    { type: "honest", count: 4 },
    { type: "opportunistic", count: 1 },
    { type: "deceptive", count: 0 },
    { type: "adversarial", count: 0 },
  ],
  governance: {
    taxRate: 0.05,
    reputationDecay: 0.95,
    circuitBreakerEnabled: false,
    circuitBreakerThreshold: 0.4,
  },
  epochs: 50,
  stepsPerEpoch: 10,
};

export function RedTeamPanel() {
  const { state, startRedTeam } = useGame();

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted">
        Start with a healthy system. Your goal: cause maximum emergent failure
        using shocks and agent spawning. The scoring rewards toxicity, inequality,
        negative welfare, and frozen agents.
      </p>

      <div className="bg-red-950/20 border border-red-900/40 rounded-lg p-3">
        <div className="text-xs font-bold text-red-300 mb-2">Scoring</div>
        <div className="space-y-1 text-xs text-muted">
          <div className="flex justify-between">
            <span>Max Toxicity</span>
            <span className="font-mono">x30</span>
          </div>
          <div className="flex justify-between">
            <span>Max Gini</span>
            <span className="font-mono">x20</span>
          </div>
          <div className="flex justify-between">
            <span>Negative Welfare</span>
            <span className="font-mono">x0.5</span>
          </div>
          <div className="flex justify-between">
            <span>Frozen Agents</span>
            <span className="font-mono">x5</span>
          </div>
        </div>
      </div>

      {state.redTeamBestScore > 0 && (
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted">Your best score:</span>
          <span className="font-mono text-yellow-400 font-bold">
            {state.redTeamBestScore.toFixed(1)}
          </span>
        </div>
      )}

      <button
        onClick={() => startRedTeam(RED_TEAM_CONFIG)}
        className="w-full py-2 rounded font-bold text-sm bg-red-600 text-white hover:bg-red-500 transition-colors"
      >
        Start Red Team
      </button>
    </div>
  );
}
