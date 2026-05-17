"use client";

import React, { useState, useCallback } from "react";
import type { ScenarioConfig } from "@/engine/sim/types";
import { DEFAULT_CONFIG } from "@/engine/sim/types";
import { runSimulation } from "@/engine/sim/orchestrator";

type SweepParam = "taxRate" | "reputationDecay" | "rho_a" | "s_plus" | "h";

const SWEEP_OPTIONS: { key: SweepParam; label: string; min: number; max: number }[] = [
  { key: "taxRate", label: "Tax Rate", min: 0, max: 0.5 },
  { key: "reputationDecay", label: "Rep. Decay", min: 0.8, max: 1.0 },
  { key: "rho_a", label: "Externality (rho_a)", min: 0, max: 1.0 },
  { key: "s_plus", label: "Surplus (s_plus)", min: 0.5, max: 5.0 },
  { key: "h", label: "Harm (h)", min: 0, max: 5.0 },
];

interface SweepResult {
  paramValue: number;
  toxicity: number;
  avgP: number;
  gini: number;
  welfare: number;
}

/** Small-multiples sweep of one parameter */
export function SweepPanel() {
  const [param, setParam] = useState<SweepParam>("taxRate");
  const [steps, setSteps] = useState(8);
  const [results, setResults] = useState<SweepResult[] | null>(null);
  const [running, setRunning] = useState(false);

  const opt = SWEEP_OPTIONS.find((o) => o.key === param)!;

  const handleSweep = useCallback(() => {
    setRunning(true);
    setTimeout(() => {
      const base = { ...DEFAULT_CONFIG };
      const sweepResults: SweepResult[] = [];

      for (let i = 0; i < steps; i++) {
        const value = opt.min + (opt.max - opt.min) * (i / (steps - 1));
        let config: ScenarioConfig;

        if (param === "taxRate" || param === "reputationDecay") {
          config = { ...base, governance: { ...base.governance, [param]: value } };
        } else {
          config = { ...base, payoff: { ...base.payoff, [param]: value } };
        }

        const data = runSimulation(config);
        const last = data.epoch_snapshots[data.epoch_snapshots.length - 1];
        sweepResults.push({
          paramValue: value,
          toxicity: last.toxicity_rate,
          avgP: last.avg_p,
          gini: last.gini_coefficient,
          welfare: last.total_welfare,
        });
      }

      setResults(sweepResults);
      setRunning(false);
    }, 10);
  }, [param, steps, opt]);

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted">Sweep one parameter across its range and compare outcomes.</p>

      <div className="flex items-center gap-2">
        <span className="text-xs text-muted shrink-0">Parameter</span>
        <select
          value={param}
          onChange={(e) => { setParam(e.target.value as SweepParam); setResults(null); }}
          className="flex-1 bg-btn text-text text-xs rounded px-2 py-1 border border-border"
        >
          {SWEEP_OPTIONS.map((o) => (
            <option key={o.key} value={o.key}>{o.label}</option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs text-muted w-14 shrink-0">Steps</span>
        <input type="range" min={4} max={16} step={1} value={steps} onChange={(e) => setSteps(parseInt(e.target.value))} className="flex-1 h-1 accent-accent" />
        <span className="text-xs font-mono w-6">{steps}</span>
      </div>

      <button
        onClick={handleSweep}
        disabled={running}
        className="w-full py-2 rounded font-bold text-sm bg-accent text-bg hover:opacity-90 disabled:opacity-50 transition-opacity"
      >
        {running ? "Sweeping..." : "Run Sweep"}
      </button>

      {results && (
        <div className="space-y-2">
          <SweepChart label="Toxicity" results={results} accessor={(r) => r.toxicity} color="#EB5757" paramLabel={opt.label} />
          <SweepChart label="Avg P" results={results} accessor={(r) => r.avgP} color="#3ECFB4" paramLabel={opt.label} />
          <SweepChart label="Gini" results={results} accessor={(r) => r.gini} color="#F2994A" paramLabel={opt.label} />
          <SweepChart label="Welfare" results={results} accessor={(r) => r.welfare} color="#5698FF" paramLabel={opt.label} />
        </div>
      )}
    </div>
  );
}

function SweepChart({ label, results, accessor, color, paramLabel }: {
  label: string;
  results: SweepResult[];
  accessor: (r: SweepResult) => number;
  color: string;
  paramLabel: string;
}) {
  const values = results.map(accessor);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const w = 200, h = 32;

  return (
    <div>
      <div className="flex justify-between text-xs mb-0.5">
        <span className="text-muted">{label}</span>
        <span className="font-mono text-muted">{min.toFixed(2)} - {max.toFixed(2)}</span>
      </div>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="bg-bg rounded border border-border">
        {/* Bars */}
        {results.map((r, i) => {
          const barW = w / results.length - 2;
          const barH = ((accessor(r) - min) / range) * (h - 4) + 2;
          const x = (i / results.length) * w + 1;
          return (
            <g key={i}>
              <rect x={x} y={h - barH} width={barW} height={barH} fill={color} opacity="0.7" rx="1" />
              <title>{paramLabel}: {r.paramValue.toFixed(3)} → {label}: {accessor(r).toFixed(3)}</title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
