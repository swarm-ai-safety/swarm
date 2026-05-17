"use client";

import React, { useState, useCallback, useMemo } from "react";
import type { ScenarioConfig } from "@/engine/sim/types";
import { DEFAULT_CONFIG } from "@/engine/sim/types";
import { runSimulation } from "@/engine/sim/orchestrator";
import type { SimulationData } from "@/data/types";

interface Props {
  onSelect: (data: SimulationData) => void;
}

/** Compare two sim runs (same seed, different governance) side by side */
export function ComparePanel({ onSelect }: Props) {
  const [baseConfig] = useState<ScenarioConfig>({ ...DEFAULT_CONFIG });
  const [taxA, setTaxA] = useState(0.0);
  const [taxB, setTaxB] = useState(0.2);
  const [cbA, setCbA] = useState(false);
  const [cbB, setCbB] = useState(true);
  const [results, setResults] = useState<{ a: SimulationData; b: SimulationData } | null>(null);
  const [running, setRunning] = useState(false);

  const handleCompare = useCallback(() => {
    setRunning(true);
    // Use setTimeout to allow UI to update
    setTimeout(() => {
      const configA: ScenarioConfig = {
        ...baseConfig,
        governance: { ...baseConfig.governance, taxRate: taxA, circuitBreakerEnabled: cbA },
      };
      const configB: ScenarioConfig = {
        ...baseConfig,
        governance: { ...baseConfig.governance, taxRate: taxB, circuitBreakerEnabled: cbB },
      };
      const a = runSimulation(configA);
      const b = runSimulation(configB);
      setResults({ a, b });
      setRunning(false);
    }, 10);
  }, [baseConfig, taxA, taxB, cbA, cbB]);

  const comparison = useMemo(() => {
    if (!results) return null;
    const lastA = results.a.epoch_snapshots[results.a.epoch_snapshots.length - 1];
    const lastB = results.b.epoch_snapshots[results.b.epoch_snapshots.length - 1];
    return {
      a: { toxicity: lastA.toxicity_rate, gini: lastA.gini_coefficient, avgP: lastA.avg_p, welfare: lastA.total_welfare },
      b: { toxicity: lastB.toxicity_rate, gini: lastB.gini_coefficient, avgP: lastB.avg_p, welfare: lastB.total_welfare },
    };
  }, [results]);

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted">Compare two governance configs on the same seed ({baseConfig.seed}).</p>

      <div className="grid grid-cols-2 gap-3">
        {/* Config A */}
        <div className="bg-bg border border-border rounded p-2 space-y-2">
          <div className="text-xs font-bold text-accent">Config A</div>
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted w-12">Tax</span>
            <input type="range" min={0} max={0.5} step={0.01} value={taxA} onChange={(e) => setTaxA(parseFloat(e.target.value))} className="flex-1 h-1 accent-accent" />
            <span className="text-xs font-mono w-8">{taxA.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted w-12">CB</span>
            <button onClick={() => setCbA(!cbA)} className={`px-1.5 py-0.5 rounded text-xs ${cbA ? "bg-accent text-bg" : "bg-btn"}`}>{cbA ? "ON" : "OFF"}</button>
          </div>
        </div>
        {/* Config B */}
        <div className="bg-bg border border-border rounded p-2 space-y-2">
          <div className="text-xs font-bold text-secondary">Config B</div>
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted w-12">Tax</span>
            <input type="range" min={0} max={0.5} step={0.01} value={taxB} onChange={(e) => setTaxB(parseFloat(e.target.value))} className="flex-1 h-1 accent-secondary" />
            <span className="text-xs font-mono w-8">{taxB.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted w-12">CB</span>
            <button onClick={() => setCbB(!cbB)} className={`px-1.5 py-0.5 rounded text-xs ${cbB ? "bg-secondary text-bg" : "bg-btn"}`}>{cbB ? "ON" : "OFF"}</button>
          </div>
        </div>
      </div>

      <button
        onClick={handleCompare}
        disabled={running}
        className="w-full py-2 rounded font-bold text-sm bg-accent text-bg hover:opacity-90 disabled:opacity-50 transition-opacity"
      >
        {running ? "Running..." : "Compare"}
      </button>

      {comparison && results && (
        <div className="space-y-2">
          <CompareRow label="Toxicity" a={comparison.a.toxicity} b={comparison.b.toxicity} format={(v) => (v * 100).toFixed(0) + "%"} lower />
          <CompareRow label="Avg P" a={comparison.a.avgP} b={comparison.b.avgP} format={(v) => v.toFixed(3)} />
          <CompareRow label="Gini" a={comparison.a.gini} b={comparison.b.gini} format={(v) => v.toFixed(3)} lower />
          <CompareRow label="Welfare" a={comparison.a.welfare} b={comparison.b.welfare} format={(v) => v.toFixed(1)} />

          <div className="flex gap-2 pt-2">
            <button onClick={() => onSelect(results.a)} className="flex-1 py-1.5 rounded text-xs font-bold bg-btn hover:bg-btn-hover transition-colors">
              View A
            </button>
            <button onClick={() => onSelect(results.b)} className="flex-1 py-1.5 rounded text-xs font-bold bg-btn hover:bg-btn-hover transition-colors">
              View B
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function CompareRow({ label, a, b, format, lower }: { label: string; a: number; b: number; format: (v: number) => string; lower?: boolean }) {
  const aWins = lower ? a < b : a > b;
  const bWins = lower ? b < a : b > a;
  return (
    <div className="flex items-center text-xs">
      <span className="text-muted w-16 shrink-0">{label}</span>
      <span className={`font-mono w-16 text-right ${aWins ? "text-accent font-bold" : "text-muted"}`}>{format(a)}</span>
      <span className="text-muted mx-2">vs</span>
      <span className={`font-mono w-16 ${bWins ? "text-secondary font-bold" : "text-muted"}`}>{format(b)}</span>
    </div>
  );
}
