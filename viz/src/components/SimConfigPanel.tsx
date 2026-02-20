"use client";

import React, { useState, useCallback } from "react";
import type { ScenarioConfig } from "@/engine/sim/types";
import { DEFAULT_CONFIG } from "@/engine/sim/types";
import { useSimWorker } from "@/state/use-sim-worker";
import type { SimulationData } from "@/data/types";

interface Props {
  onComplete: (data: SimulationData) => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted w-28 shrink-0">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 h-1 accent-accent"
      />
      <span className="text-xs font-mono w-12 text-right">{format ? format(value) : value}</span>
    </div>
  );
}

function Section({
  title,
  collapsible,
  children,
}: {
  title: string;
  collapsible?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(!collapsible);
  return (
    <div className="border-t border-border pt-3 mt-3 first:border-0 first:pt-0 first:mt-0">
      <button
        onClick={collapsible ? () => setOpen(!open) : undefined}
        className={`text-xs font-bold uppercase tracking-wider mb-2 flex items-center gap-1 ${
          collapsible ? "cursor-pointer hover:text-accent" : "cursor-default"
        }`}
      >
        {collapsible && <span className="text-muted">{open ? "\u25BC" : "\u25B6"}</span>}
        {title}
      </button>
      {open && <div className="space-y-2">{children}</div>}
    </div>
  );
}

export function SimConfigPanel({ onComplete }: Props) {
  const [config, setConfig] = useState<ScenarioConfig>({ ...DEFAULT_CONFIG });
  const { status, progress, error, run } = useSimWorker();

  const updateAgentCount = useCallback((typeIdx: number, count: number) => {
    setConfig((prev) => {
      const agents = [...prev.agents];
      agents[typeIdx] = { ...agents[typeIdx], count };
      return { ...prev, agents };
    });
  }, []);

  const updateGov = useCallback(<K extends keyof ScenarioConfig["governance"]>(key: K, val: ScenarioConfig["governance"][K]) => {
    setConfig((prev) => ({ ...prev, governance: { ...prev.governance, [key]: val } }));
  }, []);

  const updatePayoff = useCallback(<K extends keyof ScenarioConfig["payoff"]>(key: K, val: number) => {
    setConfig((prev) => ({ ...prev, payoff: { ...prev.payoff, [key]: val } }));
  }, []);

  const handleRun = useCallback(async () => {
    const totalAgents = config.agents.reduce((s, a) => s + a.count, 0);
    if (totalAgents < 2) {
      alert("Need at least 2 agents to run a simulation.");
      return;
    }
    const data = await run(config);
    if (data) onComplete(data);
  }, [config, run, onComplete]);

  const isRunning = status === "running";

  return (
    <div className="space-y-1">
      {/* Agents */}
      <Section title="Agents">
        {config.agents.map((ag, i) => (
          <Slider
            key={ag.type}
            label={ag.type.charAt(0).toUpperCase() + ag.type.slice(1)}
            value={ag.count}
            min={0}
            max={10}
            step={1}
            onChange={(v) => updateAgentCount(i, v)}
          />
        ))}
        <p className="text-xs text-muted">
          Total: {config.agents.reduce((s, a) => s + a.count, 0)} agents
        </p>
      </Section>

      {/* Governance */}
      <Section title="Governance">
        <Slider label="Tax rate" value={config.governance.taxRate} min={0} max={0.5} step={0.01} onChange={(v) => updateGov("taxRate", v)} format={(v) => v.toFixed(2)} />
        <Slider label="Rep. decay" value={config.governance.reputationDecay} min={0.8} max={1.0} step={0.01} onChange={(v) => updateGov("reputationDecay", v)} format={(v) => v.toFixed(2)} />
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted w-28 shrink-0">Circuit breaker</span>
          <button
            onClick={() => updateGov("circuitBreakerEnabled", !config.governance.circuitBreakerEnabled)}
            className={`px-2 py-0.5 rounded text-xs ${config.governance.circuitBreakerEnabled ? "bg-accent text-bg" : "bg-btn"}`}
          >
            {config.governance.circuitBreakerEnabled ? "ON" : "OFF"}
          </button>
          {config.governance.circuitBreakerEnabled && (
            <span className="text-xs text-muted">@ {config.governance.circuitBreakerThreshold}</span>
          )}
        </div>
      </Section>

      {/* Payoff (collapsible) */}
      <Section title="Payoff Parameters" collapsible>
        <Slider label="s_plus" value={config.payoff.s_plus} min={0} max={5} step={0.1} onChange={(v) => updatePayoff("s_plus", v)} format={(v) => v.toFixed(1)} />
        <Slider label="s_minus" value={config.payoff.s_minus} min={0} max={5} step={0.1} onChange={(v) => updatePayoff("s_minus", v)} format={(v) => v.toFixed(1)} />
        <Slider label="h (harm)" value={config.payoff.h} min={0} max={5} step={0.1} onChange={(v) => updatePayoff("h", v)} format={(v) => v.toFixed(1)} />
        <Slider label="theta" value={config.payoff.theta} min={0} max={1} step={0.05} onChange={(v) => updatePayoff("theta", v)} format={(v) => v.toFixed(2)} />
        <Slider label="rho_a" value={config.payoff.rho_a} min={0} max={1} step={0.05} onChange={(v) => updatePayoff("rho_a", v)} format={(v) => v.toFixed(2)} />
        <Slider label="rho_b" value={config.payoff.rho_b} min={0} max={1} step={0.05} onChange={(v) => updatePayoff("rho_b", v)} format={(v) => v.toFixed(2)} />
      </Section>

      {/* Simulation (collapsible) */}
      <Section title="Simulation" collapsible>
        <Slider label="Epochs" value={config.epochs} min={5} max={100} step={5} onChange={(v) => setConfig((p) => ({ ...p, epochs: v }))} />
        <Slider label="Steps/epoch" value={config.stepsPerEpoch} min={5} max={50} step={5} onChange={(v) => setConfig((p) => ({ ...p, stepsPerEpoch: v }))} />
        <Slider label="Seed" value={config.seed} min={1} max={9999} step={1} onChange={(v) => setConfig((p) => ({ ...p, seed: v }))} />
      </Section>

      {/* Run button + progress */}
      <div className="pt-3 mt-3 border-t border-border">
        <button
          onClick={handleRun}
          disabled={isRunning}
          className="w-full py-2 rounded font-bold text-sm bg-accent text-bg hover:opacity-90 disabled:opacity-50 transition-opacity"
        >
          {isRunning ? "Running..." : "Run Simulation"}
        </button>
        {isRunning && (
          <div className="mt-2 h-1.5 bg-btn rounded overflow-hidden">
            <div
              className="h-full bg-accent transition-all duration-200"
              style={{ width: `${Math.round(progress * 100)}%` }}
            />
          </div>
        )}
        {error && <p className="text-xs text-red-400 mt-2">{error}</p>}
      </div>
    </div>
  );
}
