/** Hook for reading/writing shareable URL state.
 *
 *  Reads ?preset=...&seed=...&autorun=1 from the URL on mount.
 *  Provides encodeShareUrl() to build a shareable link from a ScenarioConfig.
 */

"use client";

import { useMemo, useCallback } from "react";
import type { ScenarioConfig } from "@/engine/sim/types";
import { DEFAULT_CONFIG } from "@/engine/sim/types";
import { findPresetBySlug, DEMO_PRESETS } from "@/data/presets";

export interface UrlState {
  /** Matched preset slug from ?preset= */
  preset: string | null;
  /** Seed override from ?seed= */
  seed: number | null;
  /** Whether to auto-run on load */
  autorun: boolean;
}

function getSearchParams(): URLSearchParams {
  if (typeof window === "undefined") return new URLSearchParams();
  return new URLSearchParams(window.location.search);
}

/** Read URL state on mount (runs once). */
export function useUrlState(): UrlState {
  return useMemo(() => {
    const params = getSearchParams();
    const presetSlug = params.get("preset");
    const seedStr = params.get("seed");
    const autorun = params.get("autorun") === "1" || params.get("autorun") === "true";

    return {
      preset: presetSlug && findPresetBySlug(presetSlug) ? presetSlug : null,
      seed: seedStr ? parseInt(seedStr, 10) || null : null,
      autorun,
    };
  }, []);
}

/** Build a shareable URL from a ScenarioConfig. */
export function useShareUrl() {
  return useCallback((config: ScenarioConfig): string => {
    if (typeof window === "undefined") return "";

    const params = new URLSearchParams();

    // Check if config matches a known preset
    const matchedPreset = DEMO_PRESETS.find((p) => {
      const pc = p.config;
      return (
        JSON.stringify(pc.agents) === JSON.stringify(config.agents) &&
        (!pc.governance || JSON.stringify({ ...DEFAULT_CONFIG.governance, ...pc.governance }) ===
          JSON.stringify(config.governance)) &&
        (!pc.payoff || JSON.stringify({ ...DEFAULT_CONFIG.payoff, ...pc.payoff }) ===
          JSON.stringify(config.payoff))
      );
    });

    if (matchedPreset) {
      params.set("preset", matchedPreset.slug);
    } else {
      // Encode agent counts compactly: h3,o2,d1,a1
      const agentStr = config.agents
        .filter((a) => a.count > 0)
        .map((a) => `${a.type[0]}${a.count}`)
        .join(",");
      if (agentStr) params.set("agents", agentStr);

      // Encode governance knobs (only non-default values)
      const gov = config.governance;
      const dg = DEFAULT_CONFIG.governance;
      if (gov.taxRate !== dg.taxRate) params.set("tax", gov.taxRate.toString());
      if (gov.reputationDecay !== dg.reputationDecay) params.set("decay", gov.reputationDecay.toString());
      if (gov.circuitBreakerEnabled !== dg.circuitBreakerEnabled) params.set("cb", gov.circuitBreakerEnabled ? "1" : "0");
      if (gov.circuitBreakerThreshold !== dg.circuitBreakerThreshold) params.set("cbt", gov.circuitBreakerThreshold.toString());

      // Encode payoff params (only non-default)
      const pay = config.payoff;
      const dp = DEFAULT_CONFIG.payoff;
      if (pay.rho_a !== dp.rho_a) params.set("rho_a", pay.rho_a.toString());
      if (pay.rho_b !== dp.rho_b) params.set("rho_b", pay.rho_b.toString());
      if (pay.h !== dp.h) params.set("h", pay.h.toString());
    }

    // Always encode seed and epochs
    params.set("seed", config.seed.toString());
    if (config.epochs !== DEFAULT_CONFIG.epochs) params.set("epochs", config.epochs.toString());
    if (config.stepsPerEpoch !== DEFAULT_CONFIG.stepsPerEpoch) params.set("steps", config.stepsPerEpoch.toString());

    const base = window.location.origin + window.location.pathname;
    return `${base}?${params.toString()}`;
  }, []);
}

/** Build a ScenarioConfig from URL state. */
export function configFromUrlState(state: UrlState): ScenarioConfig | null {
  if (!state.preset && !state.seed) return null;

  const preset = state.preset ? findPresetBySlug(state.preset) : null;
  if (preset) {
    const config: ScenarioConfig = {
      ...DEFAULT_CONFIG,
      ...preset.config,
      governance: { ...DEFAULT_CONFIG.governance, ...preset.config.governance },
      payoff: { ...DEFAULT_CONFIG.payoff, ...preset.config.payoff },
    };
    if (state.seed) config.seed = state.seed;
    return config;
  }

  // No preset match but has seed — use default config with seed override
  if (state.seed) {
    return { ...DEFAULT_CONFIG, seed: state.seed };
  }

  return null;
}
