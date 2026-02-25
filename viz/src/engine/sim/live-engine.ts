/**
 * LiveEngine — step-by-step simulation engine for interactive sandbox play.
 *
 * Unlike `runSimulation()` which runs all epochs at once, LiveEngine
 * executes one step at a time, allowing real-time interventions between steps.
 */

import type { SimulationData, InteractionEvent, AgentSnapshot, EpochSnapshot } from "@/data/types";
import type { ScenarioConfig, SimAgentState, GovernanceConfig, PayoffParams } from "./types";
import type { StepResult } from "./metrics";
import { mulberry32 } from "./rng";
import { ProxyComputer } from "./proxy";
import { SoftPayoffEngine } from "./payoff";
import { AGENT_PROFILES } from "./agents";
import { generateObservables } from "./observable-gen";
import { applyTax, applyReputationDecay, checkCircuitBreaker, applyCircuitBreaker } from "./governance";
import { buildEpochSnapshot, buildAgentSnapshot } from "./metrics";
import type { AgentType } from "@/data/types";
import type { ShockEvent } from "./shocks";
import { applyShock } from "./shocks";

/** Snapshot of engine state for save/load */
export interface LiveEngineSnapshot {
  config: ScenarioConfig;
  agents: SimAgentState[];
  epoch: number;
  step: number;
  governance: GovernanceConfig;
  epochStepResults: StepResult[];
  simId: string;
  nextAgentIdx: number;
}

export class LiveEngine {
  agents: SimAgentState[];
  epoch: number;
  step: number;
  config: ScenarioConfig;
  governance: GovernanceConfig;

  private rng: () => number;
  private proxy: ProxyComputer;
  private payoffEngine: SoftPayoffEngine;
  private simId: string;
  private nextAgentIdx: number;

  // Accumulate step results within current epoch
  private epochStepResults: StepResult[] = [];

  // History for visualization
  private epochSnapshots: EpochSnapshot[] = [];
  private agentSnapshots: AgentSnapshot[] = [];
  private events: InteractionEvent[] = [];

  constructor(config: ScenarioConfig) {
    this.config = { ...config };
    this.governance = { ...config.governance };
    this.rng = mulberry32(config.seed);
    this.simId = `live-${Date.now()}`;
    this.proxy = new ProxyComputer(undefined, config.sigmoidK);
    this.payoffEngine = new SoftPayoffEngine(config.payoff);
    this.epoch = 0;
    this.step = 0;
    this.nextAgentIdx = 0;

    this.agents = this.createAgents(config);
    this.nextAgentIdx = this.agents.length;
  }

  private createAgents(config: ScenarioConfig): SimAgentState[] {
    const agents: SimAgentState[] = [];
    let idx = 0;

    for (const group of config.agents) {
      const profile = AGENT_PROFILES[group.type];
      if (!profile) continue;

      for (let i = 0; i < group.count; i++) {
        const nameIdx = idx % profile.names.length;
        const suffix = idx >= profile.names.length
          ? `-${Math.floor(idx / profile.names.length) + 1}`
          : "";
        agents.push({
          id: `agent-${idx}`,
          name: `${profile.names[nameIdx]}${suffix}`,
          type: group.type,
          reputation: 0.5 + (this.rng() - 0.5) * 0.2,
          resources: 10 + this.rng() * 5,
          totalPayoff: 0,
          interactionsInitiated: 0,
          interactionsReceived: 0,
          sumPInitiated: 0,
          sumPReceived: 0,
          isFrozen: false,
          isQuarantined: false,
        });
        idx++;
      }
    }

    return agents;
  }

  /** Run one simulation step. Returns the step result. */
  tick(): { result: StepResult; event: InteractionEvent; epochCompleted: boolean } {
    const activeAgents = this.agents
      .map((a, i) => ({ a, i }))
      .filter(({ a }) => !a.isFrozen);

    let result: StepResult;
    let event: InteractionEvent;

    if (activeAgents.length < 2) {
      // Not enough active agents — produce dummy
      const dummyP = 0.5;
      result = { p: dummyP, accepted: false, initiatorPayoff: 0, counterpartyPayoff: 0 };
      event = {
        event_type: "interaction",
        timestamp: new Date().toISOString(),
        epoch: this.epoch,
        step: this.step,
        interaction_id: `${this.simId}-e${this.epoch}-s${this.step}`,
        initiator: this.agents[0]?.id ?? "none",
        counterparty: this.agents[1]?.id ?? "none",
        interaction_type: "collaboration",
        accepted: false,
        p: dummyP,
        v_hat: 0,
      };
    } else {
      // Pick initiator
      const initEntry = activeAgents[Math.floor(this.rng() * activeAgents.length)];
      const initiator = initEntry.a;
      const initiatorIdx = initEntry.i;

      // Pick counterparty
      let cpIdx: number;
      do {
        cpIdx = Math.floor(this.rng() * this.agents.length);
      } while (cpIdx === initiatorIdx);
      const counterparty = this.agents[cpIdx];

      // Generate observables and compute proxy
      const profile = AGENT_PROFILES[initiator.type];
      const obs = generateObservables(profile, initiator.reputation, this.rng);
      const { vHat, p } = this.proxy.computeLabels(obs);

      // Acceptance decision
      const cpProfile = AGENT_PROFILES[counterparty.type];
      const accepted = initiator.reputation >= cpProfile.acceptanceThreshold && !counterparty.isFrozen;

      // Tax
      const tau = accepted ? applyTax(this.governance) : 0;
      const cA = 0.01;
      const cB = 0.01;

      let initiatorPayoff = 0;
      let counterpartyPayoff = 0;

      if (accepted) {
        initiatorPayoff = this.payoffEngine.payoffInitiator(p, tau, cA, initiator.reputation);
        counterpartyPayoff = this.payoffEngine.payoffCounterparty(p, tau, cB, counterparty.reputation);

        initiator.totalPayoff += initiatorPayoff;
        counterparty.totalPayoff += counterpartyPayoff;
        initiator.resources = Math.max(0, initiator.resources + initiatorPayoff * 0.1);
        counterparty.resources = Math.max(0, counterparty.resources + counterpartyPayoff * 0.1);

        // Reputation update
        const repDelta = (p - 0.5) * 0.05;
        initiator.reputation = Math.max(0, Math.min(1, initiator.reputation + repDelta));
        counterparty.reputation = Math.max(0, Math.min(1, counterparty.reputation + repDelta * 0.5));
      }

      initiator.interactionsInitiated++;
      counterparty.interactionsReceived++;
      initiator.sumPInitiated += p;
      counterparty.sumPReceived += p;

      result = { p, accepted, initiatorPayoff, counterpartyPayoff };
      event = {
        event_type: "interaction",
        timestamp: new Date().toISOString(),
        epoch: this.epoch,
        step: this.step,
        interaction_id: `${this.simId}-e${this.epoch}-s${this.step}`,
        initiator: initiator.id,
        counterparty: counterparty.id,
        interaction_type: "collaboration",
        accepted,
        p,
        v_hat: vHat,
      };
    }

    this.epochStepResults.push(result);
    this.events.push(event);
    this.step++;

    // Check if epoch is complete
    let epochCompleted = false;
    if (this.step >= this.config.stepsPerEpoch) {
      this.completeEpoch();
      epochCompleted = true;
    }

    return { result, event, epochCompleted };
  }

  /** Force-complete the current epoch (used when step count reached) */
  private completeEpoch(): void {
    // Governance: reputation decay
    applyReputationDecay(this.agents, this.governance.reputationDecay);

    // Build epoch snapshot
    const epochSnap = buildEpochSnapshot(this.epoch, this.epochStepResults, this.agents, this.simId);
    this.epochSnapshots.push(epochSnap);

    // Circuit breaker check
    if (checkCircuitBreaker(this.governance, epochSnap.toxicity_rate)) {
      applyCircuitBreaker(this.agents);
    }

    // Build per-agent snapshots
    for (const agent of this.agents) {
      this.agentSnapshots.push(buildAgentSnapshot(agent, this.epoch));
    }

    // Advance to next epoch
    this.epoch++;
    this.step = 0;
    this.epochStepResults = [];
  }

  /** Get current epoch snapshot (even if epoch is in-progress) */
  getCurrentEpochSnapshot(): EpochSnapshot {
    if (this.epochStepResults.length === 0 && this.epochSnapshots.length > 0) {
      return this.epochSnapshots[this.epochSnapshots.length - 1];
    }
    return buildEpochSnapshot(
      this.epoch,
      this.epochStepResults.length > 0 ? this.epochStepResults : [{ p: 0.5, accepted: false, initiatorPayoff: 0, counterpartyPayoff: 0 }],
      this.agents,
      this.simId,
    );
  }

  /** Get completed epoch snapshots */
  getEpochSnapshots(): EpochSnapshot[] {
    return this.epochSnapshots;
  }

  /** Get all agent snapshots */
  getAgentSnapshots(): AgentSnapshot[] {
    // Include current-epoch snapshots
    const current = this.agents.map((a) => buildAgentSnapshot(a, this.epoch));
    return [...this.agentSnapshots, ...current];
  }

  /** Export full SimulationData for visualization compatibility */
  toSimulationData(): SimulationData {
    return {
      simulation_id: this.simId,
      started_at: new Date().toISOString(),
      ended_at: null,
      n_epochs: this.epoch + 1,
      steps_per_epoch: this.config.stepsPerEpoch,
      n_agents: this.agents.length,
      seed: this.config.seed,
      epoch_snapshots: [...this.epochSnapshots, this.getCurrentEpochSnapshot()],
      agent_snapshots: this.getAgentSnapshots(),
      events: this.events,
    };
  }

  // ─── Live Interventions ──────────────────────────────────────────

  /** Spawn a new agent of a given type */
  spawnAgent(type: AgentType): SimAgentState {
    const profile = AGENT_PROFILES[type];
    const idx = this.nextAgentIdx++;
    const nameIdx = idx % profile.names.length;
    const suffix = `-${Math.floor(idx / profile.names.length) + 1}`;

    const agent: SimAgentState = {
      id: `agent-${idx}`,
      name: `${profile.names[nameIdx]}${suffix}`,
      type,
      reputation: 0.5 + (this.rng() - 0.5) * 0.2,
      resources: 10 + this.rng() * 5,
      totalPayoff: 0,
      interactionsInitiated: 0,
      interactionsReceived: 0,
      sumPInitiated: 0,
      sumPReceived: 0,
      isFrozen: false,
      isQuarantined: false,
    };

    this.agents.push(agent);
    return agent;
  }

  /** Remove an agent by ID */
  removeAgent(id: string): boolean {
    const idx = this.agents.findIndex((a) => a.id === id);
    if (idx === -1) return false;
    this.agents.splice(idx, 1);
    return true;
  }

  /** Freeze/unfreeze an agent */
  toggleFreeze(id: string): boolean {
    const agent = this.agents.find((a) => a.id === id);
    if (!agent) return false;
    agent.isFrozen = !agent.isFrozen;
    return true;
  }

  /** Quarantine/unquarantine an agent */
  toggleQuarantine(id: string): boolean {
    const agent = this.agents.find((a) => a.id === id);
    if (!agent) return false;
    agent.isQuarantined = !agent.isQuarantined;
    return true;
  }

  /** Update governance config live */
  updateGovernance(partial: Partial<GovernanceConfig>): void {
    this.governance = { ...this.governance, ...partial };
  }

  /** Update payoff parameters live */
  updatePayoff(partial: Partial<PayoffParams>): void {
    const newParams = { ...this.config.payoff, ...partial };
    this.config = { ...this.config, payoff: newParams };
    this.payoffEngine = new SoftPayoffEngine(newParams);
  }

  /** Inject a shock event */
  injectShock(shock: ShockEvent): void {
    applyShock(this, shock);
  }

  // ─── Save / Load ──────────────────────────────────────────────

  /** Serialize engine state */
  serialize(): string {
    const snapshot: LiveEngineSnapshot = {
      config: this.config,
      agents: this.agents.map((a) => ({ ...a })),
      epoch: this.epoch,
      step: this.step,
      governance: { ...this.governance },
      epochStepResults: [...this.epochStepResults],
      simId: this.simId,
      nextAgentIdx: this.nextAgentIdx,
    };
    return JSON.stringify(snapshot);
  }

  /** Restore engine from serialized state */
  static deserialize(json: string): LiveEngine {
    const snapshot: LiveEngineSnapshot = JSON.parse(json);
    const engine = new LiveEngine(snapshot.config);
    engine.agents = snapshot.agents;
    engine.epoch = snapshot.epoch;
    engine.step = snapshot.step;
    engine.governance = snapshot.governance;
    engine.epochStepResults = snapshot.epochStepResults;
    engine.simId = snapshot.simId;
    engine.nextAgentIdx = snapshot.nextAgentIdx;
    return engine;
  }
}
