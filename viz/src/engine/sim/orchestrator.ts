/** Main simulation loop — composes all engine modules, outputs SimulationData. */

import type { SimulationData, InteractionEvent, AgentSnapshot } from "@/data/types";
import type { ScenarioConfig, SimAgentState } from "./types";
import type { StepResult } from "./metrics";
import { mulberry32 } from "./rng";
import { ProxyComputer } from "./proxy";
import { SoftPayoffEngine } from "./payoff";
import { AGENT_PROFILES } from "./agents";
import { generateObservables } from "./observable-gen";
import { applyTax, applyReputationDecay, checkCircuitBreaker, applyCircuitBreaker } from "./governance";
import { buildEpochSnapshot, buildAgentSnapshot } from "./metrics";

import { storeConfig } from "./config-store";

type ProgressCallback = (epoch: number, totalEpochs: number) => void;

/** Mid-run governance change */
export interface GovernanceIntervention {
  epoch: number;
  governance: import("./types").GovernanceConfig;
}

/** Create initial agent states from config */
function createAgents(config: ScenarioConfig, rng: () => number): SimAgentState[] {
  const agents: SimAgentState[] = [];
  let idx = 0;

  for (const group of config.agents) {
    const profile = AGENT_PROFILES[group.type];
    if (!profile) continue;

    for (let i = 0; i < group.count; i++) {
      const nameIdx = idx % profile.names.length;
      const suffix = idx >= profile.names.length ? `-${Math.floor(idx / profile.names.length) + 1}` : "";
      agents.push({
        id: `agent-${idx}`,
        name: `${profile.names[nameIdx]}${suffix}`,
        type: group.type,
        reputation: 0.5 + (rng() - 0.5) * 0.2,
        resources: 10 + rng() * 5,
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

/** Pick a random counterparty different from initiator */
function pickCounterparty(agents: SimAgentState[], initiatorIdx: number, rng: () => number): number {
  if (agents.length < 2) return initiatorIdx;
  let idx: number;
  do {
    idx = Math.floor(rng() * agents.length);
  } while (idx === initiatorIdx);
  return idx;
}

/** Run a single step: pick pair, generate observables, compute payoffs */
function runStep(
  agents: SimAgentState[],
  proxy: ProxyComputer,
  payoffEngine: SoftPayoffEngine,
  taxRate: number,
  rng: () => number,
  epoch: number,
  step: number,
  simId: string,
): { result: StepResult; event: InteractionEvent } {
  // Pick initiator (non-frozen)
  const activeAgents = agents.map((a, i) => ({ a, i })).filter(({ a }) => !a.isFrozen);
  if (activeAgents.length < 2) {
    // Not enough active agents — produce a rejected dummy interaction
    const dummyP = 0.5;
    return {
      result: { p: dummyP, accepted: false, initiatorPayoff: 0, counterpartyPayoff: 0 },
      event: {
        event_type: "interaction",
        timestamp: new Date().toISOString(),
        epoch,
        step,
        interaction_id: `${simId}-e${epoch}-s${step}`,
        initiator: agents[0]?.id ?? "none",
        counterparty: agents[1]?.id ?? "none",
        interaction_type: "collaboration",
        accepted: false,
        p: dummyP,
        v_hat: 0,
      },
    };
  }

  const initEntry = activeAgents[Math.floor(rng() * activeAgents.length)];
  const initiator = initEntry.a;
  const initiatorIdx = initEntry.i;

  const cpIdx = pickCounterparty(agents, initiatorIdx, rng);
  const counterparty = agents[cpIdx];

  // Generate observables
  const profile = AGENT_PROFILES[initiator.type];
  const obs = generateObservables(profile, initiator.reputation, rng);
  const { vHat, p } = proxy.computeLabels(obs);

  // Acceptance: counterparty decides based on initiator's reputation
  const cpProfile = AGENT_PROFILES[counterparty.type];
  const accepted = initiator.reputation >= cpProfile.acceptanceThreshold && !counterparty.isFrozen;

  // Tax
  const tau = accepted ? applyTax({ taxRate, reputationDecay: 1, circuitBreakerEnabled: false, circuitBreakerThreshold: 0 }) : 0;

  // Governance cost (small fixed cost)
  const cA = 0.01;
  const cB = 0.01;

  let initiatorPayoff = 0;
  let counterpartyPayoff = 0;

  if (accepted) {
    initiatorPayoff = payoffEngine.payoffInitiator(p, tau, cA, initiator.reputation);
    counterpartyPayoff = payoffEngine.payoffCounterparty(p, tau, cB, counterparty.reputation);

    initiator.totalPayoff += initiatorPayoff;
    counterparty.totalPayoff += counterpartyPayoff;
    initiator.resources = Math.max(0, initiator.resources + initiatorPayoff * 0.1);
    counterparty.resources = Math.max(0, counterparty.resources + counterpartyPayoff * 0.1);

    // Update reputation based on p
    const repDelta = (p - 0.5) * 0.05;
    initiator.reputation = Math.max(0, Math.min(1, initiator.reputation + repDelta));
    counterparty.reputation = Math.max(0, Math.min(1, counterparty.reputation + repDelta * 0.5));
  }

  // Track interaction counts
  initiator.interactionsInitiated++;
  counterparty.interactionsReceived++;
  initiator.sumPInitiated += p;
  counterparty.sumPReceived += p;

  const event: InteractionEvent = {
    event_type: "interaction",
    timestamp: new Date().toISOString(),
    epoch,
    step,
    interaction_id: `${simId}-e${epoch}-s${step}`,
    initiator: initiator.id,
    counterparty: counterparty.id,
    interaction_type: "collaboration",
    accepted,
    p,
    v_hat: vHat,
  };

  return { result: { p, accepted, initiatorPayoff, counterpartyPayoff }, event };
}

/** Run full simulation, return SimulationData.
 *  Optional `interventions` array applies governance changes at specified epochs. */
export function runSimulation(
  config: ScenarioConfig,
  onProgress?: ProgressCallback,
  interventions?: GovernanceIntervention[],
): SimulationData {
  storeConfig(config);

  const rng = mulberry32(config.seed);
  const simId = `sim-${Date.now()}`;
  const proxy = new ProxyComputer(undefined, config.sigmoidK);
  const payoffEngine = new SoftPayoffEngine(config.payoff);
  const agents = createAgents(config, rng);

  // Sort interventions by epoch for efficient lookup
  const sortedInterventions = interventions?.slice().sort((a, b) => a.epoch - b.epoch) ?? [];
  let interventionIdx = 0;
  let currentGovernance = { ...config.governance };

  const epochSnapshots: import("@/data/types").EpochSnapshot[] = [];
  const agentSnapshots: AgentSnapshot[] = [];
  const events: InteractionEvent[] = [];

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    // Apply any intervention at this epoch
    while (interventionIdx < sortedInterventions.length && sortedInterventions[interventionIdx].epoch <= epoch) {
      currentGovernance = { ...sortedInterventions[interventionIdx].governance };
      interventionIdx++;
    }

    const stepResults: StepResult[] = [];

    for (let step = 0; step < config.stepsPerEpoch; step++) {
      const { result, event } = runStep(
        agents, proxy, payoffEngine, currentGovernance.taxRate,
        rng, epoch, step, simId,
      );
      stepResults.push(result);
      events.push(event);
    }

    // Governance: reputation decay
    applyReputationDecay(agents, currentGovernance.reputationDecay);

    // Build epoch snapshot
    const epochSnap = buildEpochSnapshot(epoch, stepResults, agents, simId);
    epochSnapshots.push(epochSnap);

    // Circuit breaker check
    if (checkCircuitBreaker(currentGovernance, epochSnap.toxicity_rate)) {
      applyCircuitBreaker(agents);
    }

    // Build per-agent snapshots for this epoch
    for (const agent of agents) {
      agentSnapshots.push(buildAgentSnapshot(agent, epoch));
    }

    onProgress?.(epoch + 1, config.epochs);
  }

  return {
    simulation_id: simId,
    started_at: new Date().toISOString(),
    ended_at: new Date().toISOString(),
    n_epochs: config.epochs,
    steps_per_epoch: config.stepsPerEpoch,
    n_agents: agents.length,
    seed: config.seed,
    epoch_snapshots: epochSnapshots,
    agent_snapshots: agentSnapshots,
    events,
  };
}
