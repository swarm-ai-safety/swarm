/** SoftPayoffEngine — port of swarm/core/payoff.py */

import type { PayoffParams } from "./types";

const DEFAULT_PARAMS: PayoffParams = {
  s_plus: 2.0,
  s_minus: 1.0,
  h: 2.0,
  theta: 0.5,
  rho_a: 0.0,
  rho_b: 0.0,
  w_rep: 1.0,
};

export class SoftPayoffEngine {
  private cfg: PayoffParams;

  constructor(params: Partial<PayoffParams> = {}) {
    this.cfg = { ...DEFAULT_PARAMS, ...params };
  }

  expectedSurplus(p: number): number {
    return p * this.cfg.s_plus - (1 - p) * this.cfg.s_minus;
  }

  expectedHarm(p: number): number {
    return (1 - p) * this.cfg.h;
  }

  /** Initiator payoff: π_a = θ·S_soft - τ - c_a - ρ_a·E_soft + w_rep·r_a */
  payoffInitiator(p: number, tau: number, cA: number, rA: number): number {
    const S = this.expectedSurplus(p);
    const E = this.expectedHarm(p);
    return this.cfg.theta * S - tau - cA - this.cfg.rho_a * E + this.cfg.w_rep * rA;
  }

  /** Counterparty payoff: π_b = (1-θ)·S_soft + τ - c_b - ρ_b·E_soft + w_rep·r_b */
  payoffCounterparty(p: number, tau: number, cB: number, rB: number): number {
    const S = this.expectedSurplus(p);
    const E = this.expectedHarm(p);
    return (1 - this.cfg.theta) * S + tau - cB - this.cfg.rho_b * E + this.cfg.w_rep * rB;
  }
}
