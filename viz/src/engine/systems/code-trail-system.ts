/** Code Dissolve Walk Trails — falling green characters behind walking agents */

import type { CodeTrailParticle } from "../types";
import { randomMatrixChar, matrixColorForP } from "../entities/matrix-chars";

export class CodeTrailSystem {
  particles: CodeTrailParticle[] = [];

  /** Spawn a burst of code trail chars at a position */
  spawnBurst(x: number, y: number, avgP: number, count: number = 3) {
    for (let i = 0; i < count; i++) {
      const lifespan = 300 + Math.random() * 250; // Variable: 300-550ms
      this.particles.push({
        x: x + (Math.random() - 0.5) * 12,
        y: y + (Math.random() - 0.5) * 6,
        char: randomMatrixChar(),
        color: matrixColorForP(avgP, 1),
        alpha: 0.7 + Math.random() * 0.3,
        life: lifespan,
        maxLife: lifespan,
        vy: 0.015 + Math.random() * 0.01,
        vx: (Math.random() - 0.5) * 0.006, // Slight horizontal drift
        gravity: 0.00003 + Math.random() * 0.00002, // Downward acceleration
        fontSize: 8 + Math.random() * 4,
      });
    }
  }

  /** Update all trail particles */
  update(dt: number) {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      // Gravity acceleration — trails speed up as they fall
      p.vy += p.gravity * dt;
      p.y += p.vy * dt;
      p.x += p.vx * dt;
      p.life -= dt;
      // Quadratic alpha fade: hold brightness longer, then drop off
      const lifeRatio = Math.max(0, p.life / p.maxLife);
      p.alpha = lifeRatio * lifeRatio * 0.8;
      if (p.life <= 0) {
        this.particles.splice(i, 1);
      }
    }
  }

  clear() {
    this.particles = [];
  }
}
