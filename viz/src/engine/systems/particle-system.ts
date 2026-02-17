import type { Particle } from "../types";

export class ParticleSystem {
  particles: Particle[] = [];

  add(particles: Particle[]) {
    this.particles.push(...particles);
  }

  update(dt: number) {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.life -= dt;
      if (p.life <= 0) {
        this.particles.splice(i, 1);
      }
    }
  }

  clear() {
    this.particles = [];
  }
}
