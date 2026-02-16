import type { Particle } from "../types";
import { rgba } from "@/utils/color";

export function renderParticles(ctx: CanvasRenderingContext2D, particles: Particle[]) {
  for (const p of particles) {
    const lifeRatio = p.life / p.maxLife;
    ctx.fillStyle = rgba(p.color, p.alpha * lifeRatio);
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size * lifeRatio, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** Render sky gradient based on threat level (0=calm blue, 1=dark red storm) */
export function renderSky(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  threatLevel: number,
) {
  const grad = ctx.createLinearGradient(0, 0, 0, height * 0.6);
  if (threatLevel < 0.3) {
    grad.addColorStop(0, "#0D1117");
    grad.addColorStop(1, "#0D1117");
  } else if (threatLevel < 0.6) {
    const t = (threatLevel - 0.3) / 0.3;
    grad.addColorStop(0, rgba("#1A1520", 1));
    grad.addColorStop(1, rgba("#0D1117", 1 - t * 0.3));
  } else {
    const t = (threatLevel - 0.6) / 0.4;
    grad.addColorStop(0, rgba("#2D1020", 0.5 + t * 0.5));
    grad.addColorStop(1, rgba("#0D1117", 0.7));
  }
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);
}

/** Render ground haze for toxicity */
export function renderHaze(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  toxicity: number,
) {
  if (toxicity < 0.1) return;
  const intensity = Math.min(toxicity, 1);
  const hazeColor = toxicity > 0.5 ? "#EB5757" : "#6FCF97";
  const grad = ctx.createLinearGradient(0, height * 0.6, 0, height);
  grad.addColorStop(0, rgba(hazeColor, 0));
  grad.addColorStop(1, rgba(hazeColor, intensity * 0.15));
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);
}

/** Create snowflake particles for frozen agents */
export function createSnowflakes(x: number, y: number, count: number): Particle[] {
  const particles: Particle[] = [];
  for (let i = 0; i < count; i++) {
    particles.push({
      x: x + (Math.random() - 0.5) * 30,
      y: y - Math.random() * 40,
      vx: (Math.random() - 0.5) * 0.3,
      vy: Math.random() * 0.5 + 0.2,
      life: 1000 + Math.random() * 500,
      maxLife: 1500,
      size: 2 + Math.random() * 2,
      color: "#A8CFF5",
      alpha: 0.7,
    });
  }
  return particles;
}

/** Create hazard particles for quarantined agents */
export function createHazardParticles(x: number, y: number, count: number): Particle[] {
  const particles: Particle[] = [];
  for (let i = 0; i < count; i++) {
    const angle = Math.random() * Math.PI * 2;
    const dist = 20 + Math.random() * 15;
    particles.push({
      x: x + Math.cos(angle) * dist,
      y: y + Math.sin(angle) * dist * 0.5,
      vx: Math.cos(angle) * 0.2,
      vy: -Math.random() * 0.5 - 0.3,
      life: 800 + Math.random() * 400,
      maxLife: 1200,
      size: 1.5 + Math.random(),
      color: "#EB5757",
      alpha: 0.5,
    });
  }
  return particles;
}
