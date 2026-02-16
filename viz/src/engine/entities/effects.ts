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

/** Interpolate between two hex colors */
function lerpColor(a: string, b: string, t: number): string {
  const parseHex = (h: string) => {
    const c = h.replace("#", "");
    return [
      parseInt(c.substring(0, 2), 16),
      parseInt(c.substring(2, 4), 16),
      parseInt(c.substring(4, 6), 16),
    ];
  };
  const [ar, ag, ab] = parseHex(a);
  const [br, bg, bb] = parseHex(b);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bv = Math.round(ab + (bb - ab) * t);
  return `rgb(${r},${g},${bv})`;
}

/** Render sky gradient based on threat level (0=calm blue, 1=dark red storm) */
export function renderSky(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  threatLevel: number,
) {
  const grad = ctx.createLinearGradient(0, 0, 0, height * 0.7);
  const t = Math.min(1, Math.max(0, threatLevel));

  // Smooth color transitions
  const topColor = lerpColor("#0D1117", "#3D1020", t);
  const botColor = lerpColor("#0D1117", "#1A0A10", t * 0.7);

  grad.addColorStop(0, topColor);
  grad.addColorStop(1, botColor);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  // Storm flicker at high threat
  if (t > 0.6) {
    const flicker = Math.sin(Date.now() / 200) * 0.5 + 0.5;
    const flickerAlpha = (t - 0.6) * 0.15 * flicker;
    ctx.fillStyle = rgba("#EB5757", flickerAlpha);
    ctx.fillRect(0, 0, width, height * 0.3);
  }
}

/** Render ground haze for toxicity */
export function renderHaze(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  toxicity: number,
) {
  if (toxicity < 0.05) return;
  const intensity = Math.min(toxicity, 1);
  const hazeColor = toxicity > 0.5 ? "#EB5757" : "#6FCF97";
  const now = Date.now() / 1000;

  // Main haze gradient
  const grad = ctx.createLinearGradient(0, height * 0.55, 0, height);
  grad.addColorStop(0, rgba(hazeColor, 0));
  grad.addColorStop(0.5, rgba(hazeColor, intensity * 0.08));
  grad.addColorStop(1, rgba(hazeColor, intensity * 0.18));
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  // Drifting wave layers
  for (let layer = 0; layer < 2; layer++) {
    const waveY = height * (0.7 + layer * 0.12);
    const waveAlpha = intensity * (0.06 - layer * 0.02);
    ctx.beginPath();
    ctx.moveTo(0, height);
    for (let x = 0; x <= width; x += 8) {
      const y = waveY + Math.sin(x * 0.01 + now * (0.5 + layer * 0.3)) * 8;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(width, height);
    ctx.closePath();
    ctx.fillStyle = rgba(hazeColor, waveAlpha);
    ctx.fill();
  }
}

/** Render a pulsing threat zone around an agent position */
export function renderThreatZone(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  threatLevel: number,
  agentType: string,
) {
  if (threatLevel < 0.2) return;

  const now = Date.now() / 1000;
  const pulse = 0.7 + Math.sin(now * 2) * 0.3;
  const radius = 30 + threatLevel * 20;
  const color = agentType === "adversarial" ? "#EB5757" : "#BB6BD9";

  const grad = ctx.createRadialGradient(x, y, 0, x, y, radius * pulse);
  grad.addColorStop(0, rgba(color, threatLevel * 0.15));
  grad.addColorStop(0.6, rgba(color, threatLevel * 0.08));
  grad.addColorStop(1, rgba(color, 0));
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(x, y, radius * pulse, 0, Math.PI * 2);
  ctx.fill();
}

/** Create snowflake particles for frozen agents */
export function createSnowflakes(x: number, y: number, count: number): Particle[] {
  const particles: Particle[] = [];
  for (let i = 0; i < count; i++) {
    const wind = Math.sin(Date.now() / 2000) * 0.2;
    particles.push({
      x: x + (Math.random() - 0.5) * 30,
      y: y - Math.random() * 40,
      vx: (Math.random() - 0.5) * 0.3 + wind,
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

/** Create glow particles for high-payoff agents */
export function createGlowParticles(x: number, y: number, payoff: number, color: string): Particle[] {
  if (payoff < 5) return [];
  const count = Math.min(4, Math.floor(payoff / 10));
  const particles: Particle[] = [];
  for (let i = 0; i < count; i++) {
    particles.push({
      x: x + (Math.random() - 0.5) * 20,
      y: y - Math.random() * 30,
      vx: (Math.random() - 0.5) * 0.15,
      vy: -Math.random() * 0.3 - 0.1,
      life: 1200 + Math.random() * 600,
      maxLife: 1800,
      size: 1.5 + Math.random(),
      color,
      alpha: 0.4,
    });
  }
  return particles;
}
