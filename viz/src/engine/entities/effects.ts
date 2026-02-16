import type { Particle, DigitalRainState, RainColumn, CodeTrailParticle, RecompileState } from "../types";
import { rgba } from "@/utils/color";
import { randomMatrixChar, matrixGreen, matrixColorForP, MATRIX_FONT, MATRIX_FONT_SMALL } from "./matrix-chars";
import { clamp } from "@/utils/math";

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

// ─── Digital Rain ──────────────────────────────────────────────────

/** Initialize digital rain columns based on canvas width */
export function initDigitalRain(width: number): DigitalRainState {
  const colWidth = 18;
  const colCount = Math.ceil(width / colWidth);
  const columns: RainColumn[] = [];

  for (let i = 0; i < colCount; i++) {
    const trailLen = 8 + Math.floor(Math.random() * 14);
    const chars: string[] = [];
    for (let j = 0; j < trailLen; j++) {
      chars.push(randomMatrixChar());
    }
    // X-position jitter (+/- 3px from grid) to break uniform spacing
    const xJitter = (Math.random() - 0.5) * 6;
    columns.push({
      x: i * colWidth + colWidth / 2 + xJitter,
      speed: 0.02 + Math.random() * 0.10, // Wider range: 0.02-0.12
      chars,
      headY: Math.random() * -500,
      charInterval: colWidth + (Math.random() - 0.5) * 4, // Variable per column
      brightness: 0.6 + Math.random() * 0.4,
      nextMutateTime: 50 + Math.random() * 150,
    });
  }
  return { columns, initialized: true };
}

/** Update rain state each frame */
export function updateDigitalRain(
  state: DigitalRainState,
  dt: number,
  height: number,
  threatLevel: number,
) {
  for (const col of state.columns) {
    // Speed increases with threat
    const speedMul = 1 + threatLevel * 0.6;
    col.headY += col.speed * dt * speedMul;

    // Wrap around when fully off screen
    const trailLen = col.chars.length * col.charInterval;
    if (col.headY - trailLen > height) {
      col.headY = Math.random() * -200 - 50;
      col.speed = 0.02 + Math.random() * 0.10; // Wider speed range on wrap
      // Occasionally change trail length when column wraps
      if (Math.random() < 0.3) {
        const newLen = 8 + Math.floor(Math.random() * 14);
        while (col.chars.length < newLen) col.chars.push(randomMatrixChar());
        if (col.chars.length > newLen) col.chars.length = newLen;
      }
    }

    // Mutate random chars periodically
    col.nextMutateTime -= dt;
    if (col.nextMutateTime <= 0) {
      const idx = Math.floor(Math.random() * col.chars.length);
      col.chars[idx] = randomMatrixChar();
      col.nextMutateTime = 50 + Math.random() * 150;
    }
  }
}

/** Render digital rain columns (screen-space, before camera transform) */
export function renderDigitalRain(
  ctx: CanvasRenderingContext2D,
  state: DigitalRainState,
  width: number,
  height: number,
  threatLevel: number,
) {
  ctx.save();
  ctx.font = MATRIX_FONT;
  ctx.textAlign = "center";

  const t = clamp(threatLevel, 0, 1);
  // Density: skip more columns at low threat for sparse feel
  const skipRate = Math.max(0, 0.4 - t * 0.4);

  for (let ci = 0; ci < state.columns.length; ci++) {
    // Sparse at calm, dense at high threat
    if (skipRate > 0 && ((ci * 7) % 10) / 10 < skipRate) continue;

    const col = state.columns[ci];
    const chars = col.chars;

    for (let j = 0; j < chars.length; j++) {
      const charY = col.headY - j * col.charInterval;
      if (charY < -20 || charY > height + 20) continue;

      const isHead = j === 0;
      // Quadratic fade: hold brightness longer, then drop off sharply
      const linearRatio = 1 - j / chars.length;
      const fadeRatio = linearRatio * linearRatio;

      if (isHead) {
        // Lead character: brightest (white-green)
        ctx.fillStyle = t > 0.5
          ? `rgba(255,${Math.round(180 - t * 100)},${Math.round(80 - t * 60)},${col.brightness * 0.95})`
          : `rgba(200,255,220,${col.brightness * 0.9})`;
      } else {
        // Trailing: green fading, red-shifted with threat
        const alpha = fadeRatio * col.brightness * 0.6;
        if (t > 0.3) {
          const redShift = (t - 0.3) / 0.7;
          const r = Math.round(redShift * 200);
          const g = Math.round(255 * (1 - redShift * 0.5));
          ctx.fillStyle = `rgba(${r},${g},65,${alpha})`;
        } else {
          ctx.fillStyle = matrixGreen(alpha);
        }
      }

      ctx.fillText(chars[j], col.x, charY);
    }
  }

  ctx.restore();
}

// ─── Code Trails ───────────────────────────────────────────────────

/** Render code trail particles (world-space, before depth-sorted entities) */
export function renderCodeTrails(
  ctx: CanvasRenderingContext2D,
  particles: CodeTrailParticle[],
) {
  if (particles.length === 0) return;
  ctx.save();
  ctx.textAlign = "center";

  for (const p of particles) {
    if (p.alpha <= 0) continue;
    ctx.font = `bold ${p.fontSize}px 'Courier New', monospace`;
    ctx.fillStyle = p.color.replace(/[\d.]+\)$/, `${p.alpha})`);
    ctx.fillText(p.char, p.x, p.y);
  }

  ctx.restore();
}

// ─── Recompile Flash ───────────────────────────────────────────────

/** Render epoch recompile scanline flash (screen-space) */
export function renderRecompileFlash(
  ctx: CanvasRenderingContext2D,
  state: RecompileState,
  width: number,
  height: number,
) {
  if (!state.active) return;

  const elapsed = Date.now() - state.startTime;
  if (elapsed > state.duration) return;

  const progress = elapsed / state.duration;

  // Scanline sweep: top → bottom over first 50% of duration
  if (progress < 0.5) {
    const scanProgress = progress / 0.5;
    const scanY = scanProgress * height;
    const bandHeight = 30;

    const grad = ctx.createLinearGradient(0, scanY - bandHeight, 0, scanY + bandHeight);
    grad.addColorStop(0, matrixGreen(0));
    grad.addColorStop(0.4, matrixGreen(0.3));
    grad.addColorStop(0.5, matrixGreen(0.6));
    grad.addColorStop(0.6, matrixGreen(0.3));
    grad.addColorStop(1, matrixGreen(0));
    ctx.fillStyle = grad;
    ctx.fillRect(0, scanY - bandHeight, width, bandHeight * 2);
  }

  // Full screen fade-out flash in second half
  if (progress > 0.3) {
    const fadeAlpha = Math.max(0, (1 - (progress - 0.3) / 0.7) * 0.08);
    ctx.fillStyle = matrixGreen(fadeAlpha);
    ctx.fillRect(0, 0, width, height);
  }
}

/** Render agent recompile character burst (world-space, above agent head) */
export function renderAgentRecompileBurst(
  ctx: CanvasRenderingContext2D,
  bx: number,
  by: number,
  scale: number,
  startTime: number,
  duration: number,
) {
  const elapsed = Date.now() - startTime;
  if (elapsed > duration || elapsed < 0) return;

  const progress = elapsed / duration;
  const alpha = 1 - progress;
  const charH = 56 * scale;
  const orbY = by - charH - 8 * scale;

  ctx.save();
  ctx.font = MATRIX_FONT_SMALL;
  ctx.textAlign = "center";

  // Show 3-5 rapidly cycling characters above head
  const charCount = 3 + Math.floor(Math.random() * 3);
  const timeBucket = Math.floor(elapsed / 40); // Change every 40ms
  for (let i = 0; i < charCount; i++) {
    const seed = timeBucket * 31 + i * 17;
    const charIdx = ((seed >>> 0) % 56);
    // Use matrixCharFromSeed logic inline for perf
    const ch = String.fromCharCode(0xff66 + (charIdx % 56));
    const offsetX = (i - (charCount - 1) / 2) * 8;
    const offsetY = -Math.sin(progress * Math.PI) * 6;
    ctx.fillStyle = matrixGreen(alpha * 0.8);
    ctx.fillText(ch, bx + offsetX, orbY + offsetY);
  }

  ctx.restore();
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
