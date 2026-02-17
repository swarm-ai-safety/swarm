import type { InteractionArc, AgentVisual, RenderEntity } from "../types";
import { gridToScreen } from "../isometric";
import { CHARACTER, ARC_STREAM } from "../constants";
import { rgba, pToHealthColor } from "@/utils/color";
import { easeOut, easeInOut, hashString, lerp } from "@/utils/math";
import { matrixCharFromSeed, errorCharFromSeed, matrixColorForP, MATRIX_FONT_SMALL } from "./matrix-chars";

/** Evaluate a quadratic bezier at parameter t */
function bezierPoint(
  x0: number, y0: number,
  cx: number, cy: number,
  x1: number, y1: number,
  t: number,
): { x: number; y: number } {
  const mt = 1 - t;
  return {
    x: mt * mt * x0 + 2 * mt * t * cx + t * t * x1,
    y: mt * mt * y0 + 2 * mt * t * cy + t * t * y1,
  };
}

/** Create render entity for an interaction arc */
export function createArcEntity(
  arc: InteractionArc,
  agents: Map<string, AgentVisual>,
  highlightedAgentId: string | null,
): RenderEntity | null {
  const from = agents.get(arc.fromId);
  const to = agents.get(arc.toId);
  if (!from || !to) return null;

  const highlight =
    highlightedAgentId === arc.fromId || highlightedAgentId === arc.toId;

  return {
    depth: Math.max(from.gridX + from.gridY, to.gridX + to.gridY) + 0.5,
    render: (ctx) => renderArc(ctx, arc, from, to, highlight),
  };
}

/** Get visual style modifiers based on interaction type */
function getArcStyle(type?: InteractionArc["interactionType"]): { dash: number[]; widthBonus: number; double: boolean } {
  switch (type) {
    case "communication": return { dash: [2, 4], widthBonus: 0, double: false };
    case "task":          return { dash: [], widthBonus: 0.5, double: false };
    case "governance":    return { dash: [], widthBonus: 0, double: true };
    default:              return { dash: [], widthBonus: 0, double: false };
  }
}

function renderArc(
  ctx: CanvasRenderingContext2D,
  arc: InteractionArc,
  from: AgentVisual,
  to: AgentVisual,
  highlight: boolean,
) {
  const fromGridScreen = gridToScreen(from.gridX, from.gridY);
  const toScreen = gridToScreen(to.gridX, to.gridY);
  const fromScreen = {
    x: fromGridScreen.x + from.walkOffsetX,
    y: fromGridScreen.y + from.walkOffsetY,
  };
  const fromY = fromScreen.y - from.scale * CHARACTER.baseHeight * 0.6;
  const toY = toScreen.y - to.scale * CHARACTER.baseHeight * 0.6;

  const color = pToHealthColor(arc.p);
  const progress = easeOut(arc.progress);

  // Control point for bezier (arc upward)
  const midX = (fromScreen.x + toScreen.x) / 2;
  const midY = Math.min(fromY, toY) - 40 - Math.abs(fromScreen.x - toScreen.x) * 0.15;

  ctx.save();

  const arcStyle = getArcStyle(arc.interactionType);

  if (arc.accepted) {
    // Ghost line showing full path
    ctx.strokeStyle = rgba(color, highlight ? 0.15 : 0.08);
    ctx.lineWidth = 1 + arcStyle.widthBonus;
    if (arcStyle.dash.length > 0) ctx.setLineDash(arcStyle.dash);
    ctx.beginPath();
    ctx.moveTo(fromScreen.x, fromY);
    ctx.quadraticCurveTo(midX, midY, toScreen.x, toY);
    ctx.stroke();
    if (arcStyle.double) {
      ctx.beginPath();
      ctx.moveTo(fromScreen.x, fromY - 2);
      ctx.quadraticCurveTo(midX, midY - 2, toScreen.x, toY - 2);
      ctx.stroke();
    }
    if (arcStyle.dash.length > 0) ctx.setLineDash([]);

    // Trailing path up to current progress
    if (progress > 0.01) {
      ctx.strokeStyle = rgba(color, highlight ? 0.5 : 0.25);
      ctx.lineWidth = (highlight ? 2 : 1) + arcStyle.widthBonus;
      if (arcStyle.dash.length > 0) ctx.setLineDash(arcStyle.dash);
      ctx.beginPath();
      ctx.moveTo(fromScreen.x, fromY);
      const steps = Math.max(8, Math.floor(progress * 30));
      for (let i = 1; i <= steps; i++) {
        const t = (i / steps) * progress;
        const pt = bezierPoint(fromScreen.x, fromY, midX, midY, toScreen.x, toY, t);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();
      if (arcStyle.double) {
        ctx.beginPath();
        ctx.moveTo(fromScreen.x, fromY - 2);
        const steps2 = Math.max(8, Math.floor(progress * 30));
        for (let i = 1; i <= steps2; i++) {
          const t = (i / steps2) * progress;
          const pt = bezierPoint(fromScreen.x, fromY - 2, midX, midY - 2, toScreen.x, toY - 2, t);
          ctx.lineTo(pt.x, pt.y);
        }
        ctx.stroke();
      }
      if (arcStyle.dash.length > 0) ctx.setLineDash([]);
    }

    // Data stream: flowing characters along the bezier
    if (progress < 1) {
      renderArcStream(ctx, arc, fromScreen.x, fromY, midX, midY, toScreen.x, toY, progress, color, highlight);
    }

    // Hold phase (0.45–0.55): genome exchange grid between agents
    if (progress >= 0.45 && progress <= 0.55) {
      renderGenomeGrid(ctx, arc, fromScreen.x, fromY, toScreen.x, toY, midX, midY, progress, color);
    }

    // Arrival burst when complete
    if (progress > 0.95) {
      const burstAlpha = (1 - (progress - 0.95) / 0.05) * 0.6;
      const burstRadius = 6 + (progress - 0.95) / 0.05 * 12;
      const burst = ctx.createRadialGradient(toScreen.x, toY, 0, toScreen.x, toY, burstRadius);
      burst.addColorStop(0, rgba(color, burstAlpha));
      burst.addColorStop(1, rgba(color, 0));
      ctx.fillStyle = burst;
      ctx.beginPath();
      ctx.arc(toScreen.x, toY, burstRadius, 0, Math.PI * 2);
      ctx.fill();
    }
  } else {
    // Rejected: dashed, fading
    const fadeAlpha = (highlight ? 0.7 : 0.3) * (1 - progress * 0.5);
    ctx.strokeStyle = rgba(color, fadeAlpha);
    ctx.lineWidth = highlight ? 2 : 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(fromScreen.x, fromY);
    ctx.quadraticCurveTo(midX, midY, toScreen.x, toY);
    ctx.stroke();

    // Garbled error character stream on rejection
    renderRejectionStream(ctx, arc, fromScreen.x, fromY, midX, midY, toScreen.x, toY, progress, highlight);

    // Rejection X marker
    if (progress > 0.2 && progress < 0.85) {
      const xT = 0.5;
      const xPt = bezierPoint(fromScreen.x, fromY, midX, midY, toScreen.x, toY, xT);
      const xFade = progress < 0.5
        ? easeInOut((progress - 0.2) / 0.3)
        : 1 - easeInOut((progress - 0.5) / 0.35);
      const xSize = 5;
      ctx.strokeStyle = rgba("#EB5757", xFade * 0.8);
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(xPt.x - xSize, xPt.y - xSize);
      ctx.lineTo(xPt.x + xSize, xPt.y + xSize);
      ctx.moveTo(xPt.x + xSize, xPt.y - xSize);
      ctx.lineTo(xPt.x - xSize, xPt.y + xSize);
      ctx.stroke();
    }

    // Shatter particles with error characters
    if (progress > 0.3 && progress < 0.8) {
      const shatterPt = bezierPoint(fromScreen.x, fromY, midX, midY, toScreen.x, toY, 0.5);
      const fade = 1 - (progress - 0.3) / 0.5;
      const arcHash = hashString(arc.id);
      ctx.font = MATRIX_FONT_SMALL;
      ctx.textAlign = "center";
      for (let i = 0; i < 6; i++) {
        const angle = (i / 6) * Math.PI * 2 + progress * 3;
        const dist = (progress - 0.3) * 35;
        const ch = errorCharFromSeed(arcHash + i * 13 + Math.floor(progress * 5));
        ctx.fillStyle = rgba("#EB5757", fade * 0.7);
        ctx.fillText(
          ch,
          shatterPt.x + Math.cos(angle) * dist,
          shatterPt.y + Math.sin(angle) * dist,
        );
      }
    }
  }

  ctx.restore();
}

/** Render a stream of flowing characters along an accepted arc */
function renderArcStream(
  ctx: CanvasRenderingContext2D,
  arc: InteractionArc,
  x0: number, y0: number,
  cx: number, cy: number,
  x1: number, y1: number,
  progress: number,
  color: string,
  highlight: boolean,
) {
  const arcHash = hashString(arc.id);
  const now = Date.now();
  const pFactor = Math.max(0, Math.min(1, arc.p));

  // P-value dependent stream parameters
  const charCount = Math.round(lerp(ARC_STREAM.charCountMin, ARC_STREAM.charCountMax, pFactor));
  const charSpacing = lerp(ARC_STREAM.charSpacingMax, ARC_STREAM.charSpacingMin, pFactor); // high-p = tighter
  const mutateInterval = lerp(ARC_STREAM.mutateIntervalMax, ARC_STREAM.mutateIntervalMin, pFactor); // high-p = faster
  const glowRadius = lerp(ARC_STREAM.glowRadiusMin, ARC_STREAM.glowRadiusMax, pFactor);

  ctx.save();
  ctx.font = MATRIX_FONT_SMALL;
  ctx.textAlign = "center";

  for (let i = 0; i < charCount; i++) {
    // Each character flows along the path, spread out behind the head
    const charT = progress - i * charSpacing;
    if (charT < 0 || charT > 1) continue;

    const pt = bezierPoint(x0, y0, cx, cy, x1, y1, charT);

    // Deterministic character that mutates based on time bucket
    const timeBucket = Math.floor(now / mutateInterval);
    const seed = arcHash + i * 31 + timeBucket;
    const ch = matrixCharFromSeed(seed);

    // Lead chars brighter, trailing ones fade
    const fadeRatio = 1 - i / charCount;
    const alpha = fadeRatio * (highlight ? 0.9 : 0.7);

    if (i === 0) {
      // Head character: brightest, with glow
      ctx.fillStyle = `rgba(255,255,255,${alpha})`;
      ctx.fillText(ch, pt.x, pt.y);
      const glow = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, glowRadius);
      glow.addColorStop(0, rgba(color, 0.4));
      glow.addColorStop(1, rgba(color, 0));
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, glowRadius, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.fillStyle = matrixColorForP(arc.p, alpha);
      ctx.fillText(ch, pt.x, pt.y);
    }
  }

  ctx.restore();
}

/** Render a 3x3 flickering genome exchange grid between agents during hold phase */
function renderGenomeGrid(
  ctx: CanvasRenderingContext2D,
  arc: InteractionArc,
  x0: number, y0: number,
  x1: number, y1: number,
  midX: number, midY: number,
  progress: number,
  color: string,
) {
  const arcHash = hashString(arc.id);
  const now = Date.now();
  const gridMidX = (x0 + x1) / 2;
  const gridMidY = (y0 + y1) / 2 - 15;
  const holdProgress = (progress - 0.45) / 0.1; // 0-1 within hold phase
  const fadeAlpha = Math.sin(holdProgress * Math.PI) * 0.7;

  ctx.save();
  ctx.font = MATRIX_FONT_SMALL;
  ctx.textAlign = "center";

  const { gridSize, gridSpacing } = ARC_STREAM;
  const timeBucket = Math.floor(now / 60);

  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const gx = gridMidX + (col - 1) * gridSpacing;
      const gy = gridMidY + (row - 1) * gridSpacing;
      const seed = arcHash + row * 7 + col * 13 + timeBucket;
      const ch = matrixCharFromSeed(seed);
      ctx.fillStyle = matrixColorForP(arc.p, fadeAlpha * (0.5 + Math.random() * 0.5));
      ctx.fillText(ch, gx, gy);
    }
  }

  ctx.restore();
}

/** Render garbled error characters along a rejected arc */
function renderRejectionStream(
  ctx: CanvasRenderingContext2D,
  arc: InteractionArc,
  x0: number, y0: number,
  cx: number, cy: number,
  x1: number, y1: number,
  progress: number,
  highlight: boolean,
) {
  if (progress < 0.15 || progress > 0.9) return;

  const arcHash = hashString(arc.id);
  const now = Date.now();
  const charCount = 6;

  ctx.save();
  ctx.font = MATRIX_FONT_SMALL;
  ctx.textAlign = "center";

  for (let i = 0; i < charCount; i++) {
    const charT = progress * 0.8 - i * 0.06;
    if (charT < 0 || charT > 1) continue;

    const pt = bezierPoint(x0, y0, cx, cy, x1, y1, charT);

    // Error characters that glitch
    const timeBucket = Math.floor(now / 100);
    const seed = arcHash + i * 17 + timeBucket;
    const ch = errorCharFromSeed(seed);

    // Scatter offset increases with progress
    const scatter = (progress - 0.15) * 15;
    const sx = pt.x + Math.sin(seed) * scatter;
    const sy = pt.y + Math.cos(seed * 1.3) * scatter;

    const fadeRatio = 1 - i / charCount;
    const alpha = fadeRatio * (highlight ? 0.7 : 0.4) * (1 - progress);
    ctx.fillStyle = `rgba(235,87,87,${alpha})`;
    ctx.fillText(ch, sx, sy);
  }

  ctx.restore();
}

/** Render collusion tendrils between flagged pairs */
export function renderCollusionTendril(
  ctx: CanvasRenderingContext2D,
  a1: AgentVisual,
  a2: AgentVisual,
  intensity: number,
) {
  const s1 = gridToScreen(a1.gridX, a1.gridY);
  const s2 = gridToScreen(a2.gridX, a2.gridY);
  const y1 = s1.y - a1.scale * CHARACTER.baseHeight * 0.3;
  const y2 = s2.y - a2.scale * CHARACTER.baseHeight * 0.3;

  ctx.save();
  const t = Date.now() / 1000;

  // Dual wavy lines
  for (let line = 0; line < 2; line++) {
    const offset = line === 0 ? 1 : -1;
    ctx.strokeStyle = rgba("#5B2D8B", intensity * (0.3 + line * 0.15));
    ctx.lineWidth = 1.5 - line * 0.5;
    ctx.beginPath();
    const segments = 20;
    for (let i = 0; i <= segments; i++) {
      const frac = i / segments;
      const baseX = s1.x + (s2.x - s1.x) * frac;
      const baseY = y1 + (y2 - y1) * frac;
      const wave = Math.sin(frac * Math.PI * 4 + t * 2.5 + line) * 6 * offset * intensity;
      const px = baseX + wave * 0.3;
      const py = baseY + wave;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  // Pulse dots along the tendril
  for (let d = 0; d < 3; d++) {
    const dotT = ((t * 0.8 + d * 0.33) % 1);
    const dx = s1.x + (s2.x - s1.x) * dotT;
    const dy = y1 + (y2 - y1) * dotT;
    const dotAlpha = Math.sin(dotT * Math.PI) * intensity * 0.6;
    ctx.fillStyle = rgba("#BB6BD9", dotAlpha);
    ctx.beginPath();
    ctx.arc(dx, dy, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

/** Render golden synergy link between cooperating agents */
export function renderSynergyLink(
  ctx: CanvasRenderingContext2D,
  a1: AgentVisual,
  a2: AgentVisual,
  intensity: number,
) {
  const s1 = gridToScreen(a1.gridX, a1.gridY);
  const s2 = gridToScreen(a2.gridX, a2.gridY);
  const y1 = s1.y - a1.scale * CHARACTER.baseHeight * 0.3;
  const y2 = s2.y - a2.scale * CHARACTER.baseHeight * 0.3;

  ctx.save();
  const t = Date.now() / 1000;

  // Soft golden bezier arc
  const mx = (s1.x + s2.x) / 2;
  const my = (y1 + y2) / 2 - 20;
  ctx.strokeStyle = rgba("#F2C94C", intensity * 0.3);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(s1.x, y1);
  ctx.quadraticCurveTo(mx, my, s2.x, y2);
  ctx.stroke();

  // Flowing particle dots along arc
  for (let d = 0; d < 4; d++) {
    const dotT = ((t * 0.5 + d * 0.25) % 1);
    const mt2 = 1 - dotT;
    const dx = mt2 * mt2 * s1.x + 2 * mt2 * dotT * mx + dotT * dotT * s2.x;
    const dy = mt2 * mt2 * y1 + 2 * mt2 * dotT * my + dotT * dotT * y2;
    const dotAlpha = Math.sin(dotT * Math.PI) * intensity * 0.5;
    ctx.fillStyle = rgba("#F2C94C", dotAlpha);
    ctx.beginPath();
    ctx.arc(dx, dy, 2, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}
