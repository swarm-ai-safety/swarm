import type { InteractionArc, AgentVisual, RenderEntity } from "../types";
import { gridToScreen } from "../isometric";
import { BUILDING } from "../constants";
import { rgba, pToHealthColor } from "@/utils/color";
import { easeOut, easeInOut } from "@/utils/math";

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

function renderArc(
  ctx: CanvasRenderingContext2D,
  arc: InteractionArc,
  from: AgentVisual,
  to: AgentVisual,
  highlight: boolean,
) {
  const fromScreen = gridToScreen(from.gridX, from.gridY);
  const toScreen = gridToScreen(to.gridX, to.gridY);
  const fromY = fromScreen.y - from.floors * BUILDING.floorHeight * 0.6;
  const toY = toScreen.y - to.floors * BUILDING.floorHeight * 0.6;

  const color = pToHealthColor(arc.p);
  const progress = easeOut(arc.progress);

  // Control point for bezier (arc upward)
  const midX = (fromScreen.x + toScreen.x) / 2;
  const midY = Math.min(fromY, toY) - 40 - Math.abs(fromScreen.x - toScreen.x) * 0.15;

  ctx.save();

  if (arc.accepted) {
    // Ghost line showing full path
    ctx.strokeStyle = rgba(color, highlight ? 0.15 : 0.08);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(fromScreen.x, fromY);
    ctx.quadraticCurveTo(midX, midY, toScreen.x, toY);
    ctx.stroke();

    // Trailing path up to current progress
    if (progress > 0.01) {
      ctx.strokeStyle = rgba(color, highlight ? 0.7 : 0.4);
      ctx.lineWidth = highlight ? 2.5 : 1.5;
      ctx.beginPath();
      ctx.moveTo(fromScreen.x, fromY);
      const steps = Math.max(8, Math.floor(progress * 30));
      for (let i = 1; i <= steps; i++) {
        const t = (i / steps) * progress;
        const pt = bezierPoint(fromScreen.x, fromY, midX, midY, toScreen.x, toY, t);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();
    }

    // Animated particle head with mini trail
    if (progress < 1) {
      const head = bezierPoint(fromScreen.x, fromY, midX, midY, toScreen.x, toY, progress);

      // Mini trail (3 dots fading behind)
      for (let i = 3; i >= 1; i--) {
        const trailT = Math.max(0, progress - i * 0.03);
        const tp = bezierPoint(fromScreen.x, fromY, midX, midY, toScreen.x, toY, trailT);
        ctx.fillStyle = rgba(color, (1 - i * 0.25) * (highlight ? 0.6 : 0.3));
        ctx.beginPath();
        ctx.arc(tp.x, tp.y, (highlight ? 3 : 2) - i * 0.4, 0, Math.PI * 2);
        ctx.fill();
      }

      // Main particle
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(head.x, head.y, highlight ? 4 : 3, 0, Math.PI * 2);
      ctx.fill();

      // Glow
      const glow = ctx.createRadialGradient(head.x, head.y, 0, head.x, head.y, 10);
      glow.addColorStop(0, rgba(color, 0.5));
      glow.addColorStop(1, rgba(color, 0));
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(head.x, head.y, 10, 0, Math.PI * 2);
      ctx.fill();
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

    // Shatter particles at rejection point
    if (progress > 0.3 && progress < 0.8) {
      const shatterPt = bezierPoint(fromScreen.x, fromY, midX, midY, toScreen.x, toY, 0.5);
      const fade = 1 - (progress - 0.3) / 0.5;
      ctx.fillStyle = rgba("#EB5757", fade * 0.6);
      for (let i = 0; i < 6; i++) {
        const angle = (i / 6) * Math.PI * 2 + progress * 3;
        const dist = (progress - 0.3) * 35;
        ctx.beginPath();
        ctx.arc(
          shatterPt.x + Math.cos(angle) * dist,
          shatterPt.y + Math.sin(angle) * dist,
          2 * fade,
          0,
          Math.PI * 2,
        );
        ctx.fill();
      }
    }
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
  const y1 = s1.y - a1.floors * BUILDING.floorHeight * 0.3;
  const y2 = s2.y - a2.floors * BUILDING.floorHeight * 0.3;

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
