import type { InteractionArc, AgentVisual, RenderEntity } from "../types";
import { gridToScreen } from "../isometric";
import { BUILDING } from "../constants";
import { rgba, pToHealthColor } from "@/utils/color";
import { easeOut } from "@/utils/math";

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
    // Solid flowing arc
    ctx.strokeStyle = rgba(color, highlight ? 0.9 : 0.5);
    ctx.lineWidth = highlight ? 2.5 : 1.5;
    ctx.beginPath();
    ctx.moveTo(fromScreen.x, fromY);
    ctx.quadraticCurveTo(midX, midY, toScreen.x, toY);
    ctx.stroke();

    // Animated particle along arc
    if (progress < 1) {
      const t = progress;
      const px = (1 - t) * (1 - t) * fromScreen.x + 2 * (1 - t) * t * midX + t * t * toScreen.x;
      const py = (1 - t) * (1 - t) * fromY + 2 * (1 - t) * t * midY + t * t * toY;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px, py, highlight ? 4 : 3, 0, Math.PI * 2);
      ctx.fill();
      // Glow
      const glow = ctx.createRadialGradient(px, py, 0, px, py, 8);
      glow.addColorStop(0, rgba(color, 0.5));
      glow.addColorStop(1, rgba(color, 0));
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fill();
    }
  } else {
    // Rejected: dashed, fading
    ctx.strokeStyle = rgba(color, (highlight ? 0.7 : 0.3) * (1 - progress * 0.5));
    ctx.lineWidth = highlight ? 2 : 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(fromScreen.x, fromY);
    ctx.quadraticCurveTo(midX, midY, toScreen.x, toY);
    ctx.stroke();

    // Shatter particles at rejection point
    if (progress > 0.3 && progress < 0.8) {
      const shatterT = 0.5;
      const sx = (1 - shatterT) * (1 - shatterT) * fromScreen.x + 2 * (1 - shatterT) * shatterT * midX + shatterT * shatterT * toScreen.x;
      const sy = (1 - shatterT) * (1 - shatterT) * fromY + 2 * (1 - shatterT) * shatterT * midY + shatterT * shatterT * toY;
      const fade = 1 - (progress - 0.3) / 0.5;
      ctx.fillStyle = rgba("#EB5757", fade * 0.6);
      for (let i = 0; i < 4; i++) {
        const angle = (i / 4) * Math.PI * 2 + progress * 3;
        const dist = (progress - 0.3) * 30;
        ctx.beginPath();
        ctx.arc(sx + Math.cos(angle) * dist, sy + Math.sin(angle) * dist, 2, 0, Math.PI * 2);
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
  ctx.strokeStyle = rgba("#5B2D8B", intensity * 0.5);
  ctx.lineWidth = 1.5;
  const t = Date.now() / 1000;
  const midX = (s1.x + s2.x) / 2 + Math.sin(t * 2) * 10;
  const midY = (y1 + y2) / 2 + Math.cos(t * 1.5) * 8;
  ctx.beginPath();
  ctx.moveTo(s1.x, y1);
  ctx.quadraticCurveTo(midX, midY, s2.x, y2);
  ctx.stroke();
  ctx.restore();
}
