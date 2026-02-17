/** Tierra-style memory strip — agents as colored segments at screen bottom */

import type { AgentVisual } from "../types";
import { AGENT_COLORS } from "../constants";
import { rgba } from "@/utils/color";

const STRIP_HEIGHT = 18;
const TIMELINE_BAR_HEIGHT = 56;
const STRIP_MARGIN = 4;

export function renderTierraStrip(
  ctx: CanvasRenderingContext2D,
  agents: AgentVisual[],
  hoveredAgent: string | null,
  width: number,
  height: number,
) {
  if (agents.length === 0) return;

  const stripY = height - STRIP_HEIGHT - STRIP_MARGIN - TIMELINE_BAR_HEIGHT;
  const totalResources = agents.reduce((sum, a) => sum + Math.max(a.resources, 1), 0);
  if (totalResources <= 0) return;

  ctx.save();

  // Dark background for the strip
  ctx.fillStyle = rgba("#000000", 0.6);
  ctx.fillRect(0, stripY - 1, width, STRIP_HEIGHT + 2);

  // Draw agent segments
  let x = 0;
  for (const agent of agents) {
    const segWidth = (Math.max(agent.resources, 1) / totalResources) * width;
    const colors = AGENT_COLORS[agent.agentType];
    const isWalking = agent.walkOffsetX !== 0 || agent.walkOffsetY !== 0;
    const isHovered = agent.id === hoveredAgent;

    // Segment fill
    ctx.fillStyle = isHovered ? "#FFFFFF" : colors.secondary;
    ctx.fillRect(x, stripY, segWidth, STRIP_HEIGHT);

    // Walking agents get a pulsing bright border
    if (isWalking) {
      const pulse = 0.5 + Math.sin(Date.now() / 200) * 0.5;
      ctx.strokeStyle = rgba(colors.accent, 0.6 + pulse * 0.4);
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, stripY + 1, segWidth - 2, STRIP_HEIGHT - 2);
    }

    // Thin separator between segments
    ctx.fillStyle = rgba("#000000", 0.5);
    ctx.fillRect(x + segWidth - 0.5, stripY, 1, STRIP_HEIGHT);

    x += segWidth;
  }

  // Scanline texture overlay (1px lines every 2px, low alpha)
  ctx.fillStyle = rgba("#000000", 0.15);
  for (let sy = stripY; sy < stripY + STRIP_HEIGHT; sy += 2) {
    ctx.fillRect(0, sy, width, 1);
  }

  // Top/bottom borders
  ctx.fillStyle = rgba("#00FF41", 0.2);
  ctx.fillRect(0, stripY - 1, width, 1);
  ctx.fillRect(0, stripY + STRIP_HEIGHT, width, 1);

  ctx.restore();
}
