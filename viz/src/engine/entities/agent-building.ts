import type { AgentVisual, RenderEntity } from "../types";
import { AGENT_COLORS, BUILDING, TILE_WIDTH, TILE_HEIGHT } from "../constants";
import { gridToScreen } from "../isometric";
import { rgba, pToHealthColor } from "@/utils/color";
import { clamp, remap } from "@/utils/math";
import type { AgentType } from "@/data/types";

/** Create a render entity for an agent building */
export function createBuildingEntity(agent: AgentVisual, hoveredId: string | null): RenderEntity {
  return {
    depth: agent.gridX + agent.gridY,
    render: (ctx) => renderBuilding(ctx, agent, agent.id === hoveredId),
  };
}

function renderBuilding(
  ctx: CanvasRenderingContext2D,
  agent: AgentVisual,
  hovered: boolean,
) {
  const { x: baseX, y: baseY } = gridToScreen(agent.gridX, agent.gridY);
  const colors = AGENT_COLORS[agent.agentType];
  const floors = agent.floors;
  const totalHeight = floors * BUILDING.floorHeight;
  const halfW = BUILDING.baseWidth / 2;
  const isoHalfW = halfW * 0.8;
  const isoHalfH = halfW * 0.4;

  ctx.save();

  // Resource glow under building
  if (agent.resources > 0) {
    const glowSize = remap(agent.resources, 0, 200, 0, 20);
    if (glowSize > 2) {
      const grad = ctx.createRadialGradient(baseX, baseY, 0, baseX, baseY, TILE_WIDTH / 2 + glowSize);
      grad.addColorStop(0, rgba(colors.secondary, 0.3));
      grad.addColorStop(1, rgba(colors.secondary, 0));
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.ellipse(baseX, baseY, TILE_WIDTH / 2 + glowSize, TILE_HEIGHT / 2 + glowSize / 2, 0, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Payoff aura
  if (Math.abs(agent.totalPayoff) > 0.5) {
    const isPositive = agent.totalPayoff > 0;
    const intensity = clamp(Math.abs(agent.totalPayoff) / 10, 0, 0.5);
    const auraColor = isPositive ? "#F2C94C" : "#1A1A2E";
    const grad = ctx.createRadialGradient(baseX, baseY - totalHeight / 2, 0, baseX, baseY - totalHeight / 2, TILE_WIDTH);
    grad.addColorStop(0, rgba(auraColor, intensity));
    grad.addColorStop(1, rgba(auraColor, 0));
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(baseX, baseY - totalHeight / 2, TILE_WIDTH, TILE_WIDTH * 0.6, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  // Quarantine barrier ring
  if (agent.isQuarantined) {
    ctx.strokeStyle = rgba("#EB5757", 0.6 + Math.sin(Date.now() / 300) * 0.3);
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.ellipse(baseX, baseY, TILE_WIDTH * 0.6, TILE_HEIGHT * 0.6, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw building based on agent type
  drawBuildingShape(ctx, agent.agentType, baseX, baseY, isoHalfW, isoHalfH, totalHeight, colors, floors);

  // Frozen overlay
  if (agent.isFrozen) {
    ctx.globalAlpha = 0.3;
    ctx.fillStyle = "#A8CFF5";
    // Cover the building
    ctx.beginPath();
    ctx.moveTo(baseX, baseY - totalHeight - isoHalfH);
    ctx.lineTo(baseX + isoHalfW, baseY - totalHeight);
    ctx.lineTo(baseX + isoHalfW, baseY);
    ctx.lineTo(baseX, baseY + isoHalfH);
    ctx.lineTo(baseX - isoHalfW, baseY);
    ctx.lineTo(baseX - isoHalfW, baseY - totalHeight);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // Roof beacon (p indicator)
  const beaconColor = pToHealthColor(agent.avgP);
  const beaconY = baseY - totalHeight - isoHalfH - 4;
  ctx.fillStyle = beaconColor;
  ctx.beginPath();
  ctx.arc(baseX, beaconY, 3, 0, Math.PI * 2);
  ctx.fill();
  // Beacon glow
  const beaconGrad = ctx.createRadialGradient(baseX, beaconY, 0, baseX, beaconY, 10);
  beaconGrad.addColorStop(0, rgba(beaconColor, 0.4));
  beaconGrad.addColorStop(1, rgba(beaconColor, 0));
  ctx.fillStyle = beaconGrad;
  ctx.beginPath();
  ctx.arc(baseX, beaconY, 10, 0, Math.PI * 2);
  ctx.fill();

  // Hover outline
  if (hovered) {
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(baseX, baseY - totalHeight - isoHalfH);
    ctx.lineTo(baseX + isoHalfW, baseY - totalHeight);
    ctx.lineTo(baseX + isoHalfW, baseY);
    ctx.lineTo(baseX, baseY + isoHalfH);
    ctx.lineTo(baseX - isoHalfW, baseY);
    ctx.lineTo(baseX - isoHalfW, baseY - totalHeight);
    ctx.closePath();
    ctx.stroke();
  }

  ctx.restore();
}

function drawBuildingShape(
  ctx: CanvasRenderingContext2D,
  agentType: AgentType,
  bx: number,
  by: number,
  hw: number,
  hh: number,
  height: number,
  colors: { primary: string; secondary: string; accent: string },
  floors: number,
) {
  // Left face
  ctx.fillStyle = colors.primary;
  ctx.beginPath();
  ctx.moveTo(bx - hw, by - height);
  ctx.lineTo(bx, by - height + hh);
  ctx.lineTo(bx, by + hh);
  ctx.lineTo(bx - hw, by);
  ctx.closePath();
  ctx.fill();

  // Right face (lighter)
  ctx.fillStyle = colors.secondary;
  ctx.beginPath();
  ctx.moveTo(bx + hw, by - height);
  ctx.lineTo(bx, by - height + hh);
  ctx.lineTo(bx, by + hh);
  ctx.lineTo(bx + hw, by);
  ctx.closePath();
  ctx.fill();

  // Top face
  ctx.fillStyle = colors.accent;
  ctx.beginPath();
  ctx.moveTo(bx, by - height - hh);
  ctx.lineTo(bx + hw, by - height);
  ctx.lineTo(bx, by - height + hh);
  ctx.lineTo(bx - hw, by - height);
  ctx.closePath();
  ctx.fill();

  // Floor lines
  ctx.strokeStyle = rgba("#000000", 0.2);
  ctx.lineWidth = 0.5;
  for (let f = 1; f < floors; f++) {
    const fy = by - f * BUILDING.floorHeight;
    ctx.beginPath();
    ctx.moveTo(bx - hw, fy);
    ctx.lineTo(bx, fy + hh);
    ctx.lineTo(bx + hw, fy);
    ctx.stroke();
  }

  // Windows based on type
  drawWindows(ctx, agentType, bx, by, hw, hh, height, floors, colors);

  // Outline
  ctx.strokeStyle = rgba("#000000", 0.3);
  ctx.lineWidth = 1;
  // Left edge
  ctx.beginPath();
  ctx.moveTo(bx - hw, by);
  ctx.lineTo(bx - hw, by - height);
  ctx.lineTo(bx, by - height - hh);
  ctx.lineTo(bx + hw, by - height);
  ctx.lineTo(bx + hw, by);
  ctx.lineTo(bx, by + hh);
  ctx.closePath();
  ctx.stroke();
}

function drawWindows(
  ctx: CanvasRenderingContext2D,
  agentType: AgentType,
  bx: number,
  by: number,
  hw: number,
  hh: number,
  height: number,
  floors: number,
  colors: { primary: string; secondary: string; accent: string },
) {
  const windowColor = rgba(colors.accent, 0.6);
  ctx.fillStyle = windowColor;

  for (let f = 0; f < floors; f++) {
    const floorY = by - f * BUILDING.floorHeight - BUILDING.floorHeight / 2;

    if (agentType === "honest" || agentType === "crewai") {
      // Regular windows on both faces
      for (let w = 0; w < 2; w++) {
        const wx = bx - hw * 0.5 + w * hw * 0.35;
        const wy = floorY + hh * 0.3 * (1 - w * 0.3);
        ctx.fillRect(wx, wy - 2, 3, 4);
      }
      for (let w = 0; w < 2; w++) {
        const wx = bx + hw * 0.15 + w * hw * 0.35;
        const wy = floorY - hh * 0.1 + w * hh * 0.3;
        ctx.fillRect(wx, wy - 2, 3, 4);
      }
    } else if (agentType === "adversarial") {
      // Narrow slits (fortress)
      ctx.fillRect(bx - hw * 0.4, floorY + hh * 0.2 - 1, 2, 5);
      ctx.fillRect(bx + hw * 0.3, floorY - 1, 2, 5);
    } else if (agentType === "deceptive") {
      // Mirror panels
      ctx.save();
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = colors.accent;
      ctx.fillRect(bx - hw * 0.6, floorY + hh * 0.1, hw * 0.5, BUILDING.floorHeight * 0.6);
      ctx.fillRect(bx + hw * 0.1, floorY - hh * 0.2, hw * 0.5, BUILDING.floorHeight * 0.6);
      ctx.restore();
      ctx.fillStyle = windowColor;
    } else if (agentType === "rlm") {
      // Data center LEDs
      for (let w = 0; w < 3; w++) {
        const wx = bx - hw * 0.5 + w * hw * 0.25;
        ctx.fillStyle = w % 2 === 0 ? "#27AE60" : colors.accent;
        ctx.fillRect(wx, floorY + hh * 0.2, 2, 2);
      }
      ctx.fillStyle = windowColor;
    } else {
      // Opportunistic - open archways
      if (f === 0) {
        ctx.fillRect(bx - hw * 0.3, floorY + hh * 0.1, 6, BUILDING.floorHeight * 0.7);
        ctx.fillRect(bx + hw * 0.2, floorY - hh * 0.1, 6, BUILDING.floorHeight * 0.7);
      } else {
        ctx.fillRect(bx - hw * 0.4, floorY + hh * 0.2, 3, 4);
        ctx.fillRect(bx + hw * 0.3, floorY, 3, 4);
      }
    }
  }
}

/** Get building hitbox bounds in screen space for click detection */
export function getBuildingBounds(agent: AgentVisual): { minX: number; minY: number; maxX: number; maxY: number } {
  const { x, y } = gridToScreen(agent.gridX, agent.gridY);
  const halfW = BUILDING.baseWidth / 2 * 0.8;
  const totalHeight = agent.floors * BUILDING.floorHeight;
  const hh = halfW * 0.5;
  return {
    minX: x - halfW,
    minY: y - totalHeight - hh - 10,
    maxX: x + halfW,
    maxY: y + hh,
  };
}
