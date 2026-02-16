import type { AgentVisual, InteractionArc, Viewport, RenderEntity, OverlayState, Particle } from "./types";
import type { EpochSnapshot } from "@/data/types";
import { COLORS } from "./constants";
import { depthSort } from "./depth-sort";
import { createCharacterEntity } from "./entities/agent-character";
import { createArcEntity, renderCollusionTendril } from "./entities/interaction-arc";
import { renderGroundGrid, renderGiniCracks } from "./entities/ground-tile";
import { renderSky, renderHaze, renderParticles, renderThreatZone } from "./entities/effects";
import { gridToScreen } from "./isometric";
import type { EnvironmentState } from "./systems/environment-system";

export interface RenderState {
  agents: AgentVisual[];
  arcs: InteractionArc[];
  viewport: Viewport;
  hoveredAgent: string | null;
  selectedAgent: string | null;
  epoch: EpochSnapshot | null;
  environment: EnvironmentState;
  overlays: OverlayState;
  particles: Particle[];
  gridSize: number;
}

export function render(ctx: CanvasRenderingContext2D, state: RenderState) {
  // Use viewport dimensions (CSS pixels) since DPR scale is already applied by caller
  const width = state.viewport.width;
  const height = state.viewport.height;

  // 1. Clear
  ctx.clearRect(0, 0, width, height);

  // 2. Sky gradient (based on threat level)
  renderSky(ctx, width, height, state.environment.threatLevel);

  // 3. Apply camera transform
  ctx.save();
  ctx.translate(state.viewport.x, state.viewport.y);
  ctx.scale(state.viewport.zoom, state.viewport.zoom);

  // 4. Ground tiles
  renderGroundGrid(ctx, state.gridSize, state.environment.toxicity);

  // 5. Gini cracks
  renderGiniCracks(ctx, state.gridSize, state.environment.giniCoefficient);

  // 6. Threat zones (rendered under buildings)
  if (state.overlays.threatZones) {
    for (const agent of state.agents) {
      if (agent.agentType === "adversarial" || agent.agentType === "deceptive") {
        const pos = gridToScreen(agent.gridX, agent.gridY);
        renderThreatZone(ctx, pos.x, pos.y, state.environment.threatLevel, agent.agentType);
      }
    }
  }

  // 7. Collect renderable entities
  const entities: RenderEntity[] = [];

  // Build agent map once
  const agentMap = new Map(state.agents.map((a) => [a.id, a]));

  // Buildings
  for (const agent of state.agents) {
    entities.push(createCharacterEntity(agent, state.hoveredAgent));
  }

  // Interaction arcs
  if (state.overlays.interactions) {
    for (const arc of state.arcs) {
      const entity = createArcEntity(arc, agentMap, state.hoveredAgent);
      if (entity) entities.push(entity);
    }
  }

  // 8. Depth sort and render
  const sorted = depthSort(entities);
  for (const entity of sorted) {
    entity.render(ctx);
  }

  // 9. Collusion tendrils (rendered above buildings)
  if (state.overlays.collusionLines && state.environment.collusionRisk > 0.1) {
    const flaggedTypes = new Set(["adversarial", "deceptive", "opportunistic"]);
    const flagged = state.agents.filter((a) => flaggedTypes.has(a.agentType));
    for (let i = 0; i < flagged.length; i++) {
      for (let j = i + 1; j < flagged.length; j++) {
        renderCollusionTendril(ctx, flagged[i], flagged[j], state.environment.collusionRisk);
      }
    }
  }

  // 10. Particles
  if (state.overlays.particles) {
    renderParticles(ctx, state.particles);
  }

  ctx.restore();

  // 11. Ground haze (screen space)
  renderHaze(ctx, width, height, state.environment.toxicity);

  // 12. Tooltip for hovered agent
  if (state.hoveredAgent) {
    const agent = state.agents.find((a) => a.id === state.hoveredAgent);
    if (agent) {
      renderTooltip(ctx, state.viewport, agent);
    }
  }
}

function renderTooltip(ctx: CanvasRenderingContext2D, vp: Viewport, agent: AgentVisual) {
  const pos = gridToScreen(agent.gridX, agent.gridY);
  const sx = pos.x * vp.zoom + vp.x;
  const sy = (pos.y - agent.scale * 56 - 30) * vp.zoom + vp.y;

  const text = `${agent.name} (${agent.agentType})`;
  const subtext = `Rep: ${agent.reputation.toFixed(2)} | P: ${agent.avgP.toFixed(2)}`;
  const statusText = agent.isFrozen ? "FROZEN" : agent.isQuarantined ? "QUARANTINED" : "";

  ctx.save();
  ctx.font = "bold 12px -apple-system, sans-serif";
  const tw = Math.max(
    ctx.measureText(text).width,
    ctx.measureText(subtext).width,
    statusText ? ctx.measureText(statusText).width : 0,
  ) + 20;
  const th = statusText ? 48 : 38;

  const rx = sx - tw / 2;
  const ry = sy - th - 8;

  // Shadow
  ctx.shadowColor = "rgba(0,0,0,0.4)";
  ctx.shadowBlur = 8;
  ctx.shadowOffsetY = 2;

  // Background
  ctx.fillStyle = COLORS.panel;
  ctx.strokeStyle = COLORS.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(rx, ry, tw, th, 6);
  ctx.fill();

  // Reset shadow for border
  ctx.shadowColor = "transparent";
  ctx.stroke();

  // Arrow
  ctx.fillStyle = COLORS.panel;
  ctx.beginPath();
  ctx.moveTo(sx - 5, ry + th);
  ctx.lineTo(sx, ry + th + 5);
  ctx.lineTo(sx + 5, ry + th);
  ctx.closePath();
  ctx.fill();

  // Text
  ctx.fillStyle = COLORS.text;
  ctx.textAlign = "center";
  ctx.fillText(text, sx, ry + 15);
  ctx.font = "11px -apple-system, sans-serif";
  ctx.fillStyle = COLORS.muted;
  ctx.fillText(subtext, sx, ry + 29);

  // Status text
  if (statusText) {
    ctx.font = "bold 10px -apple-system, sans-serif";
    ctx.fillStyle = agent.isFrozen ? "#A8CFF5" : "#EB5757";
    ctx.fillText(statusText, sx, ry + 42);
  }

  ctx.restore();
}
