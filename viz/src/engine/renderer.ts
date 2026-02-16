import type { AgentVisual, InteractionArc, Viewport, RenderEntity, OverlayState, Particle } from "./types";
import type { EpochSnapshot } from "@/data/types";
import { COLORS } from "./constants";
import { depthSort } from "./depth-sort";
import { createBuildingEntity } from "./entities/agent-building";
import { createArcEntity } from "./entities/interaction-arc";
import { renderGroundGrid, renderGiniCracks } from "./entities/ground-tile";
import { renderSky, renderHaze, renderParticles } from "./entities/effects";
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
  const { width, height } = ctx.canvas;

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

  // 6. Collect renderable entities
  const entities: RenderEntity[] = [];

  // Buildings
  for (const agent of state.agents) {
    entities.push(createBuildingEntity(agent, state.hoveredAgent));
  }

  // Interaction arcs
  if (state.overlays.interactions) {
    for (const arc of state.arcs) {
      const agentMap = new Map(state.agents.map((a) => [a.id, a]));
      const entity = createArcEntity(arc, agentMap, state.hoveredAgent);
      if (entity) entities.push(entity);
    }
  }

  // 7. Depth sort and render
  const sorted = depthSort(entities);
  for (const entity of sorted) {
    entity.render(ctx);
  }

  // 8. Particles
  if (state.overlays.particles) {
    renderParticles(ctx, state.particles);
  }

  ctx.restore();

  // 9. Ground haze (screen space)
  renderHaze(ctx, width, height, state.environment.toxicity);

  // 10. Tooltip for hovered agent
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
  const sy = (pos.y - agent.floors * 10 - 30) * vp.zoom + vp.y;

  const text = `${agent.name} (${agent.agentType})`;
  const subtext = `Rep: ${agent.reputation.toFixed(2)} | P: ${agent.avgP.toFixed(2)}`;

  ctx.save();
  ctx.font = "bold 12px -apple-system, sans-serif";
  const tw = Math.max(ctx.measureText(text).width, ctx.measureText(subtext).width) + 16;

  // Background
  ctx.fillStyle = COLORS.panel;
  ctx.strokeStyle = COLORS.border;
  ctx.lineWidth = 1;
  const rx = sx - tw / 2;
  const ry = sy - 36;
  ctx.beginPath();
  ctx.roundRect(rx, ry, tw, 34, 4);
  ctx.fill();
  ctx.stroke();

  // Text
  ctx.fillStyle = COLORS.text;
  ctx.textAlign = "center";
  ctx.fillText(text, sx, ry + 14);
  ctx.font = "11px -apple-system, sans-serif";
  ctx.fillStyle = COLORS.muted;
  ctx.fillText(subtext, sx, ry + 28);

  ctx.restore();
}
