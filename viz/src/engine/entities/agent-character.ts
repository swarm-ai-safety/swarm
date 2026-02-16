import type { AgentVisual, RenderEntity } from "../types";
import { AGENT_COLORS, AGENT_MOTION, CHARACTER, TILE_WIDTH, TILE_HEIGHT } from "../constants";
import { gridToScreen } from "../isometric";
import { rgba, pToHealthColor } from "@/utils/color";
import { clamp, remap } from "@/utils/math";
import type { AgentType } from "@/data/types";

/** Create a render entity for an agent character */
export function createCharacterEntity(agent: AgentVisual, hoveredId: string | null): RenderEntity {
  return {
    depth: agent.gridX + agent.gridY,
    render: (ctx) => drawCharacter(ctx, agent, agent.id === hoveredId),
  };
}

function drawCharacter(
  ctx: CanvasRenderingContext2D,
  agent: AgentVisual,
  hovered: boolean,
) {
  const gridPos = gridToScreen(agent.gridX, agent.gridY);
  const baseX = gridPos.x + agent.walkOffsetX;
  const baseY = gridPos.y + agent.walkOffsetY;
  const colors = AGENT_COLORS[agent.agentType];
  const scale = agent.scale;

  ctx.save();

  // Resource glow under character
  drawResourceGlow(ctx, baseX, baseY, agent.resources, colors);

  // Payoff aura
  drawPayoffAura(ctx, baseX, baseY, agent.totalPayoff, scale, colors);

  // Quarantine barrier ring
  if (agent.isQuarantined) {
    drawQuarantineBarrier(ctx, baseX, baseY);
  }

  // Character body + type features
  const isWalking = agent.walkOffsetX !== 0 || agent.walkOffsetY !== 0;
  drawBody(ctx, baseX, baseY, agent.agentType, colors, scale, agent.walkPhase, isWalking);
  drawTypeFeatures(ctx, baseX, baseY, agent.agentType, colors, scale);

  // Frozen overlay
  if (agent.isFrozen) {
    drawFrozenOverlay(ctx, baseX, baseY, scale);
  }

  // P orb above head
  drawPOrb(ctx, baseX, baseY, agent.avgP, scale);

  // Hover outline
  if (hovered) {
    drawHoverOutline(ctx, baseX, baseY, scale);
  }

  ctx.restore();
}

// --- Sub-draw functions ---

function drawResourceGlow(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  resources: number,
  colors: { secondary: string },
) {
  if (resources <= 0) return;
  const glowSize = remap(resources, 0, 200, 0, 20);
  if (glowSize <= 2) return;
  const grad = ctx.createRadialGradient(bx, by, 0, bx, by, TILE_WIDTH / 2 + glowSize);
  grad.addColorStop(0, rgba(colors.secondary, 0.3));
  grad.addColorStop(1, rgba(colors.secondary, 0));
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(bx, by, TILE_WIDTH / 2 + glowSize, TILE_HEIGHT / 2 + glowSize / 2, 0, 0, Math.PI * 2);
  ctx.fill();
}

function drawPayoffAura(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  totalPayoff: number,
  scale: number,
  colors: { secondary: string },
) {
  if (Math.abs(totalPayoff) <= 0.5) return;
  const charH = CHARACTER.baseHeight * scale;
  const isPositive = totalPayoff > 0;
  const intensity = clamp(Math.abs(totalPayoff) / 10, 0, 0.5);
  const auraColor = isPositive ? "#F2C94C" : "#1A1A2E";
  const centerY = by - charH * 0.5;
  const radius = TILE_WIDTH * 0.8;
  const grad = ctx.createRadialGradient(bx, centerY, 0, bx, centerY, radius);
  grad.addColorStop(0, rgba(auraColor, intensity));
  grad.addColorStop(1, rgba(auraColor, 0));
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.ellipse(bx, centerY, radius, radius * 0.6, 0, 0, Math.PI * 2);
  ctx.fill();
}

function drawQuarantineBarrier(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
) {
  ctx.strokeStyle = rgba("#EB5757", 0.6 + Math.sin(Date.now() / 300) * 0.3);
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.ellipse(bx, by, TILE_WIDTH * 0.6, TILE_HEIGHT * 0.6, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawBody(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  agentType: AgentType,
  colors: { primary: string; secondary: string; accent: string },
  scale: number,
  walkPhase: number = 0,
  isWalking: boolean = false,
) {
  const motion = AGENT_MOTION[agentType];
  const w = CHARACTER.baseWidth * scale;
  const h = CHARACTER.baseHeight * scale;
  const hw = w / 2;

  // Asymmetric bob: agents spend more time at top of bounce than bottom
  let bobOffset: number;
  if (isWalking) {
    const rawBob = Math.sin(walkPhase * 2);
    // Asymmetry shifts the wave — positive asymmetry = more time at top
    const asymBob = rawBob + motion.bobAsymmetry * rawBob * rawBob;
    bobOffset = Math.abs(asymBob) * motion.bobAmplitude * scale;
  } else {
    // Idle breathing — slow sinusoidal so agents are never stone-still
    bobOffset = motion.idleBob * Math.sin(Date.now() * 0.0012) * scale;
  }
  const bodyBy = by - bobOffset;

  // Feet at by, head at bodyBy - h
  const feetY = by;
  const headY = bodyBy - h;
  const shoulderY = headY + h * 0.28;
  const waistY = headY + h * 0.55;
  const hipY = headY + h * 0.62;

  // Per-type leg swing
  const legSwing = isWalking ? Math.sin(walkPhase) * hw * 0.2 * motion.legSwingScale : 0;

  // --- Legs ---
  ctx.fillStyle = colors.primary;
  // Left leg
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.3, hipY);
  ctx.lineTo(bx - hw * 0.45 + legSwing, feetY);
  ctx.lineTo(bx - hw * 0.05 + legSwing, feetY);
  ctx.lineTo(bx - hw * 0.05, hipY);
  ctx.closePath();
  ctx.fill();
  // Right leg
  ctx.beginPath();
  ctx.moveTo(bx + hw * 0.05, hipY);
  ctx.lineTo(bx + hw * 0.05 - legSwing, feetY);
  ctx.lineTo(bx + hw * 0.45 - legSwing, feetY);
  ctx.lineTo(bx + hw * 0.3, hipY);
  ctx.closePath();
  ctx.fill();

  // --- Torso ---
  ctx.fillStyle = colors.secondary;
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.45, shoulderY);
  ctx.lineTo(bx + hw * 0.45, shoulderY);
  ctx.lineTo(bx + hw * 0.35, waistY);
  ctx.lineTo(bx - hw * 0.35, waistY);
  ctx.closePath();
  ctx.fill();

  // --- Arms (contralateral: left arm swings opposite to left leg) ---
  const armAngle = agentType === "deceptive" ? 0.8 : 0.2;
  const armSwing = isWalking ? Math.sin(walkPhase) * 0.3 * motion.armSwingScale : 0;
  ctx.fillStyle = colors.primary;
  // Left arm — swings FORWARD when right leg goes forward (contralateral)
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.45, shoulderY);
  ctx.lineTo(bx - hw * (0.6 + armAngle * 0.3) + armSwing * hw, shoulderY + h * 0.22);
  ctx.lineTo(bx - hw * (0.5 + armAngle * 0.2) + armSwing * hw, shoulderY + h * 0.26);
  ctx.lineTo(bx - hw * 0.35, shoulderY + h * 0.05);
  ctx.closePath();
  ctx.fill();
  // Right arm — swings opposite to left arm
  ctx.beginPath();
  ctx.moveTo(bx + hw * 0.45, shoulderY);
  ctx.lineTo(bx + hw * (0.6 + armAngle * 0.3) - armSwing * hw, shoulderY + h * 0.22);
  ctx.lineTo(bx + hw * (0.5 + armAngle * 0.2) - armSwing * hw, shoulderY + h * 0.26);
  ctx.lineTo(bx + hw * 0.35, shoulderY + h * 0.05);
  ctx.closePath();
  ctx.fill();

  // --- Head (circle) ---
  const headRadius = hw * 0.35;
  const headCenterY = headY + headRadius + h * 0.04;
  ctx.fillStyle = colors.accent;
  ctx.beginPath();
  ctx.arc(bx, headCenterY, headRadius, 0, Math.PI * 2);
  ctx.fill();

  // Outline
  ctx.strokeStyle = rgba("#000000", 0.25);
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawTypeFeatures(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  agentType: AgentType,
  colors: { primary: string; secondary: string; accent: string },
  scale: number,
) {
  const w = CHARACTER.baseWidth * scale;
  const h = CHARACTER.baseHeight * scale;
  const hw = w / 2;
  const headY = by - h;
  const headRadius = hw * 0.35;
  const headCenterY = headY + headRadius + h * 0.04;
  const shoulderY = headY + h * 0.28;

  switch (agentType) {
    case "honest":
      drawPaladin(ctx, bx, by, headCenterY, headRadius, shoulderY, hw, h, colors, scale);
      break;
    case "opportunistic":
      drawMerchant(ctx, bx, by, headCenterY, headRadius, hw, h, colors, scale);
      break;
    case "deceptive":
      drawIllusionist(ctx, bx, by, headCenterY, headRadius, shoulderY, hw, h, colors, scale);
      break;
    case "adversarial":
      drawEnforcer(ctx, bx, by, headCenterY, headRadius, shoulderY, hw, h, colors, scale);
      break;
    case "rlm":
      drawTechnomancer(ctx, bx, by, headCenterY, headRadius, shoulderY, hw, h, colors, scale);
      break;
    case "crewai":
      drawBuilder(ctx, bx, by, headCenterY, headRadius, shoulderY, hw, h, colors, scale);
      break;
  }
}

// --- Type-specific accessories ---

function drawPaladin(
  ctx: CanvasRenderingContext2D,
  bx: number, _by: number,
  headCY: number, headR: number,
  shoulderY: number, hw: number, h: number,
  colors: { primary: string; secondary: string; accent: string },
  _scale: number,
) {
  // Knight helmet with visor
  ctx.fillStyle = colors.secondary;
  ctx.beginPath();
  // Helmet dome over head
  ctx.arc(bx, headCY - headR * 0.15, headR * 1.2, Math.PI, 0);
  ctx.lineTo(bx + headR * 1.2, headCY + headR * 0.3);
  ctx.lineTo(bx - headR * 1.2, headCY + headR * 0.3);
  ctx.closePath();
  ctx.fill();
  // Visor slit
  ctx.strokeStyle = rgba("#000000", 0.5);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(bx - headR * 0.6, headCY + headR * 0.05);
  ctx.lineTo(bx + headR * 0.6, headCY + headR * 0.05);
  ctx.stroke();

  // Shield emblem on chest
  ctx.fillStyle = colors.accent;
  ctx.beginPath();
  ctx.moveTo(bx, shoulderY + h * 0.02);
  ctx.lineTo(bx + hw * 0.15, shoulderY + h * 0.06);
  ctx.lineTo(bx, shoulderY + h * 0.14);
  ctx.lineTo(bx - hw * 0.15, shoulderY + h * 0.06);
  ctx.closePath();
  ctx.fill();
}

function drawMerchant(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  headCY: number, headR: number,
  hw: number, h: number,
  colors: { primary: string; secondary: string; accent: string },
  _scale: number,
) {
  // Wide-brimmed hat
  const hatBrim = headR * 1.8;
  const hatTop = headCY - headR * 0.9;
  ctx.fillStyle = colors.primary;
  // Brim
  ctx.beginPath();
  ctx.ellipse(bx, hatTop + headR * 0.3, hatBrim, headR * 0.25, 0, 0, Math.PI * 2);
  ctx.fill();
  // Crown
  ctx.beginPath();
  ctx.moveTo(bx - headR * 0.6, hatTop + headR * 0.3);
  ctx.lineTo(bx - headR * 0.5, hatTop - headR * 0.3);
  ctx.lineTo(bx + headR * 0.5, hatTop - headR * 0.3);
  ctx.lineTo(bx + headR * 0.6, hatTop + headR * 0.3);
  ctx.closePath();
  ctx.fill();

  // Coin purse at hip
  const hipY = by - h * 0.38;
  ctx.fillStyle = colors.accent;
  ctx.beginPath();
  ctx.ellipse(bx + hw * 0.4, hipY, hw * 0.12, hw * 0.15, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = rgba("#000000", 0.3);
  ctx.lineWidth = 0.5;
  ctx.stroke();
}

function drawIllusionist(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  headCY: number, headR: number,
  shoulderY: number, hw: number, h: number,
  colors: { primary: string; secondary: string; accent: string },
  _scale: number,
) {
  // Flowing cape
  ctx.fillStyle = rgba(colors.primary, 0.7);
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.4, shoulderY);
  ctx.quadraticCurveTo(bx - hw * 0.8, shoulderY + h * 0.3, bx - hw * 0.6, by);
  ctx.lineTo(bx + hw * 0.6, by);
  ctx.quadraticCurveTo(bx + hw * 0.8, shoulderY + h * 0.3, bx + hw * 0.4, shoulderY);
  ctx.closePath();
  ctx.fill();

  // Masquerade mask
  ctx.fillStyle = colors.accent;
  ctx.beginPath();
  // Left eye hole
  ctx.ellipse(bx - headR * 0.35, headCY - headR * 0.05, headR * 0.25, headR * 0.18, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  // Right eye hole
  ctx.ellipse(bx + headR * 0.35, headCY - headR * 0.05, headR * 0.25, headR * 0.18, 0, 0, Math.PI * 2);
  ctx.fill();
  // Mask bridge
  ctx.fillStyle = colors.secondary;
  ctx.beginPath();
  ctx.moveTo(bx - headR * 0.7, headCY - headR * 0.25);
  ctx.quadraticCurveTo(bx, headCY - headR * 0.5, bx + headR * 0.7, headCY - headR * 0.25);
  ctx.lineTo(bx + headR * 0.7, headCY + headR * 0.15);
  ctx.quadraticCurveTo(bx, headCY + headR * 0.05, bx - headR * 0.7, headCY + headR * 0.15);
  ctx.closePath();
  ctx.fill();
  // Dark eye holes
  ctx.fillStyle = rgba("#000000", 0.6);
  ctx.beginPath();
  ctx.ellipse(bx - headR * 0.3, headCY - headR * 0.05, headR * 0.13, headR * 0.1, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(bx + headR * 0.3, headCY - headR * 0.05, headR * 0.13, headR * 0.1, 0, 0, Math.PI * 2);
  ctx.fill();
}

function drawEnforcer(
  ctx: CanvasRenderingContext2D,
  bx: number, _by: number,
  headCY: number, headR: number,
  shoulderY: number, hw: number, _h: number,
  colors: { primary: string; secondary: string; accent: string },
  _scale: number,
) {
  // Demon horns
  ctx.fillStyle = colors.secondary;
  // Left horn
  ctx.beginPath();
  ctx.moveTo(bx - headR * 0.6, headCY - headR * 0.6);
  ctx.lineTo(bx - headR * 1.1, headCY - headR * 1.6);
  ctx.lineTo(bx - headR * 0.3, headCY - headR * 0.8);
  ctx.closePath();
  ctx.fill();
  // Right horn
  ctx.beginPath();
  ctx.moveTo(bx + headR * 0.6, headCY - headR * 0.6);
  ctx.lineTo(bx + headR * 1.1, headCY - headR * 1.6);
  ctx.lineTo(bx + headR * 0.3, headCY - headR * 0.8);
  ctx.closePath();
  ctx.fill();

  // Spiked pauldrons (shoulder armor)
  ctx.fillStyle = colors.accent;
  // Left pauldron
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.5, shoulderY - hw * 0.1);
  ctx.lineTo(bx - hw * 0.7, shoulderY - hw * 0.35);
  ctx.lineTo(bx - hw * 0.35, shoulderY + hw * 0.05);
  ctx.closePath();
  ctx.fill();
  // Right pauldron
  ctx.beginPath();
  ctx.moveTo(bx + hw * 0.5, shoulderY - hw * 0.1);
  ctx.lineTo(bx + hw * 0.7, shoulderY - hw * 0.35);
  ctx.lineTo(bx + hw * 0.35, shoulderY + hw * 0.05);
  ctx.closePath();
  ctx.fill();

  // Angry eyes
  ctx.fillStyle = colors.secondary;
  ctx.beginPath();
  ctx.arc(bx - headR * 0.3, headCY, headR * 0.1, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(bx + headR * 0.3, headCY, headR * 0.1, 0, Math.PI * 2);
  ctx.fill();
}

function drawTechnomancer(
  ctx: CanvasRenderingContext2D,
  bx: number, _by: number,
  headCY: number, headR: number,
  shoulderY: number, hw: number, h: number,
  colors: { primary: string; secondary: string; accent: string },
  _scale: number,
) {
  // Antenna on head
  ctx.strokeStyle = colors.accent;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(bx, headCY - headR);
  ctx.lineTo(bx, headCY - headR * 1.8);
  ctx.stroke();
  // Antenna tip
  ctx.fillStyle = colors.secondary;
  ctx.beginPath();
  ctx.arc(bx, headCY - headR * 1.8, 2.5, 0, Math.PI * 2);
  ctx.fill();

  // Data visor across eyes
  ctx.fillStyle = rgba(colors.secondary, 0.6);
  ctx.beginPath();
  ctx.rect(bx - headR * 0.7, headCY - headR * 0.2, headR * 1.4, headR * 0.3);
  ctx.fill();
  // Visor glow
  ctx.strokeStyle = colors.accent;
  ctx.lineWidth = 0.5;
  ctx.stroke();

  // Circuit patterns on torso
  ctx.strokeStyle = rgba(colors.accent, 0.5);
  ctx.lineWidth = 0.8;
  const midTorsoY = shoulderY + h * 0.12;
  // Horizontal line
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.3, midTorsoY);
  ctx.lineTo(bx + hw * 0.3, midTorsoY);
  ctx.stroke();
  // Vertical branches
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.15, midTorsoY);
  ctx.lineTo(bx - hw * 0.15, midTorsoY + h * 0.06);
  ctx.moveTo(bx + hw * 0.15, midTorsoY);
  ctx.lineTo(bx + hw * 0.15, midTorsoY + h * 0.06);
  ctx.stroke();
  // Circuit nodes
  ctx.fillStyle = colors.accent;
  ctx.beginPath();
  ctx.arc(bx - hw * 0.15, midTorsoY + h * 0.06, 1.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(bx + hw * 0.15, midTorsoY + h * 0.06, 1.5, 0, Math.PI * 2);
  ctx.fill();
}

function drawBuilder(
  ctx: CanvasRenderingContext2D,
  bx: number, _by: number,
  headCY: number, headR: number,
  shoulderY: number, hw: number, h: number,
  colors: { primary: string; secondary: string; accent: string },
  _scale: number,
) {
  // Hard hat
  ctx.fillStyle = colors.accent;
  // Hat dome
  ctx.beginPath();
  ctx.arc(bx, headCY - headR * 0.3, headR * 1.1, Math.PI, 0);
  ctx.closePath();
  ctx.fill();
  // Hat brim
  ctx.fillStyle = colors.secondary;
  ctx.beginPath();
  ctx.rect(bx - headR * 1.3, headCY - headR * 0.3, headR * 2.6, headR * 0.2);
  ctx.fill();

  // Wrench in right hand
  const handY = shoulderY + h * 0.22;
  const handX = bx + hw * 0.65;
  ctx.strokeStyle = colors.accent;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(handX, handY);
  ctx.lineTo(handX + hw * 0.15, handY - h * 0.08);
  ctx.stroke();
  // Wrench head
  ctx.beginPath();
  ctx.arc(handX + hw * 0.15, handY - h * 0.08, 2, 0, Math.PI * 2);
  ctx.fill();

  // Tool belt
  const beltY = shoulderY + h * 0.25;
  ctx.fillStyle = colors.primary;
  ctx.beginPath();
  ctx.rect(bx - hw * 0.35, beltY, hw * 0.7, h * 0.04);
  ctx.fill();
  // Belt buckle
  ctx.fillStyle = colors.accent;
  ctx.beginPath();
  ctx.rect(bx - hw * 0.06, beltY, hw * 0.12, h * 0.04);
  ctx.fill();
}

// --- Overlays ---

function drawFrozenOverlay(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  scale: number,
) {
  const w = CHARACTER.baseWidth * scale;
  const h = CHARACTER.baseHeight * scale;
  const hw = w / 2;

  ctx.save();
  ctx.globalAlpha = 0.3;
  ctx.fillStyle = "#A8CFF5";
  // Cover the character silhouette
  ctx.beginPath();
  ctx.ellipse(bx, by - h * 0.5, hw * 0.6, h * 0.5, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;

  // Ice crystal outline
  ctx.strokeStyle = rgba("#A8CFF5", 0.6);
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.ellipse(bx, by - h * 0.5, hw * 0.7, h * 0.55, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

function drawPOrb(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  avgP: number,
  scale: number,
) {
  const h = CHARACTER.baseHeight * scale;
  const orbY = by - h - 8 * scale;
  const orbColor = pToHealthColor(avgP);
  const orbRadius = 3.5 * scale;

  // Orb
  ctx.fillStyle = orbColor;
  ctx.beginPath();
  ctx.arc(bx, orbY, orbRadius, 0, Math.PI * 2);
  ctx.fill();

  // Orb glow
  const grad = ctx.createRadialGradient(bx, orbY, 0, bx, orbY, orbRadius * 3);
  grad.addColorStop(0, rgba(orbColor, 0.4));
  grad.addColorStop(1, rgba(orbColor, 0));
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(bx, orbY, orbRadius * 3, 0, Math.PI * 2);
  ctx.fill();
}

function drawHoverOutline(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  scale: number,
) {
  const w = CHARACTER.baseWidth * scale;
  const h = CHARACTER.baseHeight * scale;
  const hw = w / 2;

  ctx.strokeStyle = rgba("#FFFFFF", 0.8);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.ellipse(bx, by - h * 0.5, hw * 0.65, h * 0.52, 0, 0, Math.PI * 2);
  ctx.stroke();
}

/** Get character hitbox bounds in screen space for click detection */
export function getCharacterBounds(agent: AgentVisual): { minX: number; minY: number; maxX: number; maxY: number } {
  const { x, y } = gridToScreen(agent.gridX, agent.gridY);
  const px = x + agent.walkOffsetX;
  const py = y + agent.walkOffsetY;
  const w = CHARACTER.baseWidth * agent.scale;
  const h = CHARACTER.baseHeight * agent.scale;
  return {
    minX: px - w / 2,
    minY: py - h - 10,
    maxX: px + w / 2,
    maxY: py,
  };
}
