import type { AgentVisual, RenderEntity } from "../types";
import { AGENT_COLORS, AGENT_MOTION, CHARACTER, TILE_WIDTH, TILE_HEIGHT } from "../constants";
import { gridToScreen } from "../isometric";
import { rgba, pToHealthColor } from "@/utils/color";
import { clamp, remap } from "@/utils/math";
import type { AgentType } from "@/data/types";
import { spriteRegistry } from "./sprite-registry";

// Eagerly start loading sprites (no-op if already initialized)
if (typeof window !== "undefined") {
  spriteRegistry.init();
}

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

  // Character body + type features — flip horizontally based on facing
  const walkDist = Math.sqrt(agent.walkOffsetX ** 2 + agent.walkOffsetY ** 2);
  const isWalking = walkDist > 0.5;
  const facing = agent.facing;

  // Try sprite rendering first; fall back to procedural if sprite not available
  const motion = AGENT_MOTION[agent.agentType];

  // Apply bob offset for sprites (procedural path computes its own bob)
  let spriteBobOffset = 0;
  if (!isWalking) {
    spriteBobOffset = motion.idleBob * Math.sin(Date.now() * 0.0012) * scale;
  } else {
    const stride = Math.min(1, walkDist / 80);
    const rawBob = Math.sin(agent.walkPhase * 2);
    const asymBob = rawBob + motion.bobAsymmetry * rawBob * rawBob;
    spriteBobOffset = Math.abs(asymBob) * motion.bobAmplitude * scale * stride;
  }

  const spriteDrawn = spriteRegistry.draw(
    ctx, agent.agentType,
    baseX, baseY - spriteBobOffset, scale, facing,
  );

  if (!spriteDrawn) {
    // Procedural fallback: full body + type features
    ctx.save();
    ctx.translate(baseX, 0);
    ctx.scale(facing, 1);
    ctx.translate(-baseX, 0);
    drawBody(ctx, baseX, baseY, agent.agentType, colors, scale, agent.walkPhase, isWalking, walkDist);
    drawTypeFeatures(ctx, baseX, baseY, agent.agentType, colors, scale);
    ctx.restore();
  } else {
    // Sprite drawn — add type features on top (halo, spikes, coins, etc.)
    const spriteBaseY = baseY - spriteBobOffset;
    ctx.save();
    ctx.translate(baseX, 0);
    ctx.scale(facing, 1);
    ctx.translate(-baseX, 0);
    drawTypeFeatures(ctx, baseX, spriteBaseY, agent.agentType, colors, scale);
    ctx.restore();
  }

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
  const glowSize = remap(resources, 0, 200, 0, 30);
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

/** Hair color per agent type (neon-tinted) */
const HAIR_COLORS: Record<AgentType, string> = {
  honest: "#4AE8A0",
  opportunistic: "#F5C84C",
  deceptive: "#D084F5",
  adversarial: "#FF6B6B",
  rlm: "#5BC8FF",
  crewai: "#7BF590",
};

/** Eye glow color per type */
const EYE_COLORS: Record<AgentType, string> = {
  honest: "#FFD966",
  opportunistic: "#FFB347",
  deceptive: "#FF80FF",
  adversarial: "#FF4444",
  rlm: "#80D4FF",
  crewai: "#FFEE80",
};

/** Darker variant for shading */
function darken(hex: string, amount: number): string {
  const c = hex.replace("#", "");
  const r = Math.max(0, parseInt(c.substring(0, 2), 16) - amount);
  const g = Math.max(0, parseInt(c.substring(2, 4), 16) - amount);
  const b = Math.max(0, parseInt(c.substring(4, 6), 16) - amount);
  return `rgb(${r},${g},${b})`;
}

function drawBody(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  agentType: AgentType,
  colors: { primary: string; secondary: string; accent: string },
  scale: number,
  walkPhase: number = 0,
  isWalking: boolean = false,
  walkDist: number = 0,
) {
  const motion = AGENT_MOTION[agentType];
  const w = CHARACTER.baseWidth * scale;
  const h = CHARACTER.baseHeight * scale;
  const hw = w / 2;

  // Stride intensity
  const stride = Math.min(1, walkDist / 80);

  // Asymmetric bob
  let bobOffset: number;
  if (isWalking) {
    const rawBob = Math.sin(walkPhase * 2);
    const asymBob = rawBob + motion.bobAsymmetry * rawBob * rawBob;
    bobOffset = Math.abs(asymBob) * motion.bobAmplitude * scale * stride;
  } else {
    bobOffset = motion.idleBob * Math.sin(Date.now() * 0.0012) * scale;
  }
  const bodyBy = by - bobOffset;

  const feetY = by;
  const headY = bodyBy - h;
  const neckY = headY + h * 0.22;
  const shoulderY = headY + h * 0.26;
  const chestY = headY + h * 0.38;
  const waistY = headY + h * 0.52;
  const hipY = headY + h * 0.58;
  const kneeY = headY + h * 0.78;
  const coatHemY = feetY - h * 0.04;

  const legSwing = isWalking ? Math.sin(walkPhase) * hw * 0.14 * motion.legSwingScale * stride : 0;
  const armSwing = isWalking ? -Math.sin(walkPhase) * 0.18 * motion.armSwingScale * stride : 0;

  const now = Date.now();

  // ═══ Ground projection (holographic glow on floor) ═══
  const shadowGrad = ctx.createRadialGradient(bx, feetY + 3, 0, bx, feetY + 3, hw * 1.2);
  shadowGrad.addColorStop(0, rgba(colors.secondary, 0.22));
  shadowGrad.addColorStop(0.5, rgba(colors.secondary, 0.08));
  shadowGrad.addColorStop(1, rgba(colors.secondary, 0));
  ctx.fillStyle = shadowGrad;
  ctx.beginPath();
  ctx.ellipse(bx, feetY + 3, hw * 1.0, hw * 0.28, 0, 0, Math.PI * 2);
  ctx.fill();

  // ═══ Legs (dark silhouette with subtle shading) ═══
  // Left leg
  const leftLegGrad = ctx.createLinearGradient(bx - hw * 0.25, hipY, bx - hw * 0.15, feetY);
  leftLegGrad.addColorStop(0, "#0E0E18");
  leftLegGrad.addColorStop(0.5, "#0A0A12");
  leftLegGrad.addColorStop(1, "#060610");
  ctx.fillStyle = leftLegGrad;
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.22, hipY);
  ctx.quadraticCurveTo(bx - hw * 0.28, kneeY, bx - hw * 0.30 + legSwing, feetY);
  ctx.lineTo(bx - hw * 0.02 + legSwing, feetY);
  ctx.quadraticCurveTo(bx - hw * 0.04, kneeY, bx - hw * 0.02, hipY);
  ctx.closePath();
  ctx.fill();
  // Left leg edge highlight
  ctx.strokeStyle = rgba(colors.secondary, 0.12);
  ctx.lineWidth = 0.6;
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.22, hipY);
  ctx.quadraticCurveTo(bx - hw * 0.28, kneeY, bx - hw * 0.30 + legSwing, feetY);
  ctx.stroke();

  // Right leg (darker, shadow side)
  ctx.fillStyle = "#060610";
  ctx.beginPath();
  ctx.moveTo(bx + hw * 0.02, hipY);
  ctx.quadraticCurveTo(bx + hw * 0.04, kneeY, bx + hw * 0.02 - legSwing, feetY);
  ctx.lineTo(bx + hw * 0.30 - legSwing, feetY);
  ctx.quadraticCurveTo(bx + hw * 0.28, kneeY, bx + hw * 0.22, hipY);
  ctx.closePath();
  ctx.fill();

  // Boot cuffs (small glow bands at ankles)
  ctx.fillStyle = rgba(colors.secondary, 0.2);
  ctx.beginPath();
  ctx.ellipse(bx - hw * 0.15 + legSwing, feetY - 1, hw * 0.16, 2, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(bx + hw * 0.15 - legSwing, feetY - 1, hw * 0.16, 2, 0, 0, Math.PI * 2);
  ctx.fill();

  // ═══ Torso (dark silhouette with chest shading) ═══
  const torsoGrad = ctx.createLinearGradient(bx - hw * 0.35, shoulderY, bx + hw * 0.35, waistY);
  torsoGrad.addColorStop(0, "#10101A");
  torsoGrad.addColorStop(0.5, "#0A0A14");
  torsoGrad.addColorStop(1, "#060610");
  ctx.fillStyle = torsoGrad;
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.38, shoulderY);
  ctx.lineTo(bx + hw * 0.38, shoulderY);
  ctx.lineTo(bx + hw * 0.28, waistY);
  ctx.quadraticCurveTo(bx, waistY + 2, bx - hw * 0.28, waistY);
  ctx.closePath();
  ctx.fill();

  // ═══ Holographic coat (multi-layer, translucent) ═══
  const coatSway = Math.sin(now * 0.002 + bx * 0.01) * hw * 0.05;
  const windRipple = Math.sin(now * 0.004 + bx * 0.02) * hw * 0.02;

  // Outer coat layer (main translucent coat)
  const coatGrad = ctx.createLinearGradient(bx, shoulderY - 4, bx, coatHemY);
  coatGrad.addColorStop(0, rgba(colors.secondary, 0.4));
  coatGrad.addColorStop(0.3, rgba(colors.secondary, 0.3));
  coatGrad.addColorStop(0.7, rgba(colors.secondary, 0.2));
  coatGrad.addColorStop(1, rgba(colors.secondary, 0.08));
  ctx.fillStyle = coatGrad;
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.48, shoulderY - 2);
  ctx.lineTo(bx + hw * 0.48, shoulderY - 2);
  ctx.quadraticCurveTo(bx + hw * 0.9 + coatSway, chestY, bx + hw * 0.85 + coatSway, waistY);
  ctx.quadraticCurveTo(bx + hw * 0.88 + coatSway + windRipple, hipY, bx + hw * 0.82 + coatSway, coatHemY);
  ctx.lineTo(bx - hw * 0.82 - coatSway, coatHemY);
  ctx.quadraticCurveTo(bx - hw * 0.88 - coatSway - windRipple, hipY, bx - hw * 0.85 - coatSway, waistY);
  ctx.quadraticCurveTo(bx - hw * 0.9 - coatSway, chestY, bx - hw * 0.48, shoulderY - 2);
  ctx.closePath();
  ctx.fill();

  // Inner coat lining (darker, slightly inset)
  const liningGrad = ctx.createLinearGradient(bx, shoulderY, bx, coatHemY);
  liningGrad.addColorStop(0, rgba(colors.primary, 0.2));
  liningGrad.addColorStop(1, rgba(colors.primary, 0.05));
  ctx.fillStyle = liningGrad;
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.1, shoulderY + h * 0.06);
  ctx.lineTo(bx + hw * 0.1, shoulderY + h * 0.06);
  ctx.lineTo(bx + hw * 0.25 + coatSway * 0.5, coatHemY);
  ctx.lineTo(bx - hw * 0.25 - coatSway * 0.5, coatHemY);
  ctx.closePath();
  ctx.fill();

  // Coat collar (raised, brighter)
  ctx.fillStyle = rgba(colors.secondary, 0.5);
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.38, shoulderY - 4);
  ctx.quadraticCurveTo(bx - hw * 0.45, shoulderY - 6, bx - hw * 0.42, neckY);
  ctx.lineTo(bx - hw * 0.2, neckY + 2);
  ctx.lineTo(bx - hw * 0.28, shoulderY - 2);
  ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(bx + hw * 0.38, shoulderY - 4);
  ctx.quadraticCurveTo(bx + hw * 0.45, shoulderY - 6, bx + hw * 0.42, neckY);
  ctx.lineTo(bx + hw * 0.2, neckY + 2);
  ctx.lineTo(bx + hw * 0.28, shoulderY - 2);
  ctx.closePath();
  ctx.fill();

  // Coat edge glow lines (bright edges on both sides)
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = rgba(colors.secondary, 0.55);
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.48, shoulderY - 2);
  ctx.quadraticCurveTo(bx - hw * 0.9 - coatSway, chestY, bx - hw * 0.85 - coatSway, waistY);
  ctx.quadraticCurveTo(bx - hw * 0.88 - coatSway, hipY, bx - hw * 0.82 - coatSway, coatHemY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(bx + hw * 0.48, shoulderY - 2);
  ctx.quadraticCurveTo(bx + hw * 0.9 + coatSway, chestY, bx + hw * 0.85 + coatSway, waistY);
  ctx.quadraticCurveTo(bx + hw * 0.88 + coatSway, hipY, bx + hw * 0.82 + coatSway, coatHemY);
  ctx.stroke();
  // Bottom hem glow
  ctx.lineWidth = 0.8;
  ctx.strokeStyle = rgba(colors.secondary, 0.3);
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.82 - coatSway, coatHemY);
  ctx.lineTo(bx + hw * 0.82 + coatSway, coatHemY);
  ctx.stroke();

  // Center seam with lapel detail
  ctx.strokeStyle = rgba(colors.secondary, 0.35);
  ctx.lineWidth = 0.6;
  ctx.beginPath();
  ctx.moveTo(bx, shoulderY + h * 0.04);
  ctx.lineTo(bx, coatHemY);
  ctx.stroke();
  // Lapel lines
  ctx.lineWidth = 0.5;
  ctx.strokeStyle = rgba(colors.secondary, 0.25);
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.05, shoulderY + h * 0.04);
  ctx.lineTo(bx - hw * 0.15, chestY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(bx + hw * 0.05, shoulderY + h * 0.04);
  ctx.lineTo(bx + hw * 0.15, chestY);
  ctx.stroke();

  // Holographic scanlines (fine, animated)
  ctx.strokeStyle = rgba(colors.secondary, 0.1);
  ctx.lineWidth = 0.4;
  const scanSpacing = h * 0.04;
  const scanPhase = now * 0.001;
  for (let sy = shoulderY + scanSpacing; sy < coatHemY; sy += scanSpacing) {
    const progress = (sy - shoulderY) / (coatHemY - shoulderY);
    const wave = Math.sin(scanPhase + progress * 6) * 0.02;
    const lineHW = hw * (0.48 + progress * 0.34 + wave);
    // Scanline brightness varies
    const scanAlpha = 0.06 + Math.sin(now * 0.005 + progress * 10) * 0.04;
    ctx.strokeStyle = rgba(colors.secondary, scanAlpha);
    ctx.beginPath();
    ctx.moveTo(bx - lineHW, sy);
    ctx.lineTo(bx + lineHW, sy);
    ctx.stroke();
  }

  // Sparkle dots on coat (holographic particles)
  ctx.save();
  const sparkleCount = 6;
  const sparklePhase = now * 0.002;
  for (let i = 0; i < sparkleCount; i++) {
    const t = (sparklePhase * 0.8 + i * 1.3) % 2.5;
    if (t > 1) continue;
    const sy = shoulderY + (coatHemY - shoulderY) * ((i * 0.19 + sparklePhase * 0.07) % 1);
    const sx = bx + Math.sin(i * 2.7 + now * 0.0015) * hw * 0.5;
    const sparkleAlpha = Math.sin(t * Math.PI) * 0.7;
    const sparkleR = (1 + Math.sin(i * 3.1 + now * 0.003) * 0.5) * scale;
    ctx.globalAlpha = sparkleAlpha;
    ctx.fillStyle = colors.accent;
    ctx.beginPath();
    ctx.arc(sx, sy, sparkleR, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  ctx.restore();

  // ═══ Coat sleeves / arms ═══
  // Back arm (behind torso)
  const backArmEndX = bx + hw * (0.72) - armSwing * hw;
  const backArmEndY = shoulderY + h * 0.26;
  const backSleeveGrad = ctx.createLinearGradient(bx + hw * 0.45, shoulderY, backArmEndX, backArmEndY);
  backSleeveGrad.addColorStop(0, rgba(colors.secondary, 0.3));
  backSleeveGrad.addColorStop(1, rgba(colors.secondary, 0.15));
  ctx.fillStyle = backSleeveGrad;
  ctx.beginPath();
  ctx.moveTo(bx + hw * 0.45, shoulderY);
  ctx.quadraticCurveTo(bx + hw * 0.6, shoulderY + h * 0.1, backArmEndX, backArmEndY);
  ctx.lineTo(backArmEndX - hw * 0.12, backArmEndY + h * 0.02);
  ctx.quadraticCurveTo(bx + hw * 0.5, shoulderY + h * 0.12, bx + hw * 0.35, shoulderY + h * 0.04);
  ctx.closePath();
  ctx.fill();
  // Sleeve edge
  ctx.strokeStyle = rgba(colors.secondary, 0.3);
  ctx.lineWidth = 0.7;
  ctx.beginPath();
  ctx.moveTo(bx + hw * 0.45, shoulderY);
  ctx.quadraticCurveTo(bx + hw * 0.6, shoulderY + h * 0.1, backArmEndX, backArmEndY);
  ctx.stroke();
  // Hand glow (warm golden)
  const handGlow = ctx.createRadialGradient(backArmEndX - hw * 0.06, backArmEndY, 0, backArmEndX - hw * 0.06, backArmEndY, 4 * scale);
  handGlow.addColorStop(0, rgba(EYE_COLORS[agentType], 0.7));
  handGlow.addColorStop(1, rgba(EYE_COLORS[agentType], 0));
  ctx.fillStyle = handGlow;
  ctx.beginPath();
  ctx.arc(backArmEndX - hw * 0.06, backArmEndY, 4 * scale, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = rgba(EYE_COLORS[agentType], 0.8);
  ctx.beginPath();
  ctx.arc(backArmEndX - hw * 0.06, backArmEndY, 1.8 * scale, 0, Math.PI * 2);
  ctx.fill();

  // Front arm (in front of torso)
  const frontArmEndX = bx - hw * (0.72) + armSwing * hw;
  const frontArmEndY = shoulderY + h * 0.26;
  const frontSleeveGrad = ctx.createLinearGradient(bx - hw * 0.45, shoulderY, frontArmEndX, frontArmEndY);
  frontSleeveGrad.addColorStop(0, rgba(colors.secondary, 0.35));
  frontSleeveGrad.addColorStop(1, rgba(colors.secondary, 0.18));
  ctx.fillStyle = frontSleeveGrad;
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.45, shoulderY);
  ctx.quadraticCurveTo(bx - hw * 0.6, shoulderY + h * 0.1, frontArmEndX, frontArmEndY);
  ctx.lineTo(frontArmEndX + hw * 0.12, frontArmEndY + h * 0.02);
  ctx.quadraticCurveTo(bx - hw * 0.5, shoulderY + h * 0.12, bx - hw * 0.35, shoulderY + h * 0.04);
  ctx.closePath();
  ctx.fill();
  // Sleeve edge
  ctx.strokeStyle = rgba(colors.secondary, 0.35);
  ctx.lineWidth = 0.7;
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.45, shoulderY);
  ctx.quadraticCurveTo(bx - hw * 0.6, shoulderY + h * 0.1, frontArmEndX, frontArmEndY);
  ctx.stroke();
  // Hand glow
  const handGlow2 = ctx.createRadialGradient(frontArmEndX + hw * 0.06, frontArmEndY, 0, frontArmEndX + hw * 0.06, frontArmEndY, 4 * scale);
  handGlow2.addColorStop(0, rgba(EYE_COLORS[agentType], 0.7));
  handGlow2.addColorStop(1, rgba(EYE_COLORS[agentType], 0));
  ctx.fillStyle = handGlow2;
  ctx.beginPath();
  ctx.arc(frontArmEndX + hw * 0.06, frontArmEndY, 4 * scale, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = rgba(EYE_COLORS[agentType], 0.8);
  ctx.beginPath();
  ctx.arc(frontArmEndX + hw * 0.06, frontArmEndY, 1.8 * scale, 0, Math.PI * 2);
  ctx.fill();

  // ═══ Neck ═══
  ctx.fillStyle = "#0A0A12";
  ctx.beginPath();
  ctx.rect(bx - hw * 0.1, neckY, hw * 0.2, shoulderY - neckY + 2);
  ctx.fill();

  // ═══ Head (dark sphere with shading) ═══
  const headRadius = hw * 0.36;
  const headCenterY = headY + headRadius + h * 0.03;
  // Head shadow layer
  const headGrad = ctx.createRadialGradient(
    bx - headRadius * 0.3, headCenterY - headRadius * 0.2, headRadius * 0.1,
    bx, headCenterY, headRadius,
  );
  headGrad.addColorStop(0, "#14141E");
  headGrad.addColorStop(0.7, "#0A0A14");
  headGrad.addColorStop(1, "#06060E");
  ctx.fillStyle = headGrad;
  ctx.beginPath();
  ctx.arc(bx, headCenterY, headRadius, 0, Math.PI * 2);
  ctx.fill();

  // ═══ Hair (detailed bob cut with highlights) ═══
  const hairColor = HAIR_COLORS[agentType];
  const hairDark = darken(hairColor, 60);
  // Hair base dome
  const hairGrad = ctx.createRadialGradient(
    bx - headRadius * 0.2, headCenterY - headRadius * 0.5, headRadius * 0.2,
    bx, headCenterY - headRadius * 0.2, headRadius * 1.15,
  );
  hairGrad.addColorStop(0, hairColor);
  hairGrad.addColorStop(0.7, hairColor);
  hairGrad.addColorStop(1, hairDark);
  ctx.fillStyle = hairGrad;
  ctx.beginPath();
  ctx.arc(bx, headCenterY - headRadius * 0.12, headRadius * 1.08, Math.PI * 1.05, Math.PI * -0.05);
  ctx.closePath();
  ctx.fill();

  // Hair side panels (bob shape with curves)
  // Left side
  ctx.fillStyle = hairDark;
  ctx.beginPath();
  ctx.moveTo(bx - headRadius * 1.08, headCenterY - headRadius * 0.12);
  ctx.quadraticCurveTo(bx - headRadius * 1.12, headCenterY + headRadius * 0.2, bx - headRadius * 1.0, headCenterY + headRadius * 0.55);
  ctx.quadraticCurveTo(bx - headRadius * 0.85, headCenterY + headRadius * 0.7, bx - headRadius * 0.65, headCenterY + headRadius * 0.65);
  ctx.lineTo(bx - headRadius * 0.88, headCenterY);
  ctx.closePath();
  ctx.fill();
  // Right side
  ctx.beginPath();
  ctx.moveTo(bx + headRadius * 1.08, headCenterY - headRadius * 0.12);
  ctx.quadraticCurveTo(bx + headRadius * 1.12, headCenterY + headRadius * 0.2, bx + headRadius * 1.0, headCenterY + headRadius * 0.55);
  ctx.quadraticCurveTo(bx + headRadius * 0.85, headCenterY + headRadius * 0.7, bx + headRadius * 0.65, headCenterY + headRadius * 0.65);
  ctx.lineTo(bx + headRadius * 0.88, headCenterY);
  ctx.closePath();
  ctx.fill();
  // Lit side highlight
  ctx.fillStyle = hairColor;
  ctx.beginPath();
  ctx.moveTo(bx - headRadius * 1.06, headCenterY - headRadius * 0.1);
  ctx.quadraticCurveTo(bx - headRadius * 1.08, headCenterY + headRadius * 0.1, bx - headRadius * 0.98, headCenterY + headRadius * 0.45);
  ctx.lineTo(bx - headRadius * 0.9, headCenterY + headRadius * 0.1);
  ctx.closePath();
  ctx.fill();

  // Hair specular highlight (shiny band)
  ctx.fillStyle = rgba("#FFFFFF", 0.2);
  ctx.beginPath();
  ctx.ellipse(bx - headRadius * 0.15, headCenterY - headRadius * 0.55, headRadius * 0.5, headRadius * 0.12, -0.2, 0, Math.PI * 2);
  ctx.fill();

  // Hair glow aura
  const hairAura = ctx.createRadialGradient(bx, headCenterY - headRadius * 0.3, 0, bx, headCenterY - headRadius * 0.3, headRadius * 1.8);
  hairAura.addColorStop(0, rgba(hairColor, 0.25));
  hairAura.addColorStop(1, rgba(hairColor, 0));
  ctx.fillStyle = hairAura;
  ctx.beginPath();
  ctx.arc(bx, headCenterY - headRadius * 0.3, headRadius * 1.8, 0, Math.PI * 2);
  ctx.fill();

  // ═══ Face features ═══
  const eyeColor = EYE_COLORS[agentType];
  const eyeY = headCenterY + headRadius * 0.08;
  const eyeSpacing = headRadius * 0.36;
  const eyeW = headRadius * 0.2;
  const eyeH = headRadius * 0.16;

  // Eye glow halo
  const eyeGlow = ctx.createRadialGradient(bx, eyeY, 0, bx, eyeY, headRadius * 0.9);
  eyeGlow.addColorStop(0, rgba(eyeColor, 0.2));
  eyeGlow.addColorStop(1, rgba(eyeColor, 0));
  ctx.fillStyle = eyeGlow;
  ctx.beginPath();
  ctx.arc(bx, eyeY, headRadius * 0.9, 0, Math.PI * 2);
  ctx.fill();

  // Left eye (oval with gradient)
  const leftEyeGrad = ctx.createRadialGradient(
    bx - eyeSpacing, eyeY, eyeW * 0.2,
    bx - eyeSpacing, eyeY, eyeW,
  );
  leftEyeGrad.addColorStop(0, "#FFFFFF");
  leftEyeGrad.addColorStop(0.3, eyeColor);
  leftEyeGrad.addColorStop(1, rgba(eyeColor, 0.6));
  ctx.fillStyle = leftEyeGrad;
  ctx.beginPath();
  ctx.ellipse(bx - eyeSpacing, eyeY, eyeW, eyeH, 0, 0, Math.PI * 2);
  ctx.fill();
  // Eye glow ring
  ctx.strokeStyle = rgba(eyeColor, 0.5);
  ctx.lineWidth = 0.6;
  ctx.beginPath();
  ctx.ellipse(bx - eyeSpacing, eyeY, eyeW * 1.3, eyeH * 1.3, 0, 0, Math.PI * 2);
  ctx.stroke();

  // Right eye
  const rightEyeGrad = ctx.createRadialGradient(
    bx + eyeSpacing, eyeY, eyeW * 0.2,
    bx + eyeSpacing, eyeY, eyeW,
  );
  rightEyeGrad.addColorStop(0, "#FFFFFF");
  rightEyeGrad.addColorStop(0.3, eyeColor);
  rightEyeGrad.addColorStop(1, rgba(eyeColor, 0.6));
  ctx.fillStyle = rightEyeGrad;
  ctx.beginPath();
  ctx.ellipse(bx + eyeSpacing, eyeY, eyeW, eyeH, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = rgba(eyeColor, 0.5);
  ctx.lineWidth = 0.6;
  ctx.beginPath();
  ctx.ellipse(bx + eyeSpacing, eyeY, eyeW * 1.3, eyeH * 1.3, 0, 0, Math.PI * 2);
  ctx.stroke();

  // Subtle smile
  ctx.strokeStyle = rgba(eyeColor, 0.35);
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  ctx.arc(bx, eyeY + headRadius * 0.28, headRadius * 0.22, 0.15 * Math.PI, 0.85 * Math.PI);
  ctx.stroke();

  // Nose hint
  ctx.strokeStyle = rgba(eyeColor, 0.15);
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(bx, eyeY + headRadius * 0.02);
  ctx.lineTo(bx - headRadius * 0.05, eyeY + headRadius * 0.15);
  ctx.stroke();
}

/** Animated arms drawn on top of sprites — extracted from drawBody */
function drawArms(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  agentType: AgentType,
  colors: { primary: string; secondary: string; accent: string },
  scale: number,
  walkPhase: number = 0,
  isWalking: boolean = false,
  walkDist: number = 0,
) {
  const motion = AGENT_MOTION[agentType];
  const w = CHARACTER.baseWidth * scale;
  const h = CHARACTER.baseHeight * scale;
  const hw = w / 2;

  const stride = Math.min(1, walkDist / 80);
  const headY = by - h;
  const shoulderY = headY + h * 0.26;

  const armSwing = isWalking ? -Math.sin(walkPhase) * 0.18 * motion.armSwingScale * stride : 0;

  // Back arm (behind torso)
  const backArmEndX = bx + hw * (0.72) - armSwing * hw;
  const backArmEndY = shoulderY + h * 0.26;
  const backSleeveGrad = ctx.createLinearGradient(bx + hw * 0.45, shoulderY, backArmEndX, backArmEndY);
  backSleeveGrad.addColorStop(0, rgba(colors.secondary, 0.3));
  backSleeveGrad.addColorStop(1, rgba(colors.secondary, 0.15));
  ctx.fillStyle = backSleeveGrad;
  ctx.beginPath();
  ctx.moveTo(bx + hw * 0.45, shoulderY);
  ctx.quadraticCurveTo(bx + hw * 0.6, shoulderY + h * 0.1, backArmEndX, backArmEndY);
  ctx.lineTo(backArmEndX - hw * 0.12, backArmEndY + h * 0.02);
  ctx.quadraticCurveTo(bx + hw * 0.5, shoulderY + h * 0.12, bx + hw * 0.35, shoulderY + h * 0.04);
  ctx.closePath();
  ctx.fill();
  // Sleeve edge
  ctx.strokeStyle = rgba(colors.secondary, 0.3);
  ctx.lineWidth = 0.7;
  ctx.beginPath();
  ctx.moveTo(bx + hw * 0.45, shoulderY);
  ctx.quadraticCurveTo(bx + hw * 0.6, shoulderY + h * 0.1, backArmEndX, backArmEndY);
  ctx.stroke();
  // Hand glow
  const handGlow = ctx.createRadialGradient(backArmEndX - hw * 0.06, backArmEndY, 0, backArmEndX - hw * 0.06, backArmEndY, 4 * scale);
  handGlow.addColorStop(0, rgba(EYE_COLORS[agentType], 0.7));
  handGlow.addColorStop(1, rgba(EYE_COLORS[agentType], 0));
  ctx.fillStyle = handGlow;
  ctx.beginPath();
  ctx.arc(backArmEndX - hw * 0.06, backArmEndY, 4 * scale, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = rgba(EYE_COLORS[agentType], 0.8);
  ctx.beginPath();
  ctx.arc(backArmEndX - hw * 0.06, backArmEndY, 1.8 * scale, 0, Math.PI * 2);
  ctx.fill();

  // Front arm (in front of torso)
  const frontArmEndX = bx - hw * (0.72) + armSwing * hw;
  const frontArmEndY = shoulderY + h * 0.26;
  const frontSleeveGrad = ctx.createLinearGradient(bx - hw * 0.45, shoulderY, frontArmEndX, frontArmEndY);
  frontSleeveGrad.addColorStop(0, rgba(colors.secondary, 0.35));
  frontSleeveGrad.addColorStop(1, rgba(colors.secondary, 0.18));
  ctx.fillStyle = frontSleeveGrad;
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.45, shoulderY);
  ctx.quadraticCurveTo(bx - hw * 0.6, shoulderY + h * 0.1, frontArmEndX, frontArmEndY);
  ctx.lineTo(frontArmEndX + hw * 0.12, frontArmEndY + h * 0.02);
  ctx.quadraticCurveTo(bx - hw * 0.5, shoulderY + h * 0.12, bx - hw * 0.35, shoulderY + h * 0.04);
  ctx.closePath();
  ctx.fill();
  // Sleeve edge
  ctx.strokeStyle = rgba(colors.secondary, 0.35);
  ctx.lineWidth = 0.7;
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.45, shoulderY);
  ctx.quadraticCurveTo(bx - hw * 0.6, shoulderY + h * 0.1, frontArmEndX, frontArmEndY);
  ctx.stroke();
  // Hand glow
  const handGlow2 = ctx.createRadialGradient(frontArmEndX + hw * 0.06, frontArmEndY, 0, frontArmEndX + hw * 0.06, frontArmEndY, 4 * scale);
  handGlow2.addColorStop(0, rgba(EYE_COLORS[agentType], 0.7));
  handGlow2.addColorStop(1, rgba(EYE_COLORS[agentType], 0));
  ctx.fillStyle = handGlow2;
  ctx.beginPath();
  ctx.arc(frontArmEndX + hw * 0.06, frontArmEndY, 4 * scale, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = rgba(EYE_COLORS[agentType], 0.8);
  ctx.beginPath();
  ctx.arc(frontArmEndX + hw * 0.06, frontArmEndY, 1.8 * scale, 0, Math.PI * 2);
  ctx.fill();
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
  const headRadius = hw * 0.36;
  const headCenterY = headY + headRadius + h * 0.03;
  const shoulderY = headY + h * 0.26;

  switch (agentType) {
    case "honest":
      drawHoloHalo(ctx, bx, headCenterY, headRadius, colors);
      break;
    case "opportunistic":
      drawHoloCoins(ctx, bx, by, hw, h, shoulderY, colors);
      break;
    case "deceptive":
      drawHoloMask(ctx, bx, headCenterY, headRadius, colors);
      break;
    case "adversarial":
      drawHoloSpikes(ctx, bx, headCenterY, headRadius, shoulderY, hw, colors);
      break;
    case "rlm":
      drawHoloCircuit(ctx, bx, headCenterY, headRadius, shoulderY, hw, h, colors);
      break;
    case "crewai":
      drawHoloTools(ctx, bx, headCenterY, headRadius, shoulderY, hw, h, colors);
      break;
  }
}

// --- Type-specific holographic accents ---

function drawHoloHalo(
  ctx: CanvasRenderingContext2D,
  bx: number,
  headCY: number, headR: number,
  colors: { accent: string },
) {
  const haloY = headCY - headR * 1.4;
  const pulse = Math.sin(Date.now() * 0.003) * 0.2;
  // Outer glow
  ctx.strokeStyle = rgba(colors.accent, 0.25 + pulse);
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.ellipse(bx, haloY, headR * 0.9, headR * 0.28, 0, 0, Math.PI * 2);
  ctx.stroke();
  // Main ring
  ctx.strokeStyle = rgba(colors.accent, 0.6 + pulse);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.ellipse(bx, haloY, headR * 0.9, headR * 0.28, 0, 0, Math.PI * 2);
  ctx.stroke();
}

function drawHoloCoins(
  ctx: CanvasRenderingContext2D,
  bx: number, by: number,
  hw: number, _h: number,
  shoulderY: number,
  colors: { accent: string; secondary: string },
) {
  const now = Date.now() * 0.002;
  const hipY = (shoulderY + by) * 0.5;
  for (let i = 0; i < 4; i++) {
    const angle = now + i * (Math.PI * 2 / 4);
    const ox = bx + Math.cos(angle) * hw * 0.8;
    const oy = hipY + Math.sin(angle) * hw * 0.3;
    const a = 0.5 + Math.sin(now + i) * 0.2;
    // Glow
    ctx.fillStyle = rgba(colors.accent, a * 0.3);
    ctx.beginPath();
    ctx.arc(ox, oy, 4, 0, Math.PI * 2);
    ctx.fill();
    // Core
    ctx.fillStyle = rgba(colors.accent, a);
    ctx.beginPath();
    ctx.arc(ox, oy, 2, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawHoloMask(
  ctx: CanvasRenderingContext2D,
  bx: number,
  headCY: number, headR: number,
  colors: { secondary: string; accent: string },
) {
  ctx.strokeStyle = rgba(colors.accent, 0.5);
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(bx - headR * 0.75, headCY - headR * 0.2);
  ctx.quadraticCurveTo(bx, headCY - headR * 0.5, bx + headR * 0.75, headCY - headR * 0.2);
  ctx.lineTo(bx + headR * 0.55, headCY + headR * 0.18);
  ctx.quadraticCurveTo(bx, headCY + headR * 0.12, bx - headR * 0.55, headCY + headR * 0.18);
  ctx.closePath();
  ctx.stroke();
  ctx.fillStyle = rgba(colors.secondary, 0.08);
  ctx.fill();
}

function drawHoloSpikes(
  ctx: CanvasRenderingContext2D,
  bx: number,
  headCY: number, headR: number,
  shoulderY: number, hw: number,
  colors: { secondary: string; accent: string },
) {
  ctx.lineWidth = 2;
  // Left spike
  const lGrad = ctx.createLinearGradient(bx - headR * 0.6, headCY, bx - headR * 1.2, headCY - headR * 1.8);
  lGrad.addColorStop(0, rgba(colors.secondary, 0.3));
  lGrad.addColorStop(1, rgba(colors.secondary, 0.9));
  ctx.strokeStyle = lGrad;
  ctx.beginPath();
  ctx.moveTo(bx - headR * 0.6, headCY - headR * 0.5);
  ctx.lineTo(bx - headR * 1.2, headCY - headR * 1.8);
  ctx.stroke();
  // Right spike
  const rGrad = ctx.createLinearGradient(bx + headR * 0.6, headCY, bx + headR * 1.2, headCY - headR * 1.8);
  rGrad.addColorStop(0, rgba(colors.secondary, 0.3));
  rGrad.addColorStop(1, rgba(colors.secondary, 0.9));
  ctx.strokeStyle = rGrad;
  ctx.beginPath();
  ctx.moveTo(bx + headR * 0.6, headCY - headR * 0.5);
  ctx.lineTo(bx + headR * 1.2, headCY - headR * 1.8);
  ctx.stroke();
  // Tips
  ctx.fillStyle = rgba(colors.secondary, 0.9);
  ctx.beginPath(); ctx.arc(bx - headR * 1.2, headCY - headR * 1.8, 2, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(bx + headR * 1.2, headCY - headR * 1.8, 2, 0, Math.PI * 2); ctx.fill();
  // Shoulder spikes
  ctx.strokeStyle = rgba(colors.accent, 0.5);
  ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(bx - hw * 0.5, shoulderY); ctx.lineTo(bx - hw * 0.75, shoulderY - hw * 0.35); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(bx + hw * 0.5, shoulderY); ctx.lineTo(bx + hw * 0.75, shoulderY - hw * 0.35); ctx.stroke();
}

function drawHoloCircuit(
  ctx: CanvasRenderingContext2D,
  bx: number,
  headCY: number, headR: number,
  shoulderY: number, hw: number, h: number,
  colors: { secondary: string; accent: string },
) {
  // Antenna
  ctx.strokeStyle = rgba(colors.accent, 0.7);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(bx, headCY - headR);
  ctx.lineTo(bx, headCY - headR * 2.0);
  ctx.stroke();
  const pulse = Math.sin(Date.now() * 0.005) * 0.4 + 0.6;
  ctx.fillStyle = rgba(colors.accent, pulse);
  ctx.beginPath(); ctx.arc(bx, headCY - headR * 2.0, 2.5, 0, Math.PI * 2); ctx.fill();
  // Glow ring around tip
  ctx.strokeStyle = rgba(colors.accent, pulse * 0.3);
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(bx, headCY - headR * 2.0, 5, 0, Math.PI * 2); ctx.stroke();

  // Data visor
  ctx.strokeStyle = rgba(colors.accent, 0.5);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(bx - headR * 0.7, headCY - headR * 0.12, headR * 1.4, headR * 0.24);
  ctx.stroke();
  ctx.fillStyle = rgba(colors.secondary, 0.12);
  ctx.fill();

  // Coat circuit traces
  ctx.strokeStyle = rgba(colors.accent, 0.3);
  ctx.lineWidth = 0.7;
  const midY = shoulderY + h * 0.12;
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.35, midY);
  ctx.lineTo(bx + hw * 0.35, midY);
  ctx.moveTo(bx - hw * 0.15, midY);
  ctx.lineTo(bx - hw * 0.15, midY + h * 0.08);
  ctx.moveTo(bx + hw * 0.15, midY);
  ctx.lineTo(bx + hw * 0.15, midY + h * 0.08);
  ctx.moveTo(bx, midY);
  ctx.lineTo(bx, midY + h * 0.1);
  ctx.stroke();
  // Nodes
  ctx.fillStyle = rgba(colors.accent, 0.6);
  ctx.beginPath(); ctx.arc(bx - hw * 0.15, midY + h * 0.08, 1.5, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(bx + hw * 0.15, midY + h * 0.08, 1.5, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(bx, midY + h * 0.1, 1.5, 0, Math.PI * 2); ctx.fill();
}

function drawHoloTools(
  ctx: CanvasRenderingContext2D,
  bx: number,
  headCY: number, headR: number,
  shoulderY: number, hw: number, h: number,
  colors: { secondary: string; accent: string },
) {
  // Hard hat outline
  ctx.strokeStyle = rgba(colors.accent, 0.6);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(bx, headCY - headR * 0.2, headR * 1.1, Math.PI, 0);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(bx - headR * 1.3, headCY - headR * 0.2);
  ctx.lineTo(bx + headR * 1.3, headCY - headR * 0.2);
  ctx.stroke();

  // Tool belt
  const beltY = shoulderY + h * 0.25;
  ctx.strokeStyle = rgba(colors.accent, 0.45);
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(bx - hw * 0.4, beltY);
  ctx.lineTo(bx + hw * 0.4, beltY);
  ctx.stroke();
  ctx.fillStyle = rgba(colors.accent, 0.7);
  ctx.beginPath(); ctx.arc(bx, beltY, 2, 0, Math.PI * 2); ctx.fill();
  // Tool dots
  ctx.fillStyle = rgba(colors.accent, 0.4);
  ctx.beginPath(); ctx.arc(bx - hw * 0.2, beltY, 1.2, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(bx + hw * 0.2, beltY, 1.2, 0, Math.PI * 2); ctx.fill();
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
  ctx.beginPath();
  ctx.ellipse(bx, by - h * 0.5, hw * 0.6, h * 0.5, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;

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
  const orbY = by - h - 10 * scale;
  const orbColor = pToHealthColor(avgP);
  const orbRadius = 4 * scale;

  // Outer glow
  const outerGlow = ctx.createRadialGradient(bx, orbY, 0, bx, orbY, orbRadius * 4);
  outerGlow.addColorStop(0, rgba(orbColor, 0.35));
  outerGlow.addColorStop(1, rgba(orbColor, 0));
  ctx.fillStyle = outerGlow;
  ctx.beginPath();
  ctx.arc(bx, orbY, orbRadius * 4, 0, Math.PI * 2);
  ctx.fill();

  // Orb body
  const orbGrad = ctx.createRadialGradient(
    bx - orbRadius * 0.3, orbY - orbRadius * 0.3, orbRadius * 0.1,
    bx, orbY, orbRadius,
  );
  orbGrad.addColorStop(0, "#FFFFFF");
  orbGrad.addColorStop(0.3, orbColor);
  orbGrad.addColorStop(1, rgba(orbColor, 0.7));
  ctx.fillStyle = orbGrad;
  ctx.beginPath();
  ctx.arc(bx, orbY, orbRadius, 0, Math.PI * 2);
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

  // Outer glow
  ctx.strokeStyle = rgba("#FFFFFF", 0.3);
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.ellipse(bx, by - h * 0.5, hw * 0.7, h * 0.54, 0, 0, Math.PI * 2);
  ctx.stroke();
  // Inner crisp line
  ctx.strokeStyle = rgba("#FFFFFF", 0.8);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.ellipse(bx, by - h * 0.5, hw * 0.68, h * 0.52, 0, 0, Math.PI * 2);
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
