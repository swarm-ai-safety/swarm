import { TILE_WIDTH, TILE_HEIGHT, COLORS } from "../constants";
import { gridToScreen } from "../isometric";
import { rgba, lerpColor } from "@/utils/color";
import type { RenderEntity } from "../types";
import type { EnvironmentState } from "../systems/environment-system";
import type { EpochSnapshot } from "@/data/types";
import { spriteRegistry } from "./sprite-registry";

export interface BuildingDef {
  gridX: number;
  gridY: number;
  type: "tower" | "spire" | "node";
  seed: number;
}

// --- Deterministic PRNG (mulberry32) ---

function mulberry32(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- Layout computation (cached) ---

let cachedLayout: BuildingDef[] | null = null;
let cachedGridSize = -1;

export function computeBuildingLayout(
  gridSize: number,
  agentPositions: Set<string>,
): BuildingDef[] {
  if (cachedGridSize === gridSize && cachedLayout) return cachedLayout;

  const rng = mulberry32(7919); // fixed seed for determinism
  const buildings: BuildingDef[] = [];
  const types: BuildingDef["type"][] = ["tower", "spire", "node"];

  for (let gx = 0; gx < gridSize; gx++) {
    for (let gy = 0; gy < gridSize; gy++) {
      // Skip agent-occupied cells
      if (agentPositions.has(`${gx},${gy}`)) continue;

      // ~1 building per 6-8 empty tiles
      if (rng() > 0.15) continue;

      buildings.push({
        gridX: gx,
        gridY: gy,
        type: types[Math.floor(rng() * types.length)],
        seed: Math.floor(rng() * 10000),
      });
    }
  }

  cachedLayout = buildings;
  cachedGridSize = gridSize;
  return buildings;
}

// --- Building entity factory ---

export function createBuildingEntity(
  building: BuildingDef,
  env: EnvironmentState,
  epoch: EpochSnapshot | null,
): RenderEntity {
  return {
    depth: building.gridX + building.gridY - 0.3,
    render: (ctx) => renderBuilding(ctx, building, env, epoch),
  };
}

// --- Drawing ---

function renderBuilding(
  ctx: CanvasRenderingContext2D,
  b: BuildingDef,
  env: EnvironmentState,
  epoch: EpochSnapshot | null,
) {
  const pos = gridToScreen(b.gridX, b.gridY);
  const cx = pos.x;
  const cy = pos.y;

  // Welfare drives building height and brightness
  const welfare = epoch ? Math.max(0, Math.min(1, (epoch.total_welfare + 50) / 100)) : 0.5;
  const threat = env.threatLevel;

  // Glow color: accent when safe, shifts to alert when threatened
  const glowHex = threat > 0.4
    ? lerpColor(COLORS.accent, COLORS.alert, Math.min((threat - 0.4) / 0.4, 1))
    : COLORS.accent;

  // Overall alpha dims when welfare is low
  const baseAlpha = 0.4 + welfare * 0.6;

  ctx.save();

  switch (b.type) {
    case "tower":
      drawTower(ctx, cx, cy, welfare, glowHex, baseAlpha, b.seed);
      break;
    case "spire":
      drawSpire(ctx, cx, cy, epoch, glowHex, baseAlpha, b.seed);
      break;
    case "node":
      drawNode(ctx, cx, cy, threat, glowHex, baseAlpha, b.seed);
      break;
  }

  ctx.restore();
}

// --- Data Tower ---

function drawTower(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  welfare: number,
  glowHex: string,
  alpha: number,
  seed: number,
) {
  const hw = TILE_WIDTH * 0.22;
  const hh = TILE_HEIGHT * 0.22;
  const height = 20 + welfare * 100; // 20-120px
  const drawWidth = hw * 2 + 2; // ~42px

  // Try sprite base first
  const spriteDrawn = spriteRegistry.drawBuilding(ctx, "tower", cx, cy, drawWidth, height);

  if (!spriteDrawn) {
    // Full procedural fallback
    // Right face (lighter)
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + hw, cy - hh);
    ctx.lineTo(cx + hw, cy - hh - height);
    ctx.lineTo(cx, cy - height);
    ctx.closePath();
    ctx.fillStyle = rgba("#1A2233", alpha * 0.9);
    ctx.fill();
    ctx.strokeStyle = rgba(glowHex, alpha * 0.3);
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Left face (darker)
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx - hw, cy - hh);
    ctx.lineTo(cx - hw, cy - hh - height);
    ctx.lineTo(cx, cy - height);
    ctx.closePath();
    ctx.fillStyle = rgba("#0F1620", alpha * 0.9);
    ctx.fill();
    ctx.strokeStyle = rgba(glowHex, alpha * 0.25);
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Top diamond
    ctx.beginPath();
    ctx.moveTo(cx, cy - height - hh);
    ctx.lineTo(cx + hw, cy - hh - height);
    ctx.lineTo(cx, cy - height);
    ctx.lineTo(cx - hw, cy - hh - height);
    ctx.closePath();
    ctx.fillStyle = rgba("#243040", alpha * 0.9);
    ctx.fill();
    ctx.strokeStyle = rgba(glowHex, alpha * 0.4);
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }

  // Procedural overlays (on top of sprite or procedural base)

  // Window grid on right face
  const rng = mulberry32(seed);
  const windowRows = Math.max(2, Math.floor(height / 14));
  const windowCols = 2;
  for (let r = 0; r < windowRows; r++) {
    for (let c = 0; c < windowCols; c++) {
      const wy = cy - 6 - r * 12;
      const wx = cx + 3 + c * 8;
      if (wy < cy - height + 4) continue;
      const lit = rng() > 0.3;
      if (lit) {
        ctx.fillStyle = rgba(glowHex, alpha * (0.4 + rng() * 0.4));
        ctx.fillRect(wx, wy, 4, 3);
      }
    }
  }

  // Window grid on left face
  for (let r = 0; r < windowRows; r++) {
    for (let c = 0; c < windowCols; c++) {
      const wy = cy - 6 - r * 12;
      const wx = cx - 7 - c * 8;
      if (wy < cy - height + 4) continue;
      const lit = rng() > 0.3;
      if (lit) {
        ctx.fillStyle = rgba(glowHex, alpha * (0.3 + rng() * 0.3));
        ctx.fillRect(wx, wy, 4, 3);
      }
    }
  }

  // Antenna on top
  const antennaH = 8 + welfare * 12;
  ctx.beginPath();
  ctx.moveTo(cx, cy - height - hh);
  ctx.lineTo(cx, cy - height - hh - antennaH);
  ctx.strokeStyle = rgba(glowHex, alpha * 0.6);
  ctx.lineWidth = 1;
  ctx.stroke();

  // Antenna tip glow
  ctx.beginPath();
  ctx.arc(cx, cy - height - hh - antennaH, 1.5, 0, Math.PI * 2);
  ctx.fillStyle = rgba(glowHex, alpha * 0.8);
  ctx.fill();

  // Faint scanlines
  ctx.strokeStyle = rgba(glowHex, alpha * 0.06);
  ctx.lineWidth = 0.5;
  for (let sy = cy - height; sy < cy; sy += 3) {
    ctx.beginPath();
    ctx.moveTo(cx - hw, sy);
    ctx.lineTo(cx + hw, sy);
    ctx.stroke();
  }
}

// --- Relay Spire ---

function drawSpire(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  epoch: EpochSnapshot | null,
  glowHex: string,
  alpha: number,
  _seed: number,
) {
  const connectivity = epoch?.avg_degree ?? 2;
  const normConn = Math.min(connectivity / 8, 1);
  const height = 30 + normConn * 90;
  const drawWidth = 6;

  // Try sprite base first
  const spriteDrawn = spriteRegistry.drawBuilding(ctx, "spire", cx, cy, drawWidth, height);

  if (!spriteDrawn) {
    // Full procedural fallback: tapered pole
    ctx.beginPath();
    ctx.moveTo(cx - 3, cy);
    ctx.lineTo(cx + 3, cy);
    ctx.lineTo(cx + 1, cy - height);
    ctx.lineTo(cx - 1, cy - height);
    ctx.closePath();
    ctx.fillStyle = rgba("#1A2233", alpha * 0.8);
    ctx.fill();
    ctx.strokeStyle = rgba(glowHex, alpha * 0.3);
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }

  // Procedural overlays (on top of sprite or procedural base)

  // Pulsing signal rings (2-3 concentric ellipses)
  const now = Date.now();
  const numRings = 2 + (normConn > 0.5 ? 1 : 0);
  for (let i = 0; i < numRings; i++) {
    const phase = ((now / 1200 + i * 0.8) % 1);
    const ringY = cy - height * 0.4 - i * height * 0.2;
    const ringRx = 6 + phase * 10;
    const ringRy = 3 + phase * 5;
    const ringAlpha = (1 - phase) * alpha * 0.5 * normConn;

    ctx.beginPath();
    ctx.ellipse(cx, ringY, ringRx, ringRy, 0, 0, Math.PI * 2);
    ctx.strokeStyle = rgba(glowHex, ringAlpha);
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Tip glow
  ctx.beginPath();
  ctx.arc(cx, cy - height, 2, 0, Math.PI * 2);
  ctx.fillStyle = rgba(glowHex, alpha * 0.7);
  ctx.fill();
}

// --- Power Node ---

function drawNode(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  threatLevel: number,
  glowHex: string,
  alpha: number,
  _seed: number,
) {
  const safety = 1 - threatLevel;
  const hw = TILE_WIDTH * 0.18;
  const hh = TILE_HEIGHT * 0.18;
  const height = 12 + safety * 18;
  const drawWidth = hw * 2 + 2; // ~35px

  // Try sprite base first
  const spriteDrawn = spriteRegistry.drawBuilding(ctx, "node", cx, cy, drawWidth, height);

  if (!spriteDrawn) {
    // Full procedural fallback
    // Right face
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + hw, cy - hh);
    ctx.lineTo(cx + hw, cy - hh - height);
    ctx.lineTo(cx, cy - height);
    ctx.closePath();
    ctx.fillStyle = rgba("#1A2233", alpha * 0.85);
    ctx.fill();

    // Left face
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx - hw, cy - hh);
    ctx.lineTo(cx - hw, cy - hh - height);
    ctx.lineTo(cx, cy - height);
    ctx.closePath();
    ctx.fillStyle = rgba("#0F1620", alpha * 0.85);
    ctx.fill();

    // Top diamond
    ctx.beginPath();
    ctx.moveTo(cx, cy - height - hh);
    ctx.lineTo(cx + hw, cy - hh - height);
    ctx.lineTo(cx, cy - height);
    ctx.lineTo(cx - hw, cy - hh - height);
    ctx.closePath();
    ctx.fillStyle = rgba("#243040", alpha * 0.85);
    ctx.fill();

    // Outline
    ctx.strokeStyle = rgba(glowHex, alpha * 0.3);
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + hw, cy - hh);
    ctx.lineTo(cx + hw, cy - hh - height);
    ctx.lineTo(cx, cy - height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx - hw, cy - hh);
    ctx.lineTo(cx - hw, cy - hh - height);
    ctx.lineTo(cx, cy - height);
    ctx.stroke();
  }

  // Procedural overlays (on top of sprite or procedural base)

  // Radial energy glow halo on top
  const glowRadius = 10 + safety * 14;
  const glowIntensity = safety * alpha * 0.5;
  const grad = ctx.createRadialGradient(
    cx, cy - height - hh * 0.5,
    0,
    cx, cy - height - hh * 0.5,
    glowRadius,
  );
  grad.addColorStop(0, rgba(glowHex, glowIntensity));
  grad.addColorStop(1, rgba(glowHex, 0));
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy - height - hh * 0.5, glowRadius, 0, Math.PI * 2);
  ctx.fill();
}
