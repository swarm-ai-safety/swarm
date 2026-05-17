import { TILE_WIDTH, TILE_HEIGHT, COLORS } from "../constants";
import { gridToScreen, drawDiamond } from "../isometric";
import { rgba } from "@/utils/color";
import { spriteRegistry } from "./sprite-registry";

export function renderGroundTile(
  ctx: CanvasRenderingContext2D,
  gridX: number,
  gridY: number,
  toxicity = 0,
  variance = 0,
) {
  const { x, y } = gridToScreen(gridX, gridY);
  const hw = TILE_WIDTH / 2;
  const hh = TILE_HEIGHT / 2;

  const toxic = toxicity > 0.3;
  const highTox = toxicity > 0.5;
  const lineColor = highTox ? "#EB5757" : COLORS.accent;

  // Try sprite base first, fall back to procedural
  const spriteDrawn = spriteRegistry.drawTile(ctx, x, y, TILE_WIDTH, TILE_HEIGHT);

  if (!spriteDrawn) {
    // Procedural base tile — dark floor with subtle gradient
    drawDiamond(ctx, x, y, TILE_WIDTH, TILE_HEIGHT);
    const baseGrad = ctx.createLinearGradient(x - hw, y, x + hw, y);
    if (toxic) {
      const tint = toxicity * 0.12;
      baseGrad.addColorStop(0, `rgba(${Math.round(20 + tint * 200)}, ${Math.round(8)}, ${Math.round(14 + tint * 30)}, 0.92)`);
      baseGrad.addColorStop(1, `rgba(${Math.round(12 + tint * 150)}, ${Math.round(6)}, ${Math.round(10 + tint * 20)}, 0.92)`);
    } else {
      baseGrad.addColorStop(0, "rgba(10, 14, 22, 0.92)");
      baseGrad.addColorStop(1, "rgba(6, 10, 16, 0.92)");
    }
    ctx.fillStyle = baseGrad;
    ctx.fill();

    // Diamond outline (circuit trace)
    const outlineAlpha = toxic ? 0.18 : 0.35;
    drawDiamond(ctx, x, y, TILE_WIDTH, TILE_HEIGHT);
    ctx.strokeStyle = rgba(lineColor, outlineAlpha);
    ctx.lineWidth = 0.6;
    ctx.stroke();

    // Inner circuit pattern
    const innerAlpha = toxic ? 0.1 : 0.2;
    ctx.strokeStyle = rgba(lineColor, innerAlpha);
    ctx.lineWidth = 0.4;

    // Cross lines (iso-aligned)
    ctx.beginPath();
    ctx.moveTo(x - hw * 0.5, y);
    ctx.lineTo(x + hw * 0.5, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y - hh * 0.5);
    ctx.lineTo(x, y + hh * 0.5);
    ctx.stroke();

    // Inner diamond (circuit trace at 50% size)
    ctx.strokeStyle = rgba(lineColor, innerAlpha * 0.6);
    ctx.lineWidth = 0.3;
    drawDiamond(ctx, x, y, TILE_WIDTH * 0.5, TILE_HEIGHT * 0.5);
    ctx.stroke();

    // Diagonal traces (from center to diamond midpoints)
    ctx.strokeStyle = rgba(lineColor, innerAlpha * 0.4);
    ctx.lineWidth = 0.3;
    ctx.beginPath();
    ctx.moveTo(x - hw * 0.25, y - hh * 0.25);
    ctx.lineTo(x - hw * 0.6, y - hh * 0.1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + hw * 0.25, y - hh * 0.25);
    ctx.lineTo(x + hw * 0.6, y - hh * 0.1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - hw * 0.25, y + hh * 0.25);
    ctx.lineTo(x - hw * 0.6, y + hh * 0.1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + hw * 0.25, y + hh * 0.25);
    ctx.lineTo(x + hw * 0.6, y + hh * 0.1);
    ctx.stroke();

    // Circuit nodes at key intersections
    const nodeAlpha = toxic ? 0.2 : 0.4;
    const nodeColor = rgba(lineColor, nodeAlpha);
    ctx.fillStyle = nodeColor;
    const nodeR = 1.2;

    ctx.beginPath();
    ctx.arc(x, y, nodeR, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath(); ctx.arc(x, y - hh * 0.5, nodeR * 0.8, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(x, y + hh * 0.5, nodeR * 0.8, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(x - hw * 0.5, y, nodeR * 0.8, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(x + hw * 0.5, y, nodeR * 0.8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = rgba(lineColor, nodeAlpha * 0.6);
    const nr2 = nodeR * 0.6;
    ctx.beginPath(); ctx.arc(x, y - hh * 0.25, nr2, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(x, y + hh * 0.25, nr2, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(x - hw * 0.25, y, nr2, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(x + hw * 0.25, y, nr2, 0, Math.PI * 2); ctx.fill();
  }

  // --- Procedural overlays (applied on top of sprite OR procedural base) ---

  // Toxicity tint overlay
  if (toxic && spriteDrawn) {
    drawDiamond(ctx, x, y, TILE_WIDTH, TILE_HEIGHT);
    ctx.fillStyle = rgba("#EB5757", toxicity * 0.15);
    ctx.fill();
  }

  // Variance heatmap: amber/magenta underglow when reputation_std + payoff_std is high
  if (variance > 0.1) {
    const vIntensity = Math.min((variance - 0.1) / 0.8, 1);
    const vColor = vIntensity > 0.5 ? "#BB6BD9" : "#F2994A";
    drawDiamond(ctx, x, y, TILE_WIDTH, TILE_HEIGHT);
    ctx.fillStyle = rgba(vColor, vIntensity * 0.08);
    ctx.fill();
  }
}

export function renderGroundGrid(
  ctx: CanvasRenderingContext2D,
  gridSize: number,
  toxicity = 0,
  variance = 0,
) {
  for (let gx = 0; gx < gridSize; gx++) {
    for (let gy = 0; gy < gridSize; gy++) {
      renderGroundTile(ctx, gx, gy, toxicity, variance);
    }
  }
}

/** Render crack patterns on ground for high Gini coefficient */
export function renderGiniCracks(
  ctx: CanvasRenderingContext2D,
  gridSize: number,
  gini: number,
) {
  if (gini < 0.2) return;
  const intensity = Math.min((gini - 0.2) / 0.6, 1);
  const numCracks = Math.floor(intensity * 8) + 1;

  ctx.save();
  ctx.strokeStyle = rgba("#EB5757", intensity * 0.4);
  ctx.lineWidth = 1;

  const rng = mulberry32(42);
  for (let i = 0; i < numCracks; i++) {
    const startGx = rng() * gridSize;
    const startGy = rng() * gridSize;
    const { x: sx, y: sy } = gridToScreen(startGx, startGy);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    let cx = sx, cy = sy;
    const segs = 3 + Math.floor(rng() * 4);
    for (let s = 0; s < segs; s++) {
      cx += (rng() - 0.5) * 40;
      cy += (rng() - 0.5) * 28;
      ctx.lineTo(cx, cy);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function mulberry32(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
