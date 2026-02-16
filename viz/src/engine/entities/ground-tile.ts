import { TILE_WIDTH, TILE_HEIGHT, COLORS } from "../constants";
import { gridToScreen, drawDiamond } from "../isometric";
import { rgba } from "@/utils/color";

export function renderGroundTile(
  ctx: CanvasRenderingContext2D,
  gridX: number,
  gridY: number,
  toxicity = 0,
) {
  const { x, y } = gridToScreen(gridX, gridY);

  // Base tile
  drawDiamond(ctx, x, y, TILE_WIDTH, TILE_HEIGHT);
  const baseColor = toxicity > 0.3
    ? rgba("#EB5757", 0.05 + toxicity * 0.15)
    : rgba(COLORS.border, 0.3);
  ctx.fillStyle = baseColor;
  ctx.fill();
  ctx.strokeStyle = rgba(COLORS.border, 0.4);
  ctx.lineWidth = 0.5;
  ctx.stroke();
}

export function renderGroundGrid(
  ctx: CanvasRenderingContext2D,
  gridSize: number,
  toxicity = 0,
) {
  for (let gx = 0; gx < gridSize; gx++) {
    for (let gy = 0; gy < gridSize; gy++) {
      renderGroundTile(ctx, gx, gy, toxicity);
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

  const rng = mulberry32(42); // deterministic
  for (let i = 0; i < numCracks; i++) {
    const startGx = rng() * gridSize;
    const startGy = rng() * gridSize;
    const { x: sx, y: sy } = gridToScreen(startGx, startGy);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    let cx = sx, cy = sy;
    const segs = 3 + Math.floor(rng() * 4);
    for (let s = 0; s < segs; s++) {
      cx += (rng() - 0.5) * 30;
      cy += (rng() - 0.5) * 20;
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
