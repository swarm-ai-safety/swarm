import { TILE_WIDTH, TILE_HEIGHT } from "./constants";
import type { Point } from "./types";

/** Convert grid coordinates to screen pixel coordinates */
export function gridToScreen(gridX: number, gridY: number): Point {
  return {
    x: (gridX - gridY) * (TILE_WIDTH / 2),
    y: (gridX + gridY) * (TILE_HEIGHT / 2),
  };
}

/** Convert screen pixel coordinates to grid coordinates */
export function screenToGrid(screenX: number, screenY: number): { gridX: number; gridY: number } {
  const halfW = TILE_WIDTH / 2;
  const halfH = TILE_HEIGHT / 2;
  return {
    gridX: (screenX / halfW + screenY / halfH) / 2,
    gridY: (screenY / halfH - screenX / halfW) / 2,
  };
}

/** Draw a diamond (isometric tile) */
export function drawDiamond(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  w: number,
  h: number,
) {
  ctx.beginPath();
  ctx.moveTo(cx, cy - h / 2);
  ctx.lineTo(cx + w / 2, cy);
  ctx.lineTo(cx, cy + h / 2);
  ctx.lineTo(cx - w / 2, cy);
  ctx.closePath();
}
