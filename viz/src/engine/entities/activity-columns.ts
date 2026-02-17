/**
 * Activity pulse columns — holographic bar chart showing posts/votes/tasks.
 *
 * Three translucent isometric pillars at a fixed grid position.
 * Heights proportional to epoch activity counts.
 */

import { TILE_WIDTH, TILE_HEIGHT, COLORS } from "../constants";
import { gridToScreen } from "../isometric";
import { rgba } from "@/utils/color";
import type { EpochSnapshot } from "@/data/types";

interface ColumnDef {
  label: string;
  color: string;
  getValue: (epoch: EpochSnapshot) => number;
  offsetX: number;
}

const COLUMNS: ColumnDef[] = [
  { label: "Posts", color: COLORS.info, getValue: (e) => e.total_posts, offsetX: -18 },
  { label: "Votes", color: COLORS.secondary, getValue: (e) => e.total_votes, offsetX: 0 },
  { label: "Tasks", color: COLORS.accent, getValue: (e) => e.total_tasks_completed, offsetX: 18 },
];

export function renderActivityColumns(
  ctx: CanvasRenderingContext2D,
  gridSize: number,
  epoch: EpochSnapshot | null,
) {
  if (!epoch) return;

  // Check if there's any activity to show
  const totalActivity = epoch.total_posts + epoch.total_votes + epoch.total_tasks_completed;
  if (totalActivity === 0) return;

  // Position at top-right corner of the grid
  const pos = gridToScreen(gridSize - 2, 1);
  const baseX = pos.x;
  const baseY = pos.y;

  const hw = TILE_WIDTH * 0.06;
  const hh = TILE_HEIGHT * 0.06;

  // Max value for normalization (at least 1 to avoid division by zero)
  const maxVal = Math.max(
    1,
    epoch.total_posts,
    epoch.total_votes,
    epoch.total_tasks_completed,
  );

  ctx.save();

  for (const col of COLUMNS) {
    const value = col.getValue(epoch);
    if (value === 0) continue;

    const cx = baseX + col.offsetX;
    const cy = baseY;
    const normHeight = Math.min(value / maxVal, 1);
    const height = 8 + normHeight * 50;
    const alpha = 0.5 + normHeight * 0.3;

    // Right face
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + hw, cy - hh);
    ctx.lineTo(cx + hw, cy - hh - height);
    ctx.lineTo(cx, cy - height);
    ctx.closePath();
    ctx.fillStyle = rgba(col.color, alpha * 0.5);
    ctx.fill();

    // Left face
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx - hw, cy - hh);
    ctx.lineTo(cx - hw, cy - hh - height);
    ctx.lineTo(cx, cy - height);
    ctx.closePath();
    ctx.fillStyle = rgba(col.color, alpha * 0.35);
    ctx.fill();

    // Top diamond
    ctx.beginPath();
    ctx.moveTo(cx, cy - height - hh);
    ctx.lineTo(cx + hw, cy - hh - height);
    ctx.lineTo(cx, cy - height);
    ctx.lineTo(cx - hw, cy - hh - height);
    ctx.closePath();
    ctx.fillStyle = rgba(col.color, alpha * 0.6);
    ctx.fill();

    // Outline
    ctx.strokeStyle = rgba(col.color, alpha * 0.4);
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + hw, cy - hh);
    ctx.lineTo(cx + hw, cy - hh - height);
    ctx.lineTo(cx, cy - height);
    ctx.lineTo(cx - hw, cy - hh - height);
    ctx.lineTo(cx - hw, cy - hh);
    ctx.closePath();
    ctx.stroke();

    // Scanlines
    ctx.strokeStyle = rgba(col.color, 0.06);
    ctx.lineWidth = 0.4;
    for (let sy = cy - height; sy < cy; sy += 4) {
      ctx.beginPath();
      ctx.moveTo(cx - hw, sy);
      ctx.lineTo(cx + hw, sy);
      ctx.stroke();
    }

    // Value label above column
    ctx.font = "bold 7px 'Courier New', monospace";
    ctx.textAlign = "center";
    ctx.fillStyle = rgba(col.color, 0.8);
    ctx.fillText(String(value), cx, cy - height - hh - 4);
  }

  ctx.restore();
}
