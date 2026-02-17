import type { AgentVisual } from "../types";
import { gridToScreen } from "../isometric";
import { COLORS } from "../constants";
import { rgba } from "@/utils/color";

/**
 * Render a network topology web on the ground plane connecting agents
 * based on avgDegree (k-nearest neighbors) and avgClustering (visual intensity).
 */
export function renderNetworkWeb(
  ctx: CanvasRenderingContext2D,
  agents: AgentVisual[],
  avgDegree: number,
  avgClustering: number,
): void {
  if (avgDegree < 0.5 || agents.length < 2) return;

  const k = Math.round(avgDegree);

  // Pre-compute screen positions for each agent (ground-level, with walk offset)
  const positions = agents.map((a) => {
    const pos = gridToScreen(a.gridX, a.gridY);
    return {
      x: pos.x + a.walkOffsetX,
      y: pos.y + a.walkOffsetY,
    };
  });

  // Compute squared distances between all pairs (avoids sqrt for sorting)
  const distSq = (i: number, j: number): number => {
    const dx = positions[i].x - positions[j].x;
    const dy = positions[i].y - positions[j].y;
    return dx * dx + dy * dy;
  };

  // Build adjacency: for each agent, connect to k nearest neighbors
  // Use a Set of sorted-pair keys to avoid duplicate edges
  const edgeSet = new Set<string>();
  const adjacency: Set<number>[] = agents.map(() => new Set<number>());

  for (let i = 0; i < agents.length; i++) {
    // Collect distances to all other agents
    const neighbors: { idx: number; d: number }[] = [];
    for (let j = 0; j < agents.length; j++) {
      if (i === j) continue;
      neighbors.push({ idx: j, d: distSq(i, j) });
    }
    // Sort by distance and take k nearest
    neighbors.sort((a, b) => a.d - b.d);
    const kClamped = Math.min(k, neighbors.length);
    for (let n = 0; n < kClamped; n++) {
      const j = neighbors[n].idx;
      const edgeKey = i < j ? `${i}:${j}` : `${j}:${i}`;
      edgeSet.add(edgeKey);
      adjacency[i].add(j);
      adjacency[j].add(i);
    }
  }

  // Base alpha scales with clustering (more clustering = more visible web)
  const baseAlpha = 0.15 + avgClustering * 0.45;

  ctx.save();

  // --- Draw edges ---
  ctx.strokeStyle = rgba(COLORS.accent, baseAlpha);
  ctx.lineWidth = 1.2;
  ctx.setLineDash([4, 6]);

  for (const key of edgeSet) {
    const [iStr, jStr] = key.split(":");
    const i = Number(iStr);
    const j = Number(jStr);
    ctx.beginPath();
    ctx.moveTo(positions[i].x, positions[i].y);
    ctx.lineTo(positions[j].x, positions[j].y);
    ctx.stroke();
  }

  ctx.setLineDash([]);

  // --- Draw triangular fills for clustered triplets when avgClustering > 0.5 ---
  if (avgClustering > 0.5) {
    const triAlpha = (avgClustering - 0.5) * 0.12; // very low alpha: 0..0.06
    ctx.fillStyle = rgba(COLORS.accent, triAlpha);

    // For each edge, check if both endpoints share a common neighbor (triangle)
    const drawnTriangles = new Set<string>();
    for (const key of edgeSet) {
      const [iStr, jStr] = key.split(":");
      const i = Number(iStr);
      const j = Number(jStr);
      // Find common neighbors of i and j
      for (const c of adjacency[i]) {
        if (c !== j && adjacency[j].has(c)) {
          const tri = [i, j, c].sort((a, b) => a - b);
          const triKey = tri.join(":");
          if (drawnTriangles.has(triKey)) continue;
          drawnTriangles.add(triKey);

          ctx.beginPath();
          ctx.moveTo(positions[tri[0]].x, positions[tri[0]].y);
          ctx.lineTo(positions[tri[1]].x, positions[tri[1]].y);
          ctx.lineTo(positions[tri[2]].x, positions[tri[2]].y);
          ctx.closePath();
          ctx.fill();
        }
      }
    }
  }

  // --- Draw diamond nodes at each agent's ground position ---
  const nodeSize = 6;
  const nodeHalf = nodeSize / 2;
  ctx.fillStyle = rgba(COLORS.accent, baseAlpha + 0.15);
  ctx.strokeStyle = rgba(COLORS.accent, baseAlpha + 0.25);
  ctx.lineWidth = 0.8;

  for (const pos of positions) {
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y - nodeHalf);
    ctx.lineTo(pos.x + nodeHalf, pos.y);
    ctx.lineTo(pos.x, pos.y + nodeHalf);
    ctx.lineTo(pos.x - nodeHalf, pos.y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  ctx.restore();
}
