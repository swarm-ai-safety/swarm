import type { AgentType } from "@/data/types";

/** Canvas-space draw dimensions (must match CHARACTER.baseWidth/baseHeight) */
const DRAW_W = 48;
const DRAW_H = 84;

const AGENT_TYPES: AgentType[] = [
  "honest", "opportunistic", "deceptive", "adversarial", "rlm", "crewai",
];

interface SpriteEntry {
  img: HTMLImageElement;
  /** Cleaned sprite with background removed, ready to draw */
  cleaned: HTMLCanvasElement | null;
  loaded: boolean;
}

/**
 * SpriteRegistry loads and caches agent sprite PNGs (idle only).
 * Applies background removal on load to ensure clean alpha transparency.
 * The engine's bob offset and position changes animate walking — no separate
 * walk sprites needed.
 */
class SpriteRegistry {
  private sprites = new Map<string, SpriteEntry>();
  private initialized = false;

  /** Eagerly load all sprite images. Safe to call multiple times. */
  init(): void {
    if (this.initialized) return;
    this.initialized = true;

    for (const agentType of AGENT_TYPES) {
      const key = agentType;
      const img = new Image();
      const entry: SpriteEntry = { img, cleaned: null, loaded: false };

      img.onload = () => {
        entry.cleaned = this.removeBackground(img);
        entry.loaded = true;
      };
      img.onerror = () => { /* silently fail — procedural fallback */ };
      img.src = `/sprites/${agentType}_idle.png`;

      this.sprites.set(key, entry);
    }
  }

  /** Check if a specific sprite is ready to draw */
  isReady(agentType: AgentType): boolean {
    const entry = this.sprites.get(agentType);
    return entry?.loaded ?? false;
  }

  /**
   * Remove background from Imagen output using edge flood-fill.
   *
   * Imagen doesn't produce true alpha transparency — it renders a light
   * grey background. We flood-fill inward from all border pixels, removing
   * any connected region whose color is "close enough" to the border color.
   * This only removes the outer background, never interior character pixels.
   */
  private removeBackground(img: HTMLImageElement): HTMLCanvasElement {
    const canvas = document.createElement("canvas");
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0);

    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;
    const visited = new Uint8Array(w * h);

    // Sample background color from corners (average)
    const corners = [
      [2, 2], [w - 3, 2], [2, h - 3], [w - 3, h - 3],
    ];
    let bgR = 0, bgG = 0, bgB = 0;
    for (const [cx, cy] of corners) {
      const i = (cy * w + cx) * 4;
      bgR += data[i]; bgG += data[i + 1]; bgB += data[i + 2];
    }
    bgR /= 4; bgG /= 4; bgB /= 4;

    const tolerance = 120;

    const isBg = (idx: number): boolean => {
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      const a = data[idx + 3];
      if (a === 0) return true; // already transparent

      // RGB distance from sampled corner background color.
      // Generous tolerance is safe because flood-fill only removes
      // pixels connected to the image edge — can't eat into the character.
      const dist = Math.sqrt((r - bgR) ** 2 + (g - bgG) ** 2 + (b - bgB) ** 2);
      if (dist < tolerance) return true;

      // Semi-transparent pixels near the edge are also background
      if (a < 180) return true;

      return false;
    };

    // Seed the flood-fill queue with all border pixels that match BG
    const queue: number[] = [];
    for (let x = 0; x < w; x++) {
      for (const y of [0, h - 1]) {
        const pos = y * w + x;
        if (!visited[pos] && isBg(pos * 4)) {
          queue.push(pos);
          visited[pos] = 1;
        }
      }
    }
    for (let y = 1; y < h - 1; y++) {
      for (const x of [0, w - 1]) {
        const pos = y * w + x;
        if (!visited[pos] && isBg(pos * 4)) {
          queue.push(pos);
          visited[pos] = 1;
        }
      }
    }

    // BFS flood-fill
    let head = 0;
    while (head < queue.length) {
      const pos = queue[head++];
      const px = pos % w;
      const py = Math.floor(pos / w);

      // Mark as transparent
      data[pos * 4 + 3] = 0;

      // Check 4-connected neighbors
      const neighbors = [
        py > 0 ? pos - w : -1,
        py < h - 1 ? pos + w : -1,
        px > 0 ? pos - 1 : -1,
        px < w - 1 ? pos + 1 : -1,
      ];
      for (const nPos of neighbors) {
        if (nPos >= 0 && !visited[nPos] && isBg(nPos * 4)) {
          visited[nPos] = 1;
          queue.push(nPos);
        }
      }
    }

    // Soft-edge pass: partially fade pixels adjacent to removed background
    // so character edges don't look harsh
    const alphaData = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) alphaData[i] = data[i * 4 + 3];

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const pos = y * w + x;
        if (alphaData[pos] === 0) continue; // already transparent

        // Count transparent neighbors (from flood-fill)
        let bgNeighbors = 0;
        if (alphaData[pos - 1] === 0) bgNeighbors++;
        if (alphaData[pos + 1] === 0) bgNeighbors++;
        if (alphaData[pos - w] === 0) bgNeighbors++;
        if (alphaData[pos + w] === 0) bgNeighbors++;

        if (bgNeighbors >= 2) {
          // Edge pixel touching background on 2+ sides — soften
          data[pos * 4 + 3] = Math.round(data[pos * 4 + 3] * 0.5);
        } else if (bgNeighbors === 1) {
          data[pos * 4 + 3] = Math.round(data[pos * 4 + 3] * 0.8);
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas;
  }

  /**
   * Draw the sprite for the given agent type onto the canvas.
   *
   * @param ctx       - Canvas 2D context
   * @param agentType - One of the 6 agent types
   * @param baseX     - Screen X position (character center, at feet)
   * @param baseY     - Screen Y position (character feet)
   * @param scale     - Agent scale multiplier
   * @param facing    - 1 = right, -1 = left (matches engine convention)
   * @returns true if sprite was drawn, false if caller should fall back to procedural
   */
  draw(
    ctx: CanvasRenderingContext2D,
    agentType: AgentType,
    baseX: number,
    baseY: number,
    scale: number,
    facing: number,
  ): boolean {
    const entry = this.sprites.get(agentType);
    if (!entry?.loaded || !entry.cleaned) return false;

    const src = entry.cleaned;
    const drawW = DRAW_W * scale;
    const drawH = DRAW_H * scale;

    ctx.save();

    // Sprites are generated left-facing. Engine convention: 1 = right, -1 = left.
    // Flip when facing right (1) to mirror the left-facing sprite.
    // No flip when facing left (-1) since that's the sprite's native direction.
    if (facing > 0) {
      ctx.translate(baseX, 0);
      ctx.scale(-1, 1);
      ctx.translate(-baseX, 0);
    }

    // Draw sprite centered horizontally at baseX, with bottom at baseY
    ctx.drawImage(
      src,
      0, 0, src.width, src.height,
      baseX - drawW / 2,
      baseY - drawH,
      drawW,
      drawH,
    );

    ctx.restore();
    return true;
  }
}

/** Singleton instance */
export const spriteRegistry = new SpriteRegistry();
