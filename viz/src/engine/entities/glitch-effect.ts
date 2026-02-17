/**
 * Incoherence glitch screen effect.
 *
 * Driven by incoherence_index (mean(1 - |2p - 1|)).
 * At low incoherence: subtle horizontal jitter.
 * At high incoherence: RGB channel separation, scanline tears.
 */

import { rgba } from "@/utils/color";
import { COLORS } from "../constants";

function mulberry32(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function renderIncoherenceGlitch(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  incoherence: number,
) {
  if (incoherence < 0.15) return;

  const intensity = Math.min((incoherence - 0.15) / 0.6, 1);
  const now = Date.now();

  // Seed changes every 80ms for jittery feel
  const rng = mulberry32(Math.floor(now / 80));

  // Number of glitch bands scales with intensity
  const numBands = Math.floor(1 + intensity * 5);

  ctx.save();

  for (let i = 0; i < numBands; i++) {
    // Only trigger some bands each frame for flicker
    if (rng() > 0.3 + intensity * 0.5) continue;

    const bandY = Math.floor(rng() * height);
    const bandH = 2 + Math.floor(rng() * (4 + intensity * 12));
    const shiftX = (rng() - 0.5) * (3 + intensity * 12);

    // Grab the horizontal band and redraw it shifted
    try {
      const imgData = ctx.getImageData(0, bandY, width, Math.min(bandH, height - bandY));
      ctx.putImageData(imgData, shiftX, bandY);
    } catch {
      // Canvas tainted or unavailable — skip gracefully
      break;
    }
  }

  // At high incoherence, add color-tinted scanline overlay
  if (intensity > 0.4) {
    const scanAlpha = (intensity - 0.4) * 0.08;
    const scanSeed = Math.floor(now / 120);

    // Cyan tint on random thin lines
    ctx.fillStyle = rgba("#00FFFF", scanAlpha);
    for (let i = 0; i < 3; i++) {
      const y = ((scanSeed * 37 + i * 131) % height);
      ctx.fillRect(0, y, width, 1);
    }

    // Red tint on others
    ctx.fillStyle = rgba(COLORS.alert, scanAlpha);
    for (let i = 0; i < 2; i++) {
      const y = ((scanSeed * 53 + i * 97) % height);
      ctx.fillRect(0, y, width, 1);
    }
  }

  ctx.restore();
}
