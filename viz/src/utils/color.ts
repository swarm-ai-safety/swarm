import { clamp, lerp } from "./math";

export interface RGB {
  r: number;
  g: number;
  b: number;
}

export function hexToRgb(hex: string): RGB {
  const n = parseInt(hex.replace("#", ""), 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

export function rgbToHex(r: number, g: number, b: number): string {
  return (
    "#" +
    [r, g, b]
      .map((v) =>
        clamp(Math.round(v), 0, 255)
          .toString(16)
          .padStart(2, "0"),
      )
      .join("")
  );
}

export function lerpColor(c1: string, c2: string, t: number): string {
  const a = hexToRgb(c1);
  const b = hexToRgb(c2);
  return rgbToHex(
    lerp(a.r, b.r, t),
    lerp(a.g, b.g, t),
    lerp(a.b, b.b, t),
  );
}

export function rgba(hex: string, alpha: number): string {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha})`;
}

/** Map a 0-1 value to a green→yellow→red gradient */
export function pToColor(p: number): string {
  const clamped = clamp(p, 0, 1);
  if (clamped > 0.5) {
    return lerpColor("#EB5757", "#F2C94C", (clamped - 0.5) * 2);
  }
  return lerpColor("#F2C94C", "#27AE60", clamped * 2);
}

/** Inverse: red at low p, green at high p (for "health" style) */
export function pToHealthColor(p: number): string {
  const clamped = clamp(p, 0, 1);
  if (clamped < 0.5) {
    return lerpColor("#EB5757", "#F2C94C", clamped * 2);
  }
  return lerpColor("#F2C94C", "#27AE60", (clamped - 0.5) * 2);
}
