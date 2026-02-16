export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

export function inverseLerp(a: number, b: number, v: number): number {
  if (a === b) return 0;
  return clamp((v - a) / (b - a), 0, 1);
}

export function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

export function easeOut(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export function easeIn(t: number): number {
  return t * t * t;
}

export function remap(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number,
): number {
  const t = inverseLerp(inMin, inMax, value);
  return lerp(outMin, outMax, t);
}

export function distance(x1: number, y1: number, x2: number, y2: number): number {
  return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}

/** C2-continuous interpolation — smoother than smoothstep, no flat spots at edges */
export function smootherstep(edge0: number, edge1: number, x: number): number {
  let t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * t * (t * (t * 6 - 15) + 10);
}

/** Deterministic 1D value noise returning [-1, 1] for per-agent wobble */
export function noise1D(x: number): number {
  // Integer part and fraction
  const xi = Math.floor(x);
  const f = x - xi;
  // Two pseudo-random corner values via bit-mixing
  const a = _hash1D(xi);
  const b = _hash1D(xi + 1);
  // Smooth interpolation
  const t = f * f * (3 - 2 * f);
  return a + (b - a) * t;
}

function _hash1D(n: number): number {
  // Returns [-1, 1] deterministically
  let x = ((n * 1597334677) ^ (n * 3812015801)) >>> 0;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) >>> 0;
  return (x / 2147483648) - 1;
}

/** Returns a deterministic () => number PRNG function from a seed */
export function seededRandom(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Simple string hash → 32-bit unsigned integer for deterministic seeding */
export function hashString(s: string): number {
  let hash = 0;
  for (let i = 0; i < s.length; i++) {
    const ch = s.charCodeAt(i);
    hash = ((hash << 5) - hash + ch) | 0;
  }
  return hash >>> 0;
}
