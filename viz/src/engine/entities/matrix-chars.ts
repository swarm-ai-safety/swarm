/** Shared Matrix-style character utilities */

// Half-width katakana (U+FF66–FF9D) + digits + symbols
const KATAKANA_START = 0xff66;
const KATAKANA_END = 0xff9d;
const DIGITS = "0123456789";
const SYMBOLS = "+-*/<>=(){}[];:!?@#$%&^~";
const CHARSET: string[] = [];

for (let c = KATAKANA_START; c <= KATAKANA_END; c++) {
  CHARSET.push(String.fromCharCode(c));
}
for (const ch of DIGITS + SYMBOLS) {
  CHARSET.push(ch);
}

/** Random Matrix character (katakana + digits + symbols) */
export function randomMatrixChar(): string {
  return CHARSET[Math.floor(Math.random() * CHARSET.length)];
}

/** Deterministic character from a numeric seed */
export function matrixCharFromSeed(seed: number): string {
  return CHARSET[((seed >>> 0) % CHARSET.length + CHARSET.length) % CHARSET.length];
}

/** Error/garble characters for rejection effects */
const ERROR_CHARS = "!?X#@$%&";
export function randomErrorChar(): string {
  return ERROR_CHARS[Math.floor(Math.random() * ERROR_CHARS.length)];
}

/** Deterministic error char from seed */
export function errorCharFromSeed(seed: number): string {
  return ERROR_CHARS[((seed >>> 0) % ERROR_CHARS.length + ERROR_CHARS.length) % ERROR_CHARS.length];
}

/** Canonical Matrix green with alpha */
export function matrixGreen(alpha: number): string {
  return `rgba(0,255,65,${alpha})`;
}

/** Matrix color based on p value: green (high p) → yellow (mid) → red (low) */
export function matrixColorForP(p: number, alpha: number): string {
  const clamped = Math.max(0, Math.min(1, p));
  let r: number, g: number, b: number;
  if (clamped > 0.5) {
    // yellow → green
    const t = (clamped - 0.5) * 2;
    r = Math.round(255 * (1 - t));
    g = 255;
    b = Math.round(65 * t);
  } else {
    // red → yellow
    const t = clamped * 2;
    r = 255;
    g = Math.round(255 * t);
    b = 0;
  }
  return `rgba(${r},${g},${b},${alpha})`;
}

/** Monospace font for Matrix characters */
export const MATRIX_FONT = "bold 14px 'Courier New', monospace";
export const MATRIX_FONT_SMALL = "bold 10px 'Courier New', monospace";
