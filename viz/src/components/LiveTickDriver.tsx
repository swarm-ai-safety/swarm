"use client";

import { useLiveTick } from "@/state/use-live-tick";

/**
 * Invisible component that drives the live simulation tick loop.
 * Rendered only when gameState.isLive is true.
 */
export function LiveTickDriver() {
  useLiveTick();
  return null;
}
