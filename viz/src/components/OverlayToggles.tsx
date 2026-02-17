"use client";

import React from "react";
import { useSimulation } from "@/state/use-simulation";
import type { OverlayState } from "@/engine/types";

const TOGGLE_ITEMS: { key: keyof OverlayState; label: string }[] = [
  { key: "interactions", label: "Arcs" },
  { key: "metricsHud", label: "HUD" },
  { key: "particles", label: "FX" },
  { key: "minimap", label: "Map" },
  { key: "collusionLines", label: "Collusion" },
  { key: "threatZones", label: "Threats" },
  { key: "digitalRain", label: "Rain" },
  { key: "tierraStrip", label: "Memory" },
  { key: "networkWeb", label: "Network" },
];

export function OverlayToggles() {
  const { overlays, toggleOverlay, data } = useSimulation();

  if (!data) return null;

  return (
    <div className="absolute bottom-24 left-4 flex flex-wrap gap-1 z-20 max-w-xs">
      {TOGGLE_ITEMS.map(({ key, label }) => (
        <button
          key={key}
          onClick={() => toggleOverlay(key)}
          className={`px-2.5 py-1 text-[10px] rounded-full border transition-colors ${
            overlays[key]
              ? "bg-accent/20 border-accent/50 text-accent"
              : "bg-btn border-border text-muted hover:text-text"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
