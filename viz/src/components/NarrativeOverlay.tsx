"use client";

import React, { useMemo, useCallback } from "react";
import { useSimulation } from "@/state/use-simulation";
import { detectNarrativeEvents, type NarrativeEvent } from "@/engine/narrative";

const TYPE_STYLES: Record<NarrativeEvent["type"], { bg: string; border: string; icon: string }> = {
  danger:  { bg: "rgba(235,87,87,0.12)", border: "#EB5757", icon: "\u26A0" },
  warning: { bg: "rgba(242,153,74,0.12)", border: "#F2994A", icon: "\u25B2" },
  info:    { bg: "rgba(86,152,255,0.12)", border: "#5698FF", icon: "\u25CF" },
  success: { bg: "rgba(111,207,151,0.12)", border: "#6FCF97", icon: "\u2713" },
};

/**
 * NarrativeOverlay — shows auto-detected key moments as toast banners.
 * Uses CSS animation for fade-in/out instead of JS timers, avoiding
 * setState-in-effect issues entirely.
 */
export function NarrativeOverlay() {
  const { data, currentEpoch } = useSimulation();

  const allEvents = useMemo(() => {
    if (!data) return [];
    return detectNarrativeEvents(data);
  }, [data]);

  // Build a map of epoch → best event for O(1) lookup
  const eventByEpoch = useMemo(() => {
    const map = new Map<number, NarrativeEvent>();
    const priority: NarrativeEvent["type"][] = ["danger", "warning", "info", "success"];
    for (const evt of allEvents) {
      const existing = map.get(evt.epoch);
      if (!existing || priority.indexOf(evt.type) < priority.indexOf(existing.type)) {
        map.set(evt.epoch, evt);
      }
    }
    return map;
  }, [allEvents]);

  const event = eventByEpoch.get(currentEpoch) ?? null;

  if (!event) return null;

  const style = TYPE_STYLES[event.type];

  return (
    <div
      key={`${event.epoch}-${event.message}`}
      className="absolute top-20 left-1/2 -translate-x-1/2 z-30 pointer-events-none animate-narrative-toast"
    >
      <div
        className="px-4 py-2.5 rounded-lg backdrop-blur-sm border text-sm font-medium shadow-lg"
        style={{
          background: style.bg,
          borderColor: style.border,
          color: style.border,
        }}
      >
        <span className="mr-2">{style.icon}</span>
        <span className="text-text">
          Epoch {event.epoch}:
        </span>{" "}
        {event.message}
      </div>
    </div>
  );
}
