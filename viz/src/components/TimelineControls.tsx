"use client";

import React, { useCallback } from "react";
import { usePlayback } from "@/state/use-playback";
import { useSimulation } from "@/state/use-simulation";
import { GovernanceIntervention } from "./GovernanceIntervention";

const SPEEDS = [0.5, 1, 2, 4];

function exportSimData(data: import("@/data/types").SimulationData) {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${data.simulation_id || "history"}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export function TimelineControls() {
  const { play, pause, setEpoch, setSpeed, playing, speed, currentEpoch, maxEpoch } =
    usePlayback();
  const { data } = useSimulation();

  const handleExport = useCallback(() => {
    if (data) exportSimData(data);
  }, [data]);

  return (
    <div className="absolute bottom-0 left-0 right-0 h-14 bg-panel border-t border-border flex items-center gap-3 px-4 z-[10000]">
      {/* Play/Pause */}
      <button
        onClick={() => (playing ? pause() : play())}
        className="w-9 h-9 flex items-center justify-center rounded bg-btn hover:bg-btn-hover transition-colors"
        title={playing ? "Pause" : "Play"}
      >
        {playing ? (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
            <rect x="2" y="1" width="4" height="12" rx="1" />
            <rect x="8" y="1" width="4" height="12" rx="1" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
            <path d="M3 1.5v11l9-5.5z" />
          </svg>
        )}
      </button>

      {/* Epoch label */}
      <span className="text-xs text-muted w-20 text-center font-mono">
        Epoch {currentEpoch}/{maxEpoch}
      </span>

      {/* Scrubber */}
      <input
        type="range"
        min={0}
        max={maxEpoch}
        value={currentEpoch}
        onChange={(e) => setEpoch(Number(e.target.value))}
        className="flex-1 h-1.5 appearance-none bg-border rounded-full cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
          [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:cursor-pointer"
      />

      {/* Speed selector */}
      <div className="flex items-center gap-1">
        <span className="text-xs text-muted mr-1">Speed</span>
        {SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => setSpeed(s)}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              speed === s
                ? "bg-accent text-bg font-bold"
                : "bg-btn hover:bg-btn-hover text-muted"
            }`}
          >
            {s}x
          </button>
        ))}
      </div>

      {/* Governance intervention */}
      {data && <GovernanceIntervention />}

      {/* Export button */}
      {data && (
        <button
          onClick={handleExport}
          className="w-9 h-9 flex items-center justify-center rounded bg-btn hover:bg-btn-hover transition-colors"
          title="Export as history.json"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M7 1v8M4 6l3 3 3-3M2 11v1.5h10V11" />
          </svg>
        </button>
      )}
    </div>
  );
}
