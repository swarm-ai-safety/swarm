"use client";

import React from "react";
import { usePlayback } from "@/state/use-playback";

const SPEEDS = [0.5, 1, 2, 4];

export function TimelineControls() {
  const { play, pause, setEpoch, setSpeed, playing, speed, currentEpoch, maxEpoch } =
    usePlayback();

  return (
    <div className="absolute bottom-0 left-0 right-0 h-14 bg-panel border-t border-border flex items-center gap-3 px-4 z-20">
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
    </div>
  );
}
