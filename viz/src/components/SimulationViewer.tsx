"use client";

import React from "react";
import { SimulationProvider } from "@/state/simulation-context";
import { IsometricCanvas } from "./IsometricCanvas";
import { TimelineControls } from "./TimelineControls";
import { InfoPanel } from "./InfoPanel";
import { MetricsOverlay } from "./MetricsOverlay";
import { Minimap } from "./Minimap";
import { DataLoader } from "./DataLoader";
import { OverlayToggles } from "./OverlayToggles";
import { SplashScreen } from "./SplashScreen";
import { NarrativeOverlay } from "./NarrativeOverlay";
import { Leaderboard } from "./Leaderboard";

export function SimulationViewer() {
  return (
    <SimulationProvider>
      <div className="relative w-screen h-screen overflow-hidden bg-bg">
        {/* Canvas layer */}
        <div className="absolute inset-0">
          <IsometricCanvas />
        </div>

        {/* UI overlays */}
        <DataLoader />
        <InfoPanel />
        <MetricsOverlay />
        <NarrativeOverlay />
        <Minimap />
        <Leaderboard />
        <OverlayToggles />
        <TimelineControls />
        <SplashScreen />
      </div>
    </SimulationProvider>
  );
}
