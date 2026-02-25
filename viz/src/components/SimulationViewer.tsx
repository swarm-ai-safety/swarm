"use client";

import React from "react";
import { SimulationProvider } from "@/state/simulation-context";
import { GameProvider, useGame } from "@/state/game-context";
import { IsometricCanvas } from "./IsometricCanvas";
import { TimelineControls } from "./TimelineControls";
import { LiveControlBar } from "./LiveControlBar";
import { InfoPanel } from "./InfoPanel";
import { MetricsOverlay } from "./MetricsOverlay";
import { Minimap } from "./Minimap";
import { DataLoader } from "./DataLoader";
import { OverlayToggles } from "./OverlayToggles";
import { SplashScreen } from "./SplashScreen";
import { NarrativeOverlay } from "./NarrativeOverlay";
import { ToastContainer } from "./Toast";
import { LevelEndOverlay } from "./CampaignPanel";
import { LiveTickDriver } from "./LiveTickDriver";
import dynamic from "next/dynamic";
const Leaderboard = dynamic(() => import("./Leaderboard").then((m) => m.Leaderboard), { ssr: false });
import { EventFeed } from "./EventFeed";

function SimulationViewerInner() {
  const { state: gameState } = useGame();

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-bg">
      {/* Live tick driver (invisible, just runs the hook) */}
      {gameState.isLive && <LiveTickDriver />}

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
      <EventFeed />
      <ToastContainer />
      <LevelEndOverlay />

      {/* Show LiveControlBar in live mode, TimelineControls in replay mode */}
      {gameState.isLive ? <LiveControlBar /> : <TimelineControls />}

      <SplashScreen />
    </div>
  );
}

export function SimulationViewer() {
  return (
    <SimulationProvider>
      <GameProvider>
        <SimulationViewerInner />
      </GameProvider>
    </SimulationProvider>
  );
}
