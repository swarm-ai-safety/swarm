"use client";

import React, { useState, useCallback } from "react";
import { SimulationProvider } from "@/state/simulation-context";
import { GameProvider, useGame } from "@/state/game-context";
import { useSimulation } from "@/state/use-simulation";
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

// ─── Share Button ──────────────────────────────────────────────────

function ShareButton() {
  const { data } = useSimulation();
  const [copied, setCopied] = useState(false);

  const handleShare = useCallback(() => {
    if (!data || typeof window === "undefined") return;

    const params = new URLSearchParams();
    if (data.seed != null) params.set("seed", data.seed.toString());
    if (data.n_epochs) params.set("epochs", data.n_epochs.toString());
    if (data.steps_per_epoch) params.set("steps", data.steps_per_epoch.toString());

    const base = window.location.origin + window.location.pathname;
    const url = params.toString() ? `${base}?${params.toString()}` : base;

    navigator.clipboard.writeText(url).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [data]);

  if (!data) return null;

  return (
    <button
      onClick={handleShare}
      className="absolute top-2 right-24 z-[10001] px-2.5 py-1 text-xs rounded bg-btn hover:bg-btn-hover text-muted hover:text-text transition-colors"
      title="Copy shareable URL"
    >
      {copied ? "\u2713 Copied!" : "Share"}
    </button>
  );
}

// ─── Main Viewer ───────────────────────────────────────────────────

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
      <ShareButton />
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
