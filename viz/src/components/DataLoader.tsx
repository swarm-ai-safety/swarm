"use client";

import React, { useCallback, useRef, useState } from "react";
import { useSimulation } from "@/state/use-simulation";
import { loadSimulationData } from "@/data/loader";
import { SimConfigPanel } from "./SimConfigPanel";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const SAMPLE_FILES = [
  { name: "Adversarial Red Team (75 epochs)", path: `${BASE_PATH}/sample-data/adversarial-redteam-75.json` },
  { name: "RLM Governance Lag (50 epochs)", path: `${BASE_PATH}/sample-data/rlm-governance-lag-50.json` },
  { name: "Collusion Detection (60 epochs)", path: `${BASE_PATH}/sample-data/collusion-detection-60.json` },
  { name: "24-Agent Mixed Demo", path: `${BASE_PATH}/sample-data/demo-history-24.json` },
  { name: "6-Agent Mixed Demo", path: `${BASE_PATH}/sample-data/demo-history.json` },
];

type Tab = "load" | "simulate";

export function DataLoader() {
  const { data, loadData } = useSimulation();
  const fileRef = useRef<HTMLInputElement>(null);
  const [tab, setTab] = useState<Tab>("load");

  const handleFile = useCallback(
    async (file: File) => {
      try {
        const sim = await loadSimulationData(file);
        loadData(sim);
      } catch (e) {
        alert(`Error loading file: ${e instanceof Error ? e.message : e}`);
      }
    },
    [loadData],
  );

  const handleSample = useCallback(
    async (path: string) => {
      try {
        const sim = await loadSimulationData(path);
        loadData(sim);
      } catch (e) {
        alert(`Error loading sample: ${e instanceof Error ? e.message : e}`);
      }
    },
    [loadData],
  );

  if (data) return null; // Hide once data is loaded

  return (
    <div className="absolute inset-0 flex items-center justify-center z-30 bg-bg">
      <div className="bg-panel border border-border rounded-xl p-8 max-w-md w-full mx-4 shadow-2xl max-h-[90vh] overflow-y-auto">
        <h1 className="text-xl font-bold mb-1">SWARM Isometric Viewer</h1>
        <p className="text-sm text-muted mb-4">
          Visualize multi-agent simulation data as an interactive isometric city
        </p>

        {/* Tab toggle */}
        <div className="flex gap-1 mb-4 bg-btn rounded-lg p-0.5">
          <button
            onClick={() => setTab("load")}
            className={`flex-1 py-1.5 rounded text-xs font-bold transition-colors ${
              tab === "load" ? "bg-panel text-text" : "text-muted hover:text-text"
            }`}
          >
            Load File
          </button>
          <button
            onClick={() => setTab("simulate")}
            className={`flex-1 py-1.5 rounded text-xs font-bold transition-colors ${
              tab === "simulate" ? "bg-panel text-text" : "text-muted hover:text-text"
            }`}
          >
            Run Simulation
          </button>
        </div>

        {tab === "load" ? (
          <>
            {/* File upload */}
            <div
              className="border-2 border-dashed border-border rounded-lg p-6 text-center cursor-pointer hover:border-accent transition-colors mb-4"
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                e.currentTarget.classList.add("border-accent");
              }}
              onDragLeave={(e) => {
                e.currentTarget.classList.remove("border-accent");
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.currentTarget.classList.remove("border-accent");
                const file = e.dataTransfer.files[0];
                if (file) handleFile(file);
              }}
            >
              <div className="text-3xl mb-2 text-accent">+</div>
              <p className="text-sm text-muted">
                Drop a <span className="text-text font-mono">history.json</span> here
              </p>
              <p className="text-xs text-muted mt-1">or click to browse</p>
            </div>
            <input
              ref={fileRef}
              type="file"
              accept=".json"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
              }}
            />

            {/* Sample data */}
            <div className="border-t border-border pt-4">
              <p className="text-xs text-muted mb-2">Or load sample data:</p>
              <div className="space-y-2">
                {SAMPLE_FILES.map((s) => (
                  <button
                    key={s.path}
                    onClick={() => handleSample(s.path)}
                    className="w-full text-left px-3 py-2 rounded bg-btn hover:bg-btn-hover transition-colors text-sm"
                  >
                    {s.name}
                  </button>
                ))}
              </div>
            </div>
          </>
        ) : (
          <SimConfigPanel onComplete={loadData} />
        )}
      </div>
    </div>
  );
}
