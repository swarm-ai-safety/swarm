"use client";

import { useState, useRef, useCallback } from "react";
import type { SimulationData } from "@/data/types";
import type { ScenarioConfig, WorkerResponse } from "@/engine/sim/types";
import { runSimulation } from "@/engine/sim/orchestrator";

export type SimWorkerStatus = "idle" | "running" | "done" | "error";

export function useSimWorker() {
  const [status, setStatus] = useState<SimWorkerStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SimulationData | null>(null);
  const cancelRef = useRef(false);

  const run = useCallback((config: ScenarioConfig): Promise<SimulationData | null> => {
    setStatus("running");
    setProgress(0);
    setError(null);
    setResult(null);
    cancelRef.current = false;

    // Try Web Worker first, fall back to main thread
    return new Promise((resolve) => {
      try {
        const worker = new Worker(
          new URL("@/engine/sim/worker.ts", import.meta.url),
          { type: "module" },
        );

        worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
          const msg = e.data;
          if (msg.type === "progress") {
            setProgress(msg.epoch / msg.totalEpochs);
          } else if (msg.type === "complete") {
            setStatus("done");
            setProgress(1);
            setResult(msg.data);
            worker.terminate();
            resolve(msg.data);
          } else if (msg.type === "error") {
            setStatus("error");
            setError(msg.message);
            worker.terminate();
            resolve(null);
          }
        };

        worker.onerror = () => {
          // Worker failed to load — fall back to main thread
          worker.terminate();
          runOnMainThread(config, resolve);
        };

        worker.postMessage({ type: "run", config });
      } catch {
        // Workers not available — run on main thread
        runOnMainThread(config, resolve);
      }
    });

    function runOnMainThread(cfg: ScenarioConfig, resolve: (d: SimulationData | null) => void) {
      try {
        const data = runSimulation(cfg, (epoch, total) => {
          setProgress(epoch / total);
        });
        setStatus("done");
        setProgress(1);
        setResult(data);
        resolve(data);
      } catch (err) {
        setStatus("error");
        setError(err instanceof Error ? err.message : String(err));
        resolve(null);
      }
    }
  }, []);

  const cancel = useCallback(() => {
    cancelRef.current = true;
    setStatus("idle");
  }, []);

  return { status, progress, error, result, run, cancel };
}
