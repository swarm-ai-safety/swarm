/** Web Worker entry point for simulation. */

import type { WorkerRequest, WorkerResponse } from "./types";
import { runSimulation } from "./orchestrator";

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;

  if (msg.type === "run") {
    try {
      const data = runSimulation(msg.config, (epoch, total) => {
        const resp: WorkerResponse = { type: "progress", epoch, totalEpochs: total };
        self.postMessage(resp);
      });
      const resp: WorkerResponse = { type: "complete", data };
      self.postMessage(resp);
    } catch (err) {
      const resp: WorkerResponse = {
        type: "error",
        message: err instanceof Error ? err.message : String(err),
      };
      self.postMessage(resp);
    }
  }
};
