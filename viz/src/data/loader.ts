import type { SimulationData, InteractionEvent } from "./types";

/** Load simulation data from a history.json file or File object */
export async function loadSimulationData(source: string | File): Promise<SimulationData> {
  let raw: string;
  if (typeof source === "string") {
    const resp = await fetch(source);
    if (!resp.ok) throw new Error(`Failed to load: ${resp.statusText}`);
    raw = await resp.text();
  } else {
    raw = await source.text();
  }

  const data = JSON.parse(raw) as SimulationData;
  validateData(data);
  return data;
}

/** Parse JSONL events file */
export async function loadEvents(source: string | File): Promise<InteractionEvent[]> {
  let raw: string;
  if (typeof source === "string") {
    const resp = await fetch(source);
    if (!resp.ok) throw new Error(`Failed to load events: ${resp.statusText}`);
    raw = await resp.text();
  } else {
    raw = await source.text();
  }

  return raw
    .trim()
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as InteractionEvent)
    .filter((evt) => evt.event_type === "interaction");
}

function validateData(data: SimulationData) {
  if (!data.epoch_snapshots || !Array.isArray(data.epoch_snapshots)) {
    throw new Error("Missing epoch_snapshots array");
  }
  if (!Array.isArray(data.agent_snapshots)) {
    // Agent snapshots might be empty - that's ok, we'll derive from epochs
    data.agent_snapshots = data.agent_snapshots ?? [];
  }
  if (data.epoch_snapshots.length === 0) {
    throw new Error("No epoch snapshots found");
  }
}
