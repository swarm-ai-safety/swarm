/** Module-level store for the last-used ScenarioConfig. Used by governance intervention. */

import type { ScenarioConfig } from "./types";

let _lastConfig: ScenarioConfig | null = null;

export function storeConfig(config: ScenarioConfig): void {
  _lastConfig = { ...config };
}

export function getStoredConfig(): ScenarioConfig | null {
  return _lastConfig;
}
