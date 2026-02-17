"use client";

import { useContext } from "react";
import { SimContext } from "./simulation-context";
import type { SimulationData } from "@/data/types";

export function useSimulation() {
  const { state, dispatch, codeTrailSystem, digitalRainRef, recompileStateRef } = useContext(SimContext);

  return {
    ...state,
    agentSnapshots: state.data?.agent_snapshots ?? [],
    loadData: (data: SimulationData) => dispatch({ type: "LOAD_DATA", data }),
    setHovered: (id: string | null) => dispatch({ type: "SET_HOVERED", agentId: id }),
    setSelected: (id: string | null) => dispatch({ type: "SET_SELECTED", agentId: id }),
    toggleOverlay: (key: keyof typeof state.overlays) =>
      dispatch({ type: "TOGGLE_OVERLAY", key }),
    codeTrailSystem,
    digitalRainRef,
    recompileStateRef,
  };
}
