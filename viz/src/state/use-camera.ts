"use client";

import { useContext, useCallback } from "react";
import { SimContext } from "./simulation-context";
import * as cam from "@/engine/camera";
import { gridToScreen } from "@/engine/isometric";
import { AGENT_GRID_SPACING } from "@/engine/constants";

export function useCamera() {
  const { state, dispatch, agentPositions } = useContext(SimContext);

  const setViewport = useCallback(
    (viewport: typeof state.viewport) => dispatch({ type: "SET_VIEWPORT", viewport }),
    [dispatch],
  );

  const handlePan = useCallback(
    (dx: number, dy: number) => {
      setViewport(cam.pan(state.viewport, dx, dy));
    },
    [state.viewport, setViewport],
  );

  const handleZoom = useCallback(
    (delta: number, focusX: number, focusY: number) => {
      setViewport(cam.zoom(state.viewport, delta, focusX, focusY));
    },
    [state.viewport, setViewport],
  );

  const resetCamera = useCallback(() => {
    if (agentPositions.current.size === 0) {
      setViewport(cam.centerOn(state.viewport, 0, 0));
      return;
    }

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const pos of agentPositions.current.values()) {
      const screen = gridToScreen(pos.gridX, pos.gridY);
      minX = Math.min(minX, screen.x);
      minY = Math.min(minY, screen.y - 100);
      maxX = Math.max(maxX, screen.x);
      maxY = Math.max(maxY, screen.y + 20);
    }

    setViewport(cam.fitBounds(state.viewport, minX, minY, maxX, maxY));
  }, [state.viewport, setViewport, agentPositions]);

  const resize = useCallback(
    (width: number, height: number) => {
      setViewport({ ...state.viewport, width, height });
    },
    [state.viewport, setViewport],
  );

  return {
    viewport: state.viewport,
    handlePan,
    handleZoom,
    resetCamera,
    resize,
  };
}
