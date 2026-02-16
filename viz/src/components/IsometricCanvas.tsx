"use client";

import React, { useRef, useEffect, useCallback } from "react";
import { useSimulation } from "@/state/use-simulation";
import { useCamera } from "@/state/use-camera";
import { render } from "@/engine/renderer";
import { screenToWorld } from "@/engine/camera";
import { screenToGrid } from "@/engine/isometric";
import { getBuildingBounds } from "@/engine/entities/agent-building";

export function IsometricCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const {
    agents,
    arcs,
    viewport,
    hoveredAgent,
    selectedAgent,
    currentEpochSnap,
    environment,
    overlays,
    particles,
    gridSize,
    setHovered,
    setSelected,
  } = useSimulation();
  const { handlePan, handleZoom, resetCamera, resize } = useCamera();

  const isDragging = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  // Resize observer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        canvas.width = width * devicePixelRatio;
        canvas.height = height * devicePixelRatio;
        canvas.style.width = width + "px";
        canvas.style.height = height + "px";
        resize(width, height);
      }
    });
    obs.observe(parent);
    return () => obs.disconnect();
  }, [resize]);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.save();
    ctx.scale(devicePixelRatio, devicePixelRatio);

    render(ctx, {
      agents,
      arcs,
      viewport,
      hoveredAgent,
      selectedAgent,
      epoch: currentEpochSnap,
      environment,
      overlays,
      particles,
      gridSize,
    });

    ctx.restore();
  });

  // Hit testing for hover/click
  const hitTest = useCallback(
    (clientX: number, clientY: number): string | null => {
      const world = screenToWorld(viewport, clientX, clientY);
      // Check agents in reverse depth order (front to back)
      const sorted = [...agents].sort(
        (a, b) => b.gridX + b.gridY - (a.gridX + a.gridY),
      );
      for (const agent of sorted) {
        const bounds = getBuildingBounds(agent);
        if (
          world.x >= bounds.minX &&
          world.x <= bounds.maxX &&
          world.y >= bounds.minY &&
          world.y <= bounds.maxY
        ) {
          return agent.id;
        }
      }
      return null;
    },
    [agents, viewport],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      isDragging.current = true;
      lastMouse.current = { x: e.clientX, y: e.clientY };
    },
    [],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      if (isDragging.current) {
        const dx = e.clientX - lastMouse.current.x;
        const dy = e.clientY - lastMouse.current.y;
        handlePan(dx, dy);
        lastMouse.current = { x: e.clientX, y: e.clientY };
      } else {
        const hit = hitTest(mx, my);
        setHovered(hit);
      }
    },
    [handlePan, hitTest, setHovered],
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      const wasDragging = isDragging.current;
      isDragging.current = false;

      if (!wasDragging || (Math.abs(e.clientX - lastMouse.current.x) < 3 && Math.abs(e.clientY - lastMouse.current.y) < 3)) {
        // Click - not a drag
        const rect = canvasRef.current?.getBoundingClientRect();
        if (!rect) return;
        const hit = hitTest(e.clientX - rect.left, e.clientY - rect.top);
        setSelected(hit);
      }
    },
    [hitTest, setSelected],
  );

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      handleZoom(e.deltaY, e.clientX - rect.left, e.clientY - rect.top);
    },
    [handleZoom],
  );

  const handleDoubleClick = useCallback(() => {
    resetCamera();
  }, [resetCamera]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 cursor-grab active:cursor-grabbing"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => {
        isDragging.current = false;
        setHovered(null);
      }}
      onWheel={handleWheel}
      onDoubleClick={handleDoubleClick}
    />
  );
}
