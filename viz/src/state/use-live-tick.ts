"use client";

import { useContext, useRef, useEffect, useCallback } from "react";
import { SimContext } from "./simulation-context";
import { useGame } from "./game-context";
import { interpolateAgents, computeAgentPositions } from "@/engine/systems/animation-system";
import { interpolateEnvironment } from "@/engine/systems/environment-system";
import { groupAgentsByEpoch } from "@/data/normalizer";
import { gridToScreen } from "@/engine/isometric";
import { createSnowflakes, createHazardParticles, initDigitalRain, updateDigitalRain } from "@/engine/entities/effects";

/**
 * Live tick hook — drives the simulation forward step-by-step
 * and feeds updated data into the existing visualization pipeline.
 *
 * This runs alongside the existing replay tick (use-playback) but only
 * activates when the game is in live mode (isLive = true).
 */
export function useLiveTick() {
  const {
    state: simState,
    dispatch: simDispatch,
    interactionSystem: interactionSystemRef,
    particleSystem: particleSystemRef,
    codeTrailSystem: codeTrailSystemRef,
    digitalRainRef,
    recompileStateRef,
    agentPositions: agentPositionsRef,
    agentsByEpoch: agentsByEpochRef,
  } = useContext(SimContext);

  const { state: gameState, engineRef, checkConditions, dispatch: gameDispatch, computeRedTeamScore } = useGame();

  const rafRef = useRef<number>(0);
  const lastTickTime = useRef<number>(0);
  const accumulatedTime = useRef<number>(0);
  const lastParticleSpawn = useRef<number>(0);
  // Use a ref to break the self-reference cycle (same pattern as use-playback.ts)
  const tickRef = useRef<(timestamp: number) => void>(() => {});

  const liveTick = useCallback(
    (timestamp: number) => {
      const engine = engineRef.current;
      if (!engine || !gameState.isLive) return;

      const dt = lastTickTime.current ? timestamp - lastTickTime.current : 16;
      lastTickTime.current = timestamp;

      // Accumulate time and tick simulation at configured rate
      if (!gameState.isPaused) {
        accumulatedTime.current += dt;

        let steppedThisFrame = false;
        while (accumulatedTime.current >= gameState.tickRate) {
          accumulatedTime.current -= gameState.tickRate;

          const { event, epochCompleted } = engine.tick();
          steppedThisFrame = true;

          // Feed interaction arc to the interaction system
          if (event.accepted) {
            interactionSystemRef.current.addFromEventsAtStep([event], engine.epoch, engine.step);
          }

          // If epoch completed, check campaign conditions
          if (epochCompleted) {
            if (gameState.mode === "campaign") {
              checkConditions(engine.getEpochSnapshots());
            }
            if (gameState.mode === "redteam") {
              const score = computeRedTeamScore();
              gameDispatch({ type: "UPDATE_REDTEAM_SCORE", score });
            }

            // Trigger recompile flash on epoch boundary
            recompileStateRef.current = {
              active: true,
              startTime: Date.now(),
              duration: 400,
              scanlineY: 0,
            };
          }
        }

        // Rebuild visualization data from engine state
        if (steppedThisFrame) {
          const simData = engine.toSimulationData();

          // Update agent positions (may have spawned/removed agents)
          const allIds = engine.agents.map((a) => a.id);
          const currentPositions = agentPositionsRef.current;
          let needsReposition = false;
          for (const id of allIds) {
            if (!currentPositions.has(id)) {
              needsReposition = true;
              break;
            }
          }
          if (needsReposition) {
            agentPositionsRef.current = computeAgentPositions(allIds);
          }

          // Update agent-by-epoch map for interpolation
          agentsByEpochRef.current = groupAgentsByEpoch(simData.agent_snapshots);

          // Load data into sim context if not already loaded
          if (!simState.data || simState.data.simulation_id !== simData.simulation_id) {
            simDispatch({ type: "LOAD_DATA", data: simData });
          }
        }
      }

      // Update visual systems (always run for smooth animation)
      interactionSystemRef.current.update(dt, engine.epoch);
      particleSystemRef.current.update(dt);
      codeTrailSystemRef.current.update(dt);

      // Expire recompile flash
      const recompile = recompileStateRef.current;
      if (recompile.active && Date.now() - recompile.startTime > recompile.duration) {
        recompile.active = false;
      }

      // Digital rain
      if (!digitalRainRef.current && simState.viewport.width > 0) {
        digitalRainRef.current = initDigitalRain(simState.viewport.width);
      }
      if (digitalRainRef.current) {
        updateDigitalRain(digitalRainRef.current, dt, simState.viewport.height, simState.environment.threatLevel);
      }

      // Build agent visuals from current engine state.
      // In live mode we pass the same snapshot for prev/current with fraction 0
      // because there is no replay interpolation — the engine state IS the truth.
      const epochSnap = engine.getCurrentEpochSnapshot();
      const prevAgents = agentsByEpochRef.current.get(engine.epoch) ?? new Map();
      const agents = interpolateAgents(prevAgents, prevAgents, 0, agentPositionsRef.current);

      // Particle effects
      if (timestamp - lastParticleSpawn.current > 500) {
        lastParticleSpawn.current = timestamp;
        for (const agent of agents) {
          const screen = gridToScreen(agent.gridX, agent.gridY);
          if (agent.isFrozen) {
            particleSystemRef.current.add(createSnowflakes(screen.x, screen.y - agent.scale * 56, 2));
          }
          if (agent.isQuarantined) {
            particleSystemRef.current.add(createHazardParticles(screen.x, screen.y, 2));
          }
        }
      }

      // Environment — same snapshot on both sides (no replay interpolation in live mode)
      const environment = interpolateEnvironment(epochSnap, epochSnap, 0);

      // Dispatch tick to simulation context
      simDispatch({
        type: "TICK",
        agents,
        arcs: interactionSystemRef.current.arcs,
        environment,
        particles: particleSystemRef.current.particles,
        epoch: engine.epoch,
        fraction: engine.step / Math.max(engine.config.stepsPerEpoch, 1),
        epochSnap,
      });

      // Schedule next frame via ref (avoids self-reference lint error)
      rafRef.current = requestAnimationFrame((ts) => tickRef.current(ts));
    },
    [
      gameState.isLive,
      gameState.isPaused,
      gameState.tickRate,
      gameState.mode,
      engineRef,
      simDispatch,
      simState.data,
      simState.viewport.width,
      simState.viewport.height,
      simState.environment.threatLevel,
      interactionSystemRef,
      particleSystemRef,
      codeTrailSystemRef,
      digitalRainRef,
      recompileStateRef,
      agentPositionsRef,
      agentsByEpochRef,
      checkConditions,
      computeRedTeamScore,
      gameDispatch,
    ],
  );

  // Keep tickRef in sync so RAF callback always calls latest liveTick
  useEffect(() => {
    tickRef.current = liveTick;
  }, [liveTick]);

  // Start/stop the live tick loop based on game state
  useEffect(() => {
    if (gameState.isLive) {
      lastTickTime.current = 0;
      accumulatedTime.current = 0;
      rafRef.current = requestAnimationFrame(liveTick);
    }
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [gameState.isLive, liveTick]);
}
