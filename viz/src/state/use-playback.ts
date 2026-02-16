"use client";

import { useContext, useCallback, useRef, useEffect } from "react";
import { SimContext } from "./simulation-context";
import { interpolateAgents } from "@/engine/systems/animation-system";
import { interpolateEnvironment } from "@/engine/systems/environment-system";
import { interpolateEpoch } from "@/data/interpolator";
import { createSnowflakes, createHazardParticles, initDigitalRain, updateDigitalRain } from "@/engine/entities/effects";
import { gridToScreen } from "@/engine/isometric";
import { ANIM, AGENT_MOTION } from "@/engine/constants";
import { smootherstep, hashString } from "@/utils/math";

export function usePlayback() {
  const { state, dispatch, interactionSystem, particleSystem, codeTrailSystem, digitalRainRef, recompileStateRef, agentPositions, agentsByEpoch } =
    useContext(SimContext);

  const rafRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const epochTimeRef = useRef<number>(0);
  const lastParticleSpawn = useRef<number>(0);
  const lastTrailSpawn = useRef<number>(0);
  const prevEpochRef = useRef<number>(-1);
  const tickRef = useRef<(timestamp: number) => void>(() => {});

  const tick = useCallback(
    (timestamp: number) => {
      if (!state.data) return;

      const dt = lastTimeRef.current ? timestamp - lastTimeRef.current : 16;
      lastTimeRef.current = timestamp;

      let epoch = state.currentEpoch;
      let fraction = state.epochFraction;

      if (state.playing) {
        epochTimeRef.current += dt * state.speed;
        fraction = epochTimeRef.current / ANIM.epochTransition;

        if (fraction >= 1) {
          fraction = 0;
          epochTimeRef.current = 0;
          epoch = Math.min(epoch + 1, state.data.epoch_snapshots.length - 1);

          // Spawn arcs for new epoch
          const epochSnap = state.data.epoch_snapshots[epoch];
          if (epochSnap) {
            const agentIds = [...agentPositions.current.keys()];
            const acceptRate =
              epochSnap.total_interactions > 0
                ? epochSnap.accepted_interactions / epochSnap.total_interactions
                : 0.5;
            interactionSystem.current.addSyntheticArcs(
              agentIds,
              epoch,
              epochSnap.total_interactions,
              epochSnap.avg_p,
              acceptRate,
            );
          }

          if (epoch >= state.data.epoch_snapshots.length - 1) {
            dispatch({ type: "SET_PLAYING", playing: false });
          }
        }
      }

      // Detect epoch transitions for recompile flash
      if (prevEpochRef.current >= 0 && epoch !== prevEpochRef.current) {
        recompileStateRef.current = {
          active: true,
          startTime: Date.now(),
          duration: 400,
          scanlineY: 0,
        };
      }
      prevEpochRef.current = epoch;

      // Expire recompile flash
      const recompile = recompileStateRef.current;
      if (recompile.active && Date.now() - recompile.startTime > recompile.duration) {
        recompile.active = false;
      }

      // Update systems
      interactionSystem.current.update(dt, epoch);
      particleSystem.current.update(dt);
      codeTrailSystem.current.update(dt);

      // Initialize / update digital rain
      if (!digitalRainRef.current && state.viewport.width > 0) {
        digitalRainRef.current = initDigitalRain(state.viewport.width);
      }
      if (digitalRainRef.current) {
        // Threat level is computed later but we use prev frame's estimate
        const prevThreat = state.environment.threatLevel;
        updateDigitalRain(digitalRainRef.current, dt, state.viewport.height, prevThreat);
      }

      // Interpolate agent states
      const prevEpochAgents = agentsByEpoch.current.get(epoch) ?? new Map();
      const nextEpochIdx = Math.min(epoch + 1, state.data.epoch_snapshots.length - 1);
      const nextEpochAgents = agentsByEpoch.current.get(nextEpochIdx) ?? prevEpochAgents;
      const agents = interpolateAgents(prevEpochAgents, nextEpochAgents, fraction, agentPositions.current);

      // Compute walk offsets from active arcs
      const agentMap = new Map(agents.map((a) => [a.id, a]));
      const walkingAgents = new Set<string>();
      const activeArcs = interactionSystem.current.arcs;

      for (const arc of activeArcs) {
        if (arc.progress >= 1) continue;
        const fromAgent = agentMap.get(arc.fromId);
        const toAgent = agentMap.get(arc.toId);
        if (!fromAgent || !toAgent) continue;
        if (walkingAgents.has(arc.fromId)) continue;

        const homePos = gridToScreen(fromAgent.gridX, fromAgent.gridY);
        const targetPos = gridToScreen(toAgent.gridX, toAgent.gridY);
        const motion = AGENT_MOTION[fromAgent.agentType];

        // Smootherstep for gradual acceleration/deceleration
        const p = arc.progress;
        let walkT: number;
        if (p < 0.45) {
          walkT = smootherstep(0, 0.45, p);
        } else if (p < 0.55) {
          walkT = 1.0;
        } else {
          walkT = 1 - smootherstep(0.55, 1, p);
        }

        // Base linear offset toward target
        const dx = (targetPos.x - homePos.x) * walkT * 0.85;
        const dy = (targetPos.y - homePos.y) * walkT * 0.85;

        // Perpendicular lateral sway
        const dist = Math.sqrt((targetPos.x - homePos.x) ** 2 + (targetPos.y - homePos.y) ** 2);
        const perpX = dist > 0 ? -(targetPos.y - homePos.y) / dist : 0;
        const perpY = dist > 0 ? (targetPos.x - homePos.x) / dist : 0;
        const swayAmount = motion.swayAmplitude * Math.sin(fromAgent.walkPhase * 3.5) * 4;

        // Subtle path curve — deterministic per agent
        const curveBias = ((hashString(fromAgent.id) % 200) - 100) / 100 * 0.15;
        const curveOffset = Math.sin(walkT * Math.PI) * curveBias * dist;

        fromAgent.walkOffsetX = dx + (perpX * swayAmount) + (perpX * curveOffset);
        fromAgent.walkOffsetY = dy + (perpY * swayAmount) + (perpY * curveOffset);
        walkingAgents.add(arc.fromId);
      }

      // Advance walk phase for walking agents using per-type walk rate
      for (const id of walkingAgents) {
        const agent = agentMap.get(id)!;
        const motion = AGENT_MOTION[agent.agentType];
        agent.walkPhase += dt * motion.walkRate;
      }

      // Spawn code trail particles — interval oscillates 25-55ms for organic feel
      const trailInterval = 40 + Math.sin(timestamp * 0.003) * 15;
      if (timestamp - lastTrailSpawn.current > trailInterval) {
        lastTrailSpawn.current = timestamp;
        for (const id of walkingAgents) {
          const agent = agentMap.get(id)!;
          const screen = gridToScreen(agent.gridX, agent.gridY);
          const px = screen.x + agent.walkOffsetX;
          const py = screen.y + agent.walkOffsetY;
          codeTrailSystem.current.spawnBurst(px, py - agent.scale * 28, agent.avgP, 2);
        }
      }

      // Spawn status particles periodically
      if (timestamp - lastParticleSpawn.current > 500) {
        lastParticleSpawn.current = timestamp;
        for (const agent of agents) {
          const screen = gridToScreen(agent.gridX, agent.gridY);
          const px = screen.x + agent.walkOffsetX;
          const py = screen.y + agent.walkOffsetY;
          if (agent.isFrozen) {
            particleSystem.current.add(createSnowflakes(px, py - agent.scale * 56, 2));
          }
          if (agent.isQuarantined) {
            particleSystem.current.add(createHazardParticles(px, py, 2));
          }
        }
      }

      // Interpolate environment
      const prevEpochSnap = state.data.epoch_snapshots[epoch];
      const nextEpochSnap = state.data.epoch_snapshots[nextEpochIdx];
      const environment = interpolateEnvironment(prevEpochSnap, nextEpochSnap, fraction);

      // Interpolate epoch snapshot for HUD
      const epochSnap = prevEpochSnap && nextEpochSnap
        ? interpolateEpoch(prevEpochSnap, nextEpochSnap, fraction)
        : prevEpochSnap ?? null;

      dispatch({
        type: "TICK",
        agents,
        arcs: interactionSystem.current.arcs,
        environment,
        particles: particleSystem.current.particles,
        epoch,
        fraction,
        epochSnap: epochSnap,
      });

      rafRef.current = requestAnimationFrame((ts) => tickRef.current(ts));
    },
    [state.data, state.playing, state.speed, state.currentEpoch, state.epochFraction, dispatch, interactionSystem, particleSystem, codeTrailSystem, digitalRainRef, recompileStateRef, state.environment.threatLevel, state.viewport.height, state.viewport.width, agentPositions, agentsByEpoch],
  );

  // Keep tickRef in sync so RAF callback always calls latest tick
  useEffect(() => {
    tickRef.current = tick;
  }, [tick]);

  const play = useCallback(() => {
    dispatch({ type: "SET_PLAYING", playing: true });
  }, [dispatch]);

  const pause = useCallback(() => {
    dispatch({ type: "SET_PLAYING", playing: false });
  }, [dispatch]);

  const setEpoch = useCallback(
    (epoch: number) => {
      epochTimeRef.current = 0;
      dispatch({ type: "SET_EPOCH", epoch });

      // Generate arcs for this epoch
      if (state.data) {
        interactionSystem.current.clear();
        const epochSnap = state.data.epoch_snapshots[epoch];
        if (epochSnap) {
          const agentIds = [...agentPositions.current.keys()];
          const acceptRate =
            epochSnap.total_interactions > 0
              ? epochSnap.accepted_interactions / epochSnap.total_interactions
              : 0.5;
          interactionSystem.current.addSyntheticArcs(
            agentIds,
            epoch,
            epochSnap.total_interactions,
            epochSnap.avg_p,
            acceptRate,
          );
        }
      }
    },
    [dispatch, state.data, interactionSystem, agentPositions],
  );

  const setSpeed = useCallback(
    (speed: number) => dispatch({ type: "SET_SPEED", speed }),
    [dispatch],
  );

  // Start/stop RAF loop
  useEffect(() => {
    if (state.data) {
      rafRef.current = requestAnimationFrame(tick);
    }
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [tick, state.data]);

  return {
    play,
    pause,
    setEpoch,
    setSpeed,
    playing: state.playing,
    speed: state.speed,
    currentEpoch: state.currentEpoch,
    maxEpoch: state.data ? state.data.epoch_snapshots.length - 1 : 0,
  };
}
