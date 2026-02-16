"use client";

import { useContext, useCallback, useRef, useEffect } from "react";
import { SimContext } from "./simulation-context";
import { interpolateAgents } from "@/engine/systems/animation-system";
import { interpolateEnvironment } from "@/engine/systems/environment-system";
import { interpolateEpoch } from "@/data/interpolator";
import { createSnowflakes, createHazardParticles } from "@/engine/entities/effects";
import { gridToScreen } from "@/engine/isometric";
import { ANIM } from "@/engine/constants";

export function usePlayback() {
  const { state, dispatch, interactionSystem, particleSystem, agentPositions, agentsByEpoch } =
    useContext(SimContext);

  const rafRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const epochTimeRef = useRef<number>(0);
  const lastParticleSpawn = useRef<number>(0);

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

      // Update systems
      interactionSystem.current.update(dt, epoch);
      particleSystem.current.update(dt);

      // Interpolate agent states
      const prevEpochAgents = agentsByEpoch.current.get(epoch) ?? new Map();
      const nextEpochIdx = Math.min(epoch + 1, state.data.epoch_snapshots.length - 1);
      const nextEpochAgents = agentsByEpoch.current.get(nextEpochIdx) ?? prevEpochAgents;
      const agents = interpolateAgents(prevEpochAgents, nextEpochAgents, fraction, agentPositions.current);

      // Spawn status particles periodically
      if (timestamp - lastParticleSpawn.current > 500) {
        lastParticleSpawn.current = timestamp;
        for (const agent of agents) {
          const screen = gridToScreen(agent.gridX, agent.gridY);
          if (agent.isFrozen) {
            particleSystem.current.add(createSnowflakes(screen.x, screen.y - agent.floors * 10, 2));
          }
          if (agent.isQuarantined) {
            particleSystem.current.add(createHazardParticles(screen.x, screen.y, 2));
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

      rafRef.current = requestAnimationFrame(tick);
    },
    [state.data, state.playing, state.speed, state.currentEpoch, state.epochFraction, dispatch, interactionSystem, particleSystem, agentPositions, agentsByEpoch],
  );

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
