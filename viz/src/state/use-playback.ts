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
  const stepTimeRef = useRef<number>(0);
  const lastParticleSpawn = useRef<number>(0);
  const lastTrailSpawn = useRef<number>(0);
  const prevEpochRef = useRef<number>(-1);
  const prevStepRef = useRef<number>(-1);
  const walkPhaseMap = useRef<Map<string, number>>(new Map());
  const facingMap = useRef<Map<string, number>>(new Map());
  const tickRef = useRef<(timestamp: number) => void>(() => {});

  const tick = useCallback(
    (timestamp: number) => {
      if (!state.data) return;

      const dt = lastTimeRef.current ? timestamp - lastTimeRef.current : 16;
      lastTimeRef.current = timestamp;

      let epoch = state.currentEpoch;
      let fraction = state.epochFraction;
      let step = state.currentStep;
      let maxStepInEpoch = state.maxStepInEpoch;
      const isStepMode = state.stepPlayback && state.eventIndex;

      if (state.playing) {
        if (isStepMode) {
          // ── Step-level playback ──
          const stepsInEpoch = maxStepInEpoch + 1;
          const stepDuration = ANIM.epochTransition / Math.max(stepsInEpoch, 1);
          stepTimeRef.current += dt * state.speed;

          if (stepTimeRef.current >= stepDuration) {
            stepTimeRef.current = 0;

            if (step < maxStepInEpoch) {
              // Advance to next step within epoch
              step = step + 1;
            } else {
              // End of epoch — advance to next epoch, reset step
              epoch = Math.min(epoch + 1, state.data.epoch_snapshots.length - 1);
              step = 0;
              maxStepInEpoch = state.eventIndex!.maxStep(epoch);
            }

            // Spawn arcs for this specific step
            const stepInteractions = state.eventIndex!.interactionsAt(epoch, step);
            if (stepInteractions.length > 0) {
              interactionSystem.current.addFromEventsAtStep(stepInteractions, epoch, step);
            }

            if (epoch >= state.data.epoch_snapshots.length - 1 && step >= maxStepInEpoch) {
              dispatch({ type: "SET_PLAYING", playing: false });
            }
          }

          // Fraction within epoch derived from step progress
          fraction = stepsInEpoch > 0 ? (step + stepTimeRef.current / stepDuration) / stepsInEpoch : 0;
          fraction = Math.min(fraction, 0.999);
        } else {
          // ── Epoch-level playback (legacy) ──
          epochTimeRef.current += dt * state.speed;
          fraction = epochTimeRef.current / ANIM.epochTransition;

          if (fraction >= 1) {
            fraction = 0;
            epochTimeRef.current = 0;
            epoch = Math.min(epoch + 1, state.data.epoch_snapshots.length - 1);

            // Spawn arcs for new epoch — use real events when available
            const epochSnap = state.data.epoch_snapshots[epoch];
            if (epochSnap) {
              const events = state.data.events;
              if (events && events.length > 0) {
                interactionSystem.current.addFromEvents(events, epoch);
              } else {
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

            if (epoch >= state.data.epoch_snapshots.length - 1) {
              dispatch({ type: "SET_PLAYING", playing: false });
            }
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
      prevStepRef.current = step;

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

        // Update facing based on walk direction
        const walkDirX = targetPos.x - homePos.x;
        if (Math.abs(walkDirX) > 1) {
          facingMap.current.set(arc.fromId, walkDirX > 0 ? 1 : -1);
        }
        walkingAgents.add(arc.fromId);
      }

      // Advance walk phase using persistent map so it accumulates across frames
      for (const id of walkingAgents) {
        const agent = agentMap.get(id)!;
        const motion = AGENT_MOTION[agent.agentType];
        const prev = walkPhaseMap.current.get(id) ?? 0;
        const next = prev + dt * motion.walkRate;
        walkPhaseMap.current.set(id, next);
        agent.walkPhase = next;
      }
      // Apply facing (persists after walking stops)
      for (const agent of agents) {
        agent.facing = facingMap.current.get(agent.id) ?? 1;
      }
      // Decay phase for agents that stopped walking (so they don't resume mid-swing)
      for (const [id] of walkPhaseMap.current) {
        if (!walkingAgents.has(id)) {
          walkPhaseMap.current.delete(id);
        }
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
        step,
        maxStepInEpoch,
      });

      rafRef.current = requestAnimationFrame((ts) => tickRef.current(ts));
    },
    [state.data, state.playing, state.speed, state.currentEpoch, state.epochFraction, state.currentStep, state.maxStepInEpoch, state.stepPlayback, state.eventIndex, dispatch, interactionSystem, particleSystem, codeTrailSystem, digitalRainRef, recompileStateRef, state.environment.threatLevel, state.viewport.height, state.viewport.width, agentPositions, agentsByEpoch],
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
      stepTimeRef.current = 0;
      dispatch({ type: "SET_EPOCH", epoch });

      // Generate arcs for this epoch — use real events when available
      if (state.data) {
        interactionSystem.current.clear();

        if (state.stepPlayback && state.eventIndex) {
          // In step mode, spawn arcs for step 0 of the new epoch
          const stepInteractions = state.eventIndex.interactionsAt(epoch, 0);
          if (stepInteractions.length > 0) {
            interactionSystem.current.addFromEventsAtStep(stepInteractions, epoch, 0);
          }
        } else {
          const epochSnap = state.data.epoch_snapshots[epoch];
          if (epochSnap) {
            const events = state.data.events;
            if (events && events.length > 0) {
              interactionSystem.current.addFromEvents(events, epoch);
            } else {
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
        }
      }
    },
    [dispatch, state.data, state.stepPlayback, state.eventIndex, interactionSystem, agentPositions],
  );

  const setStep = useCallback(
    (step: number) => {
      stepTimeRef.current = 0;
      dispatch({ type: "SET_STEP", step });

      // Rebuild arcs for the target step: clear and replay all steps up to this one
      if (state.data && state.eventIndex) {
        interactionSystem.current.clear();
        // Spawn arcs for all steps up to and including the target step
        // so that in-progress arcs from earlier steps are visible
        const epoch = state.currentEpoch;
        for (let s = Math.max(0, step - 2); s <= step; s++) {
          const stepInteractions = state.eventIndex.interactionsAt(epoch, s);
          if (stepInteractions.length > 0) {
            interactionSystem.current.addFromEventsAtStep(stepInteractions, epoch, s);
          }
        }
      }
    },
    [dispatch, state.data, state.eventIndex, state.currentEpoch, interactionSystem],
  );

  const setSpeed = useCallback(
    (speed: number) => dispatch({ type: "SET_SPEED", speed }),
    [dispatch],
  );

  const toggleStepPlayback = useCallback(
    () => dispatch({ type: "SET_STEP_PLAYBACK", enabled: !state.stepPlayback }),
    [dispatch, state.stepPlayback],
  );

  const stepForward = useCallback(() => {
    if (!state.data || !state.eventIndex) return;
    const maxStep = state.eventIndex.maxStep(state.currentEpoch);
    if (state.currentStep < maxStep) {
      setStep(state.currentStep + 1);
    } else if (state.currentEpoch < state.data.epoch_snapshots.length - 1) {
      // Advance to next epoch, step 0
      setEpoch(state.currentEpoch + 1);
    }
  }, [state.data, state.eventIndex, state.currentEpoch, state.currentStep, setStep, setEpoch]);

  const stepBack = useCallback(() => {
    if (!state.eventIndex) return;
    if (state.currentStep > 0) {
      setStep(state.currentStep - 1);
    } else if (state.currentEpoch > 0) {
      // Go to last step of previous epoch
      const prevEpoch = state.currentEpoch - 1;
      epochTimeRef.current = 0;
      stepTimeRef.current = 0;
      dispatch({ type: "SET_EPOCH", epoch: prevEpoch });
      const prevMax = state.eventIndex.maxStep(prevEpoch);
      setStep(prevMax);
    }
  }, [state.eventIndex, state.currentEpoch, state.currentStep, setStep, dispatch]);

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
    setStep,
    setSpeed,
    stepForward,
    stepBack,
    toggleStepPlayback,
    playing: state.playing,
    speed: state.speed,
    currentEpoch: state.currentEpoch,
    currentStep: state.currentStep,
    maxStepInEpoch: state.maxStepInEpoch,
    stepPlayback: state.stepPlayback,
    maxEpoch: state.data ? state.data.epoch_snapshots.length - 1 : 0,
  };
}
