"use client";

import React from "react";
import { useGame } from "@/state/game-context";
import { CAMPAIGN_LEVELS, describeWinCondition, describeLoseCondition } from "@/engine/sim/campaign";
import type { ChallengeLevel } from "@/engine/sim/campaign";

const DIFFICULTY_COLORS: Record<string, string> = {
  easy: "#6FCF97",
  medium: "#F2994A",
  hard: "#EB5757",
  expert: "#9B51E0",
};

export function CampaignPanel() {
  const { state, startCampaignLevel } = useGame();

  const isUnlocked = (level: ChallengeLevel): boolean => {
    if (!level.unlockAfter) return true;
    return state.completedLevels.includes(level.unlockAfter);
  };

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted">
        Complete challenges to master governance mechanics. Each level introduces new complexity.
      </p>

      <div className="space-y-2">
        {CAMPAIGN_LEVELS.map((level) => {
          const unlocked = isUnlocked(level);
          const completed = state.completedLevels.includes(level.id);

          return (
            <div
              key={level.id}
              className={`border rounded-lg p-3 transition-colors ${
                unlocked
                  ? "border-border hover:border-accent cursor-pointer"
                  : "border-border/50 opacity-50 cursor-not-allowed"
              } ${completed ? "bg-green-950/20" : "bg-panel"}`}
              onClick={() => unlocked && startCampaignLevel(level)}
            >
              <div className="flex items-center gap-2 mb-1">
                {completed && <span className="text-green-400 text-sm">&#x2713;</span>}
                {!unlocked && <span className="text-muted text-sm">&#x1F512;</span>}
                <span className="text-sm font-bold">{level.name}</span>
                <span
                  className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded-full ml-auto"
                  style={{ color: DIFFICULTY_COLORS[level.difficulty], backgroundColor: `${DIFFICULTY_COLORS[level.difficulty]}20` }}
                >
                  {level.difficulty}
                </span>
              </div>
              <p className="text-xs text-muted">{level.description}</p>
              {unlocked && (
                <div className="mt-2 text-[10px] text-muted space-y-0.5">
                  <div>Win: {describeWinCondition(level.winCondition)}</div>
                  <div>Lose: {describeLoseCondition(level.loseCondition)}</div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** Overlay shown when a level ends (win or lose) */
export function LevelEndOverlay() {
  const { state, startCampaignLevel, stopGame } = useGame();

  if (state.levelStatus !== "won" && state.levelStatus !== "lost") return null;

  const won = state.levelStatus === "won";
  const level = state.currentLevel;

  // Find next level
  const currentIdx = level ? CAMPAIGN_LEVELS.findIndex((l) => l.id === level.id) : -1;
  const nextLevel = currentIdx >= 0 && currentIdx < CAMPAIGN_LEVELS.length - 1
    ? CAMPAIGN_LEVELS[currentIdx + 1]
    : null;

  return (
    <div className="absolute inset-0 flex items-center justify-center z-[10002] bg-black/60 backdrop-blur-sm">
      <div className="bg-panel border border-border rounded-xl p-8 max-w-sm w-full mx-4 shadow-2xl text-center">
        <div className={`text-5xl mb-4 ${won ? "text-green-400" : "text-red-400"}`}>
          {won ? "\u2713" : "\u2717"}
        </div>
        <h2 className={`text-xl font-bold mb-2 ${won ? "text-green-400" : "text-red-400"}`}>
          {won ? "Level Complete!" : "Level Failed"}
        </h2>
        {level && (
          <p className="text-sm text-muted mb-4">{level.name}</p>
        )}
        {!won && level && (
          <p className="text-xs text-muted mb-4">
            {level.hints[state.hintIndex] ?? level.hints[0]}
          </p>
        )}
        <div className="flex gap-2 justify-center">
          {!won && level && (
            <button
              onClick={() => startCampaignLevel(level)}
              className="px-4 py-2 rounded bg-accent text-bg font-bold text-sm hover:opacity-90 transition-opacity"
            >
              Retry
            </button>
          )}
          {won && nextLevel && (
            <button
              onClick={() => startCampaignLevel(nextLevel)}
              className="px-4 py-2 rounded bg-accent text-bg font-bold text-sm hover:opacity-90 transition-opacity"
            >
              Next Level
            </button>
          )}
          <button
            onClick={stopGame}
            className="px-4 py-2 rounded bg-btn hover:bg-btn-hover text-sm transition-colors"
          >
            Back to Menu
          </button>
        </div>
      </div>
    </div>
  );
}
