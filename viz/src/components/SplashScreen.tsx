"use client";

import React, { useState, useEffect, useCallback } from "react";

export function SplashScreen() {
  const [phase, setPhase] = useState<"enter" | "exit" | "done">("enter");

  const skip = useCallback(() => {
    if (phase !== "done") setPhase("exit");
  }, [phase]);

  useEffect(() => {
    // After enter + glow hold (2.5s), start exit
    const timer = setTimeout(() => {
      setPhase((p) => (p === "enter" ? "exit" : p));
    }, 2500);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (phase === "exit") {
      const timer = setTimeout(() => setPhase("done"), 700);
      return () => clearTimeout(timer);
    }
  }, [phase]);

  if (phase === "done") return null;

  return (
    <div
      className={`absolute inset-0 z-50 flex items-center justify-center bg-bg cursor-pointer ${
        phase === "exit" ? "animate-[splash-exit_0.7s_ease-in_forwards]" : ""
      }`}
      onClick={skip}
    >
      <img
        src={`${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/splash.png`}
        alt="SWARM"
        className={`w-64 h-64 object-contain ${
          phase === "enter"
            ? "animate-[splash-enter_0.8s_ease-out_forwards,splash-glow_1.7s_ease-in-out_0.8s_infinite]"
            : ""
        }`}
      />
    </div>
  );
}
