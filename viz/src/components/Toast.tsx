"use client";

import React, { useEffect } from "react";
import { useGame } from "@/state/game-context";

const TOAST_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  info: { bg: "bg-blue-900/80", border: "border-blue-700", text: "text-blue-200" },
  success: { bg: "bg-green-900/80", border: "border-green-700", text: "text-green-200" },
  warning: { bg: "bg-yellow-900/80", border: "border-yellow-700", text: "text-yellow-200" },
  error: { bg: "bg-red-900/80", border: "border-red-700", text: "text-red-200" },
};

export function ToastContainer() {
  const { state, dispatch } = useGame();

  // Auto-dismiss toasts after 4 seconds
  useEffect(() => {
    if (state.toasts.length === 0) return;
    const timers = state.toasts.map((toast) => {
      const age = Date.now() - toast.timestamp;
      const remaining = Math.max(0, 4000 - age);
      return setTimeout(() => {
        dispatch({ type: "REMOVE_TOAST", id: toast.id });
      }, remaining);
    });
    return () => timers.forEach(clearTimeout);
  }, [state.toasts, dispatch]);

  if (state.toasts.length === 0) return null;

  return (
    <div className="fixed top-16 right-4 z-[10001] space-y-2 pointer-events-none">
      {state.toasts.map((toast) => {
        const colors = TOAST_COLORS[toast.type] ?? TOAST_COLORS.info;
        return (
          <div
            key={toast.id}
            className={`${colors.bg} ${colors.border} ${colors.text} border rounded-lg px-4 py-2 text-sm shadow-lg backdrop-blur-sm animate-slide-in pointer-events-auto`}
          >
            {toast.text}
          </div>
        );
      })}
    </div>
  );
}
