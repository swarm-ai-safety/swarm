import React from "react";
import { AbsoluteFill, useCurrentFrame, spring, interpolate } from "remotion";
import { colors, fonts } from "../theme";

export const Problem: React.FC = () => {
  const frame = useCurrentFrame();

  const titleProgress = spring({ frame, fps: 30, config: { damping: 200 } });
  const titleY = interpolate(titleProgress, [0, 1], [30, 0]);

  const safeP = Math.max(
    0,
    spring({ frame: frame - 30, fps: 30, config: { damping: 200 } }),
  );
  const unsafeP = Math.max(
    0,
    spring({ frame: frame - 40, fps: 30, config: { damping: 200 } }),
  );
  const strikeP = Math.max(
    0,
    spring({ frame: frame - 70, fps: 30, config: { damping: 100 } }),
  );
  const msgP = Math.max(
    0,
    spring({ frame: frame - 90, fps: 30, config: { damping: 200 } }),
  );

  const labelBox = (
    text: string,
    color: string,
    progress: number,
  ): React.CSSProperties => ({
    opacity: progress,
    transform: `scale(${interpolate(progress, [0, 1], [0.8, 1])})`,
    background: `${color}15`,
    border: `2px solid ${color}40`,
    borderRadius: 16,
    padding: "24px 48px",
    position: "relative" as const,
  });

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 50% 50%, ${colors.danger}08, ${colors.bg} 70%)`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: fonts.heading,
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: `
            linear-gradient(${colors.gridLine} 1px, transparent 1px),
            linear-gradient(90deg, ${colors.gridLine} 1px, transparent 1px)
          `,
          backgroundSize: "80px 80px",
          opacity: 0.3,
        }}
      />

      <div
        style={{
          fontSize: 64,
          fontWeight: 700,
          color: colors.text,
          opacity: titleProgress,
          transform: `translateY(${titleY}px)`,
          marginBottom: 60,
        }}
      >
        The Measurement Gap
      </div>

      <div style={{ display: "flex", gap: 80, marginBottom: 50 }}>
        <div style={labelBox("SAFE \u2713", colors.success, safeP)}>
          <span
            style={{ fontSize: 48, fontWeight: 700, color: colors.success }}
          >
            SAFE &#x2713;
          </span>
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              opacity: strikeP,
            }}
          >
            <div
              style={{
                width: "110%",
                height: 4,
                background: colors.danger,
                transform: "rotate(-15deg)",
                position: "absolute",
              }}
            />
          </div>
        </div>

        <div style={labelBox("UNSAFE \u2717", colors.danger, unsafeP)}>
          <span style={{ fontSize: 48, fontWeight: 700, color: colors.danger }}>
            UNSAFE &#x2717;
          </span>
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              opacity: strikeP,
            }}
          >
            <div
              style={{
                width: "110%",
                height: 4,
                background: colors.danger,
                transform: "rotate(-15deg)",
                position: "absolute",
              }}
            />
          </div>
        </div>
      </div>

      <div
        style={{
          fontSize: 36,
          color: colors.textDim,
          opacity: msgP,
          textAlign: "center",
          lineHeight: 1.6,
          maxWidth: 800,
        }}
      >
        Binary labels destroy the signal.
      </div>
      <div
        style={{
          fontSize: 32,
          color: colors.warning,
          opacity: msgP,
          textAlign: "center",
          marginTop: 16,
          fontWeight: 600,
        }}
      >
        Individual agents pass. The ecosystem fails.
      </div>
    </AbsoluteFill>
  );
};
