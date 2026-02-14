import React from "react";
import { AbsoluteFill, useCurrentFrame, spring, interpolate } from "remotion";
import { colors, fonts } from "../theme";

export const Opening: React.FC = () => {
  const frame = useCurrentFrame();

  const gradientX = interpolate(frame, [0, 165], [30, 70]);
  const gradientY = interpolate(frame, [0, 165], [30, 60]);

  const titleProgress = spring({ frame, fps: 30, config: { damping: 200 } });
  const titleScale = interpolate(titleProgress, [0, 1], [0.8, 1]);

  const lineWidth = interpolate(
    spring({ frame: frame - 15, fps: 30, config: { damping: 100 } }),
    [0, 1],
    [0, 400],
  );

  const subtitleProgress = spring({
    frame: frame - 20,
    fps: 30,
    config: { damping: 200 },
  });
  const sub = Math.max(0, subtitleProgress);

  const taglineProgress = spring({
    frame: frame - 45,
    fps: 30,
    config: { damping: 200 },
  });
  const tag = Math.max(0, taglineProgress);

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at ${gradientX}% ${gradientY}%, ${colors.accentDim}15, ${colors.bg} 70%)`,
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
          fontSize: 140,
          fontWeight: 800,
          color: colors.text,
          letterSpacing: "0.15em",
          opacity: titleProgress,
          transform: `scale(${titleScale})`,
        }}
      >
        SWARM
      </div>

      <div
        style={{
          width: lineWidth,
          height: 2,
          background: `linear-gradient(90deg, transparent, ${colors.accent}, transparent)`,
          marginTop: 16,
          marginBottom: 24,
        }}
      />

      <div
        style={{
          fontSize: 42,
          fontWeight: 600,
          color: colors.accent,
          opacity: sub,
          transform: `translateY(${interpolate(sub, [0, 1], [20, 0])}px)`,
          letterSpacing: "0.08em",
        }}
      >
        Distributional AGI Safety
      </div>

      <div
        style={{
          fontSize: 28,
          color: colors.textDim,
          opacity: tag,
          transform: `translateY(${interpolate(tag, [0, 1], [20, 0])}px)`,
          marginTop: 32,
          maxWidth: 700,
          textAlign: "center",
          lineHeight: 1.5,
          fontStyle: "italic",
        }}
      >
        When many AI agents interact, new risks emerge that no single agent
        reveals.
      </div>
    </AbsoluteFill>
  );
};
