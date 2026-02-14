import React from "react";
import { AbsoluteFill, useCurrentFrame, spring, interpolate } from "remotion";
import { colors, fonts } from "../theme";

export const CallToAction: React.FC = () => {
  const frame = useCurrentFrame();

  const gradientX = interpolate(frame, [0, 135], [30, 70]);

  const logoP = spring({ frame, fps: 30, config: { damping: 200 } });
  const logoScale = interpolate(logoP, [0, 1], [0.8, 1]);

  const installP = Math.max(
    0,
    spring({ frame: frame - 20, fps: 30, config: { damping: 200 } }),
  );
  const linksP = Math.max(
    0,
    spring({ frame: frame - 40, fps: 30, config: { damping: 200 } }),
  );
  const tagP = Math.max(
    0,
    spring({ frame: frame - 55, fps: 30, config: { damping: 200 } }),
  );

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at ${gradientX}% 50%, ${colors.accentDim}15, ${colors.bg} 70%)`,
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
          fontSize: 100,
          fontWeight: 800,
          color: colors.text,
          letterSpacing: "0.15em",
          opacity: logoP,
          transform: `scale(${logoScale})`,
          marginBottom: 40,
        }}
      >
        SWARM
      </div>

      <div
        style={{
          opacity: installP,
          background: `${colors.textMuted}20`,
          border: `1px solid ${colors.accent}30`,
          borderRadius: 12,
          padding: "16px 40px",
          marginBottom: 30,
        }}
      >
        <span
          style={{
            fontSize: 32,
            fontFamily: fonts.mono,
            color: colors.accent,
          }}
        >
          pip install swarm-safety
        </span>
      </div>

      <div
        style={{
          display: "flex",
          gap: 32,
          opacity: linksP,
          marginBottom: 30,
          fontSize: 26,
          color: colors.textDim,
        }}
      >
        <span>GitHub</span>
        <span style={{ color: colors.textMuted }}>&middot;</span>
        <span>Documentation</span>
        <span style={{ color: colors.textMuted }}>&middot;</span>
        <span>Papers</span>
      </div>

      <div
        style={{
          fontSize: 28,
          color: colors.accent,
          opacity: tagP,
          fontWeight: 500,
        }}
      >
        Distributional AGI Safety
      </div>
    </AbsoluteFill>
  );
};
