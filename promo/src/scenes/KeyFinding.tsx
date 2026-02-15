import React from "react";
import { AbsoluteFill, useCurrentFrame, spring, interpolate } from "remotion";
import { colors, fonts } from "../theme";

const StatCard: React.FC<{
  value: string;
  label: string;
  color: string;
  frame: number;
  delay: number;
}> = ({ value, label, color, frame, delay }) => {
  const p = Math.max(
    0,
    spring({ frame: frame - delay, fps: 30, config: { damping: 200 } }),
  );

  return (
    <div
      style={{
        opacity: p,
        transform: `translateY(${interpolate(p, [0, 1], [30, 0])}px)`,
        background: `${color}10`,
        border: `1px solid ${color}30`,
        borderRadius: 16,
        padding: "28px 36px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        width: 260,
        gap: 8,
      }}
    >
      <span
        style={{
          fontSize: 44,
          fontWeight: 800,
          color,
          fontFamily: fonts.mono,
        }}
      >
        {value}
      </span>
      <span
        style={{
          fontSize: 20,
          color: colors.textDim,
          textAlign: "center",
        }}
      >
        {label}
      </span>
    </div>
  );
};

export const KeyFinding: React.FC = () => {
  const frame = useCurrentFrame();

  const titleP = spring({ frame, fps: 30, config: { damping: 200 } });
  const bottomP = Math.max(
    0,
    spring({ frame: frame - 85, fps: 30, config: { damping: 200 } }),
  );

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 50% 50%, ${colors.success}06, ${colors.bg} 70%)`,
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
          fontSize: 56,
          fontWeight: 700,
          color: colors.text,
          opacity: titleP,
          textAlign: "center",
          marginBottom: 50,
        }}
      >
        Rigorous by Design
      </div>

      <div
        style={{
          display: "flex",
          gap: 32,
          marginBottom: 44,
        }}
      >
        <StatCard
          value="700+"
          label="pre-registered runs"
          color={colors.accent}
          frame={frame}
          delay={20}
        />
        <StatCard
          value="10"
          label="seeds per config"
          color={colors.accent}
          frame={frame}
          delay={35}
        />
        <StatCard
          value="Bonferroni"
          label="correction on all tests"
          color={colors.success}
          frame={frame}
          delay={50}
        />
      </div>

      <div
        style={{
          fontSize: 30,
          color: colors.textDim,
          opacity: bottomP,
          textAlign: "center",
          lineHeight: 1.6,
          maxWidth: 800,
        }}
      >
        200-run study found nothing significant.
        <br />
        <span style={{ color: colors.accent }}>We report that too.</span>
      </div>
    </AbsoluteFill>
  );
};
