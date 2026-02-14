import React from "react";
import { AbsoluteFill, useCurrentFrame, spring, interpolate } from "remotion";
import { colors, fonts } from "../theme";

const AnimatedNumber: React.FC<{
  value: number;
  suffix?: string;
  label: string;
  frame: number;
  delay: number;
}> = ({ value, suffix = "", label, frame, delay }) => {
  const progress = Math.max(
    0,
    spring({ frame: frame - delay, fps: 30, config: { damping: 50 } }),
  );
  const display = Math.round(interpolate(progress, [0, 1], [0, value]));

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        opacity: progress,
        transform: `scale(${interpolate(progress, [0, 1], [0.8, 1])})`,
      }}
    >
      <span
        style={{
          fontSize: 80,
          fontWeight: 800,
          color: colors.accent,
          fontFamily: fonts.mono,
        }}
      >
        {display.toLocaleString()}
        {suffix}
      </span>
      <span
        style={{
          fontSize: 24,
          color: colors.textDim,
          marginTop: 8,
          fontWeight: 500,
        }}
      >
        {label}
      </span>
    </div>
  );
};

export const Stats: React.FC = () => {
  const frame = useCurrentFrame();

  const titleP = spring({ frame, fps: 30, config: { damping: 200 } });
  const bottomP = Math.max(
    0,
    spring({ frame: frame - 70, fps: 30, config: { damping: 200 } }),
  );

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 50% 50%, ${colors.accent}08, ${colors.bg} 70%)`,
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
          marginBottom: 60,
        }}
      >
        Production-Ready at Scale
      </div>

      <div style={{ display: "flex", gap: 80, marginBottom: 50 }}>
        <AnimatedNumber
          value={83287}
          label="lines of code"
          frame={frame}
          delay={15}
        />
        <AnimatedNumber
          value={2922}
          label="tests"
          frame={frame}
          delay={25}
        />
        <AnimatedNumber
          value={55}
          suffix="+"
          label="scenarios"
          frame={frame}
          delay={35}
        />
        <AnimatedNumber
          value={24}
          suffix="+"
          label="governance levers"
          frame={frame}
          delay={45}
        />
      </div>

      <div
        style={{
          fontSize: 28,
          color: colors.textDim,
          opacity: bottomP,
          display: "flex",
          gap: 24,
          alignItems: "center",
        }}
      >
        <span style={{ color: colors.accent }}>23 agent types</span>
        <span style={{ color: colors.textMuted }}>&middot;</span>
        <span style={{ color: colors.accent }}>8 framework bridges</span>
        <span style={{ color: colors.textMuted }}>&middot;</span>
        <span style={{ color: colors.accent }}>3 research papers</span>
      </div>
    </AbsoluteFill>
  );
};
