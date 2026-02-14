import React from "react";
import { AbsoluteFill, useCurrentFrame, spring, interpolate } from "remotion";
import { colors, fonts } from "../theme";

const MetricRow: React.FC<{
  label: string;
  hard: string;
  soft: string;
  frame: number;
  delay: number;
}> = ({ label, hard, soft, frame, delay }) => {
  const p = Math.max(
    0,
    spring({ frame: frame - delay, fps: 30, config: { damping: 200 } }),
  );

  return (
    <div
      style={{
        display: "flex",
        width: 900,
        alignItems: "center",
        opacity: p,
        transform: `translateX(${interpolate(p, [0, 1], [30, 0])}px)`,
        gap: 20,
      }}
    >
      <span
        style={{
          width: 300,
          fontSize: 26,
          color: colors.textDim,
          textAlign: "right",
        }}
      >
        {label}
      </span>
      <span
        style={{
          width: 250,
          fontSize: 26,
          color: colors.danger,
          fontWeight: 600,
          textAlign: "center",
        }}
      >
        {hard}
      </span>
      <span
        style={{
          width: 250,
          fontSize: 26,
          color: colors.success,
          fontWeight: 600,
          textAlign: "center",
        }}
      >
        {soft}
      </span>
    </div>
  );
};

export const KeyFinding: React.FC = () => {
  const frame = useCurrentFrame();

  const titleP = spring({ frame, fps: 30, config: { damping: 200 } });
  const headerP = Math.max(
    0,
    spring({ frame: frame - 20, fps: 30, config: { damping: 200 } }),
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
          fontSize: 48,
          fontWeight: 700,
          color: colors.text,
          opacity: titleP,
          textAlign: "center",
          marginBottom: 50,
          lineHeight: 1.4,
        }}
      >
        Every soft metric detects
        <br />
        <span style={{ color: colors.accent }}>
          what every hard metric misses.
        </span>
      </div>

      <div
        style={{
          display: "flex",
          width: 900,
          alignItems: "center",
          opacity: headerP,
          gap: 20,
          marginBottom: 24,
          paddingBottom: 16,
          borderBottom: `1px solid ${colors.textMuted}40`,
        }}
      >
        <span style={{ width: 300 }} />
        <span
          style={{
            width: 250,
            fontSize: 28,
            color: colors.danger,
            fontWeight: 700,
            textAlign: "center",
          }}
        >
          Hard Metrics
        </span>
        <span
          style={{
            width: 250,
            fontSize: 28,
            color: colors.success,
            fontWeight: 700,
            textAlign: "center",
          }}
        >
          Soft Metrics
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <MetricRow
          label="Acceptance rate"
          hard="passes &#x2717;"
          soft="toxicity &#x2713;"
          frame={frame}
          delay={40}
        />
        <MetricRow
          label="Benchmark score"
          hard="passes &#x2717;"
          soft="quality gap &#x2713;"
          frame={frame}
          delay={55}
        />
        <MetricRow
          label="Distribution shift"
          hard="invisible &#x2717;"
          soft="KS test &#x2713;"
          frame={frame}
          delay={70}
        />
      </div>
    </AbsoluteFill>
  );
};
