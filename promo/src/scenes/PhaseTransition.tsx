import React from "react";
import { AbsoluteFill, useCurrentFrame, spring, interpolate } from "remotion";
import { colors, fonts } from "../theme";

const RegimeCard: React.FC<{
  title: string;
  adversarial: string;
  welfare: string;
  color: string;
  frame: number;
  delay: number;
}> = ({ title, adversarial, welfare, color, frame, delay }) => {
  const p = Math.max(
    0,
    spring({ frame: frame - delay, fps: 30, config: { damping: 200 } }),
  );

  return (
    <div
      style={{
        opacity: p,
        transform: `translateY(${interpolate(p, [0, 1], [40, 0])}px)`,
        background: `${color}10`,
        border: `2px solid ${color}40`,
        borderRadius: 20,
        padding: "36px 44px",
        width: 340,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 16,
      }}
    >
      <span style={{ fontSize: 36, fontWeight: 700, color }}>{title}</span>
      <span style={{ fontSize: 22, color: colors.textDim }}>
        Adversarial: {adversarial}
      </span>
      <span style={{ fontSize: 22, color: colors.textDim }}>
        Welfare: {welfare}
      </span>
    </div>
  );
};

export const PhaseTransition: React.FC = () => {
  const frame = useCurrentFrame();

  const titleP = spring({ frame, fps: 30, config: { damping: 200 } });
  const msgP = Math.max(
    0,
    spring({ frame: frame - 80, fps: 30, config: { damping: 200 } }),
  );

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 50% 60%, ${colors.warning}08, ${colors.bg} 70%)`,
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
          fontSize: 60,
          fontWeight: 700,
          color: colors.text,
          opacity: titleP,
          marginBottom: 50,
        }}
      >
        The Phase Transition
      </div>

      <div style={{ display: "flex", gap: 40, marginBottom: 50 }}>
        <RegimeCard
          title="Cooperative"
          adversarial="0\u201320%"
          welfare="Stable"
          color={colors.success}
          frame={frame}
          delay={20}
        />
        <RegimeCard
          title="Contested"
          adversarial="20\u201337.5%"
          welfare="Declining"
          color={colors.warning}
          frame={frame}
          delay={35}
        />
        <RegimeCard
          title="Collapse"
          adversarial="50%+"
          welfare="Zero"
          color={colors.danger}
          frame={frame}
          delay={50}
        />
      </div>

      <div
        style={{
          fontSize: 34,
          fontWeight: 600,
          color: colors.warning,
          opacity: msgP,
          textAlign: "center",
        }}
      >
        The transition is abrupt, not gradual.
      </div>
    </AbsoluteFill>
  );
};
