import React from "react";
import { AbsoluteFill, useCurrentFrame, spring, interpolate } from "remotion";
import { colors, fonts } from "../theme";

const levers = [
  "Transaction Taxes",
  "Circuit Breakers",
  "Collusion Detection",
  "Reputation Decay",
  "Random Audits",
  "Dynamic Friction",
  "Sybil Detection",
  "Staking",
  "Council Governance",
];

export const Governance: React.FC = () => {
  const frame = useCurrentFrame();

  const titleP = spring({ frame, fps: 30, config: { damping: 200 } });
  const bottomP = Math.max(
    0,
    spring({ frame: frame - 80, fps: 30, config: { damping: 200 } }),
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
          fontSize: 60,
          fontWeight: 700,
          color: colors.text,
          opacity: titleP,
          marginBottom: 50,
        }}
      >
        24+ Governance Levers
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
          gap: 20,
          maxWidth: 1200,
          marginBottom: 50,
        }}
      >
        {levers.map((lever, i) => {
          const p = Math.max(
            0,
            spring({
              frame: frame - 20 - i * 5,
              fps: 30,
              config: { damping: 200 },
            }),
          );
          return (
            <div
              key={lever}
              style={{
                opacity: p,
                transform: `scale(${interpolate(p, [0, 1], [0.8, 1])})`,
                background: `${colors.accent}12`,
                border: `1px solid ${colors.accent}30`,
                borderRadius: 12,
                padding: "16px 28px",
                fontSize: 26,
                color: colors.accent,
                fontWeight: 500,
              }}
            >
              {lever}
            </div>
          );
        })}
      </div>

      <div
        style={{
          fontSize: 30,
          color: colors.textDim,
          opacity: bottomP,
          textAlign: "center",
          lineHeight: 1.6,
        }}
      >
        Test interventions before deployment.
        <br />
        <span style={{ color: colors.accent }}>
          Measure trade-offs empirically.
        </span>
      </div>
    </AbsoluteFill>
  );
};
