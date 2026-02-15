import React from "react";
import { AbsoluteFill, useCurrentFrame, spring, interpolate } from "remotion";
import { colors, fonts } from "../theme";

export const PhaseTransition: React.FC = () => {
  const frame = useCurrentFrame();

  const titleP = spring({ frame, fps: 30, config: { damping: 200 } });

  const taxP = Math.max(
    0,
    spring({ frame: frame - 25, fps: 30, config: { damping: 200 } }),
  );
  const arrowP = Math.max(
    0,
    spring({ frame: frame - 45, fps: 30, config: { damping: 100 } }),
  );
  const welfareP = Math.max(
    0,
    spring({ frame: frame - 55, fps: 30, config: { damping: 200 } }),
  );
  const statsP = Math.max(
    0,
    spring({ frame: frame - 75, fps: 30, config: { damping: 200 } }),
  );
  const replicatedP = Math.max(
    0,
    spring({ frame: frame - 95, fps: 30, config: { damping: 200 } }),
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
          fontSize: 52,
          fontWeight: 700,
          color: colors.text,
          opacity: titleP,
          marginBottom: 50,
          textAlign: "center",
          lineHeight: 1.3,
        }}
      >
        Replicated Finding:
        <br />
        <span style={{ color: colors.accent }}>The Tax-Welfare Tradeoff</span>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 40,
          marginBottom: 40,
        }}
      >
        <div
          style={{
            opacity: taxP,
            transform: `scale(${interpolate(taxP, [0, 1], [0.8, 1])})`,
            background: `${colors.warning}15`,
            border: `2px solid ${colors.warning}40`,
            borderRadius: 16,
            padding: "20px 36px",
          }}
        >
          <span
            style={{
              fontSize: 48,
              fontWeight: 800,
              color: colors.warning,
              fontFamily: fonts.mono,
            }}
          >
            10% tax
          </span>
        </div>

        <div
          style={{
            fontSize: 48,
            color: colors.textDim,
            opacity: arrowP,
          }}
        >
          {"\u2192"}
        </div>

        <div
          style={{
            opacity: welfareP,
            transform: `scale(${interpolate(welfareP, [0, 1], [0.8, 1])})`,
            background: `${colors.danger}15`,
            border: `2px solid ${colors.danger}40`,
            borderRadius: 16,
            padding: "20px 36px",
          }}
        >
          <span
            style={{
              fontSize: 48,
              fontWeight: 800,
              color: colors.danger,
              fontFamily: fonts.mono,
            }}
          >
            {"\u22128.1% welfare"}
          </span>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: 48,
          opacity: statsP,
          marginBottom: 36,
        }}
      >
        {[
          { label: "p", value: "0.0007" },
          { label: "Cohen\u2019s d", value: "0.80" },
          { label: "runs", value: "160" },
          { label: "seeds", value: "10" },
        ].map(({ label, value }) => (
          <div
            key={label}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <span
              style={{
                fontSize: 36,
                fontWeight: 700,
                color: colors.accent,
                fontFamily: fonts.mono,
              }}
            >
              {value}
            </span>
            <span
              style={{ fontSize: 20, color: colors.textDim, marginTop: 4 }}
            >
              {label}
            </span>
          </div>
        ))}
      </div>

      <div
        style={{
          fontSize: 28,
          color: colors.success,
          opacity: replicatedP,
          fontWeight: 600,
          textAlign: "center",
        }}
      >
        Survives Bonferroni correction. Replicated across 5 independent studies.
      </div>
    </AbsoluteFill>
  );
};
