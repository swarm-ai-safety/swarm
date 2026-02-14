import React from "react";
import { AbsoluteFill, useCurrentFrame, spring, interpolate } from "remotion";
import { colors, fonts } from "../theme";

export const Solution: React.FC = () => {
  const frame = useCurrentFrame();

  const titleP = spring({ frame, fps: 30, config: { damping: 200 } });
  const formulaP = Math.max(
    0,
    spring({ frame: frame - 25, fps: 30, config: { damping: 200 } }),
  );
  const barP = Math.max(
    0,
    spring({ frame: frame - 45, fps: 30, config: { damping: 80 } }),
  );
  const barWidth = interpolate(barP, [0, 1], [0, 900]);
  const labelsP = Math.max(
    0,
    spring({ frame: frame - 55, fps: 30, config: { damping: 200 } }),
  );
  const bottomP = Math.max(
    0,
    spring({ frame: frame - 80, fps: 30, config: { damping: 200 } }),
  );

  const pPos = interpolate(frame, [60, 140], [0.2, 0.85], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const indicatorP = Math.max(
    0,
    spring({ frame: frame - 60, fps: 30, config: { damping: 200 } }),
  );

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 50% 40%, ${colors.accent}10, ${colors.bg} 70%)`,
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
          opacity: titleP,
          transform: `translateY(${interpolate(titleP, [0, 1], [30, 0])}px)`,
          marginBottom: 40,
        }}
      >
        Soft Probabilistic Labels
      </div>

      <div
        style={{
          fontSize: 40,
          fontFamily: fonts.mono,
          color: colors.accent,
          opacity: formulaP,
          marginBottom: 50,
        }}
      >
        p = P(interaction is beneficial)
      </div>

      <div
        style={{
          width: 900,
          height: 48,
          borderRadius: 24,
          background: `${colors.textMuted}30`,
          position: "relative",
          overflow: "hidden",
          marginBottom: 20,
        }}
      >
        <div
          style={{
            width: barWidth,
            height: "100%",
            borderRadius: 24,
            background: `linear-gradient(90deg, ${colors.danger}, ${colors.warning}, ${colors.success})`,
            opacity: 0.8,
          }}
        />
        {frame > 60 && (
          <div
            style={{
              position: "absolute",
              top: -12,
              left: pPos * 900 - 2,
              width: 4,
              height: 72,
              background: colors.text,
              borderRadius: 2,
              opacity: indicatorP,
            }}
          >
            <div
              style={{
                position: "absolute",
                top: -32,
                left: "50%",
                transform: "translateX(-50%)",
                fontSize: 24,
                fontFamily: fonts.mono,
                color: colors.text,
                fontWeight: 700,
                whiteSpace: "nowrap",
              }}
            >
              p = {pPos.toFixed(2)}
            </div>
          </div>
        )}
      </div>

      <div
        style={{
          display: "flex",
          width: 900,
          justifyContent: "space-between",
          opacity: labelsP,
          marginBottom: 50,
        }}
      >
        <span style={{ fontSize: 24, color: colors.danger, fontWeight: 600 }}>
          0.0 harmful
        </span>
        <span style={{ fontSize: 24, color: colors.warning, fontWeight: 600 }}>
          0.5 uncertain
        </span>
        <span style={{ fontSize: 24, color: colors.success, fontWeight: 600 }}>
          1.0 beneficial
        </span>
      </div>

      <div
        style={{
          fontSize: 32,
          color: colors.textDim,
          opacity: bottomP,
          textAlign: "center",
          maxWidth: 750,
          lineHeight: 1.5,
        }}
      >
        Preserves information that binary labels destroy.
      </div>
    </AbsoluteFill>
  );
};
