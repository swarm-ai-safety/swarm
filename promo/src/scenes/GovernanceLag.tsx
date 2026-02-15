import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  spring,
  interpolate,
} from "remotion";
import { colors, fonts } from "../theme";

// Real epoch data from rlm_governance_lag (50 epochs, 10 agents, seed 42)
const epochData = [
  { toxicity: 0.343, welfare: 51.76 },
  { toxicity: 0.329, welfare: 60.01 },
  { toxicity: 0.336, welfare: 58.65 },
  { toxicity: 0.340, welfare: 56.76 },
  { toxicity: 0.315, welfare: 60.78 },
  { toxicity: 0.343, welfare: 56.23 },
  { toxicity: 0.328, welfare: 56.40 },
  { toxicity: 0.323, welfare: 63.07 },
  { toxicity: 0.334, welfare: 58.10 },
  { toxicity: 0.328, welfare: 60.24 },
  { toxicity: 0.348, welfare: 54.40 },
  { toxicity: 0.329, welfare: 59.90 },
  { toxicity: 0.337, welfare: 53.81 },
  { toxicity: 0.351, welfare: 51.99 },
  { toxicity: 0.336, welfare: 59.41 },
  { toxicity: 0.331, welfare: 59.62 },
  { toxicity: 0.329, welfare: 61.87 },
  { toxicity: 0.334, welfare: 56.26 },
  { toxicity: 0.308, welfare: 64.14 },
  { toxicity: 0.329, welfare: 59.92 },
  { toxicity: 0.323, welfare: 54.46 },
  { toxicity: 0.336, welfare: 53.91 },
  { toxicity: 0.350, welfare: 53.16 },
  { toxicity: 0.337, welfare: 56.50 },
  { toxicity: 0.329, welfare: 61.89 },
  { toxicity: 0.332, welfare: 58.52 },
  { toxicity: 0.324, welfare: 60.02 },
  { toxicity: 0.329, welfare: 59.98 },
  { toxicity: 0.348, welfare: 55.31 },
  { toxicity: 0.334, welfare: 53.34 },
  { toxicity: 0.332, welfare: 58.46 },
  { toxicity: 0.309, welfare: 61.96 },
  { toxicity: 0.327, welfare: 62.22 },
  { toxicity: 0.327, welfare: 58.54 },
  { toxicity: 0.324, welfare: 62.88 },
  { toxicity: 0.314, welfare: 63.05 },
  { toxicity: 0.345, welfare: 54.00 },
  { toxicity: 0.334, welfare: 60.77 },
  { toxicity: 0.340, welfare: 57.79 },
  { toxicity: 0.329, welfare: 59.00 },
  { toxicity: 0.322, welfare: 62.45 },
  { toxicity: 0.314, welfare: 63.92 },
  { toxicity: 0.348, welfare: 56.06 },
  { toxicity: 0.319, welfare: 64.86 },
  { toxicity: 0.330, welfare: 57.99 },
  { toxicity: 0.321, welfare: 57.83 },
  { toxicity: 0.329, welfare: 58.04 },
  { toxicity: 0.319, welfare: 59.09 },
  { toxicity: 0.339, welfare: 56.94 },
  { toxicity: 0.336, welfare: 58.63 },
];

function lerpEpoch(frame: number, fps: number, key: "toxicity" | "welfare") {
  const framesPerEpoch = fps * 0.3;
  const totalFrames = epochData.length * framesPerEpoch;
  const looped = frame % totalFrames;
  const epochFloat = looped / framesPerEpoch;
  const e0 = Math.floor(epochFloat) % epochData.length;
  const e1 = Math.min(e0 + 1, epochData.length - 1);
  const t = epochFloat - Math.floor(epochFloat);
  return epochData[e0][key] * (1 - t) + epochData[e1][key] * t;
}

function currentEpochIdx(frame: number, fps: number) {
  const framesPerEpoch = fps * 0.3;
  return Math.floor((frame % (epochData.length * framesPerEpoch)) / framesPerEpoch) % epochData.length;
}

// Sparkline component
const Sparkline: React.FC<{
  data: number[];
  x: number;
  y: number;
  w: number;
  h: number;
  minV: number;
  maxV: number;
  color: string;
  label: string;
  currentValue: string;
}> = ({ data, x, y, w, h, minV, maxV, color, label, currentValue }) => {
  if (data.length < 2) return null;
  const points = data
    .map((v, i) => {
      const px = x + (i / (data.length - 1)) * w;
      const py = y + h - interpolate(v, [minV, maxV], [0, h], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      });
      return `${px},${py}`;
    })
    .join(" ");

  const lastY = y + h - interpolate(data[data.length - 1], [minV, maxV], [0, h], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <g>
      <rect x={x - 8} y={y - 24} width={w + 16} height={h + 44} rx={6} fill={colors.bg} opacity={0.8} />
      <text x={x} y={y - 8} fill={colors.textMuted} fontSize={10} fontFamily={fonts.mono}>
        {label}
      </text>
      <text x={x + w} y={y - 8} fill={color} fontSize={12} fontFamily={fonts.mono} textAnchor="end" fontWeight={700}>
        {currentValue}
      </text>
      <polyline points={points} fill="none" stroke={color} strokeWidth={2} opacity={0.8} />
      <circle cx={x + w} cy={lastY} r={4} fill={color} />
    </g>
  );
};

export const GovernanceLag: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  const introP = spring({ frame, fps, config: { damping: 200 } });
  const titleP = spring({ frame: frame - 5, fps, config: { damping: 200 } });

  const dataFrame = Math.max(0, frame - 30);
  const toxicity = lerpEpoch(dataFrame, fps, "toxicity");
  const welfare = lerpEpoch(dataFrame, fps, "welfare");
  const epochIdx = currentEpochIdx(dataFrame, fps);

  // Governance boundary Y position driven by toxicity
  const govY = interpolate(toxicity, [0.30, 0.36], [height * 0.3, height * 0.55], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Collect sparkline history from epoch data up to current epoch
  const historySlice = epochData.slice(0, epochIdx + 1);
  const toxHistory = historySlice.map((e) => e.toxicity);
  const welHistory = historySlice.map((e) => e.welfare);

  // Generate particles deterministically from frame
  const numParticles = 120;
  const particles = Array.from({ length: numParticles }, (_, i) => {
    const seed = i * 137.5;
    const px = ((seed * 1.1 + frame * (0.3 + (i % 5) * 0.1)) % width);
    const baseY = height * 0.15 + ((seed * 0.7) % (height * 0.7));
    const py = baseY + Math.sin(frame * 0.02 + i * 0.5) * 15;
    const isAbove = py < govY;
    const nearGov = Math.abs(py - govY) < 40;
    return { x: px, y: py, isAbove, nearGov, size: 2 + (i % 4) };
  });

  const welfareNorm = interpolate(welfare, [50, 66], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 50% 50%, ${colors.accent}06, ${colors.bg} 70%)`,
        fontFamily: fonts.heading,
      }}
    >
      {/* Grid */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: `
            linear-gradient(${colors.gridLine} 1px, transparent 1px),
            linear-gradient(90deg, ${colors.gridLine} 1px, transparent 1px)
          `,
          backgroundSize: "80px 80px",
          opacity: 0.15,
        }}
      />

      {/* Particle field + governance boundary */}
      <svg
        width={width}
        height={height}
        style={{ position: "absolute", inset: 0, opacity: introP }}
      >
        {/* Governance boundary line */}
        {Array.from({ length: Math.floor(width / 3) }, (_, i) => {
          const x = i * 3;
          const wobble = Math.sin(x * 0.01 + frame * 0.06) * 6 + Math.sin(x * 0.03 + frame * 0.03) * 3;
          const y = govY + wobble;
          const alpha = 0.3 + Math.sin(x * 0.05 + frame * 0.08) * 0.15;
          return (
            <circle key={`g-${i}`} cx={x} cy={y} r={1.5} fill="#BB6BD9" opacity={alpha} />
          );
        })}

        {/* Zone labels */}
        <text x={60} y={govY - 20} fill={colors.warning} fontSize={11} fontFamily={fonts.mono} opacity={0.4}>
          UNMONITORED ZONE
        </text>
        <text x={60} y={govY + 28} fill={colors.accent} fontSize={11} fontFamily={fonts.mono} opacity={0.4}>
          GOVERNED ZONE
        </text>

        {/* Particles */}
        {particles.map((p, i) => {
          let fill: string;
          let opacity: number;
          if (p.nearGov) {
            fill = "#BB6BD9";
            opacity = 0.5;
          } else if (p.isAbove) {
            const toxNorm = interpolate(toxicity, [0.30, 0.36], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
            fill = interpolate(toxNorm, [0, 1], [0, 1]) > 0.5 ? colors.warning : colors.danger;
            opacity = 0.3 + toxNorm * 0.3;
          } else {
            fill = colors.accent;
            opacity = 0.2 + welfareNorm * 0.3;
          }
          return (
            <circle
              key={`p-${i}`}
              cx={p.x}
              cy={p.y}
              r={p.size * (p.nearGov ? 1.3 : 1)}
              fill={fill}
              opacity={opacity}
            />
          );
        })}

        {/* Sparklines (bottom right) */}
        <Sparkline
          data={toxHistory}
          x={width - 340}
          y={height - 160}
          w={260}
          h={45}
          minV={0.28}
          maxV={0.38}
          color={colors.danger}
          label="TOXICITY"
          currentValue={toxicity.toFixed(3)}
        />
        <Sparkline
          data={welHistory}
          x={width - 340}
          y={height - 90}
          w={260}
          h={45}
          minV={48}
          maxV={68}
          color={colors.accent}
          label="WELFARE"
          currentValue={welfare.toFixed(1)}
        />
      </svg>

      {/* Title */}
      <div
        style={{
          position: "absolute",
          top: 50,
          left: 60,
          opacity: Math.max(0, titleP),
          transform: `translateY(${interpolate(Math.max(0, titleP), [0, 1], [10, 0])}px)`,
        }}
      >
        <div style={{ fontSize: 42, fontWeight: 700, color: colors.text, letterSpacing: "0.08em" }}>
          GOVERNANCE LAG
        </div>
        <div style={{ fontSize: 16, color: colors.textDim, marginTop: 8, fontFamily: fonts.mono }}>
          rlm_governance_lag / 10 agents / 50 epochs / seed 42
        </div>
      </div>

      {/* Big stats */}
      <div
        style={{
          position: "absolute",
          left: 60,
          top: 140,
          opacity: Math.max(0, spring({ frame: frame - 20, fps, config: { damping: 200 } })),
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div>
            <span style={{ fontSize: 48, fontWeight: 800, color: colors.danger, fontFamily: fonts.mono }}>
              {toxicity.toFixed(3)}
            </span>
            <span style={{ fontSize: 14, color: colors.textMuted, marginLeft: 12, fontFamily: fonts.mono }}>
              TOXICITY
            </span>
          </div>
          <div>
            <span style={{ fontSize: 48, fontWeight: 800, color: colors.accent, fontFamily: fonts.mono }}>
              {welfare.toFixed(1)}
            </span>
            <span style={{ fontSize: 14, color: colors.textMuted, marginLeft: 12, fontFamily: fonts.mono }}>
              WELFARE
            </span>
          </div>
        </div>
      </div>

      {/* Epoch counter */}
      <div
        style={{
          position: "absolute",
          left: 60,
          top: 280,
          fontSize: 14,
          color: colors.textMuted,
          fontFamily: fonts.mono,
          opacity: Math.max(0, spring({ frame: frame - 25, fps, config: { damping: 200 } })),
        }}
      >
        EPOCH {epochIdx}/{epochData.length - 1}
        <div style={{ width: 200, height: 3, borderRadius: 2, background: `${colors.textMuted}20`, marginTop: 8 }}>
          <div
            style={{
              width: `${(epochIdx / (epochData.length - 1)) * 100}%`,
              height: "100%",
              borderRadius: 2,
              background: "#BB6BD9",
            }}
          />
        </div>
      </div>

      {/* Attribution */}
      <div
        style={{
          position: "absolute",
          bottom: 50,
          right: 60,
          fontSize: 12,
          color: colors.textMuted,
          fontFamily: fonts.mono,
          opacity: Math.max(0, introP),
        }}
      >
        distributional-agi-safety
      </div>
    </AbsoluteFill>
  );
};
