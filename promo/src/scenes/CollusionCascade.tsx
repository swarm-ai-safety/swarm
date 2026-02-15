import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  spring,
  interpolate,
  Sequence,
} from "remotion";
import { colors, fonts } from "../theme";

// Real epoch data from rlm_recursive_collusion (30 epochs, 12 agents, seed 42)
const epochData = [
  { toxicity: 0.347, quality_gap: 0.043, welfare: 72.4, accepted: 77, rejected: 32 },
  { toxicity: 0.338, quality_gap: 0.04, welfare: 89.0, accepted: 92, rejected: 12 },
  { toxicity: 0.335, quality_gap: 0.019, welfare: 97.4, accepted: 100, rejected: 9 },
  { toxicity: 0.348, quality_gap: 0.047, welfare: 91.7, accepted: 98, rejected: 7 },
  { toxicity: 0.334, quality_gap: 0.075, welfare: 118.4, accepted: 121, rejected: 6 },
  { toxicity: 0.341, quality_gap: 0.0, welfare: 128.3, accepted: 134, rejected: 0 },
  { toxicity: 0.34, quality_gap: 0.0, welfare: 132.4, accepted: 138, rejected: 0 },
  { toxicity: 0.343, quality_gap: 0.122, welfare: 127.4, accepted: 134, rejected: 1 },
  { toxicity: 0.353, quality_gap: -0.08, welfare: 120.6, accepted: 131, rejected: 1 },
  { toxicity: 0.341, quality_gap: -0.091, welfare: 128.3, accepted: 134, rejected: 1 },
  { toxicity: 0.349, quality_gap: 0.0, welfare: 126.1, accepted: 135, rejected: 0 },
  { toxicity: 0.336, quality_gap: -0.085, welfare: 130.5, accepted: 134, rejected: 1 },
  { toxicity: 0.335, quality_gap: 0.0, welfare: 129.7, accepted: 133, rejected: 0 },
  { toxicity: 0.327, quality_gap: 0.052, welfare: 131.0, accepted: 131, rejected: 1 },
  { toxicity: 0.33, quality_gap: 0.132, welfare: 130.6, accepted: 132, rejected: 1 },
  { toxicity: 0.338, quality_gap: 0.0, welfare: 128.5, accepted: 133, rejected: 0 },
  { toxicity: 0.336, quality_gap: -0.009, welfare: 124.5, accepted: 128, rejected: 3 },
  { toxicity: 0.33, quality_gap: 0.056, welfare: 131.8, accepted: 133, rejected: 1 },
  { toxicity: 0.337, quality_gap: 0.036, welfare: 126.0, accepted: 130, rejected: 3 },
  { toxicity: 0.338, quality_gap: -0.103, welfare: 127.7, accepted: 132, rejected: 1 },
  { toxicity: 0.33, quality_gap: 0.065, welfare: 129.6, accepted: 131, rejected: 1 },
  { toxicity: 0.335, quality_gap: -0.051, welfare: 125.6, accepted: 129, rejected: 1 },
  { toxicity: 0.334, quality_gap: -0.017, welfare: 125.2, accepted: 128, rejected: 3 },
  { toxicity: 0.324, quality_gap: 0.0, welfare: 133.1, accepted: 132, rejected: 0 },
  { toxicity: 0.326, quality_gap: 0.0, welfare: 135.3, accepted: 135, rejected: 0 },
  { toxicity: 0.346, quality_gap: -0.045, welfare: 121.4, accepted: 129, rejected: 2 },
  { toxicity: 0.327, quality_gap: 0.002, welfare: 129.7, accepted: 130, rejected: 2 },
  { toxicity: 0.335, quality_gap: 0.0, welfare: 128.6, accepted: 132, rejected: 0 },
  { toxicity: 0.336, quality_gap: 0.0, welfare: 126.3, accepted: 130, rejected: 0 },
  { toxicity: 0.348, quality_gap: 0.0, welfare: 125.4, accepted: 134, rejected: 0 },
];

const N_AGENTS = 12;

// Pre-computed agent ring positions (deterministic)
function agentPositions(cx: number, cy: number, radius: number) {
  return Array.from({ length: N_AGENTS }, (_, i) => {
    const angle = (i / N_AGENTS) * Math.PI * 2 - Math.PI / 2;
    return {
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius,
      isColluder: i < 4,
    };
  });
}

// Connections: colluding pairs (first 4) + some cross-links
const colludingPairs: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [0, 3], [0, 2], [1, 3],
];
const honestLinks: [number, number][] = [
  [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 4],
  [4, 8], [6, 10],
];
const crossLinks: [number, number][] = [
  [0, 4], [1, 5], [2, 9], [3, 11],
];

function lerpEpoch(frame: number, fps: number, key: keyof typeof epochData[0]) {
  const framesPerEpoch = fps * 0.4; // 12 frames per epoch at 30fps
  const totalFrames = epochData.length * framesPerEpoch;
  const looped = frame % totalFrames;
  const epochFloat = looped / framesPerEpoch;
  const e0 = Math.floor(epochFloat) % epochData.length;
  const e1 = Math.min(e0 + 1, epochData.length - 1);
  const t = epochFloat - Math.floor(epochFloat);
  return (epochData[e0][key] as number) * (1 - t) + (epochData[e1][key] as number) * t;
}

function currentEpochIdx(frame: number, fps: number) {
  const framesPerEpoch = fps * 0.4;
  const totalFrames = epochData.length * framesPerEpoch;
  const looped = frame % totalFrames;
  return Math.floor(looped / framesPerEpoch) % epochData.length;
}

const MetricRow: React.FC<{
  label: string;
  value: string;
  color: string;
  barFill: number;
  opacity: number;
}> = ({ label, value, color, barFill, opacity }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 16, opacity }}>
    <span style={{ fontSize: 11, color: colors.textMuted, width: 90, fontFamily: fonts.mono, textTransform: "uppercase" }}>
      {label}
    </span>
    <span style={{ fontSize: 20, fontWeight: 700, color, fontFamily: fonts.mono, width: 80 }}>
      {value}
    </span>
    <div style={{ width: 120, height: 4, borderRadius: 2, background: `${colors.textMuted}25` }}>
      <div style={{ width: `${Math.max(0, Math.min(100, barFill * 100))}%`, height: "100%", borderRadius: 2, background: color }} />
    </div>
  </div>
);

export const CollusionCascade: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  const cx = width * 0.55;
  const cy = height * 0.5;
  const ringRadius = Math.min(width, height) * 0.25;
  const agents = agentPositions(cx, cy, ringRadius);

  // Intro animation
  const introP = spring({ frame, fps, config: { damping: 200 } });
  const titleP = spring({ frame: frame - 5, fps, config: { damping: 200 } });
  const metricsP = spring({ frame: frame - 30, fps, config: { damping: 200 } });

  // Current data
  const toxicity = lerpEpoch(Math.max(0, frame - 40), fps, "toxicity");
  const welfare = lerpEpoch(Math.max(0, frame - 40), fps, "welfare");
  const qualityGap = lerpEpoch(Math.max(0, frame - 40), fps, "quality_gap");
  const epochIdx = currentEpochIdx(Math.max(0, frame - 40), fps);

  const collusionIntensity = interpolate(toxicity, [0.32, 0.36], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const adverseSelection = qualityGap < -0.02
    ? interpolate(qualityGap, [-0.12, 0], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
    : 0;

  // Agent wobble from frame
  const wobble = (i: number) => ({
    dx: Math.sin(frame * 0.03 + i * 2.5) * 8,
    dy: Math.cos(frame * 0.025 + i * 3.1) * 8,
  });

  // Colluders drift toward center
  const colluderDrift = collusionIntensity * 0.15;

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 55% 50%, ${colors.accent}06, ${colors.bg} 70%)`,
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
          opacity: 0.2,
        }}
      />

      {/* Adverse selection flash */}
      {adverseSelection > 0.3 && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: colors.danger,
            opacity: Math.sin(frame * 0.15) * 0.03 * adverseSelection,
          }}
        />
      )}

      {/* Agent network */}
      <svg
        width={width}
        height={height}
        style={{ position: "absolute", inset: 0, opacity: introP }}
      >
        {/* Honest links */}
        {honestLinks.map(([i, j], idx) => {
          const a = agents[i];
          const b = agents[j];
          const wa = wobble(i);
          const wb = wobble(j);
          const delay = idx * 2 + 10;
          const lineP = Math.max(0, spring({ frame: frame - delay, fps, config: { damping: 200 } }));
          return (
            <line
              key={`h-${idx}`}
              x1={a.x + wa.dx}
              y1={a.y + wa.dy}
              x2={b.x + wb.dx}
              y2={b.y + wb.dy}
              stroke={colors.accent}
              strokeWidth={0.8}
              opacity={lineP * 0.15}
            />
          );
        })}

        {/* Cross links */}
        {crossLinks.map(([i, j], idx) => {
          const a = agents[i];
          const b = agents[j];
          const wa = wobble(i);
          const wb = wobble(j);
          const delay = idx * 3 + 20;
          const lineP = Math.max(0, spring({ frame: frame - delay, fps, config: { damping: 200 } }));
          return (
            <line
              key={`x-${idx}`}
              x1={a.x + wa.dx + (a.isColluder ? (cx - a.x) * colluderDrift : 0)}
              y1={a.y + wa.dy + (a.isColluder ? (cy - a.y) * colluderDrift : 0)}
              x2={b.x + wb.dx}
              y2={b.y + wb.dy}
              stroke={colors.warning}
              strokeWidth={0.8}
              opacity={lineP * 0.2 * collusionIntensity}
            />
          );
        })}

        {/* Colluding links - glow with intensity */}
        {colludingPairs.map(([i, j], idx) => {
          const a = agents[i];
          const b = agents[j];
          const wa = wobble(i);
          const wb = wobble(j);
          const ax = a.x + wa.dx + (cx - a.x) * colluderDrift;
          const ay = a.y + wa.dy + (cy - a.y) * colluderDrift;
          const bx = b.x + wb.dx + (cx - b.x) * colluderDrift;
          const by = b.y + wb.dy + (cy - b.y) * colluderDrift;
          const delay = idx * 2 + 5;
          const lineP = Math.max(0, spring({ frame: frame - delay, fps, config: { damping: 200 } }));
          return (
            <g key={`c-${idx}`}>
              {/* Glow */}
              <line
                x1={ax} y1={ay} x2={bx} y2={by}
                stroke={colors.warning}
                strokeWidth={4}
                opacity={lineP * 0.1 * collusionIntensity}
              />
              {/* Core */}
              <line
                x1={ax} y1={ay} x2={bx} y2={by}
                stroke={colors.warning}
                strokeWidth={1.5}
                opacity={lineP * (0.15 + collusionIntensity * 0.5)}
              />
            </g>
          );
        })}

        {/* Agent dots */}
        {agents.map((a, i) => {
          const w = wobble(i);
          const delay = i * 2;
          const dotP = Math.max(0, spring({ frame: frame - delay, fps, config: { damping: 150 } }));
          const ax = a.x + w.dx + (a.isColluder ? (cx - a.x) * colluderDrift : 0);
          const ay = a.y + w.dy + (a.isColluder ? (cy - a.y) * colluderDrift : 0);
          const breathe = Math.sin(frame * 0.04 + i) * 0.2 + 1;
          const baseR = a.isColluder ? 10 : 7;
          const r = baseR * breathe * dotP;
          const col = a.isColluder
            ? interpolateColor(colors.accent, colors.warning, collusionIntensity)
            : colors.accent;

          return (
            <g key={`a-${i}`}>
              <circle cx={ax} cy={ay} r={r * 3} fill={col} opacity={0.06 * dotP * (a.isColluder ? 2 : 1)} />
              <circle cx={ax} cy={ay} r={r} fill={col} opacity={0.85 * dotP} />
            </g>
          );
        })}
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
          COLLUSION CASCADE
        </div>
        <div style={{ fontSize: 16, color: colors.textDim, marginTop: 8, fontFamily: fonts.mono }}>
          rlm_recursive_collusion / 12 agents / seed 42
        </div>
      </div>

      {/* Metrics panel */}
      <div
        style={{
          position: "absolute",
          left: 60,
          bottom: 80,
          display: "flex",
          flexDirection: "column",
          gap: 12,
          opacity: Math.max(0, metricsP),
          background: `${colors.bg}CC`,
          padding: "20px 24px",
          borderRadius: 12,
          border: `1px solid ${colors.textMuted}20`,
        }}
      >
        <div style={{ fontSize: 11, color: colors.textMuted, fontFamily: fonts.mono, marginBottom: 4 }}>
          EPOCH {epochIdx}/{epochData.length - 1}
        </div>
        <MetricRow label="Toxicity" value={toxicity.toFixed(3)} color={colors.danger} barFill={interpolate(toxicity, [0.3, 0.36], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })} opacity={1} />
        <MetricRow label="Welfare" value={welfare.toFixed(1)} color={colors.accent} barFill={interpolate(welfare, [70, 140], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })} opacity={1} />
        <MetricRow label="Q-Gap" value={qualityGap.toFixed(3)} color={qualityGap < 0 ? colors.danger : colors.success} barFill={interpolate(Math.abs(qualityGap), [0, 0.15], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })} opacity={1} />
        <MetricRow label="Collusion" value={collusionIntensity.toFixed(2)} color={colors.warning} barFill={collusionIntensity} opacity={1} />
      </div>

      {/* Epoch progress bar */}
      <Sequence from={40}>
        <div
          style={{
            position: "absolute",
            bottom: 40,
            left: 60,
            right: 60,
          }}
        >
          <div style={{ width: "100%", height: 3, borderRadius: 2, background: `${colors.textMuted}20` }}>
            <div
              style={{
                width: `${(epochIdx / (epochData.length - 1)) * 100}%`,
                height: "100%",
                borderRadius: 2,
                background: `linear-gradient(90deg, ${colors.accent}, ${colors.warning})`,
                transition: "width 0.1s",
              }}
            />
          </div>
        </div>
      </Sequence>

      {/* Attribution */}
      <div
        style={{
          position: "absolute",
          bottom: 50,
          right: 60,
          fontSize: 12,
          color: colors.textMuted,
          fontFamily: fonts.mono,
          opacity: Math.max(0, metricsP),
        }}
      >
        distributional-agi-safety
      </div>
    </AbsoluteFill>
  );
};

function interpolateColor(from: string, to: string, t: number): string {
  // Simple hex interpolation
  const f = hexToRgb(from);
  const toRgb = hexToRgb(to);
  const r = Math.round(f[0] + (toRgb[0] - f[0]) * t);
  const g = Math.round(f[1] + (toRgb[1] - f[1]) * t);
  const b = Math.round(f[2] + (toRgb[2] - f[2]) * t);
  return `rgb(${r},${g},${b})`;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ];
}
