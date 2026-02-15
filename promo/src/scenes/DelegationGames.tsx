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

// Real data from delegation_games_study (80 runs, 8 configs, 10 seeds each)
const configs = [
  { tax: 0.0,  cb: false, label: "No Tax, No CB",  welfare: 85.2,  toxicity: 0.335 },
  { tax: 0.0,  cb: true,  label: "No Tax, CB On",  welfare: 94.6,  toxicity: 0.334 },
  { tax: 0.05, cb: false, label: "5% Tax, No CB",  welfare: 88.1,  toxicity: 0.332 },
  { tax: 0.05, cb: true,  label: "5% Tax, CB On",  welfare: 101.1, toxicity: 0.333 },
  { tax: 0.1,  cb: false, label: "10% Tax, No CB", welfare: 92.7,  toxicity: 0.33 },
  { tax: 0.1,  cb: true,  label: "10% Tax, CB On", welfare: 105.7, toxicity: 0.329 },
  { tax: 0.15, cb: false, label: "15% Tax, No CB", welfare: 98.4,  toxicity: 0.328 },
  { tax: 0.15, cb: true,  label: "15% Tax, CB On", welfare: 119.6, toxicity: 0.325 },
];

const sorted = [...configs].sort((a, b) => b.welfare - a.welfare);
const maxWelfare = Math.max(...configs.map((c) => c.welfare));

function configColor(c: { tax: number; cb: boolean }): string {
  if (c.cb && c.tax >= 0.15) return colors.accent;
  if (c.cb) return "#64B4DC";
  if (c.tax >= 0.1) return "#BB6BD9";
  return colors.warning;
}

// Bar race scene (first half)
const BarRace: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  const titleP = spring({ frame, fps, config: { damping: 200 } });

  const barH = 48;
  const gap = 14;
  const startX = 320;
  const maxBarW = width - startX - 160;
  const startY = height * 0.12;

  return (
    <AbsoluteFill>
      {/* Rank label */}
      <div
        style={{
          position: "absolute",
          top: startY - 36,
          left: startX,
          fontSize: 12,
          color: colors.textMuted,
          fontFamily: fonts.mono,
          opacity: Math.max(0, titleP),
        }}
      >
        RANKED BY TOTAL WELFARE (10 seeds per config)
      </div>

      {sorted.map((c, rank) => {
        const delay = rank * 6 + 15;
        const barP = Math.max(0, spring({ frame: frame - delay, fps, config: { damping: 80 } }));
        const barW = interpolate(barP, [0, 1], [0, (c.welfare / maxWelfare) * maxBarW]);
        const y = startY + rank * (barH + gap);
        const col = configColor(c);
        const isTop = rank === 0;

        return (
          <div key={c.label} style={{ position: "absolute", top: y, left: 0, right: 0 }}>
            {/* Label */}
            <div
              style={{
                position: "absolute",
                right: width - startX + 16,
                width: startX - 16,
                textAlign: "right",
                fontSize: isTop ? 18 : 15,
                fontWeight: isTop ? 700 : 500,
                color: isTop ? colors.text : colors.textDim,
                fontFamily: fonts.heading,
                lineHeight: `${barH}px`,
                opacity: barP,
              }}
            >
              {c.label}
            </div>

            {/* Bar track */}
            <div
              style={{
                position: "absolute",
                left: startX,
                width: maxBarW,
                height: barH,
                borderRadius: 8,
                background: `${colors.textMuted}10`,
              }}
            />

            {/* Bar fill */}
            <div
              style={{
                position: "absolute",
                left: startX,
                width: barW,
                height: barH,
                borderRadius: 8,
                background: col,
                opacity: isTop ? 0.9 : 0.6,
                boxShadow: isTop ? `0 0 20px ${col}40` : "none",
              }}
            />

            {/* Value */}
            <div
              style={{
                position: "absolute",
                left: startX + barW + 12,
                fontSize: 16,
                fontWeight: 700,
                color: col,
                fontFamily: fonts.mono,
                lineHeight: `${barH}px`,
                opacity: barP,
              }}
            >
              {c.welfare.toFixed(1)}
            </div>

            {/* Toxicity dot */}
            {isTop && (
              <div
                style={{
                  position: "absolute",
                  left: startX + barW - 20,
                  top: barH / 2 - 4,
                  width: 8,
                  height: 8,
                  borderRadius: 4,
                  background: colors.danger,
                  opacity: 0.6 * barP,
                }}
              />
            )}
          </div>
        );
      })}
    </AbsoluteFill>
  );
};

// Bubble comparison scene (second half)
const BubbleView: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  const cx = width * 0.5;
  const cy = height * 0.48;
  const ringR = Math.min(width, height) * 0.26;

  return (
    <AbsoluteFill>
      <svg width={width} height={height} style={{ position: "absolute", inset: 0 }}>
        {configs.map((c, i) => {
          const angle = (i / configs.length) * Math.PI * 2 - Math.PI / 2;
          const bx = cx + Math.cos(angle) * ringR;
          const by = cy + Math.sin(angle) * ringR;
          const size = interpolate(c.welfare, [80, 125], [30, 70], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const delay = i * 5;
          const bubbleP = Math.max(0, spring({ frame: frame - delay, fps, config: { damping: 100 } }));
          const col = configColor(c);
          const breathe = Math.sin(frame * 0.04 + i * 1.2) * 2;

          // Draw connection to CB pair
          const pairIdx = c.cb ? i - 1 : -1;
          const showLink = c.cb && pairIdx >= 0;

          return (
            <g key={i}>
              {showLink && (() => {
                const pAngle = (pairIdx / configs.length) * Math.PI * 2 - Math.PI / 2;
                const px = cx + Math.cos(pAngle) * ringR;
                const py = cy + Math.sin(pAngle) * ringR;
                return (
                  <line
                    x1={bx} y1={by} x2={px} y2={py}
                    stroke="#BB6BD9"
                    strokeWidth={1}
                    opacity={0.2 * bubbleP}
                    strokeDasharray="4,4"
                  />
                );
              })()}
              {/* Glow */}
              <circle cx={bx} cy={by} r={(size + breathe) * 2 * bubbleP} fill={col} opacity={0.05 * bubbleP} />
              {/* Bubble */}
              <circle cx={bx} cy={by} r={(size + breathe) * bubbleP} fill={col} opacity={0.7 * bubbleP} />
              {/* Label */}
              <text
                x={bx}
                y={by + size + 20}
                textAnchor="middle"
                fill={colors.textDim}
                fontSize={12}
                fontFamily={fonts.mono}
                opacity={bubbleP}
              >
                {c.label}
              </text>
              {/* Value inside */}
              <text
                x={bx}
                y={by + 5}
                textAnchor="middle"
                fill={colors.text}
                fontSize={16}
                fontWeight={700}
                fontFamily={fonts.mono}
                opacity={bubbleP}
              >
                {c.welfare.toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>
    </AbsoluteFill>
  );
};

// Findings overlay
const Findings: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const p1 = Math.max(0, spring({ frame: frame - 10, fps, config: { damping: 200 } }));
  const p2 = Math.max(0, spring({ frame: frame - 25, fps, config: { damping: 200 } }));

  return (
    <div
      style={{
        position: "absolute",
        bottom: 80,
        left: 60,
        background: `${colors.bg}DD`,
        padding: "18px 24px",
        borderRadius: 10,
        border: `1px solid ${colors.textMuted}20`,
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div style={{ fontSize: 10, color: colors.textMuted, fontFamily: fonts.mono, letterSpacing: "0.1em" }}>
        KEY FINDINGS (Bonferroni-corrected)
      </div>
      <div style={{ fontSize: 14, color: colors.accent, fontFamily: fonts.mono, opacity: p1 }}>
        ** TAX 0% vs 15% (CB on): d=1.56, p=0.004
      </div>
      <div style={{ fontSize: 14, color: colors.warning, fontFamily: fonts.mono, opacity: p2 }}>
        {"   "}CB ON vs OFF (pooled): d=0.56, p=0.015
      </div>
    </div>
  );
};

export const DelegationGames: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const titleP = spring({ frame, fps, config: { damping: 200 } });
  const half = Math.floor(durationInFrames / 2);
  const isSecondHalf = frame >= half;

  // Cross-fade between views
  const fadeOut = isSecondHalf
    ? interpolate(frame, [half, half + 15], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
    : 1;
  const fadeIn = isSecondHalf
    ? interpolate(frame, [half, half + 15], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
    : 0;

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

      {/* Bar race view */}
      <div style={{ position: "absolute", inset: 0, opacity: isSecondHalf ? fadeOut : 1 }}>
        <Sequence from={0} durationInFrames={half + 15}>
          <BarRace />
        </Sequence>
      </div>

      {/* Bubble view */}
      {isSecondHalf && (
        <div style={{ position: "absolute", inset: 0, opacity: fadeIn }}>
          <Sequence from={half}>
            <BubbleView />
          </Sequence>
        </div>
      )}

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
          DELEGATION GAMES
        </div>
        <div style={{ fontSize: 16, color: colors.textDim, marginTop: 8, fontFamily: fonts.mono }}>
          80 runs / 8 configs / 10 seeds / Bonferroni-corrected
        </div>
      </div>

      {/* View indicator */}
      <div
        style={{
          position: "absolute",
          top: 110,
          left: 60,
          fontSize: 12,
          fontFamily: fonts.mono,
          color: "#BB6BD9",
          opacity: 0.7,
        }}
      >
        {isSecondHalf ? "BUBBLE VIEW" : "BAR RACE"}
      </div>

      {/* Findings */}
      <Sequence from={60}>
        <Findings />
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
          opacity: Math.max(0, titleP),
        }}
      >
        distributional-agi-safety
      </div>
    </AbsoluteFill>
  );
};
