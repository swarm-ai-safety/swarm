import React from "react";
import { AbsoluteFill, useCurrentFrame, spring, interpolate } from "remotion";
import { colors, fonts } from "./theme";

// Particle positions - pre-computed for deterministic layout
const particles = [
  // Inner cluster (high-p agents, green-ish)
  { x: 480, y: 240, r: 6, p: 0.92 },
  { x: 520, y: 210, r: 5, p: 0.88 },
  { x: 540, y: 260, r: 7, p: 0.95 },
  { x: 500, y: 280, r: 4, p: 0.85 },
  { x: 460, y: 220, r: 5, p: 0.91 },
  { x: 510, y: 235, r: 6, p: 0.89 },
  // Mid ring (uncertain agents, yellow/orange)
  { x: 380, y: 180, r: 5, p: 0.55 },
  { x: 600, y: 190, r: 6, p: 0.48 },
  { x: 620, y: 310, r: 5, p: 0.52 },
  { x: 370, y: 320, r: 4, p: 0.45 },
  { x: 420, y: 150, r: 5, p: 0.58 },
  { x: 580, y: 340, r: 4, p: 0.42 },
  // Outer ring (low-p agents, red-ish)
  { x: 300, y: 150, r: 4, p: 0.15 },
  { x: 680, y: 140, r: 5, p: 0.12 },
  { x: 700, y: 350, r: 4, p: 0.18 },
  { x: 290, y: 360, r: 5, p: 0.1 },
  { x: 340, y: 100, r: 3, p: 0.2 },
  { x: 660, y: 380, r: 3, p: 0.08 },
  // Scattered
  { x: 240, y: 250, r: 3, p: 0.3 },
  { x: 740, y: 250, r: 3, p: 0.22 },
  { x: 500, y: 120, r: 3, p: 0.65 },
  { x: 500, y: 400, r: 3, p: 0.35 },
];

// Connection lines between nearby particles
const connections: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5],
  [0, 6], [2, 7], [3, 8], [4, 9],
  [6, 10], [7, 11], [10, 12], [7, 13],
  [8, 14], [9, 15], [6, 16], [11, 17],
];

function pToColor(p: number): string {
  if (p > 0.7) return colors.success;
  if (p > 0.4) return colors.warning;
  return colors.danger;
}

export const HeroImage: React.FC = () => {
  const frame = useCurrentFrame();

  // Subtle breathing animation for particles
  const breathe = Math.sin(frame * 0.05) * 0.15 + 1;

  // Staggered fade-in for connections
  const connFade = spring({ frame, fps: 30, config: { damping: 200 } });

  // Title animations
  const titleP = spring({ frame: frame - 5, fps: 30, config: { damping: 200 } });
  const subtitleP = spring({ frame: frame - 15, fps: 30, config: { damping: 200 } });
  const tagP = spring({ frame: frame - 30, fps: 30, config: { damping: 200 } });
  const barP = spring({ frame: frame - 40, fps: 30, config: { damping: 80 } });

  // Probability bar animated fill
  const barWidth = interpolate(Math.max(0, barP), [0, 1], [0, 320]);

  // Scanning indicator position
  const scanPos = interpolate(frame, [50, 120], [0.15, 0.82], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const scanOpacity = frame > 50 && frame < 125
    ? spring({ frame: frame - 50, fps: 30, config: { damping: 200 } })
    : frame >= 125 ? Math.max(0, 1 - (frame - 125) * 0.05) : 0;

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 40% 45%, ${colors.accentDim}12, ${colors.bg} 65%)`,
        fontFamily: fonts.heading,
      }}
    >
      {/* Grid background */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: `
            linear-gradient(${colors.gridLine} 1px, transparent 1px),
            linear-gradient(90deg, ${colors.gridLine} 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
          opacity: 0.25,
        }}
      />

      {/* Particle network - left side */}
      <svg
        width={800}
        height={500}
        style={{ position: "absolute", left: 0, top: 65 }}
        viewBox="0 0 800 500"
      >
        {/* Connection lines */}
        {connections.map(([i, j], idx) => {
          const a = particles[i];
          const b = particles[j];
          const avgP = (a.p + b.p) / 2;
          const delay = idx * 2;
          const lineOpacity = interpolate(
            Math.max(0, spring({ frame: frame - delay, fps: 30, config: { damping: 200 } })),
            [0, 1],
            [0, 0.2 + avgP * 0.15],
          );
          return (
            <line
              key={`c-${idx}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={pToColor(avgP)}
              strokeWidth={1}
              opacity={lineOpacity * connFade}
            />
          );
        })}

        {/* Particles */}
        {particles.map((pt, idx) => {
          const delay = idx * 1.5;
          const ptP = Math.max(
            0,
            spring({ frame: frame - delay, fps: 30, config: { damping: 150 } }),
          );
          const col = pToColor(pt.p);
          const radius = pt.r * breathe * ptP;
          return (
            <g key={`p-${idx}`}>
              {/* Glow */}
              <circle
                cx={pt.x}
                cy={pt.y}
                r={radius * 3}
                fill={col}
                opacity={0.08 * ptP}
              />
              {/* Core */}
              <circle
                cx={pt.x}
                cy={pt.y}
                r={radius}
                fill={col}
                opacity={0.9 * ptP}
              />
            </g>
          );
        })}
      </svg>

      {/* Right side: text + probability bar */}
      <div
        style={{
          position: "absolute",
          right: 80,
          top: 0,
          bottom: 0,
          width: 480,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
        }}
      >
        {/* Title */}
        <div
          style={{
            fontSize: 72,
            fontWeight: 800,
            color: colors.text,
            letterSpacing: "0.12em",
            opacity: Math.max(0, titleP),
            transform: `translateY(${interpolate(Math.max(0, titleP), [0, 1], [15, 0])}px)`,
          }}
        >
          SWARM
        </div>

        {/* Accent line */}
        <div
          style={{
            width: interpolate(Math.max(0, subtitleP), [0, 1], [0, 200]),
            height: 2,
            background: `linear-gradient(90deg, ${colors.accent}, transparent)`,
            marginTop: 12,
            marginBottom: 16,
          }}
        />

        {/* Subtitle */}
        <div
          style={{
            fontSize: 22,
            fontWeight: 600,
            color: colors.accent,
            opacity: Math.max(0, subtitleP),
            letterSpacing: "0.04em",
            lineHeight: 1.4,
          }}
        >
          Distributional AGI Safety
        </div>

        {/* Tagline */}
        <div
          style={{
            fontSize: 15,
            color: colors.textDim,
            opacity: Math.max(0, tagP),
            marginTop: 16,
            lineHeight: 1.6,
            maxWidth: 400,
          }}
        >
          Probabilistic measurement of emergent risks
          in multi-agent AI systems
        </div>

        {/* Mini probability bar */}
        <div
          style={{
            marginTop: 28,
            opacity: Math.max(0, barP),
          }}
        >
          <div
            style={{
              width: 320,
              height: 20,
              borderRadius: 10,
              background: `${colors.textMuted}25`,
              position: "relative",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: barWidth,
                height: "100%",
                borderRadius: 10,
                background: `linear-gradient(90deg, ${colors.danger}, ${colors.warning}, ${colors.success})`,
                opacity: 0.85,
              }}
            />
            {/* Scanning indicator */}
            {scanOpacity > 0 && (
              <div
                style={{
                  position: "absolute",
                  top: -4,
                  left: scanPos * 320 - 1.5,
                  width: 3,
                  height: 28,
                  background: colors.text,
                  borderRadius: 1.5,
                  opacity: scanOpacity,
                }}
              />
            )}
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              width: 320,
              marginTop: 6,
            }}
          >
            <span
              style={{
                fontSize: 11,
                fontFamily: fonts.mono,
                color: colors.danger,
                fontWeight: 600,
                opacity: Math.max(0, barP),
              }}
            >
              p=0
            </span>
            <span
              style={{
                fontSize: 11,
                fontFamily: fonts.mono,
                color: colors.textDim,
                opacity: Math.max(0, barP),
              }}
            >
              P(beneficial)
            </span>
            <span
              style={{
                fontSize: 11,
                fontFamily: fonts.mono,
                color: colors.success,
                fontWeight: 600,
                opacity: Math.max(0, barP),
              }}
            >
              p=1
            </span>
          </div>
        </div>

        {/* Key stats row */}
        <div
          style={{
            display: "flex",
            gap: 28,
            marginTop: 24,
            opacity: Math.max(0, spring({ frame: frame - 55, fps: 30, config: { damping: 200 } })),
          }}
        >
          {[
            { label: "soft labels", icon: "~" },
            { label: "multi-agent", icon: "+" },
            { label: "reproducible", icon: "=" },
          ].map(({ label, icon }) => (
            <div
              key={label}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              <span
                style={{
                  fontSize: 16,
                  fontFamily: fonts.mono,
                  color: colors.accent,
                  fontWeight: 700,
                }}
              >
                {icon}
              </span>
              <span style={{ fontSize: 12, color: colors.textDim }}>
                {label}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom edge gradient fade */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: 40,
          background: `linear-gradient(transparent, ${colors.bg})`,
        }}
      />
    </AbsoluteFill>
  );
};
