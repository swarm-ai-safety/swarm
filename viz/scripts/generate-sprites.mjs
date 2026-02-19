#!/usr/bin/env node
/**
 * generate-sprites.mjs
 *
 * Generates sprite PNGs for the 6 SWARM agent types using Gemini Imagen 4.
 * Produces 12 images: 6 types x 2 states (idle, walk).
 *
 * - Idle: single 192x336 frame (4x canvas resolution for crisp rendering)
 * - Walk: 4-frame vertical strip (192x1344), stitched together
 *
 * Usage:
 *   GEMINI_API_KEY=... node viz/scripts/generate-sprites.mjs
 *   # or
 *   cd viz && npm run generate-sprites
 *
 * Requirements:
 *   npm install --save-dev @google/genai @napi-rs/canvas
 *
 * Manual fallback:
 *   If API access is unavailable, paste the prompts below into gemini.google.com
 *   with Imagen enabled, download the PNGs, and place them in viz/public/sprites/.
 */

import { GoogleGenAI } from "@google/genai";
import { createCanvas, loadImage } from "@napi-rs/canvas";
import { writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(__dirname, "..", "public", "sprites");
mkdirSync(OUT_DIR, { recursive: true });

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
  console.error("ERROR: Set GEMINI_API_KEY environment variable");
  console.error("  export GEMINI_API_KEY=your-key-here");
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

// Frame dimensions (4x canvas size of 48x84)
const FRAME_W = 192;
const FRAME_H = 336;
const WALK_FRAMES = 4;

/** Agent type visual descriptions for prompts */
const AGENT_TYPES = {
  honest: {
    colors: "emerald green and teal",
    features: "serene expression, clean geometric line patterns on coat, small emblem on chest",
    hair: "bright green bob cut",
    personality: "trustworthy and calm",
  },
  opportunistic: {
    colors: "gold and amber",
    features: "sly confident smirk, flashy coat with gold trim details, ornate belt buckle",
    hair: "golden-yellow styled hair",
    personality: "cunning and resourceful",
  },
  deceptive: {
    colors: "purple and magenta",
    features: "enigmatic expression, shifting iridescent coat patterns, high collar",
    hair: "violet-purple bob cut",
    personality: "mysterious and secretive",
  },
  adversarial: {
    colors: "red and crimson",
    features: "aggressive stance, jagged coat edges, sharp shoulder pads, spiked hair",
    hair: "fiery red spiked hair",
    personality: "aggressive and threatening",
  },
  rlm: {
    colors: "cyan and electric blue",
    features: "data visor across eyes, circuit trace patterns on coat, small antenna on head",
    hair: "blue-cyan neat hair",
    personality: "analytical and robotic",
  },
  crewai: {
    colors: "lime green and yellow-green",
    features: "hard hat on head, tool belt at waist, utility worker look, sturdy boots",
    hair: "bright green short hair under hard hat",
    personality: "industrious and collaborative",
  },
};

/**
 * Build the image generation prompt for a given agent type and state.
 *
 * Walk sprites request a 4-frame horizontal strip showing a walk cycle,
 * which we then rotate into a vertical strip.
 */
function buildPrompt(agentType, config, state) {
  const base = [
    "Isometric 3/4 view character sprite, facing left.",
    "Plain white background, no colored background, no gradient background.",
    "Cyberpunk holographic aesthetic with dark body silhouette and glowing translucent coat.",
    `Color scheme: ${config.colors}.`,
    `Character has ${config.hair} and ${config.features}.`,
    `Personality: ${config.personality}.`,
    "The character wears a long translucent holographic coat over a dark bodysuit.",
    "Stylized 3D rendering, clean edges, suitable for a game sprite.",
    "No text, no UI elements, no watermarks.",
  ];

  if (state === "idle") {
    base.push(
      `Single character in neutral standing pose, ${FRAME_W}x${FRAME_H} pixels.`,
      "Slight idle sway, weight on both feet."
    );
  } else {
    base.push(
      `Horizontal sprite sheet of 4 frames showing walk cycle, total ${FRAME_W * WALK_FRAMES}x${FRAME_H} pixels.`,
      "Frames left to right: left foot forward, passing, right foot forward, passing.",
      "Each frame is exactly ${FRAME_W} pixels wide.",
      "Consistent character proportions across all frames."
    );
  }

  return base.join(" ");
}

/**
 * Generate a single sprite image via Imagen 4.
 * Returns a Buffer containing the PNG data.
 */
async function generateImage(prompt) {
  const response = await ai.models.generateImages({
    model: "imagen-4.0-generate-001",
    prompt,
    config: {
      numberOfImages: 1,
      outputMimeType: "image/png",
    },
  });

  if (!response.generatedImages || response.generatedImages.length === 0) {
    throw new Error("No images generated");
  }

  return Buffer.from(response.generatedImages[0].image.imageBytes, "base64");
}

/**
 * For walk sprites: convert Imagen output into a vertical strip of 4 frames.
 *
 * - If Imagen returns a horizontal strip (wide image), split into 4 columns
 *   and stack vertically at native resolution.
 * - If it returns a single image (square/tall), duplicate it 4x at native
 *   resolution with subtle vertical bounce offsets.
 *
 * The sprite registry reads actual image dimensions at runtime, so we
 * preserve native resolution instead of downscaling to FRAME_W x FRAME_H.
 */
async function processWalkSprite(pngBuffer) {
  const img = await loadImage(pngBuffer);

  const isHorizontalStrip = img.width > img.height * 1.5;

  if (isHorizontalStrip) {
    // Horizontal strip -> vertical strip at native frame size
    const frameW = Math.floor(img.width / WALK_FRAMES);
    const frameH = img.height;
    const outCanvas = createCanvas(frameW, frameH * WALK_FRAMES);
    const ctx = outCanvas.getContext("2d");
    for (let i = 0; i < WALK_FRAMES; i++) {
      ctx.drawImage(
        img,
        i * frameW, 0, frameW, frameH,       // source
        0, i * frameH, frameW, frameH         // dest (native res)
      );
    }
    return outCanvas.toBuffer("image/png");
  } else {
    // Single image — stack 4 copies vertically at native resolution
    // with subtle bounce offsets to simulate walk bob
    const frameW = img.width;
    const frameH = img.height;
    const bouncePixels = Math.round(frameH * 0.004); // ~4px at 1024
    const bounceOffsets = [0, -bouncePixels, 0, -bouncePixels];
    const outCanvas = createCanvas(frameW, frameH * WALK_FRAMES);
    const ctx = outCanvas.getContext("2d");
    for (let i = 0; i < WALK_FRAMES; i++) {
      ctx.drawImage(
        img,
        0, 0, frameW, frameH,
        0, i * frameH + bounceOffsets[i], frameW, frameH
      );
    }
    return outCanvas.toBuffer("image/png");
  }
}

// --- Environment sprite definitions (tiles + buildings) ---

const ENV_SPRITES = {
  tile: {
    width: 384,
    height: 192,
    prompt: [
      "A single rhombus (diamond) shape centered in the image, like a playing card diamond rotated 90 degrees.",
      "The diamond points left, right, up, and down. It is wider than tall (2:1 width to height ratio).",
      "Plain white background filling the entire image outside the diamond.",
      "The diamond interior is dark navy/charcoal with faint glowing teal circuit-board traces and grid lines.",
      "Tiny glowing solder-point dots at circuit intersections. Cyberpunk PCB aesthetic.",
      "384x192 pixels. Perfectly symmetrical diamond. No text, no UI, no watermarks.",
    ].join(" "),
  },
  tower: {
    width: 128,
    height: 512,
    prompt: [
      "Isometric 3/4 view of a tall cyberpunk server rack / data tower building.",
      "Plain white background, no colored background, no gradient.",
      "Dark navy metallic panels, teal-glowing seams, small rectangular windows in grid pattern.",
      "Isometric box shape: narrow width, tall height. Flat top with small antenna.",
      "128x512 pixels. No text, no UI elements, no watermarks.",
    ].join(" "),
  },
  spire: {
    width: 64,
    height: 512,
    prompt: [
      "Isometric 3/4 view of a thin tall cyberpunk relay antenna spire.",
      "Plain white background, no colored background, no gradient.",
      "Dark metallic pole tapering to a point, with a small satellite dish partway up.",
      "Faint teal glow at the tip. Very thin and vertical.",
      "64x512 pixels. No text, no UI elements, no watermarks.",
    ].join(" "),
  },
  node: {
    width: 128,
    height: 128,
    prompt: [
      "Isometric 3/4 view of a compact cyberpunk power cube / energy node.",
      "Plain white background, no colored background, no gradient.",
      "Small dark isometric box with glowing teal energy core visible through panel gaps.",
      "Subtle holographic shimmer on top face. Compact and squat shape.",
      "128x128 pixels. No text, no UI elements, no watermarks.",
    ].join(" "),
  },
};

async function main() {
  const types = Object.entries(AGENT_TYPES);
  const envEntries = Object.entries(ENV_SPRITES);
  let generated = 0;
  const total = types.length * 2 + envEntries.length;

  console.log(`Generating ${total} sprite images in ${OUT_DIR}/\n`);

  // --- Character sprites ---
  for (const [agentType, config] of types) {
    for (const state of ["idle", "walk"]) {
      const outPath = join(OUT_DIR, `${agentType}_${state}.png`);
      const prompt = buildPrompt(agentType, config, state);

      console.log(`[${++generated}/${total}] ${agentType}_${state}...`);
      // Log prompt for manual fallback reference
      console.log(`  Prompt: ${prompt.substring(0, 120)}...`);

      try {
        const pngBuffer = await generateImage(prompt);

        if (state === "walk") {
          const processed = await processWalkSprite(pngBuffer);
          writeFileSync(outPath, processed);
        } else {
          writeFileSync(outPath, pngBuffer);
        }

        console.log(`  -> ${outPath}`);
      } catch (err) {
        console.error(`  ERROR: ${err.message}`);
        console.error(`  Skipping ${agentType}_${state}. Use manual fallback.`);
      }
    }
  }

  // --- Environment sprites (tile + buildings) ---
  console.log("\n--- Environment sprites ---\n");
  for (const [name, def] of envEntries) {
    const outPath = join(OUT_DIR, `${name}.png`);
    console.log(`[${++generated}/${total}] ${name}...`);
    console.log(`  Prompt: ${def.prompt.substring(0, 120)}...`);

    try {
      const pngBuffer = await generateImage(def.prompt);
      writeFileSync(outPath, pngBuffer);
      console.log(`  -> ${outPath}`);
    } catch (err) {
      console.error(`  ERROR: ${err.message}`);
      console.error(`  Skipping ${name}. Use manual fallback.`);
    }
  }

  console.log("\nDone! Place any missing sprites manually in viz/public/sprites/");
  console.log("Expected files:");
  for (const [agentType] of types) {
    console.log(`  ${agentType}_idle.png  (${FRAME_W}x${FRAME_H})`);
    console.log(`  ${agentType}_walk.png  (${FRAME_W}x${FRAME_H * WALK_FRAMES} vertical strip)`);
  }
  for (const [name, def] of envEntries) {
    console.log(`  ${name}.png  (${def.width}x${def.height})`);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});

/*
 * ============================================================
 * MANUAL FALLBACK PROMPTS
 * ============================================================
 * If the API is unavailable, paste these into gemini.google.com
 * or any Imagen-capable tool. Save outputs as the filenames shown.
 *
 * --- honest_idle.png (192x336) ---
 * Isometric 3/4 view character sprite, facing left. Transparent background.
 * Cyberpunk holographic aesthetic with dark body silhouette and glowing translucent coat.
 * Color scheme: emerald green and teal. Character has bright green bob cut and
 * glowing halo ring floating above head, serene expression, clean geometric coat patterns.
 * Single standing pose, stylized 3D, no text.
 *
 * --- honest_walk.png (4 frames, each 192x336, stacked vertically = 192x1344) ---
 * Same as above but 4-frame walk cycle sprite sheet.
 *
 * (Repeat for: opportunistic, deceptive, adversarial, rlm, crewai)
 *
 * --- tile.png (384x192) ---
 * Isometric diamond-shaped floor tile. White background.
 * Dark cyberpunk circuit-board aesthetic with faint glowing teal traces.
 *
 * --- tower.png (128x512) ---
 * Isometric 3/4 view cyberpunk server rack / data tower. White background.
 * Dark navy panels, teal-glowing seams, window grid pattern.
 *
 * --- spire.png (64x512) ---
 * Isometric 3/4 view thin cyberpunk relay antenna. White background.
 * Dark metallic pole tapering to point, small satellite dish.
 *
 * --- node.png (128x128) ---
 * Isometric 3/4 view compact cyberpunk power cube. White background.
 * Dark isometric box with glowing teal energy core.
 * ============================================================
 */
