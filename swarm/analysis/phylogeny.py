"""Animated agent phylogeny visualization.

Generates self-contained HTML files with Canvas/JS animation showing agents
as particles drifting through a Perlin noise flow field, positioned by
reputation (X) and cumulative payoff (Y), colored by agent type.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Agent type colors matching dashboard conventions
AGENT_COLORS: Dict[str, str] = {
    "honest": "#22c55e",
    "opportunistic": "#facc15",
    "deceptive": "#8b5cf6",
    "adversarial": "#ef4444",
    "rlm": "#06b6d4",
}
DEFAULT_COLOR = "#6b7280"


def extract_agent_trajectories(
    event_log_path: Union[str, Path],
    replay_index: int = -1,
) -> Dict[str, Any]:
    """Extract per-agent-per-epoch state from a JSONL event log.

    Streams the file line-by-line for memory efficiency.
    Detects replay boundaries via ``simulation_started`` events and
    processes only the target replay (default: last).

    Returns dict with keys:
        agents: {agent_id: {agent_type, epochs: [{epoch, reputation, ...}]}}
        n_epochs: int
        n_agents: int
    """
    event_log_path = Path(event_log_path)

    # Buffers for agent_created events that precede simulation_started
    pending_agents: Dict[str, str] = {}

    # Per-replay state (reset on each simulation_started)
    agent_registry: Dict[str, str] = {}
    cumulative_payoffs: Dict[str, float] = {}
    latest_reputation: Dict[str, float] = {}
    epoch_interactions: Dict[tuple, int] = {}
    epoch_p_sum: Dict[tuple, float] = {}
    epoch_p_count: Dict[tuple, int] = {}
    frozen_agents: set = set()
    epoch_snapshots: Dict[tuple, dict] = {}
    max_epoch: int = -1
    n_replays: int = 0

    with open(event_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            etype = event.get("event_type", "")

            if etype == "agent_created":
                aid = event.get("agent_id", "")
                atype = event.get("payload", {}).get("agent_type", "unknown")
                # Always buffer; simulation_started will move to registry
                pending_agents[aid] = atype

            elif etype == "simulation_started":
                n_replays += 1
                # Reset all state for new replay
                agent_registry = dict(pending_agents)
                pending_agents = {}
                cumulative_payoffs = dict.fromkeys(agent_registry, 0.0)
                latest_reputation = dict.fromkeys(agent_registry, 0.0)
                epoch_interactions = {}
                epoch_p_sum = {}
                epoch_p_count = {}
                frozen_agents = set()
                epoch_snapshots = {}
                max_epoch = -1

            elif etype == "reputation_updated":
                aid = event.get("agent_id")
                if aid and aid in agent_registry:
                    latest_reputation[aid] = event["payload"]["new_reputation"]

            elif etype == "payoff_computed":
                epoch = event.get("epoch", 0) or 0
                payload = event.get("payload", {})
                p = payload.get("components", {}).get("p", 0.5)
                initiator = event.get("initiator_id")
                counterparty = event.get("counterparty_id")

                for aid, payoff_key in [
                    (initiator, "payoff_initiator"),
                    (counterparty, "payoff_counterparty"),
                ]:
                    if aid and aid in agent_registry:
                        cumulative_payoffs[aid] += payload.get(payoff_key, 0.0)
                        key = (aid, epoch)
                        epoch_interactions[key] = epoch_interactions.get(key, 0) + 1
                        epoch_p_sum[key] = epoch_p_sum.get(key, 0.0) + p
                        epoch_p_count[key] = epoch_p_count.get(key, 0) + 1

            elif etype == "agent_state_updated":
                aid = event.get("agent_id")
                status = event.get("payload", {}).get("status")
                if aid and status == "frozen":
                    frozen_agents.add(aid)
                elif aid and status == "active":
                    frozen_agents.discard(aid)

            elif etype == "epoch_completed":
                epoch = event.get("payload", {}).get(
                    "epoch", event.get("epoch", 0)
                )
                if epoch is None:
                    continue
                max_epoch = max(max_epoch, epoch)
                # Snapshot all agents at end of this epoch
                for aid in agent_registry:
                    key = (aid, epoch)
                    pc = epoch_p_count.get(key, 0)
                    avg_p = (
                        epoch_p_sum.get(key, 0.0) / pc if pc > 0 else 0.5
                    )
                    epoch_snapshots[key] = {
                        "epoch": epoch,
                        "reputation": latest_reputation.get(aid, 0.0),
                        "cumulative_payoff": cumulative_payoffs.get(aid, 0.0),
                        "n_interactions": epoch_interactions.get(key, 0),
                        "avg_p": avg_p,
                        "is_frozen": aid in frozen_agents,
                    }

    # Build result with carry-forward for missing epochs
    agents_result: Dict[str, Any] = {}
    for aid, atype in agent_registry.items():
        epochs_list: List[dict] = []
        for e in range(max_epoch + 1):
            key = (aid, e)
            if key in epoch_snapshots:
                epochs_list.append(epoch_snapshots[key])
            elif epochs_list:
                # Carry forward from previous epoch
                prev = dict(epochs_list[-1])
                prev["epoch"] = e
                prev["n_interactions"] = 0
                prev["avg_p"] = 0.5
                epochs_list.append(prev)
            else:
                epochs_list.append(
                    {
                        "epoch": e,
                        "reputation": 0.0,
                        "cumulative_payoff": 0.0,
                        "n_interactions": 0,
                        "avg_p": 0.5,
                        "is_frozen": False,
                    }
                )
        agents_result[aid] = {
            "agent_type": atype,
            "epochs": epochs_list,
        }

    return {
        "agents": agents_result,
        "n_epochs": max_epoch + 1 if max_epoch >= 0 else 0,
        "n_agents": len(agent_registry),
    }


def extract_ecosystem_metrics(
    source_path: Union[str, Path],
) -> Dict[str, Any]:
    """Extract ecosystem-level metrics from CSV or history.json.

    Returns dict with key:
        epochs: [{epoch, toxicity_rate, ecosystem_threat_level, quality_gap,
                  n_agents, n_frozen, total_welfare}]
    """
    source_path = Path(source_path)
    epochs: List[dict] = []

    if source_path.suffix == ".csv":
        with open(source_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(
                    {
                        "epoch": int(row.get("epoch", 0)),
                        "toxicity_rate": float(
                            row.get("toxicity_rate", 0)
                        ),
                        "ecosystem_threat_level": float(
                            row.get("ecosystem_threat_level", 0)
                        ),
                        "quality_gap": float(row.get("quality_gap", 0)),
                        "n_agents": int(row.get("n_agents", 0)),
                        "n_frozen": int(row.get("n_frozen", 0)),
                        "total_welfare": float(
                            row.get("total_welfare", 0)
                        ),
                    }
                )
    elif source_path.suffix == ".json":
        with open(source_path) as f:
            data = json.load(f)
        snapshots = data.get("epoch_snapshots", [])
        if isinstance(snapshots, list):
            for snap in snapshots:
                epochs.append(
                    {
                        "epoch": snap.get("epoch", 0),
                        "toxicity_rate": snap.get("toxicity_rate", 0),
                        "ecosystem_threat_level": snap.get(
                            "ecosystem_threat_level", 0
                        ),
                        "quality_gap": snap.get("quality_gap", 0),
                        "n_agents": snap.get("n_agents", 0),
                        "n_frozen": snap.get("n_frozen", 0),
                        "total_welfare": snap.get("total_welfare", 0),
                    }
                )

    return {"epochs": epochs}


def generate_phylogeny_html(
    agent_data: Dict[str, Any],
    ecosystem_data: Dict[str, Any],
    output_path: Union[str, Path],
    title: str = "Agent Phylogeny",
    width: int = 1200,
    height: int = 800,
) -> Path:
    """Generate self-contained HTML with animated phylogeny visualization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "width": width,
        "height": height,
        "colors": AGENT_COLORS,
        "default_color": DEFAULT_COLOR,
    }

    html = _HTML_TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__WIDTH__", str(width))
    html = html.replace("__HEIGHT__", str(height))
    html = html.replace("__AGENT_DATA_JSON__", json.dumps(agent_data))
    html = html.replace("__ECOSYSTEM_DATA_JSON__", json.dumps(ecosystem_data))
    html = html.replace("__CONFIG_JSON__", json.dumps(config))

    output_path.write_text(html)
    return output_path


def generate_phylogeny(
    run_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Auto-discover data in *run_dir* and generate phylogeny animation."""
    run_dir = Path(run_dir)

    # Find event log
    jsonl_files = sorted(run_dir.glob("artifacts/*.jsonl"))
    if not jsonl_files:
        jsonl_files = sorted(run_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL event log found in {run_dir}")
    event_log = jsonl_files[0]

    # Find ecosystem data (CSV preferred, then history.json)
    csv_files = sorted(run_dir.glob("csv/*epoch*.csv"))
    if not csv_files:
        csv_files = sorted(run_dir.glob("csv/*.csv"))
    json_file = run_dir / "history.json"

    eco_source: Optional[Path] = None
    if csv_files:
        eco_source = csv_files[0]
    elif json_file.exists():
        eco_source = json_file

    agent_data = extract_agent_trajectories(event_log)
    ecosystem_data = (
        extract_ecosystem_metrics(eco_source)
        if eco_source
        else {"epochs": []}
    )

    if output_path is None:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        output_path = plots_dir / "phylogeny.html"

    run_title = f"Agent Phylogeny \u2014 {run_dir.name}"
    return generate_phylogeny_html(
        agent_data, ecosystem_data, output_path, title=run_title
    )


# ---------------------------------------------------------------------------
# HTML / JS template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a1a;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;display:flex;justify-content:center;padding:20px}
#container{max-width:__WIDTH__px;width:100%}
h1{font-size:1.1em;color:#fff;margin-bottom:12px;font-weight:500}
#canvas-wrap{position:relative}
canvas{display:block;border-radius:6px;border:1px solid rgba(255,255,255,0.08)}
#controls{display:flex;align-items:center;gap:10px;margin-top:10px;padding:8px 0;flex-wrap:wrap}
#playBtn{background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:#fff;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:14px;min-width:38px;text-align:center}
#playBtn:hover{background:rgba(255,255,255,0.18)}
input[type=range]{-webkit-appearance:none;height:4px;background:rgba(255,255,255,0.15);border-radius:2px;outline:none}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:#fff;border-radius:50%;cursor:pointer}
#epochSlider{flex:1;min-width:120px}
#epoch-display{color:#888;font-size:.85em;font-family:monospace;min-width:110px}
#legend{display:flex;gap:16px;margin-top:6px;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:6px;font-size:.8em;color:#aaa}
.legend-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
#tooltip{position:absolute;display:none;background:rgba(0,0,0,0.88);border:1px solid rgba(255,255,255,0.15);border-radius:6px;padding:8px 12px;font-size:.8em;font-family:monospace;line-height:1.5;pointer-events:none;z-index:10;color:#e0e0e0;max-width:250px}
.speed-label{margin-left:auto;font-size:.8em;color:#888;display:flex;align-items:center;gap:6px}
#speedSlider{width:80px}
</style>
</head>
<body>
<div id="container">
  <h1>__TITLE__</h1>
  <div id="canvas-wrap">
    <canvas id="canvas"></canvas>
    <div id="tooltip"></div>
  </div>
  <div id="controls">
    <button id="playBtn">&#9654;</button>
    <input type="range" id="epochSlider" min="0" max="1" value="0" step="0.01">
    <span id="epoch-display">Epoch 0 / 0</span>
    <span class="speed-label">Speed
      <input type="range" id="speedSlider" min="1" max="20" value="5">
    </span>
  </div>
  <div id="legend"></div>
</div>

<script>
(function(){
"use strict";

// === Embedded data ===
var AGENT_DATA = __AGENT_DATA_JSON__;
var ECOSYSTEM_DATA = __ECOSYSTEM_DATA_JSON__;
var CONFIG = __CONFIG_JSON__;

// === Constants ===
var COLORS = CONFIG.colors || {};
var DEFAULT_COLOR = CONFIG.default_color || "#6b7280";
var W = CONFIG.width || 1200;
var H = CONFIG.height || 800;

// === Canvas setup ===
var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
var dpr = window.devicePixelRatio || 1;
canvas.width = W * dpr;
canvas.height = H * dpr;
canvas.style.width = W + "px";
canvas.style.height = H + "px";
ctx.scale(dpr, dpr);

// === Plot margins ===
var margin = {top: 40, right: 30, bottom: 50, left: 70};
var plotW = W - margin.left - margin.right;
var plotH = H - margin.top - margin.bottom;

// === Compute data ranges ===
var repMin = Infinity, repMax = -Infinity;
var payMin = Infinity, payMax = -Infinity;
var agentIds = Object.keys(AGENT_DATA.agents);

for (var i = 0; i < agentIds.length; i++) {
    var ag = AGENT_DATA.agents[agentIds[i]];
    for (var j = 0; j < ag.epochs.length; j++) {
        var ep = ag.epochs[j];
        if (ep.reputation < repMin) repMin = ep.reputation;
        if (ep.reputation > repMax) repMax = ep.reputation;
        if (ep.cumulative_payoff < payMin) payMin = ep.cumulative_payoff;
        if (ep.cumulative_payoff > payMax) payMax = ep.cumulative_payoff;
    }
}

// Add padding
var repPad = Math.max((repMax - repMin) * 0.12, 0.05);
var payPad = Math.max((payMax - payMin) * 0.12, 0.1);
repMin -= repPad; repMax += repPad;
payMin -= payPad; payMax += payPad;

var maxEpoch = Math.max(AGENT_DATA.n_epochs - 1, 0);

// === Perlin noise (3D, classic) ===
var Perlin = (function() {
    var p = new Uint8Array(512);
    var perm = new Uint8Array(256);
    for (var i = 0; i < 256; i++) perm[i] = i;
    // Fisher-Yates shuffle
    for (var i = 255; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    for (var i = 0; i < 512; i++) p[i] = perm[i & 255];

    function fade(t) { return t * t * t * (t * (t * 6 - 15) + 10); }
    function lerp(a, b, t) { return a + t * (b - a); }
    function grad3(hash, x, y, z) {
        var h = hash & 15;
        var u = h < 8 ? x : y;
        var v = h < 4 ? y : (h === 12 || h === 14 ? x : z);
        return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
    }

    return {
        noise: function(x, y, z) {
            var X = Math.floor(x) & 255;
            var Y = Math.floor(y) & 255;
            var Z = Math.floor(z) & 255;
            x -= Math.floor(x);
            y -= Math.floor(y);
            z -= Math.floor(z);
            var u = fade(x), v = fade(y), w = fade(z);
            var A  = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z;
            var B  = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;
            return lerp(
                lerp(
                    lerp(grad3(p[AA], x, y, z), grad3(p[BA], x-1, y, z), u),
                    lerp(grad3(p[AB], x, y-1, z), grad3(p[BB], x-1, y-1, z), u),
                    v),
                lerp(
                    lerp(grad3(p[AA+1], x, y, z-1), grad3(p[BA+1], x-1, y, z-1), u),
                    lerp(grad3(p[AB+1], x, y-1, z-1), grad3(p[BB+1], x-1, y-1, z-1), u),
                    v),
                w);
        }
    };
})();

// === Build agent objects ===
function hexToRGB(hex) {
    return [parseInt(hex.slice(1,3),16), parseInt(hex.slice(3,5),16), parseInt(hex.slice(5,7),16)];
}

var agents = [];
for (var i = 0; i < agentIds.length; i++) {
    var id = agentIds[i];
    var d = AGENT_DATA.agents[id];
    var color = COLORS[d.agent_type] || DEFAULT_COLOR;
    var rgb = hexToRGB(color);
    agents.push({
        id: id,
        type: d.agent_type,
        epochs: d.epochs,
        color: color,
        rgbStr: rgb[0] + "," + rgb[1] + "," + rgb[2],
        x: W / 2,
        y: H / 2,
        trail: []
    });
}

// === State ===
var currentEpoch = 0;
var playing = false;
var baseSpeed = 0.4;
var flowTime = 0;
var lastTS = null;

// === Controls ===
var playBtn = document.getElementById("playBtn");
var epochSlider = document.getElementById("epochSlider");
var speedSlider = document.getElementById("speedSlider");
var epochDisplay = document.getElementById("epoch-display");
var tooltip = document.getElementById("tooltip");

epochSlider.max = maxEpoch || 1;
epochSlider.step = 0.01;

playBtn.addEventListener("click", function() {
    playing = !playing;
    playBtn.innerHTML = playing ? "&#9646;&#9646;" : "&#9654;";
    if (playing && currentEpoch >= maxEpoch) currentEpoch = 0;
});

epochSlider.addEventListener("input", function() {
    currentEpoch = parseFloat(epochSlider.value);
    playing = false;
    playBtn.innerHTML = "&#9654;";
});

// === Helpers ===
function lerp(a, b, t) { return a + (b - a) * t; }

function dataToCanvas(rep, pay) {
    var x = margin.left + (rep - repMin) / (repMax - repMin) * plotW;
    var y = margin.top + plotH - (pay - payMin) / (payMax - payMin) * plotH;
    return {x: x, y: y};
}

function getAgentState(agent, epoch) {
    var eps = agent.epochs;
    if (!eps || eps.length === 0) return null;
    var idx = Math.floor(epoch);
    var frac = epoch - idx;
    if (idx >= eps.length - 1) return eps[eps.length - 1];
    var a = eps[idx];
    var b = eps[Math.min(idx + 1, eps.length - 1)];
    return {
        reputation: lerp(a.reputation, b.reputation, frac),
        cumulative_payoff: lerp(a.cumulative_payoff, b.cumulative_payoff, frac),
        n_interactions: lerp(a.n_interactions, b.n_interactions, frac),
        avg_p: lerp(a.avg_p, b.avg_p, frac),
        is_frozen: frac < 0.5 ? a.is_frozen : b.is_frozen
    };
}

function getEcoState(epoch) {
    var eps = ECOSYSTEM_DATA.epochs;
    if (!eps || eps.length === 0) return {toxicity_rate:0, ecosystem_threat_level:0, quality_gap:0, total_welfare:0};
    var idx = Math.floor(epoch);
    var frac = epoch - idx;
    if (idx >= eps.length - 1) return eps[eps.length - 1];
    var a = eps[idx];
    var b = eps[Math.min(idx + 1, eps.length - 1)];
    return {
        toxicity_rate: lerp(a.toxicity_rate||0, b.toxicity_rate||0, frac),
        ecosystem_threat_level: lerp(a.ecosystem_threat_level||0, b.ecosystem_threat_level||0, frac),
        quality_gap: lerp(a.quality_gap||0, b.quality_gap||0, frac),
        total_welfare: lerp(a.total_welfare||0, b.total_welfare||0, frac)
    };
}

// === Drawing functions ===

function drawBackground(eco) {
    var tox = Math.min(eco.toxicity_rate || 0, 1);
    var grad = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, W * 0.7);
    var r = Math.round(10 + tox * 40);
    var g = Math.round(10 + (1 - tox) * 15);
    grad.addColorStop(0, "rgb(" + r + "," + g + ",26)");
    grad.addColorStop(1, "#0a0a1a");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);
}

function drawGrid() {
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    var nx = 6, ny = 6;
    for (var i = 0; i <= nx; i++) {
        var x = margin.left + (i / nx) * plotW;
        ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, H - margin.bottom); ctx.stroke();
    }
    for (var i = 0; i <= ny; i++) {
        var y = margin.top + (i / ny) * plotH;
        ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(W - margin.right, y); ctx.stroke();
    }
    ctx.restore();
}

function drawAxes() {
    ctx.save();
    ctx.fillStyle = "#666";
    ctx.font = "11px monospace";
    ctx.textAlign = "center";

    var nx = 6;
    for (var i = 0; i <= nx; i++) {
        var x = margin.left + (i / nx) * plotW;
        var val = repMin + (i / nx) * (repMax - repMin);
        ctx.fillText(val.toFixed(2), x, H - margin.bottom + 16);
    }
    ctx.fillStyle = "#888";
    ctx.fillText("Reputation", W / 2, H - 8);

    ctx.textAlign = "right";
    ctx.fillStyle = "#666";
    var ny = 6;
    for (var i = 0; i <= ny; i++) {
        var y = margin.top + (i / ny) * plotH;
        var val = payMax - (i / ny) * (payMax - payMin);
        ctx.fillText(val.toFixed(1), margin.left - 8, y + 4);
    }

    ctx.save();
    ctx.translate(14, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillStyle = "#888";
    ctx.fillText("Cumulative Payoff", 0, 0);
    ctx.restore();

    ctx.restore();
}

function drawFlowField(eco) {
    var step = 55;
    var turb = 0.3 + (eco.toxicity_rate || 0) * 0.7;
    ctx.save();
    ctx.globalAlpha = 0.04 + turb * 0.04;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 0.8;
    for (var x = margin.left + step / 2; x < W - margin.right; x += step) {
        for (var y = margin.top + step / 2; y < H - margin.bottom; y += step) {
            var angle = Perlin.noise(x * 0.005 * turb, y * 0.005 * turb, flowTime * 0.3) * Math.PI * 2;
            var len = 10 + turb * 6;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + Math.cos(angle) * len, y + Math.sin(angle) * len);
            ctx.stroke();
        }
    }
    ctx.restore();
}

function drawTrail(agent) {
    var trail = agent.trail;
    if (trail.length < 2) return;
    for (var i = 1; i < trail.length; i++) {
        var alpha = (i / trail.length) * 0.35;
        ctx.beginPath();
        ctx.moveTo(trail[i-1].x, trail[i-1].y);
        ctx.lineTo(trail[i].x, trail[i].y);
        ctx.strokeStyle = "rgba(" + agent.rgbStr + "," + alpha + ")";
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}

function drawAgent(agent, state) {
    var radius = Math.max(3, Math.min(3 + Math.sqrt(Math.max(state.n_interactions, 0)) * 2.2, 14));

    ctx.save();
    if (state.is_frozen) ctx.globalAlpha = 0.4;

    // Glow
    ctx.shadowColor = agent.color;
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.arc(agent.x, agent.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = agent.color;
    ctx.fill();
    ctx.shadowBlur = 0;

    // Inner bright core
    ctx.beginPath();
    ctx.arc(agent.x, agent.y, radius * 0.45, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.fill();

    // Frozen ring
    if (state.is_frozen) {
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = "#88ccff";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(agent.x, agent.y, radius + 3, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    ctx.restore();
}

function drawHUD(eco) {
    ctx.save();
    ctx.fillStyle = "#666";
    ctx.font = "11px monospace";
    ctx.textAlign = "right";
    var lines = [
        "Toxicity: " + ((eco.toxicity_rate || 0) * 100).toFixed(1) + "%",
        "Threat: " + ((eco.ecosystem_threat_level || 0) * 100).toFixed(1) + "%",
        "Welfare: " + (eco.total_welfare || 0).toFixed(1)
    ];
    for (var i = 0; i < lines.length; i++) {
        ctx.fillText(lines[i], W - margin.right, margin.top + 15 + i * 16);
    }
    ctx.restore();
}

// === Tooltip ===
canvas.addEventListener("mousemove", function(e) {
    var rect = canvas.getBoundingClientRect();
    var mx = (e.clientX - rect.left) * (W / rect.width);
    var my = (e.clientY - rect.top) * (H / rect.height);

    var closest = null;
    var closestDist = 22;

    for (var i = 0; i < agents.length; i++) {
        var dx = agents[i].x - mx;
        var dy = agents[i].y - my;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < closestDist) {
            closestDist = dist;
            closest = agents[i];
        }
    }

    if (closest) {
        var state = getAgentState(closest, currentEpoch);
        tooltip.style.display = "block";
        var tx = e.clientX - rect.left + 15;
        var ty = e.clientY - rect.top - 10;
        if (tx + 200 > rect.width) tx = e.clientX - rect.left - 210;
        tooltip.style.left = tx + "px";
        tooltip.style.top = ty + "px";
        tooltip.innerHTML =
            "<b>" + closest.id + "</b> (" + closest.type + ")<br>" +
            "Rep: " + state.reputation.toFixed(3) + "<br>" +
            "Payoff: " + state.cumulative_payoff.toFixed(3) + "<br>" +
            "Avg p: " + state.avg_p.toFixed(3) + "<br>" +
            "Interactions: " + Math.round(state.n_interactions);
    } else {
        tooltip.style.display = "none";
    }
});

canvas.addEventListener("mouseleave", function() {
    tooltip.style.display = "none";
});

// === Build legend ===
var legendEl = document.getElementById("legend");
var typesPresent = {};
for (var i = 0; i < agents.length; i++) typesPresent[agents[i].type] = true;
var typeOrder = ["honest", "opportunistic", "deceptive", "adversarial", "rlm"];
for (var i = 0; i < typeOrder.length; i++) {
    var t = typeOrder[i];
    if (!typesPresent[t]) continue;
    var item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = '<span class="legend-dot" style="background:' + (COLORS[t] || DEFAULT_COLOR) + '"></span>' + t;
    legendEl.appendChild(item);
}

// === Animation loop ===
function animate(ts) {
    if (lastTS === null) lastTS = ts;
    var dt = Math.min((ts - lastTS) / 1000, 0.1);  // clamp to avoid jumps
    lastTS = ts;

    if (playing) {
        var speedFactor = parseFloat(speedSlider.value) / 5;
        currentEpoch += dt * baseSpeed * speedFactor;
        if (currentEpoch >= maxEpoch) {
            currentEpoch = maxEpoch;
            playing = false;
            playBtn.innerHTML = "&#9654;";
        }
        epochSlider.value = currentEpoch;
    }

    flowTime += dt;

    epochDisplay.textContent = "Epoch " + Math.floor(currentEpoch) + " / " + maxEpoch;

    var eco = getEcoState(currentEpoch);
    var turb = 0.3 + (eco.toxicity_rate || 0) * 0.7;
    var threatDrift = (eco.ecosystem_threat_level || 0) * 0.5;

    // Clear & draw background
    drawBackground(eco);
    drawGrid();
    drawFlowField(eco);

    // Update agents
    for (var i = 0; i < agents.length; i++) {
        var agent = agents[i];
        var state = getAgentState(agent, currentEpoch);
        if (!state) continue;

        var pos = dataToCanvas(state.reputation, state.cumulative_payoff);

        // Flow field perturbation
        var angle = Perlin.noise(pos.x * 0.008, pos.y * 0.008, flowTime * 0.4) * Math.PI * 2;
        var pertStr = 2 + turb * 5;
        agent.x = pos.x + Math.cos(angle) * pertStr;
        agent.y = pos.y + Math.sin(angle) * pertStr + threatDrift * 2;

        // Clamp to plot area
        agent.x = Math.max(margin.left, Math.min(W - margin.right, agent.x));
        agent.y = Math.max(margin.top, Math.min(H - margin.bottom, agent.y));

        // Trail
        agent.trail.push({x: agent.x, y: agent.y});
        if (agent.trail.length > 45) agent.trail.shift();

        drawTrail(agent);
        drawAgent(agent, state);
    }

    drawAxes();
    drawHUD(eco);

    requestAnimationFrame(animate);
}

// Start
requestAnimationFrame(animate);

})();
</script>
</body>
</html>
"""
