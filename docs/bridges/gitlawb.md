---
description: "Live monitoring of Gitlawb decentralized git agent interactions using SWARM safety metrics."
---

# SWARM–Gitlawb Live Monitor

Real-time safety metrics for AI agent interactions on the [Gitlawb](https://gitlawb.com) decentralized git network.

<div id="gitlawb-monitor" style="margin-top: 1.5em;">

<div id="gl-status" style="padding: 0.8em 1em; border-radius: 8px; background: var(--md-code-bg-color, #1e1e2e); margin-bottom: 1em; font-family: var(--md-code-font, 'JetBrains Mono', monospace); font-size: 0.85em;">
  <span id="gl-conn-status" style="color: #f38ba8;">Connecting...</span>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.8em; margin-bottom: 1.5em;">
  <div class="gl-metric-card">
    <div class="gl-metric-label">Interactions</div>
    <div class="gl-metric-value" id="gl-count">0</div>
  </div>
  <div class="gl-metric-card">
    <div class="gl-metric-label">Toxicity Rate</div>
    <div class="gl-metric-value" id="gl-toxicity">--</div>
  </div>
  <div class="gl-metric-card">
    <div class="gl-metric-label">Avg Quality</div>
    <div class="gl-metric-value" id="gl-quality">--</div>
  </div>
  <div class="gl-metric-card">
    <div class="gl-metric-label">Repos Active</div>
    <div class="gl-metric-value" id="gl-repos">0</div>
  </div>
</div>

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1em; margin-bottom: 1.5em;">
  <div>
    <h4 style="margin-bottom: 0.5em; font-size: 0.9em; opacity: 0.7;">Quality Distribution</h4>
    <canvas id="gl-quality-chart" height="160"></canvas>
  </div>
  <div>
    <h4 style="margin-bottom: 0.5em; font-size: 0.9em; opacity: 0.7;">Events Over Time</h4>
    <canvas id="gl-timeline-chart" height="160"></canvas>
  </div>
</div>

<h4 style="margin-bottom: 0.5em; font-size: 0.9em; opacity: 0.7;">Live Event Feed</h4>
<div id="gl-feed" style="max-height: 360px; overflow-y: auto; font-family: var(--md-code-font, 'JetBrains Mono', monospace); font-size: 0.8em; background: var(--md-code-bg-color, #1e1e2e); border-radius: 8px; padding: 0.8em;">
  <div style="opacity: 0.5;">Waiting for events...</div>
</div>

</div>

<style>
.gl-metric-card {
  background: var(--md-code-bg-color, #1e1e2e);
  border-radius: 8px;
  padding: 1em;
  text-align: center;
}
.gl-metric-label {
  font-size: 0.75em;
  opacity: 0.6;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.3em;
}
.gl-metric-value {
  font-size: 1.6em;
  font-weight: 700;
  font-family: var(--md-code-font, 'JetBrains Mono', monospace);
  color: var(--md-accent-fg-color, #89b4fa);
}
.gl-event {
  padding: 0.4em 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  display: flex;
  gap: 0.8em;
  align-items: baseline;
}
.gl-event-time { opacity: 0.4; min-width: 6em; }
.gl-event-type { font-weight: 600; min-width: 5em; }
.gl-event-type.push { color: #a6e3a1; }
.gl-event-type.task { color: #89b4fa; }
.gl-event-type.delegation { color: #f9e2af; }
.gl-event-detail { opacity: 0.8; }
.gl-event-quality { margin-left: auto; }
.gl-q-high { color: #a6e3a1; }
.gl-q-mid { color: #f9e2af; }
.gl-q-low { color: #f38ba8; }
</style>

<script>
(function() {
  const WS_URL = "wss://node.gitlawb.com/graphql/ws";
  const HTTP_URL = "https://node.gitlawb.com/graphql";
  const MAX_FEED = 100;
  const MAX_TIMELINE = 50;

  const state = {
    interactions: [],
    repos: new Set(),
    timeline: [],
    qualityBuckets: [0, 0, 0, 0, 0], // 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    ws: null,
    connected: false,
  };

  // --- DOM refs ---
  const $conn = document.getElementById("gl-conn-status");
  const $count = document.getElementById("gl-count");
  const $toxicity = document.getElementById("gl-toxicity");
  const $quality = document.getElementById("gl-quality");
  const $repos = document.getElementById("gl-repos");
  const $feed = document.getElementById("gl-feed");

  // --- Chart setup (lightweight inline bar charts) ---
  function drawBarChart(canvasId, data, labels, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = canvas.offsetHeight * 2;
    ctx.clearRect(0, 0, w, h);

    const max = Math.max(...data, 1);
    const barW = w / data.length * 0.7;
    const gap = w / data.length * 0.3;

    data.forEach((v, i) => {
      const barH = (v / max) * (h - 30);
      const x = i * (barW + gap) + gap / 2;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.8;
      ctx.fillRect(x, h - 20 - barH, barW, barH);
      ctx.globalAlpha = 0.5;
      ctx.font = "18px JetBrains Mono, monospace";
      ctx.fillStyle = "#cdd6f4";
      ctx.textAlign = "center";
      ctx.fillText(labels[i], x + barW / 2, h - 4);
      if (v > 0) {
        ctx.globalAlpha = 0.9;
        ctx.fillText(v, x + barW / 2, h - 24 - barH);
      }
    });
  }

  function updateCharts() {
    drawBarChart("gl-quality-chart", state.qualityBuckets,
      ["0-20", "20-40", "40-60", "60-80", "80-100"], "#89b4fa");

    const tl = state.timeline.slice(-20);
    if (tl.length > 1) {
      const counts = {};
      tl.forEach(t => { counts[t] = (counts[t] || 0) + 1; });
      const labels = Object.keys(counts);
      const vals = Object.values(counts);
      drawBarChart("gl-timeline-chart", vals, labels.map(l => l.slice(11, 16)), "#a6e3a1");
    }
  }

  // --- Heuristic quality score ---
  function scoreEvent(event, type) {
    if (type === "push") {
      let base = 0.7;
      const ref = event.refName || "";
      if (/temp|test|wip/i.test(ref)) base -= 0.15;
      const old = event.oldSha || "";
      if (old && /^0+$/.test(old)) base -= 0.2;
      return Math.max(0.1, Math.min(0.95, base));
    }
    if (type === "task") {
      const s = (event.newStatus || "").toLowerCase();
      if (s === "completed") return 0.8;
      if (s === "failed") return 0.2;
      if (s === "claimed") return 0.5;
      return 0.5;
    }
    return 0.6;
  }

  function qualityClass(p) {
    if (p >= 0.6) return "gl-q-high";
    if (p >= 0.4) return "gl-q-mid";
    return "gl-q-low";
  }

  // --- Add interaction ---
  function addInteraction(event, type) {
    const p = scoreEvent(event, type);
    const now = new Date().toISOString();
    state.interactions.push({ event, type, p, ts: now });
    if (state.interactions.length > 2000) state.interactions.shift();

    // Update quality buckets
    const bucket = Math.min(4, Math.floor(p * 5));
    state.qualityBuckets[bucket]++;

    // Update timeline
    state.timeline.push(now.slice(0, 16));

    // Track repos
    if (event.repo) state.repos.add(event.repo);

    // Update metrics
    const n = state.interactions.length;
    const toxicity = state.interactions.filter(i => i.p < 0.3).length / n;
    const avgQ = state.interactions.reduce((s, i) => s + i.p, 0) / n;

    $count.textContent = n;
    $toxicity.textContent = (toxicity * 100).toFixed(1) + "%";
    $toxicity.style.color = toxicity > 0.3 ? "#f38ba8" : toxicity > 0.15 ? "#f9e2af" : "#a6e3a1";
    $quality.textContent = avgQ.toFixed(2);
    $quality.style.color = qualityClass(avgQ);
    $repos.textContent = state.repos.size;

    // Add to feed
    const feedFirst = $feed.querySelector("div[style]");
    if (feedFirst && feedFirst.textContent.includes("Waiting")) $feed.innerHTML = "";

    const el = document.createElement("div");
    el.className = "gl-event";
    const time = now.slice(11, 19);
    let detail = "";
    if (type === "push") {
      detail = `${shortDid(event.pusherDid)} pushed to ${event.repo}:${(event.refName || "").replace("refs/heads/", "")}`;
    } else if (type === "task") {
      detail = `Task ${shortId(event.taskId)}: ${event.oldStatus} -> ${event.newStatus}`;
    }
    el.innerHTML = `
      <span class="gl-event-time">${time}</span>
      <span class="gl-event-type ${type === "push" ? "push" : type === "task" ? "task" : "delegation"}">${type}</span>
      <span class="gl-event-detail">${detail}</span>
      <span class="gl-event-quality ${qualityClass(p)}">${p.toFixed(2)}</span>
    `;
    $feed.prepend(el);
    while ($feed.children.length > MAX_FEED) $feed.lastChild.remove();

    updateCharts();
  }

  function shortDid(did) {
    if (!did) return "anon";
    const key = did.replace("did:key:", "");
    return key.slice(0, 8) + "...";
  }
  function shortId(id) {
    if (!id) return "?";
    return id.slice(0, 8);
  }

  // --- Backfill recent data via HTTP ---
  async function backfill() {
    try {
      const res = await fetch(HTTP_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: `query { refUpdates(limit: 20) { repo refName oldSha newSha pusherDid nodeDid timestamp } tasks(limit: 20) { id status delegatorDid assigneeDid createdAt } }`
        })
      });
      const json = await res.json();
      const data = json.data || {};
      (data.refUpdates || []).forEach(e => addInteraction(e, "push"));
      (data.tasks || []).forEach(e => addInteraction(e, "task"));
    } catch (e) {
      console.warn("Backfill failed:", e);
    }
  }

  // --- WebSocket subscriptions ---
  function connect() {
    $conn.textContent = "Connecting...";
    $conn.style.color = "#f38ba8";

    const ws = new WebSocket(WS_URL, "graphql-ws");
    state.ws = ws;

    ws.onopen = () => {
      state.connected = true;
      $conn.textContent = "Connected to Gitlawb";
      $conn.style.color = "#a6e3a1";

      // Send connection init
      ws.send(JSON.stringify({ type: "connection_init" }));
    };

    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data);

      if (data.type === "connection_ack") {
        // Subscribe to ref updates
        ws.send(JSON.stringify({
          id: "sub-refs",
          type: "subscribe",
          payload: {
            query: `subscription { refUpdates { repo refName oldSha newSha pusherDid nodeDid timestamp } }`
          }
        }));
        // Subscribe to task events
        ws.send(JSON.stringify({
          id: "sub-tasks",
          type: "subscribe",
          payload: {
            query: `subscription { taskEvents { taskId oldStatus newStatus byDid at } }`
          }
        }));
      }

      if ((data.type === "next" || data.type === "data") && data.payload && data.payload.data) {
        const d = data.payload.data;
        if (d.refUpdates) addInteraction(d.refUpdates, "push");
        if (d.taskEvents) addInteraction(d.taskEvents, "task");
      }
    };

    ws.onerror = () => {
      $conn.textContent = "Connection error";
      $conn.style.color = "#f38ba8";
    };

    ws.onclose = () => {
      state.connected = false;
      $conn.textContent = "Disconnected — reconnecting...";
      $conn.style.color = "#f9e2af";
      setTimeout(connect, 5000);
    };
  }

  // --- Init ---
  backfill().then(() => connect());
})();
</script>

---

## About This Monitor

This dashboard connects directly to the [Gitlawb](https://gitlawb.com) node via WebSocket and subscribes to two real-time event streams:

- **refUpdates** — every git push to any repo on the network
- **taskEvents** — every agent task status change (claimed, completed, failed)

Each event is scored with a heuristic quality probability (`p`) derived from SWARM's soft-label framework. The metrics shown are simplified versions of the full `SoftMetrics` computation available via the Python bridge.

### Metrics

| Metric | Description |
|--------|-------------|
| **Interactions** | Total events observed this session |
| **Toxicity Rate** | Fraction of interactions with quality < 0.3 |
| **Avg Quality** | Mean quality probability across all interactions |
| **Repos Active** | Number of distinct repos with activity |

### Running the Full Bridge

For production monitoring with LLM judge scoring, persistence, and full SWARM metrics:

```bash
pip install swarm[gitlawb]

# One-shot analysis
python -m swarm.bridges.gitlawb --mode oneshot --repos swarm --limit 100

# Continuous daemon
python -m swarm.bridges.gitlawb --mode daemon --repos swarm
```

### Source

- [Bridge code](https://github.com/swarm-ai-safety/swarm/tree/main/swarm/bridges/gitlawb)
- [Gitlawb network](https://gitlawb.com)
