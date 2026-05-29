/* Knowledge-graph viewer for the SWARM docs.
 * Loads assets/kb_graph.json, renders a force-directed graph (Cytoscape),
 * supports click-to-navigate, section filtering, node search, and a BFS
 * "speedrun" shortest-path finder between any two pages.
 */
(function () {
  "use strict";

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  function baseUrl() {
    // graph.md renders at <site>/graph/ ; assets live one level up.
    var marker = "/graph/";
    var p = window.location.pathname;
    var i = p.indexOf(marker);
    return i >= 0 ? p.slice(0, i + 1) : "/";
  }

  var SECTION_COLORS = {};
  var PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
                 "#0891b2", "#db2777", "#65a30d", "#9333ea", "#ea580c",
                 "#0d9488", "#c026d3", "#4f46e5", "#16a34a", "#b45309"];

  function colorFor(section, idx) {
    if (!(section in SECTION_COLORS)) {
      SECTION_COLORS[section] = PALETTE[Object.keys(SECTION_COLORS).length % PALETTE.length];
    }
    return SECTION_COLORS[section];
  }

  function bfsPath(adj, start, goal) {
    if (start === goal) return [start];
    var queue = [start];
    var prev = {}; prev[start] = null;
    while (queue.length) {
      var cur = queue.shift();
      var nbrs = adj[cur] || [];
      for (var k = 0; k < nbrs.length; k++) {
        var n = nbrs[k];
        if (!(n in prev)) {
          prev[n] = cur;
          if (n === goal) {
            var path = [n]; var c = cur;
            while (c !== null) { path.unshift(c); c = prev[c]; }
            return path;
          }
          queue.push(n);
        }
      }
    }
    return null;
  }

  ready(function () {
    var container = document.getElementById("kb-graph");
    if (!container || typeof cytoscape === "undefined") return;
    var base = baseUrl();

    fetch(base + "assets/kb_graph.json")
      .then(function (r) { return r.json(); })
      .then(function (graph) { init(graph, base); })
      .catch(function (e) {
        container.innerHTML = '<p style="padding:1rem">Could not load graph data: ' + e + "</p>";
      });

    function init(graph, base) {
      // adjacency for pathfinding (directed, following links forward)
      var adj = {};
      var byId = {};
      graph.nodes.forEach(function (n) { adj[n.id] = []; byId[n.id] = n; });
      graph.edges.forEach(function (e) { adj[e.source].push(e.target); });

      var elements = [];
      graph.nodes.forEach(function (n) {
        elements.push({ data: {
          id: n.id, label: n.title, section: n.section,
          color: colorFor(n.section), deg: (n.indegree + n.outdegree),
          url: base + n.url, orphan: n.orphan
        }});
      });
      graph.edges.forEach(function (e) {
        elements.push({ data: { id: e.source + "->" + e.target,
          source: e.source, target: e.target, kind: e.kind }});
      });

      var cy = cytoscape({
        container: container,
        elements: elements,
        wheelSensitivity: 0.2,
        style: [
          { selector: "node", style: {
            "background-color": "data(color)",
            "label": "data(label)",
            "font-size": 6,
            "width": "mapData(deg, 0, 30, 8, 42)",
            "height": "mapData(deg, 0, 30, 8, 42)",
            "text-wrap": "wrap", "text-max-width": 70,
            "color": "#334155", "min-zoomed-font-size": 7,
            "border-width": 0
          }},
          { selector: "node[?orphan]", style: { "border-width": 1.5, "border-color": "#94a3b8", "border-style": "dashed" }},
          { selector: "edge", style: {
            "width": 0.6, "line-color": "#cbd5e1", "target-arrow-color": "#cbd5e1",
            "target-arrow-shape": "triangle", "arrow-scale": 0.5,
            "curve-style": "bezier", "opacity": 0.5
          }},
          { selector: ".faded", style: { "opacity": 0.08 }},
          { selector: ".highlight", style: { "opacity": 1 }},
          { selector: "node.path", style: { "border-width": 3, "border-color": "#111827", "z-index": 99 }},
          { selector: "edge.path", style: { "width": 3, "line-color": "#111827", "target-arrow-color": "#111827", "opacity": 1, "z-index": 99 }}
        ],
        layout: { name: "cose", animate: false, nodeRepulsion: 6000, idealEdgeLength: 60, padding: 30 }
      });

      // click a node -> open its page; hover -> highlight neighborhood
      cy.on("tap", "node", function (evt) { window.location.href = evt.target.data("url"); });
      cy.on("mouseover", "node", function (evt) {
        var n = evt.target;
        cy.elements().addClass("faded");
        n.closedNeighborhood().removeClass("faded").addClass("highlight");
      });
      cy.on("mouseout", "node", function () {
        cy.elements().removeClass("faded").removeClass("highlight");
      });

      // ---- section filter legend ----
      var legend = document.getElementById("kb-legend");
      var sections = Object.keys(SECTION_COLORS).sort();
      var hidden = {};
      sections.forEach(function (s) {
        var chip = document.createElement("button");
        chip.className = "kb-chip";
        chip.style.borderColor = SECTION_COLORS[s];
        chip.innerHTML = '<span style="background:' + SECTION_COLORS[s] + '"></span>' + s;
        chip.onclick = function () {
          hidden[s] = !hidden[s];
          chip.classList.toggle("off", hidden[s]);
          cy.nodes('[section = "' + s + '"]').style("display", hidden[s] ? "none" : "element");
        };
        legend.appendChild(chip);
      });

      // ---- search box ----
      var search = document.getElementById("kb-search");
      search.addEventListener("input", function () {
        var q = search.value.trim().toLowerCase();
        if (!q) { cy.elements().removeClass("faded").removeClass("highlight"); return; }
        var matches = cy.nodes().filter(function (n) {
          return n.data("label").toLowerCase().indexOf(q) >= 0 || n.id().toLowerCase().indexOf(q) >= 0;
        });
        cy.elements().addClass("faded");
        matches.removeClass("faded").addClass("highlight");
        if (matches.length) cy.animate({ fit: { eles: matches, padding: 80 } }, { duration: 300 });
      });

      // ---- BFS "speedrun" pathfinder ----
      var fromSel = document.getElementById("kb-from");
      var toSel = document.getElementById("kb-to");
      var pathOut = document.getElementById("kb-path-result");
      graph.nodes.slice().sort(function (a, b) {
        return a.title.localeCompare(b.title);
      }).forEach(function (n) {
        [fromSel, toSel].forEach(function (sel) {
          var o = document.createElement("option");
          o.value = n.id; o.textContent = n.title + "  (" + n.section + ")";
          sel.appendChild(o);
        });
      });

      document.getElementById("kb-find-path").onclick = function () {
        cy.elements().removeClass("path");
        var a = fromSel.value, b = toSel.value;
        var path = bfsPath(adj, a, b);
        if (!path) {
          pathOut.innerHTML = '<span class="kb-nopath">No link path from <b>' +
            byId[a].title + "</b> to <b>" + byId[b].title +
            "</b> (try the reverse, or densify links).</span>";
          return;
        }
        for (var i = 0; i < path.length; i++) {
          cy.getElementById(path[i]).addClass("path");
          if (i > 0) cy.getElementById(path[i - 1] + "->" + path[i]).addClass("path");
        }
        cy.animate({ fit: { eles: cy.elements(".path"), padding: 80 } }, { duration: 400 });
        var hops = path.map(function (id) {
          return '<a href="' + base + byId[id].url + '">' + byId[id].title + "</a>";
        }).join(' <span class="kb-arrow">→</span> ');
        pathOut.innerHTML = "<b>" + (path.length - 1) + " hop" +
          (path.length === 2 ? "" : "s") + ":</b> " + hops;
      };
    }
  });
})();
