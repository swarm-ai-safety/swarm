---
title: Knowledge Graph
description: "Interactive map of the SWARM knowledge base. Traverse pages by their links, filter by section, search, and find the shortest link path between any two concepts."
hide:
  - toc
---

# Knowledge Graph

Every page in these docs is a node; every link between pages is an edge. Drag to
pan, scroll to zoom, **hover** a node to spotlight its neighborhood, and **click**
to open the page. Dashed nodes are *orphans* — nothing links to them yet.

<div class="kb-controls" markdown="0">
  <input id="kb-search" type="search" placeholder="Search pages…" autocomplete="off" />
  <div class="kb-pathfinder">
    <span class="kb-pf-label">Speedrun:</span>
    <select id="kb-from" aria-label="From page"></select>
    <span class="kb-arrow">→</span>
    <select id="kb-to" aria-label="To page"></select>
    <button id="kb-find-path">Find shortest path</button>
  </div>
  <div id="kb-path-result" class="kb-path-result"></div>
  <div id="kb-legend" class="kb-legend"></div>
</div>

<div id="kb-graph" class="kb-graph-canvas"></div>

<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<script src="../javascripts/kb_graph.js"></script>

> The graph is regenerated on every docs build from the live markdown corpus.
> To rebuild it locally and see which pages are weakly linked, run
> `python scripts/build_kb_graph.py`.
