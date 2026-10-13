import json
import html
import math
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import networkx as nx
import numpy as np
from pyvis.network import Network

from src.topology.schemas import (
    NodeMetrics,
    CommunityPartition,
    StructuralCluster,
    ThemeInheritance,
    TopologyResult,
    HubPartition
)
from src.topology.resolver import TargetResolver


class TopologyVisualizer:
    """
    Dedicated HTML canvas and interactive visualization engine for Stage 3 Topology.
    Handles ForceAtlas2 physics stabilization, responsive UI panels, node inspectors,
    and co-located output export across Stage 3.1, 3.2, and 3.3.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.full_config = config or {}
        self.config = self.full_config.get('topology', {})

    def generate_stage_3_1_overview_visuals(self, *args, **kwargs):
        return self._generate_stage_3_1_overview_visuals(*args, **kwargs)

    def generate_stage_3_2_reduced_core_visuals(self, *args, **kwargs):
        return self._generate_stage_3_2_reduced_core_visuals(*args, **kwargs)

    def generate_stage_3_3_community_visuals(self, *args, **kwargs):
        return self._generate_stage_3_3_community_visuals(*args, **kwargs)

    # =========================================================================
    # VISUALIZATION ENGINE (4-TIER VISUAL ARCHITECTURE)
    # 01_overview: Operations on entire graph
    # 02_hubs_and_connectors: Reduced graph (hubs removed)
    # 03_community: Dedicated community path visuals
    # 03_embedding: Dedicated embedding path visuals
    # =========================================================================

    def _generate_stage_3_1_overview_visuals(
        self, dg: nx.DiGraph, ug: nx.Graph, global_hubs: List[str],
        node_metrics: Dict[str, NodeMetrics], raw_diagnostics: Dict[str, Any],
        vis_dir: Path, community_result: TopologyResult = None
    ):
        vis_dir.mkdir(parents=True, exist_ok=True)
        # 1. Complete Topology Graph (Reference hairball, 0% CPU, 2D precomputed layout)
        self._generate_complete_topology_hairball(
            dg=dg,
            result=community_result,
            output_path=vis_dir / "complete_topology_graph.html",
            path_label="Stage 3.1: Global Overview",
            path_type="community"
        )
        # 2. Participation Dispersion Map
        self._generate_participation_dispersion_map(
            dg=dg,
            node_metrics=node_metrics,
            global_hubs=global_hubs,
            output_path=vis_dir / "participation_dispersion_map.html"
        )
        # 3. Workflow Chokepoints (Betweenness Centrality Map)
        self._generate_workflow_chokepoints_map(
            dg=dg,
            node_metrics=node_metrics,
            output_path=vis_dir / "workflow_chokepoints.html"
        )

    def _get_shared_interactive_css(self) -> str:
        return """
        :root {
            --bg-body: #0d0e12;
            --bg-header: rgba(20, 22, 31, 0.95);
            --bg-card: #1c202d;
            --bg-card-glass: rgba(20, 22, 31, 0.92);
            --border: #262c3e;
            --border-hover: #3b82f6;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --text-dim: #64748b;
            --accent-blue: #38bdf8;
            --accent-green: #34d399;
            --accent-emerald: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #fbbf24;
            --accent-purple: #a855f7;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-body);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            transition: background-color 0.2s, color 0.2s;
        }
        header {
            background: var(--bg-header);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
            padding: 10px 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }
        .header-title h1 {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
            letter-spacing: -0.01em;
        }
        .header-sub {
            font-size: 11px;
            color: var(--text-sub);
            margin-top: 3px;
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }
        .stat-badge {
            background: rgba(56, 189, 248, 0.12);
            color: var(--accent-blue);
            font-weight: 600;
            padding: 1px 7px;
            border-radius: 4px;
            font-size: 10px;
            border: 1px solid rgba(56, 189, 248, 0.25);
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .stat-badge-success {
            background: rgba(52, 211, 153, 0.12);
            color: var(--accent-green);
            border-color: rgba(52, 211, 153, 0.25);
        }
        .stat-badge-danger {
            background: rgba(239, 68, 68, 0.12);
            color: var(--accent-red);
            border-color: rgba(239, 68, 68, 0.25);
        }
        .stat-badge-warning {
            background: rgba(251, 191, 36, 0.12);
            color: var(--accent-amber);
            border-color: rgba(251, 191, 36, 0.25);
        }
        .stat-badge-blue {
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border-color: rgba(56, 189, 248, 0.3);
        }
        .header-actions {
            display: flex;
            gap: 7px;
            align-items: center;
            flex-wrap: wrap;
        }
        .btn {
            background: #1e2230;
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 5px 11px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            user-select: none;
        }
        .btn:hover {
            background: #282f44;
            border-color: var(--border-hover);
        }
        .btn-primary {
            background: #2563eb;
            border-color: #3b82f6;
            color: #ffffff;
        }
        .btn-primary:hover {
            background: #1d4ed8;
        }
        .btn-guide {
            background: rgba(168, 85, 247, 0.18);
            border-color: rgba(168, 85, 247, 0.4);
            color: #d8b4fe;
        }
        .btn-guide:hover {
            background: rgba(168, 85, 247, 0.3);
            border-color: #c084fc;
            color: #ffffff;
        }
        .btn-paper {
            background: #334155;
            border-color: #475569;
        }
        .search-box {
            position: relative;
        }
        .search-input {
            background: #181c28;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 5px 10px;
            border-radius: 6px;
            font-size: 11px;
            width: 175px;
            outline: none;
            transition: width 0.2s, border-color 0.2s;
        }
        .search-input:focus {
            width: 230px;
            border-color: #3b82f6;
        }
        .filter-select {
            background: #181c28;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 5px 9px;
            border-radius: 6px;
            font-size: 11px;
            outline: none;
            cursor: pointer;
        }
        .canvas-container {
            flex: 1;
            position: relative;
            width: 100%;
            height: calc(100vh - 58px);
        }
        #network {
            width: 100%;
            height: 100%;
        }
        #loadingBar {
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(13, 14, 18, 0.85);
            backdrop-filter: blur(6px);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 20;
            transition: opacity 0.4s ease, visibility 0.4s ease;
        }
        #loadingBar.hidden {
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
        }
        .loader-card {
            background: #1a1e2b;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px 28px;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
            width: 320px;
        }
        .loader-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 12px;
        }
        .progress-track {
            width: 100%;
            height: 8px;
            background: #252b3d;
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-fill {
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            border-radius: 4px;
            transition: width 0.15s ease;
        }
        .progress-label {
            margin-top: 10px;
            font-size: 11px;
            color: var(--text-sub);
        }
        .legend-card {
            position: absolute;
            top: 14px;
            right: 14px;
            background: var(--bg-card-glass);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 11px;
            color: var(--text-sub);
            z-index: 5;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
            max-width: 330px;
        }
        .legend-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .legend-title {
            font-weight: 700;
            text-transform: uppercase;
            font-size: 10px;
            color: var(--accent-blue);
            letter-spacing: 0.05em;
        }
        .legend-section {
            margin-top: 8px;
        }
        .legend-section-title {
            font-size: 9px;
            text-transform: uppercase;
            color: var(--text-dim);
            font-weight: 700;
            margin-bottom: 4px;
            letter-spacing: 0.03em;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 7px;
            margin-bottom: 5px;
            line-height: 1.3;
        }
        .legend-icon {
            width: 14px;
            height: 14px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            flex-shrink: 0;
        }
        .legend-line {
            width: 16px;
            height: 2px;
            border-radius: 1px;
            flex-shrink: 0;
        }
        .inspector-card {
            position: absolute;
            bottom: 14px;
            left: 14px;
            background: rgba(20, 22, 31, 0.94);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 16px;
            font-size: 11px;
            color: var(--text-sub);
            z-index: 5;
            box-shadow: 0 8px 28px rgba(0,0,0,0.6);
            max-width: 380px;
            max-height: 280px;
            overflow-y: auto;
            display: none;
        }
        .inspector-card.visible { display: block; }
        .inspector-close {
            position: absolute;
            top: 8px;
            right: 10px;
            background: transparent;
            border: none;
            color: var(--text-dim);
            cursor: pointer;
            font-size: 16px;
            line-height: 1;
        }
        .inspector-close:hover { color: var(--text-main); }
        .inspector-name {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 8px;
            padding-right: 18px;
            word-break: break-word;
        }
        .metric-pill {
            display: inline-block;
            background: #252b3d;
            color: #94a3b8;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 10px;
            margin-right: 4px;
            margin-bottom: 4px;
        }
        .metric-pill b { color: #f8fafc; }
        .guide-drawer {
            position: absolute;
            top: 0; right: 0;
            width: 400px;
            height: 100%;
            background: rgba(18, 20, 29, 0.97);
            backdrop-filter: blur(14px);
            border-left: 1px solid var(--border);
            z-index: 15;
            display: flex;
            flex-direction: column;
            box-shadow: -10px 0 30px rgba(0,0,0,0.6);
            transform: translateX(100%);
            transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .guide-drawer.open {
            transform: translateX(0);
        }
        .guide-header {
            padding: 14px 18px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .guide-header h2 {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .guide-close {
            background: transparent;
            border: none;
            color: var(--text-dim);
            font-size: 18px;
            cursor: pointer;
        }
        .guide-close:hover { color: var(--text-main); }
        .guide-content {
            padding: 16px 18px;
            overflow-y: auto;
            flex: 1;
            font-size: 12px;
            line-height: 1.5;
            color: var(--text-sub);
        }
        .guide-card {
            background: #171b26;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 14px;
        }
        .guide-card-title {
            font-size: 12px;
            font-weight: 700;
            color: var(--accent-blue);
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .formula-box {
            background: #0f121a;
            border: 1px solid #23293d;
            border-radius: 4px;
            padding: 6px 10px;
            font-family: monospace;
            font-size: 11px;
            color: #38bdf8;
            margin: 6px 0;
            display: block;
        }
        .takeaway-list {
            margin-left: 16px;
            margin-top: 6px;
        }
        .takeaway-list li {
            margin-bottom: 6px;
        }
        body.paper-white {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        body.paper-white header {
            background: #f8fafc !important;
            border-bottom: 1px solid #cbd5e1 !important;
            color: #0f172a !important;
        }
        body.paper-white .header-title h1 { color: #0f172a !important; }
        body.paper-white .header-sub { color: #475569 !important; }
        body.paper-white .btn { background: #f1f5f9; color: #0f172a; border-color: #cbd5e1; }
        body.paper-white .btn-primary { background: #2563eb; color: #ffffff; }
        body.paper-white .search-input { background: #f8fafc; border-color: #cbd5e1; color: #0f172a; }
        body.paper-white .filter-select { background: #f8fafc; border-color: #cbd5e1; color: #0f172a; }
        body.paper-white .legend-card { background: rgba(255, 255, 255, 0.94) !important; border-color: #cbd5e1 !important; color: #334155 !important; }
        body.paper-white .legend-title { color: #0284c7 !important; }
        body.paper-white .inspector-card { background: rgba(255, 255, 255, 0.96) !important; border-color: #cbd5e1 !important; color: #334155 !important; }
        body.paper-white .inspector-name { color: #0f172a !important; }
        body.paper-white .metric-pill { background: #e2e8f0 !important; color: #475569 !important; }
        body.paper-white .metric-pill b { color: #0f172a !important; }
        body.paper-white .guide-drawer { background: rgba(255, 255, 255, 0.98) !important; border-color: #cbd5e1 !important; color: #334155 !important; }
        body.paper-white .guide-card { background: #f8fafc !important; border-color: #e2e8f0 !important; }
        body.paper-white .guide-header { border-color: #cbd5e1 !important; }
        body.paper-white .guide-header h2 { color: #0f172a !important; }
        body.paper-white .formula-box { background: #f1f5f9 !important; border-color: #cbd5e1 !important; color: #0369a1 !important; }
        """

    def _generate_participation_dispersion_map(self, dg: nx.DiGraph, node_metrics: Dict[str, NodeMetrics], global_hubs: List[str], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hubs_set = set(global_hubs)
        css = self._get_shared_interactive_css()

        enriched_nodes = []
        total_nodes = 0
        hubs_count = 0
        high_disp_count = 0
        provincial_count = 0
        max_p = 0.0

        for node in dg.nodes():
            metrics = node_metrics.get(node)
            p_val = metrics.participation_coefficient if metrics else 0.0
            if p_val > max_p:
                max_p = p_val
            is_hub = (node in hubs_set)
            if is_hub:
                hubs_count += 1
            if p_val >= 0.70:
                high_disp_count += 1
            if p_val == 0.0:
                provincial_count += 1
            total_nodes += 1

            if is_hub:
                role_desc = "Global Hub Router — Cross-community routing candidate connecting 3+ modules"
            elif p_val >= 0.70:
                role_desc = "Cross-Community Bridge — High dispersion; links uniformly into disparate clusters"
            elif p_val >= 0.30:
                role_desc = "Intermediate Connector — Mixed local and cross-community reach"
            else:
                role_desc = "Provincial Entity — Low dispersion; insulated within local thematic cluster"

            tooltip = (
                f"<b>{html.escape(str(node))}</b><br>"
                f"Participation Coefficient (P_i): {p_val:.4f}<br>"
                f"Global Hub: {'Yes (★)' if is_hub else 'No'}<br>"
                f"Role: {role_desc}"
            )

            color = f"hsl({int((1.0 - p_val) * 200)}, 80%, 50%)"
            size = 15 + (p_val * 40)
            deg = dg.degree(node)

            enriched_nodes.append({
                "id": str(node),
                "label": str(node) if (is_hub or deg >= 15) else "",
                "full_label": str(node),
                "shape": "star" if is_hub else "dot",
                "size": round(size, 1),
                "color": color,
                "borderWidth": 2.5 if is_hub else 1.5,
                "font": {"color": "#ffffff", "size": 13 if is_hub else 10, "face": "Inter, sans-serif"},
                "title": tooltip,
                "p_val": round(float(p_val), 4),
                "is_hub": is_hub,
                "role_desc": role_desc
            })

        enriched_edges = []
        for idx, (u, v, data) in enumerate(dg.edges(data=True)):
            enriched_edges.append({
                "id": f"e_{idx}",
                "from": str(u),
                "to": str(v),
                "arrows": "to",
                "title": data.get("predicate", ""),
                "color": {"color": "rgba(200, 200, 200, 0.3)", "highlight": "#38bdf8"},
                "width": 1
            })

        total_edges = len(enriched_edges)
        nodes_json = json.dumps(enriched_nodes)
        edges_json = json.dumps(enriched_edges)

        html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stage 3.1: Global Overview — Participation Dispersion Map (P_i)</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <style>
{css}
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <div>
                <h1>Stage 3.1: Global Overview — Participation Dispersion Map (P_i)</h1>
                <div class="header-sub">
                    <span>Topological Dispersion: Cross-Community Connectivity vs. Local Silos</span>
                    <span class="stat-badge">Entities: {total_nodes}</span>
                    <span class="stat-badge">Directed Edges: {total_edges}</span>
                    <span class="stat-badge stat-badge-warning">Global Hubs (★): {hubs_count}</span>
                    <span class="stat-badge stat-badge-success">High Dispersion (P_i ≥ 0.7): {high_disp_count}</span>
                    <span class="stat-badge">Provincial (P_i = 0): {provincial_count}</span>
                </div>
            </div>
        </div>
        <div class="header-actions">
            <div class="search-box">
                <input type="text" class="search-input" id="nodeSearch" placeholder="Search entity in map..." onkeydown="if(event.key==='Enter') searchNode(this.value)">
            </div>
            <select class="filter-select" id="dispFilter" onchange="filterDispersion(this.value)">
                <option value="all">All Entities ({total_nodes})</option>
                <option value="hubs">Global Hubs Only ({hubs_count})</option>
                <option value="high">High Dispersion P_i ≥ 0.70 ({high_disp_count})</option>
                <option value="provincial">Provincial P_i = 0 ({provincial_count})</option>
            </select>
            <button class="btn" id="labelBtn" onclick="toggleAllLabels()">🏷️ Show All Labels</button>
            <button class="btn btn-paper" id="paperBtn" onclick="togglePaperMode()">📄 Paper Mode (White)</button>
            <button class="btn btn-guide" onclick="toggleGuide()">ℹ️ Guide & Reference</button>
            <button class="btn btn-primary" onclick="downloadSnapshot()">📷 Snapshot</button>
            <button class="btn" onclick="network.fit({{animation: {{duration: 500}}}})">Fit</button>
        </div>
    </header>

    <div class="canvas-container">
        <div id="loadingBar">
            <div class="loader-card">
                <div class="loader-title">Stabilizing ForceAtlas2 Physics Layout...</div>
                <div class="progress-track">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-label" id="progressText">Stabilizing 0%</div>
            </div>
        </div>

        <div class="legend-card" id="legendCard">
            <div class="legend-header">
                <span class="legend-title">Topological Legend</span>
                <span style="font-size: 10px; color: var(--text-dim);">P_i Dispersion</span>
            </div>
            <div class="legend-section">
                <div class="legend-section-title">Node Role & Shape</div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: #fbbf24;">★</span>
                    <span><b>Global Hub</b> (P_i ≥ 0.80 & multi-community router)</span>
                </div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: #38bdf8;">●</span>
                    <span><b>Standard Entity</b></span>
                </div>
            </div>
            <div class="legend-section">
                <div class="legend-section-title">Dispersion Gradient (P_i)</div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: hsl(30, 80%, 50%);">●</span>
                    <span><b>High Dispersion (P_i ≥ 0.70)</b> — Cross-community bridge</span>
                </div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: hsl(110, 80%, 50%);">●</span>
                    <span><b>Moderate Dispersion (0.30–0.70)</b> — Mixed reach</span>
                </div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: hsl(200, 80%, 50%);">●</span>
                    <span><b>Low / Siloed (P_i &lt; 0.30)</b> — Local provincial node</span>
                </div>
            </div>
            <div class="legend-section">
                <div class="legend-section-title">Edges & Sizing</div>
                <div class="legend-item">
                    <div class="legend-line" style="background: rgba(200, 200, 200, 0.4);"></div>
                    <span><b>Directed Semantic Edge</b> (predicate link)</span>
                </div>
                <div class="legend-item" style="color: var(--text-dim); font-size: 10px; margin-top: 4px;">
                    Node diameter is proportional to P_i score (15px → 55px).
                </div>
            </div>
        </div>

        <div class="inspector-card" id="inspectorCard">
            <button class="inspector-close" onclick="closeInspector()">×</button>
            <div class="inspector-name" id="inspName">Entity</div>
            <div id="inspContent"></div>
        </div>

        <div class="guide-drawer" id="guideDrawer">
            <div class="guide-header">
                <h2>📖 Participation Dispersion Guide</h2>
                <button class="guide-close" onclick="toggleGuide()">×</button>
            </div>
            <div class="guide-content">
                <div class="guide-card">
                    <div class="guide-card-title">🎯 Visualization Objective</div>
                    <div>
                        Maps the extent to which each concept bridges across distinct thematic modules versus remaining confined within its local cluster. Used in Stage 3.1 to detect candidates for Stage 3.2 global hub separation.
                    </div>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">📐 Mathematical Definition</div>
                    <div>The <b>Participation Coefficient</b> (P_i) is defined as:</div>
                    <code class="formula-box">P_i = 1 - Σ (k_is / k_i)²  for s = 1...N_M</code>
                    <div style="font-size: 11px; color: var(--text-dim); margin-top: 4px;">
                        Where <i>k_is</i> is the number of links from node <i>i</i> to community <i>s</i>, and <i>k_i</i> is node <i>i</i>'s total degree.
                    </div>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">🔍 How to Interpret Patterns</div>
                    <ul class="takeaway-list">
                        <li><b>Warm Stars (P_i → 0.85+):</b> Global hubs in this graph. Their edges disperse evenly across 3+ communities, acting as structural linchpins.</li>
                        <li><b>Warm Red/Orange Dots:</b> High-dispersion bridge entities that create transitive paths between disparate knowledge domains.</li>
                        <li><b>Cool Cyan/Blue Dots (P_i = 0):</b> Provincial entities whose connections exist 100% within their home community.</li>
                        <li><b>Stage 3.2 Implication:</b> When global hubs are stripped in Stage 3.2, core graph modularity increases significantly, proving that high P_i hubs were obscuring the underlying modular topology.</li>
                    </ul>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">💡 Interactive Navigation</div>
                    <ul class="takeaway-list">
                        <li><b>Search:</b> Type any concept in the header box to zoom directly to its position.</li>
                        <li><b>Filter Dropdown:</b> Isolate Global Hubs or Provincial entities.</li>
                        <li><b>Node Click:</b> Click any node to open the Inspector and read its exact P_i score and role.</li>
                        <li><b>Paper Mode:</b> Switch to clean white mode for high-contrast viewing and print figures.</li>
                    </ul>
                </div>
            </div>
        </div>

        <div id="network"></div>
    </div>

    <script>
        const allNodes = {nodes_json};
        const allEdges = {edges_json};

        const nodesDataSet = new vis.DataSet(allNodes);
        const edgesDataSet = new vis.DataSet(allEdges);

        const container = document.getElementById('network');
        const data = {{ nodes: nodesDataSet, edges: edgesDataSet }};

        const options = {{
            physics: {{
                enabled: true,
                forceAtlas2Based: {{
                    gravitationalConstant: -80,
                    centralGravity: 0.005,
                    springLength: 200,
                    springConstant: 0.08
                }},
                solver: 'forceAtlas2Based',
                stabilization: {{ enabled: true, iterations: 150, fit: true }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 80,
                hideEdgesOnDrag: true,
                zoomView: true,
                dragView: true
            }}
        }};

        const network = new vis.Network(container, data, options);

        network.on('stabilizationProgress', function(params) {{
            const pct = Math.round((params.iterations / params.total) * 100);
            document.getElementById('progressFill').style.width = pct + '%';
            document.getElementById('progressText').innerText = 'Stabilizing Layout: ' + pct + '%';
        }});

        network.on('stabilizationIterationsDone', function() {{
            document.getElementById('loadingBar').classList.add('hidden');
        }});

        setTimeout(() => {{
            const bar = document.getElementById('loadingBar');
            if (bar && !bar.classList.contains('hidden')) {{
                bar.classList.add('hidden');
            }}
        }}, 2500);

        let labelsVisible = false;
        function toggleAllLabels() {{
            labelsVisible = !labelsVisible;
            const updates = allNodes.map(n => ({{
                id: n.id,
                label: labelsVisible ? n.full_label : (n.is_hub ? n.full_label : '')
            }}));
            nodesDataSet.update(updates);
            document.getElementById('labelBtn').innerText = labelsVisible ? '🏷️ Auto Labels' : '🏷️ Show All Labels';
        }}

        function filterDispersion(val) {{
            let filteredNodes;
            if (val === 'hubs') {{
                filteredNodes = allNodes.map(n => ({{ id: n.id, hidden: !n.is_hub }}));
            }} else if (val === 'high') {{
                filteredNodes = allNodes.map(n => ({{ id: n.id, hidden: !(n.p_val >= 0.70) }}));
            }} else if (val === 'provincial') {{
                filteredNodes = allNodes.map(n => ({{ id: n.id, hidden: !(n.p_val === 0.0) }}));
            }} else {{
                filteredNodes = allNodes.map(n => ({{ id: n.id, hidden: false }}));
            }}
            nodesDataSet.update(filteredNodes);
            network.fit({{ animation: {{ duration: 500 }} }});
        }}

        let paperMode = false;
        function togglePaperMode() {{
            paperMode = !paperMode;
            document.body.classList.toggle('paper-white', paperMode);
            const edgeColor = paperMode ? 'rgba(71, 85, 105, 0.35)' : 'rgba(200, 200, 200, 0.3)';
            const fontColor = paperMode ? '#0f172a' : '#ffffff';
            const updates = allNodes.map(n => ({{
                id: n.id,
                font: {{ color: fontColor, size: n.is_hub ? 13 : 10, face: 'Inter, sans-serif' }}
            }}));
            nodesDataSet.update(updates);
            const edgeUpdates = allEdges.map(e => ({{
                id: e.id,
                color: {{ color: edgeColor, highlight: '#2563eb' }}
            }}));
            edgesDataSet.update(edgeUpdates);
            document.getElementById('paperBtn').innerText = paperMode ? '🌙 Dark Mode' : '📄 Paper Mode (White)';
        }}

        function toggleGuide() {{
            document.getElementById('guideDrawer').classList.toggle('open');
        }}

        function searchNode(query) {{
            if (!query) return;
            const q = query.toLowerCase().trim();
            const found = allNodes.find(n => n.full_label.toLowerCase().includes(q));
            if (found) {{
                network.focus(found.id, {{
                    scale: 1.8,
                    animation: {{ duration: 700, easingFunction: 'easeInOutQuad' }}
                }});
                network.selectNodes([found.id]);
                showInspector(found.id);
            }} else {{
                alert('Entity not found in participation dispersion map: "' + query + '"');
            }}
        }}

        function showInspector(nodeId) {{
            const n = allNodes.find(item => item.id === nodeId);
            if (!n) return;
            document.getElementById('inspName').innerText = n.full_label;
            const html = `
                <div style="margin-bottom: 8px;">
                    <span class="metric-pill">P_i Score: <b>${{n.p_val.toFixed(4)}}</b></span>
                    <span class="metric-pill">Global Hub: <b>${{n.is_hub ? 'Yes (★)' : 'No'}}</b></span>
                </div>
                <div style="color: var(--text-sub); font-size: 11px; line-height: 1.4; margin-bottom: 6px;">
                    ${{n.role_desc}}
                </div>
                <div style="color: var(--text-dim); font-size: 10px;">
                    Node participates across graph modules with participation coefficient P_i = ${{n.p_val.toFixed(4)}}.
                </div>
            `;
            document.getElementById('inspContent').innerHTML = html;
            document.getElementById('inspectorCard').classList.add('visible');
        }}

        function closeInspector() {{
            document.getElementById('inspectorCard').classList.remove('visible');
        }}

        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                showInspector(params.nodes[0]);
            }} else {{
                closeInspector();
            }}
        }});

        function downloadSnapshot() {{
            network.fit({{ animation: false }});
            setTimeout(() => {{
                const canvas = container.getElementsByTagName('canvas')[0];
                if (!canvas) return;
                const link = document.createElement('a');
                link.download = 'Stage_3_1_Participation_Dispersion_Map.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            }}, 300);
        }}
    </script>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_out)

    def _generate_workflow_chokepoints_map(self, dg: nx.DiGraph, node_metrics: Dict[str, NodeMetrics], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        css = self._get_shared_interactive_css()

        enriched_nodes = []
        total_nodes = 0
        choke_count = 0
        max_betw = 0.0

        for node in dg.nodes():
            metrics = node_metrics.get(node)
            betw = metrics.betweenness_centrality if metrics else 0.0
            if betw > max_betw:
                max_betw = betw
            is_choke = (betw > 0.0001)
            if is_choke:
                choke_count += 1
            total_nodes += 1

            if is_choke:
                role_desc = "Workflow Narrative Chokepoint — Critical shortest-path transit bottleneck across knowledge domains"
            elif betw > 0.00002:
                role_desc = "Secondary Narrative Bridge — Moderate shortest-path participation"
            else:
                role_desc = "Peripheral / Local Node — Embedded within local topical flows"

            tooltip = (
                f"<b>{html.escape(str(node))}</b><br>"
                f"Betweenness Centrality: {betw:.6f}<br>"
                f"Is Narrative Chokepoint: {'Yes (⬥)' if is_choke else 'No'}<br>"
                f"Role: {role_desc}"
            )

            color = "#ef4444" if is_choke else "#38bdf8"
            shape = "diamond" if is_choke else "dot"
            size = 15 + min(60, betw * 30000)
            deg = dg.degree(node)

            enriched_nodes.append({
                "id": str(node),
                "label": str(node) if (is_choke or deg >= 15) else "",
                "full_label": str(node),
                "shape": shape,
                "size": round(size, 1),
                "color": color,
                "borderWidth": 3 if is_choke else 1,
                "font": {"color": "#ffffff", "size": 13 if is_choke else 10, "face": "Inter, sans-serif"},
                "title": tooltip,
                "betw": round(float(betw), 6),
                "is_choke": is_choke,
                "role_desc": role_desc
            })

        enriched_edges = []
        for idx, (u, v, data) in enumerate(dg.edges(data=True)):
            enriched_edges.append({
                "id": f"e_{idx}",
                "from": str(u),
                "to": str(v),
                "arrows": "to",
                "title": data.get("predicate", ""),
                "color": {"color": "rgba(200, 200, 255, 0.4)", "highlight": "#38bdf8"},
                "width": 1
            })

        standard_count = total_nodes - choke_count
        total_edges = len(enriched_edges)
        top_chokepoints = sorted(enriched_nodes, key=lambda x: x["betw"], reverse=True)[:3]
        chokepoints_guide_items = "".join([
            f"<li><b>{html.escape(node['full_label'])} (C_B = {node['betw']:.6f}):</b> {html.escape(node['role_desc'])}</li>"
            for node in top_chokepoints
        ])
        nodes_json = json.dumps(enriched_nodes)
        edges_json = json.dumps(enriched_edges)

        html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stage 3.1: Global Overview — Workflow Narratives & Betweenness Chokepoints</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <style>
{css}
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <div>
                <h1>Stage 3.1: Global Overview — Workflow Narratives & Betweenness Chokepoints</h1>
                <div class="header-sub">
                    <span>Shortest-Path Graph Bottlenecks: Identifying Critical Bridges in Narrative Flow</span>
                    <span class="stat-badge">Entities: {total_nodes}</span>
                    <span class="stat-badge">Directed Flow Edges: {total_edges}</span>
                    <span class="stat-badge stat-badge-danger">Narrative Chokepoints (⬥): {choke_count}</span>
                    <span class="stat-badge stat-badge-success">Standard Entities (●): {standard_count}</span>
                    <span class="stat-badge">Max Betweenness: {max_betw:.6f}</span>
                </div>
            </div>
        </div>
        <div class="header-actions">
            <div class="search-box">
                <input type="text" class="search-input" id="nodeSearch" placeholder="Search entity in flow..." onkeydown="if(event.key==='Enter') searchNode(this.value)">
            </div>
            <select class="filter-select" id="chokeFilter" onchange="filterChokepoints(this.value)">
                <option value="all">All Entities ({total_nodes})</option>
                <option value="chokepoints">Chokepoints Only ({choke_count})</option>
                <option value="standard">Standard Entities ({standard_count})</option>
            </select>
            <button class="btn" id="labelBtn" onclick="toggleAllLabels()">🏷️ Show All Labels</button>
            <button class="btn btn-paper" id="paperBtn" onclick="togglePaperMode()">📄 Paper Mode (White)</button>
            <button class="btn btn-guide" onclick="toggleGuide()">ℹ️ Guide & Reference</button>
            <button class="btn btn-primary" onclick="downloadSnapshot()">📷 Snapshot</button>
            <button class="btn" onclick="network.fit({{animation: {{duration: 500}}}})">Fit</button>
        </div>
    </header>

    <div class="canvas-container">
        <div id="loadingBar">
            <div class="loader-card">
                <div class="loader-title">Stabilizing ForceAtlas2 Physics Layout...</div>
                <div class="progress-track">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-label" id="progressText">Stabilizing 0%</div>
            </div>
        </div>

        <div class="legend-card" id="legendCard">
            <div class="legend-header">
                <span class="legend-title">Topological Legend</span>
                <span style="font-size: 10px; color: var(--text-dim);">Betweenness Flow</span>
            </div>
            <div class="legend-section">
                <div class="legend-section-title">Node Classification & Role</div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: #ef4444; font-size: 14px;">⬥</span>
                    <span><b>Workflow Chokepoint (C_B &gt; 10⁻⁴)</b> — Critical narrative transit bottleneck</span>
                </div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: #38bdf8;">●</span>
                    <span><b>Standard Entity (C_B ≤ 10⁻⁴)</b> — Local concept within sub-narratives</span>
                </div>
            </div>
            <div class="legend-section">
                <div class="legend-section-title">Visual Attributes</div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: #ef4444;">✦</span>
                    <span><b>Size & Shadow:</b> Scaled with Betweenness Centrality score</span>
                </div>
                <div class="legend-item">
                    <div class="legend-line" style="background: rgba(200, 200, 255, 0.4);"></div>
                    <span><b>Directed Flow Edge:</b> Semantic narrative transition predicate</span>
                </div>
            </div>
        </div>

        <div class="inspector-card" id="inspectorCard">
            <button class="inspector-close" onclick="closeInspector()">×</button>
            <div class="inspector-name" id="inspName">Entity</div>
            <div id="inspContent"></div>
        </div>

        <div class="guide-drawer" id="guideDrawer">
            <div class="guide-header">
                <h2>📖 Workflow Chokepoint Guide</h2>
                <button class="guide-close" onclick="toggleGuide()">×</button>
            </div>
            <div class="guide-content">
                <div class="guide-card">
                    <div class="guide-card-title">🎯 Visualization Objective</div>
                    <div>
                        Identifies the critical structural choke-points and informational bridges in the knowledge graph. Highlights nodes where multi-hop narrative traversals must transit between otherwise isolated topical clusters.
                    </div>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">📐 Mathematical Definition</div>
                    <div><b>Betweenness Centrality</b> (C_B) measures shortest-path traffic:</div>
                    <code class="formula-box">C_B(v) = Σ (σ_st(v) / σ_st)  for s ≠ v ≠ t</code>
                    <div style="font-size: 11px; color: var(--text-dim); margin-top: 4px;">
                        Where <i>σ_st</i> is the total count of shortest paths between nodes <i>s</i> and <i>t</i>, and <i>σ_st(v)</i> is the count passing through node <i>v</i>.
                    </div>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">🔍 Identified Chokepoints in Corpus</div>
                    <ul class="takeaway-list">
                        {chokepoints_guide_items}
                    </ul>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">⚠️ Vulnerability & RAG Impact</div>
                    <div>
                        Disrupting or hallucinating on a chokepoint node fragments knowledge paths into disconnected components. These nodes represent high-leverage targets for knowledge verification and reasoning chain preservation.
                    </div>
                </div>
            </div>
        </div>

        <div id="network"></div>
    </div>

    <script>
        const allNodes = {nodes_json};
        const allEdges = {edges_json};

        const nodesDataSet = new vis.DataSet(allNodes);
        const edgesDataSet = new vis.DataSet(allEdges);

        const container = document.getElementById('network');
        const data = {{ nodes: nodesDataSet, edges: edgesDataSet }};

        const options = {{
            physics: {{
                enabled: true,
                forceAtlas2Based: {{
                    gravitationalConstant: -80,
                    centralGravity: 0.005,
                    springLength: 200,
                    springConstant: 0.08
                }},
                solver: 'forceAtlas2Based',
                stabilization: {{ enabled: true, iterations: 150, fit: true }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 80,
                hideEdgesOnDrag: true,
                zoomView: true,
                dragView: true
            }}
        }};

        const network = new vis.Network(container, data, options);

        network.on('stabilizationProgress', function(params) {{
            const pct = Math.round((params.iterations / params.total) * 100);
            document.getElementById('progressFill').style.width = pct + '%';
            document.getElementById('progressText').innerText = 'Stabilizing Layout: ' + pct + '%';
        }});

        network.on('stabilizationIterationsDone', function() {{
            document.getElementById('loadingBar').classList.add('hidden');
        }});

        setTimeout(() => {{
            const bar = document.getElementById('loadingBar');
            if (bar && !bar.classList.contains('hidden')) {{
                bar.classList.add('hidden');
            }}
        }}, 2500);

        let labelsVisible = false;
        function toggleAllLabels() {{
            labelsVisible = !labelsVisible;
            const updates = allNodes.map(n => ({{
                id: n.id,
                label: labelsVisible ? n.full_label : (n.is_choke ? n.full_label : '')
            }}));
            nodesDataSet.update(updates);
            document.getElementById('labelBtn').innerText = labelsVisible ? '🏷️ Auto Labels' : '🏷️ Show All Labels';
        }}

        function filterChokepoints(val) {{
            let filteredNodes;
            if (val === 'chokepoints') {{
                filteredNodes = allNodes.map(n => ({{ id: n.id, hidden: !n.is_choke }}));
            }} else if (val === 'standard') {{
                filteredNodes = allNodes.map(n => ({{ id: n.id, hidden: n.is_choke }}));
            }} else {{
                filteredNodes = allNodes.map(n => ({{ id: n.id, hidden: false }}));
            }}
            nodesDataSet.update(filteredNodes);
            network.fit({{ animation: {{ duration: 500 }} }});
        }}

        let paperMode = false;
        function togglePaperMode() {{
            paperMode = !paperMode;
            document.body.classList.toggle('paper-white', paperMode);
            const edgeColor = paperMode ? 'rgba(71, 85, 105, 0.35)' : 'rgba(200, 200, 255, 0.4)';
            const fontColor = paperMode ? '#0f172a' : '#ffffff';
            const updates = allNodes.map(n => ({{
                id: n.id,
                font: {{ color: fontColor, size: n.is_choke ? 13 : 10, face: 'Inter, sans-serif' }}
            }}));
            nodesDataSet.update(updates);
            const edgeUpdates = allEdges.map(e => ({{
                id: e.id,
                color: {{ color: edgeColor, highlight: '#2563eb' }}
            }}));
            edgesDataSet.update(edgeUpdates);
            document.getElementById('paperBtn').innerText = paperMode ? '🌙 Dark Mode' : '📄 Paper Mode (White)';
        }}

        function toggleGuide() {{
            document.getElementById('guideDrawer').classList.toggle('open');
        }}

        function searchNode(query) {{
            if (!query) return;
            const q = query.toLowerCase().trim();
            const found = allNodes.find(n => n.full_label.toLowerCase().includes(q));
            if (found) {{
                network.focus(found.id, {{
                    scale: 1.8,
                    animation: {{ duration: 700, easingFunction: 'easeInOutQuad' }}
                }});
                network.selectNodes([found.id]);
                showInspector(found.id);
            }} else {{
                alert('Entity not found in chokepoints map: "' + query + '"');
            }}
        }}

        function showInspector(nodeId) {{
            const n = allNodes.find(item => item.id === nodeId);
            if (!n) return;
            document.getElementById('inspName').innerText = n.full_label;
            const html = `
                <div style="margin-bottom: 8px;">
                    <span class="metric-pill">Betweenness C_B: <b>${{n.betw.toFixed(6)}}</b></span>
                    <span class="metric-pill">Chokepoint: <b>${{n.is_choke ? 'Yes (⬥)' : 'No'}}</b></span>
                </div>
                <div style="color: var(--text-sub); font-size: 11px; line-height: 1.4; margin-bottom: 6px;">
                    ${{n.role_desc}}
                </div>
                <div style="color: var(--text-dim); font-size: 10px;">
                    ${{n.is_choke ? 'Crucial transit point in multi-hop traversal paths.' : 'Local concept contributing to modular intra-cluster flow.'}}
                </div>
            `;
            document.getElementById('inspContent').innerHTML = html;
            document.getElementById('inspectorCard').classList.add('visible');
        }}

        function closeInspector() {{
            document.getElementById('inspectorCard').classList.remove('visible');
        }}

        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                showInspector(params.nodes[0]);
            }} else {{
                closeInspector();
            }}
        }});

        function downloadSnapshot() {{
            network.fit({{ animation: false }});
            setTimeout(() => {{
                const canvas = container.getElementsByTagName('canvas')[0];
                if (!canvas) return;
                const link = document.createElement('a');
                link.download = 'Stage_3_1_Workflow_Chokepoints.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            }}, 300);
        }}
    </script>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_out)

    def _generate_stage_3_2_reduced_core_visuals(
        self, core_dg: nx.DiGraph, cluster_ug: nx.Graph,
        global_hubs: List[str], separation_report: Dict[str, Any], vis_dir: Path,
        dg: nx.DiGraph = None, hub_partitions: List[HubPartition] = None,
        node_metrics: Dict[str, NodeMetrics] = None
    ):
        vis_dir.mkdir(parents=True, exist_ok=True)
        self._generate_reduced_core_graph(
            core_dg=core_dg,
            cluster_ug=cluster_ug,
            global_hubs=global_hubs,
            separation_report=separation_report,
            output_path=vis_dir / "reduced_core_graph.html"
        )
        if dg is not None and global_hubs:
            self._generate_hub_backbone_network_graph(
                dg=dg,
                global_hubs=global_hubs,
                hub_partitions=hub_partitions or [],
                node_metrics=node_metrics or {},
                output_path=vis_dir / "hub_backbone_network.html"
            )

    def _generate_reduced_core_graph(
        self, core_dg: nx.DiGraph, cluster_ug: nx.Graph,
        global_hubs: List[str], separation_report: Dict[str, Any], output_path: Path
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        core_ug = core_dg.to_undirected()
        
        # 1. Deterministic 2D coordinates for core_ug
        pos = nx.spring_layout(core_ug, k=0.12, iterations=35, seed=42)
        
        # 2. Connected components mapping
        comps = sorted(list(nx.connected_components(core_ug)), key=len, reverse=True)
        comp_map = {}
        for c_idx, comp in enumerate(comps):
            for n in comp:
                comp_map[n] = c_idx
                
        palette = [
            "#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f472b6",
            "#a78bfa", "#fb923c", "#4ade80", "#2dd4bf", "#e879f9",
            "#60a5fa", "#f87171"
        ]
        
        vis_nodes = []
        for node in core_dg.nodes():
            x, y = pos[node]
            deg = core_dg.degree(node)
            c_idx = comp_map.get(node, 0)
            comp_size = len(comps[c_idx]) if c_idx < len(comps) else 1
            
            if c_idx == 0:
                node_color = "#38bdf8"
                node_border = "#0284c7"
            else:
                node_color = palette[c_idx % len(palette)]
                node_border = "rgba(255, 255, 255, 0.4)"
                
            node_size = 6 + min(18, deg * 1.0)
            label_text = node if deg >= 12 else ""
            
            tooltip = (
                f"<b>Node:</b> {node}<br>"
                f"<b>Core Degree:</b> {deg} (In: {core_dg.in_degree(node)}, Out: {core_dg.out_degree(node)})<br>"
                f"<b>Component:</b> #{c_idx + 1} ({comp_size} nodes)<br>"
                f"<b>Status:</b> Retained in Core (Global Hubs Excluded)"
            )
            
            vis_nodes.append({
                "id": node,
                "label": label_text,
                "full_label": node,
                "x": round(float(x * 2200), 1),
                "y": round(float(y * 2200), 1),
                "size": round(node_size, 1),
                "shape": "dot",
                "color": {
                    "background": node_color,
                    "border": node_border,
                    "highlight": {"background": "#ffffff", "border": node_color}
                },
                "borderWidth": 1.5,
                "font": {"color": "#ffffff", "size": 10, "face": "Inter, sans-serif"},
                "title": tooltip,
                "degree": deg,
                "component": c_idx + 1,
                "comp_size": comp_size,
                "in_degree": core_dg.in_degree(node),
                "out_degree": core_dg.out_degree(node)
            })
            
        vis_edges = []
        for idx, (u, v, data) in enumerate(core_dg.edges(data=True)):
            vis_edges.append({
                "id": f"e_{idx}",
                "from": u,
                "to": v,
                "arrows": "to",
                "title": data.get("predicate", ""),
                "color": {
                    "color": "rgba(148, 163, 184, 0.18)",
                    "highlight": "#38bdf8"
                },
                "width": 0.8,
                "smooth": False
            })
            
        total_nodes = len(core_dg.nodes())
        total_edges = len(core_dg.edges())
        total_hubs_stripped = len(global_hubs)
        raw_q = separation_report.get("modularity_enhancement", {}).get("raw_modularity_q", 0.0)
        core_q = separation_report.get("modularity_enhancement", {}).get("core_modularity_q", 0.0)
        q_gain = separation_report.get("modularity_enhancement", {}).get("percent_modularity_increase", 0.0)
        total_comps = len(comps)
        giant_size = len(comps[0]) if comps else 0
        subnetworks = [c for c in comps if len(c) > 1 and c != comps[0]]
        subnetworks_count = len(subnetworks)
        subnetworks_nodes = sum(len(c) for c in subnetworks)
        singletons_count = sum(1 for c in comps if len(c) == 1)
        
        nodes_json = json.dumps(vis_nodes)
        edges_json = json.dumps(vis_edges)
        
        template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stage 3.2: Decoupled Core Topology — Global Hubs Removed</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <style>
        :root {{
            --bg-body: #0d0e12;
            --bg-header: rgba(20, 22, 31, 0.95);
            --bg-card: #1c202d;
            --bg-card-glass: rgba(20, 22, 31, 0.92);
            --border: #262c3e;
            --border-hover: #3b82f6;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --text-dim: #64748b;
            --accent-blue: #38bdf8;
            --accent-green: #34d399;
            --accent-purple: #a855f7;
            --accent-amber: #fbbf24;
            --accent-red: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-body);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            transition: background-color 0.2s, color 0.2s;
        }}
        header {{
            background: var(--bg-header);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
            padding: 8px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }}
        .header-title h1 {{
            font-size: 15px;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
            letter-spacing: -0.01em;
        }}
        .header-sub {{
            font-size: 11px;
            color: var(--text-sub);
            margin-top: 2px;
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .stat-badge {{
            background: rgba(56, 189, 248, 0.12);
            color: var(--accent-blue);
            font-weight: 600;
            padding: 1px 7px;
            border-radius: 4px;
            font-size: 10px;
            border: 1px solid rgba(56, 189, 248, 0.25);
            display: inline-flex;
            align-items: center;
            gap: 3px;
        }}
        .stat-badge-success {{
            background: rgba(52, 211, 153, 0.12);
            color: var(--accent-green);
            border-color: rgba(52, 211, 153, 0.25);
        }}
        .header-actions {{
            display: flex;
            gap: 6px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .btn {{
            background: #1e2230;
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 5px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            user-select: none;
        }}
        .btn:hover {{
            background: #282f44;
            border-color: var(--border-hover);
        }}
        .btn-primary {{
            background: #2563eb;
            border-color: #3b82f6;
            color: #ffffff;
        }}
        .btn-primary:hover {{
            background: #1d4ed8;
        }}
        .btn-warning {{
            background: rgba(251, 191, 36, 0.2);
            border-color: #fbbf24;
            color: #fde047;
        }}
        .btn-guide {{
            background: rgba(168, 85, 247, 0.18);
            border-color: rgba(168, 85, 247, 0.4);
            color: #d8b4fe;
        }}
        .btn-guide:hover {{
            background: rgba(168, 85, 247, 0.3);
            border-color: #c084fc;
            color: #ffffff;
        }}
        .btn-paper {{
            background: #334155;
            border-color: #475569;
        }}
        .search-box {{
            position: relative;
        }}
        .search-input {{
            background: #181c28;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 5px 9px;
            border-radius: 6px;
            font-size: 11px;
            width: 160px;
            outline: none;
            transition: width 0.2s, border-color 0.2s;
        }}
        .search-input:focus {{
            width: 220px;
            border-color: #3b82f6;
        }}
        .filter-select {{
            background: #181c28;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 5px 8px;
            border-radius: 6px;
            font-size: 11px;
            outline: none;
            cursor: pointer;
        }}
        .slider-control {{
            background: #181c28;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 4px 8px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: var(--text-sub);
            user-select: none;
        }}
        .slider-label {{
            font-size: 10px;
            white-space: nowrap;
        }}
        .slider-label b {{
            color: var(--accent-blue);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        .slider-input {{
            width: 75px;
            accent-color: #38bdf8;
            cursor: pointer;
            vertical-align: middle;
        }}
        .canvas-container {{
            flex: 1;
            position: relative;
            width: 100%;
            height: calc(100vh - 58px);
        }}
        #network {{
            width: 100%;
            height: 100%;
        }}
        /* Loading Progress Bar */
        #loadingBar {{
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(13, 14, 18, 0.85);
            backdrop-filter: blur(6px);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 20;
            transition: opacity 0.4s ease, visibility 0.4s ease;
        }}
        #loadingBar.hidden {{
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
        }}
        .loader-card {{
            background: #1a1e2b;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px 28px;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
            width: 320px;
        }}
        .loader-title {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 12px;
        }}
        .progress-track {{
            width: 100%;
            height: 8px;
            background: #252b3d;
            border-radius: 4px;
            overflow: hidden;
        }}
        .progress-fill {{
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            border-radius: 4px;
            transition: width 0.15s ease;
        }}
        .progress-label {{
            margin-top: 10px;
            font-size: 11px;
            color: var(--text-sub);
        }}
        /* Legend Card */
        .legend-card {{
            position: absolute;
            top: 14px;
            right: 14px;
            background: var(--bg-card-glass);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 11px;
            color: var(--text-sub);
            z-index: 5;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
            max-width: 330px;
        }}
        .legend-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }}
        .legend-title {{
            font-weight: 700;
            text-transform: uppercase;
            font-size: 10px;
            color: var(--accent-blue);
            letter-spacing: 0.05em;
        }}
        .legend-section {{
            margin-top: 8px;
        }}
        .legend-section-title {{
            font-size: 9px;
            text-transform: uppercase;
            color: var(--text-dim);
            font-weight: 700;
            margin-bottom: 4px;
            letter-spacing: 0.03em;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 7px;
            margin-bottom: 5px;
            line-height: 1.3;
        }}
        .legend-icon {{
            width: 14px;
            height: 14px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            flex-shrink: 0;
        }}
        .legend-line {{
            width: 16px;
            height: 2px;
            border-radius: 1px;
            flex-shrink: 0;
        }}
        /* Inspector Card */
        .inspector-card {{
            position: absolute;
            bottom: 14px;
            left: 14px;
            background: rgba(20, 22, 31, 0.94);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 16px;
            font-size: 11px;
            color: var(--text-sub);
            z-index: 5;
            box-shadow: 0 8px 28px rgba(0,0,0,0.6);
            max-width: 370px;
            display: none;
        }}
        .inspector-card.visible {{ display: block; }}
        .inspector-close {{
            position: absolute;
            top: 8px;
            right: 10px;
            background: transparent;
            border: none;
            color: var(--text-dim);
            cursor: pointer;
            font-size: 16px;
            line-height: 1;
        }}
        .inspector-close:hover {{ color: var(--text-main); }}
        .inspector-name {{
            font-size: 13px;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 8px;
            padding-right: 18px;
            word-break: break-word;
        }}
        .metric-pill {{
            display: inline-block;
            background: #252b3d;
            color: #94a3b8;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 10px;
            margin-right: 4px;
            margin-bottom: 4px;
        }}
        .metric-pill b {{ color: #f8fafc; }}
        /* Guide Drawer */
        .guide-drawer {{
            position: absolute;
            top: 0; right: 0;
            width: 390px;
            height: 100%;
            background: rgba(18, 20, 29, 0.97);
            backdrop-filter: blur(12px);
            border-left: 1px solid var(--border);
            z-index: 15;
            display: flex;
            flex-direction: column;
            box-shadow: -10px 0 30px rgba(0,0,0,0.6);
            transform: translateX(100%);
            transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        .guide-drawer.open {{
            transform: translateX(0);
        }}
        .guide-header {{
            padding: 14px 18px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .guide-header h2 {{
            font-size: 14px;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .guide-close {{
            background: transparent;
            border: none;
            color: var(--text-dim);
            font-size: 18px;
            cursor: pointer;
        }}
        .guide-close:hover {{ color: var(--text-main); }}
        .guide-content {{
            padding: 16px 18px;
            overflow-y: auto;
            flex: 1;
            font-size: 12px;
            line-height: 1.5;
            color: var(--text-sub);
        }}
        .guide-card {{
            background: #171b26;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 14px;
        }}
        .guide-card-title {{
            font-size: 12px;
            font-weight: 700;
            color: var(--accent-blue);
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .takeaway-list {{
            margin-left: 16px;
            margin-top: 6px;
        }}
        .takeaway-list li {{
            margin-bottom: 6px;
        }}
        /* Paper Print Mode */
        body.paper-white {{
            background-color: #ffffff !important;
            color: #0f172a !important;
        }}
        body.paper-white header {{
            background: #f8fafc !important;
            border-bottom: 1px solid #cbd5e1 !important;
            color: #0f172a !important;
        }}
        body.paper-white .header-title h1 {{ color: #0f172a !important; }}
        body.paper-white .header-sub {{ color: #475569 !important; }}
        body.paper-white .btn {{ background: #f1f5f9; color: #0f172a; border-color: #cbd5e1; }}
        body.paper-white .btn-primary {{ background: #2563eb; color: #ffffff; }}
        body.paper-white .btn-warning {{ background: #fef3c7; color: #b45309; border-color: #fcd34d; }}
        body.paper-white .search-input {{ background: #f8fafc; border-color: #cbd5e1; color: #0f172a; }}
        body.paper-white .filter-select {{ background: #f8fafc; border-color: #cbd5e1; color: #0f172a; }}
        body.paper-white .slider-control {{ background: #f8fafc; border-color: #cbd5e1; color: #475569; }}
        body.paper-white .slider-label b {{ color: #0284c7; }}
        body.paper-white .legend-card {{ background: rgba(255, 255, 255, 0.94) !important; border-color: #cbd5e1 !important; color: #334155 !important; }}
        body.paper-white .legend-title {{ color: #0284c7 !important; }}
        body.paper-white .inspector-card {{ background: rgba(255, 255, 255, 0.96) !important; border-color: #cbd5e1 !important; color: #334155 !important; }}
        body.paper-white .inspector-name {{ color: #0f172a !important; }}
        body.paper-white .metric-pill {{ background: #e2e8f0 !important; color: #475569 !important; }}
        body.paper-white .metric-pill b {{ color: #0f172a !important; }}
        body.paper-white .guide-drawer {{ background: rgba(255, 255, 255, 0.98) !important; border-color: #cbd5e1 !important; color: #334155 !important; }}
        body.paper-white .guide-card {{ background: #f8fafc !important; border-color: #e2e8f0 !important; }}
        body.paper-white .guide-header {{ border-color: #cbd5e1 !important; }}
        body.paper-white .guide-header h2 {{ color: #0f172a !important; }}
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <div>
                <h1>Stage 3.2: Decoupled Core Topology (Global Hubs Removed)</h1>
                <div class="header-sub">
                    <span>Decoupled Modular Core</span>
                    <span class="stat-badge">Active Entities: {total_nodes}</span>
                    <span class="stat-badge">Internal Edges: {total_edges}</span>
                    <span class="stat-badge stat-badge-success">Hubs Stripped: {total_hubs_stripped} (0 in Graph)</span>
                    <span class="stat-badge stat-badge-success">Core Modularity Q: {core_q:.4f} (+{q_gain:.1f}%)</span>
                    <span class="stat-badge">Components: {total_comps}</span>
                </div>
            </div>
        </div>
        <div class="header-actions">
            <div class="search-box">
                <input type="text" class="search-input" id="nodeSearch" placeholder="Search entity in core graph..." onkeydown="if(event.key==='Enter') searchNode(this.value)">
            </div>
            <select class="filter-select" id="compFilter" onchange="filterComponent(this.value)">
                <option value="all">All Components ({total_comps})</option>
                <option value="giant">Giant Core #1 ({giant_size} nodes)</option>
                <option value="subnetworks">Decoupled Sub-networks ({subnetworks_count} clusters, {subnetworks_nodes} nodes)</option>
                <option value="singletons">Isolated Singletons ({singletons_count} nodes)</option>
            </select>
            <div class="slider-control" title="Central Gravity: pulls peripheral components and clusters toward network center">
                <span class="slider-label">Gravity: <b id="gravVal">0.012</b></span>
                <input type="range" class="slider-input" id="gravSlider" min="0.002" max="0.040" step="0.002" value="0.012" oninput="updateGravity(this.value)">
            </div>
            <div class="slider-control" title="Cluster Separation: ForceAtlas2 repulsive force between node communities">
                <span class="slider-label">Separation: <b id="sepVal">120</b></span>
                <input type="range" class="slider-input" id="sepSlider" min="40" max="300" step="10" value="120" oninput="updateSeparation(this.value)">
            </div>
            <button class="btn" id="physicsBtn" onclick="togglePhysics()">⚡ Freeze Physics</button>
            <button class="btn" id="labelBtn" onclick="toggleAllLabels()">🏷️ Show All Labels</button>
            <button class="btn btn-paper" id="paperBtn" onclick="togglePaperMode()">📄 Paper Mode</button>
            <button class="btn btn-guide" onclick="toggleGuide()">ℹ️ Guide & Reference</button>
            <button class="btn btn-primary" onclick="downloadSnapshot()">📷 Snapshot</button>
            <button class="btn" onclick="network.fit({{animation: {{duration: 500}}}})">Fit</button>
        </div>
    </header>

    <div class="canvas-container">
        <div id="loadingBar">
            <div class="loader-card">
                <div class="loader-title">Stabilizing ForceAtlas2 Physics Layout...</div>
                <div class="progress-track">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-label" id="progressText">Stabilizing Layout: 0%</div>
            </div>
        </div>

        <div class="legend-card" id="legendCard">
            <div class="legend-header">
                <span class="legend-title">Core Graph Structure</span>
                <span style="font-size: 10px; color: var(--text-dim);">Stage 3.2</span>
            </div>
            <div class="legend-section">
                <div class="legend-section-title">Component Membership</div>
                <div class="legend-item"><span class="legend-icon" style="color: #38bdf8;">●</span> <span><b>Giant Core #1</b> ({giant_size} nodes)</span></div>
                <div class="legend-item"><span class="legend-icon" style="color: #a78bfa;">●</span> <span><b>Decoupled Sub-Networks</b> ({subnetworks_count} clusters)</span></div>
                <div class="legend-item"><span class="legend-icon" style="color: #fbbf24;">●</span> <span><b>Isolated Singletons</b> ({singletons_count} nodes)</span></div>
            </div>
            <div class="legend-section">
                <div class="legend-section-title">Network & Exclusions</div>
                <div class="legend-item"><div class="legend-line" style="background: rgba(148, 163, 184, 0.25);"></div> <span>Core Intra-Network Directed Edge</span></div>
                <div class="legend-item"><span class="legend-icon" style="color: #ef4444;">✕</span> <span>Global Hubs: 100% Excluded ({total_hubs_stripped} stripped)</span></div>
            </div>
            <div class="legend-section">
                <div class="legend-section-title">Physics & Forces</div>
                <div class="legend-item" style="color: var(--text-dim); font-size: 10px; line-height: 1.3;">
                    ForceAtlas2 live gravity pulls peripheral nodes inward; cluster separation pushes modular communities apart.
                </div>
            </div>
        </div>

        <div class="inspector-card" id="inspectorCard">
            <button class="inspector-close" onclick="closeInspector()">×</button>
            <div class="inspector-name" id="inspName">Entity</div>
            <div id="inspContent"></div>
        </div>

        <div class="guide-drawer" id="guideDrawer">
            <div class="guide-header">
                <h2>📖 Decoupled Core Topology Guide</h2>
                <button class="guide-close" onclick="toggleGuide()">×</button>
            </div>
            <div class="guide-content">
                <div class="guide-card">
                    <div class="guide-card-title">🎯 Visualization Objective</div>
                    <div>
                        Examines the modular core network revealed after removing the top {total_hubs_stripped} Global Hub routers identified in Stage 3.1. Without the artificial bridge edges created by ubiquitous hubs, previously obscured thematic clusters detach into their natural community boundaries.
                    </div>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">📈 Modularity Gain (+{q_gain:.1f}%)</div>
                    <div>
                        Core Modularity increases from <b>Q = {raw_q:.4f}</b> to <b>Q = {core_q:.4f}</b> (+{q_gain:.1f}%), confirming that global hubs acted as cross-domain bridges that masked true community independence.
                    </div>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">🪐 Physics: Gravity & Separation</div>
                    <ul class="takeaway-list">
                        <li><b>Central Gravity (Default 0.012):</b> Applies an inward orbital pull toward the network center (0, 0). Because {total_comps - 1} peripheral components were decoupled when global hubs were stripped, central gravity prevents disconnected clusters and singletons from drifting infinitely into outer space, creating a cohesive orbital halo.</li>
                        <li><b>Cluster Separation (Default 120):</b> Inverse-square repulsion in ForceAtlas2 that pushes densely interlinked modular clusters away from each other, allowing internal community fissures to be clearly inspected.</li>
                        <li><b>⚡ Freeze Physics:</b> Halts the live physics solver to fix node coordinates in place, providing silk-smooth 60fps zooming, panning, and inspection. Click again to unfreeze live physical dynamics.</li>
                    </ul>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">📦 Component Isolation Filter</div>
                    <ul class="takeaway-list">
                        <li><b>Giant Core Component #1 ({giant_size} nodes):</b> The primary interconnected backbone of the domain.</li>
                        <li><b>Decoupled Sub-networks ({subnetworks_count} components, {subnetworks_nodes} nodes):</b> Self-contained multi-node clusters that only linked into the main network via the removed global hubs.</li>
                        <li><b>Isolated Singletons ({singletons_count} nodes):</b> Individual leaf entities that were solely connected to global hubs.</li>
                    </ul>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">💡 Interactive Navigation</div>
                    <ul class="takeaway-list">
                        <li><b>Search:</b> Type any concept in the header box to zoom directly to its position.</li>
                        <li><b>Sliders:</b> Live-adjust Gravity and Separation in real time.</li>
                        <li><b>Node Click:</b> Click any node to open the Inspector and read its cluster membership and degree.</li>
                        <li><b>Paper Mode:</b> Switch to clean white mode for high-contrast viewing and print figures.</li>
                    </ul>
                </div>
            </div>
        </div>

        <div id="network"></div>
    </div>

    <script>
        const allNodes = {nodes_json};
        const allEdges = {edges_json};

        const nodesDataSet = new vis.DataSet(allNodes);
        const edgesDataSet = new vis.DataSet(allEdges);

        const container = document.getElementById('network');
        const data = {{ nodes: nodesDataSet, edges: edgesDataSet }};

        let physicsRunning = true;
        let currentGravity = 0.012;
        let currentSeparation = -120;
        let currentFilter = 'all';

        const options = {{
            nodes: {{
                shape: 'dot',
                font: {{ color: '#ffffff', size: 10, face: 'Inter, sans-serif' }},
                borderWidth: 1.5
            }},
            edges: {{
                arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
                smooth: false,
                width: 0.8
            }},
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: currentSeparation,
                    centralGravity: currentGravity,
                    springLength: 180,
                    springConstant: 0.08,
                    damping: 0.4,
                    avoidOverlap: 0.8
                }},
                stabilization: {{
                    enabled: true,
                    iterations: 150,
                    updateInterval: 10,
                    fit: true
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 80,
                hideEdgesOnDrag: true,
                navigationButtons: false,
                keyboard: true,
                zoomView: true,
                dragView: true
            }}
        }};

        const network = new vis.Network(container, data, options);

        network.on('stabilizationProgress', function(params) {{
            const pct = Math.round((params.iterations / params.total) * 100);
            document.getElementById('progressFill').style.width = pct + '%';
            document.getElementById('progressText').innerText = 'Stabilizing Physics: ' + pct + '%';
        }});

        network.on('stabilizationIterationsDone', function() {{
            document.getElementById('loadingBar').classList.add('hidden');
        }});

        setTimeout(() => {{
            const bar = document.getElementById('loadingBar');
            if (bar && !bar.classList.contains('hidden')) {{
                bar.classList.add('hidden');
            }}
        }}, 3000);

        function togglePhysics() {{
            physicsRunning = !physicsRunning;
            network.setOptions({{ physics: {{ enabled: physicsRunning }} }});
            const btn = document.getElementById('physicsBtn');
            if (physicsRunning) {{
                btn.innerHTML = '⚡ Freeze Physics';
                btn.classList.remove('btn-warning');
            }} else {{
                btn.innerHTML = '▶️ Run Physics';
                btn.classList.add('btn-warning');
            }}
        }}

        function updateGravity(val) {{
            currentGravity = parseFloat(val);
            document.getElementById('gravVal').innerText = currentGravity.toFixed(3);
            network.setOptions({{
                physics: {{
                    enabled: true,
                    forceAtlas2Based: {{ centralGravity: currentGravity }}
                }}
            }});
            if (!physicsRunning) {{
                physicsRunning = true;
                const btn = document.getElementById('physicsBtn');
                btn.innerHTML = '⚡ Freeze Physics';
                btn.classList.remove('btn-warning');
            }}
        }}

        function updateSeparation(val) {{
            const repMag = parseFloat(val);
            currentSeparation = -repMag;
            document.getElementById('sepVal').innerText = Math.round(repMag);
            network.setOptions({{
                physics: {{
                    enabled: true,
                    forceAtlas2Based: {{ gravitationalConstant: currentSeparation }}
                }}
            }});
            if (!physicsRunning) {{
                physicsRunning = true;
                const btn = document.getElementById('physicsBtn');
                btn.innerHTML = '⚡ Freeze Physics';
                btn.classList.remove('btn-warning');
            }}
        }}

        function filterComponent(val) {{
            currentFilter = val;
            let updates;
            if (val === 'giant') {{
                updates = allNodes.map(n => ({{ id: n.id, hidden: n.component !== 1 }}));
            }} else if (val === 'subnetworks') {{
                updates = allNodes.map(n => ({{ id: n.id, hidden: (n.component === 1 || n.comp_size <= 1) }}));
            }} else if (val === 'singletons') {{
                updates = allNodes.map(n => ({{ id: n.id, hidden: n.comp_size > 1 }}));
            }} else {{
                updates = allNodes.map(n => ({{ id: n.id, hidden: false }}));
            }}
            nodesDataSet.update(updates);
            network.fit({{ animation: {{ duration: 500 }} }});
        }}

        let labelsVisible = false;
        function toggleAllLabels() {{
            labelsVisible = !labelsVisible;
            const updates = allNodes.map(n => ({{
                id: n.id,
                label: labelsVisible ? n.full_label : (n.degree >= 12 ? n.full_label : '')
            }}));
            nodesDataSet.update(updates);
            document.getElementById('labelBtn').innerText = labelsVisible ? '🏷️ Auto Labels' : '🏷️ Show All Labels';
        }}

        let paperMode = false;
        function togglePaperMode() {{
            paperMode = !paperMode;
            document.body.classList.toggle('paper-white', paperMode);
            const edgeColor = paperMode ? 'rgba(71, 85, 105, 0.25)' : 'rgba(148, 163, 184, 0.18)';
            const fontColor = paperMode ? '#0f172a' : '#ffffff';
            const updates = allNodes.map(n => ({{
                id: n.id,
                font: {{ color: fontColor, size: 10, face: 'Inter, sans-serif' }}
            }}));
            nodesDataSet.update(updates);
            const edgeUpdates = allEdges.map(e => ({{
                id: e.id,
                color: {{ color: edgeColor, highlight: '#2563eb' }}
            }}));
            edgesDataSet.update(edgeUpdates);
            document.getElementById('paperBtn').innerText = paperMode ? '🌙 Dark Mode' : '📄 Paper Mode (White)';
        }}

        function toggleGuide() {{
            document.getElementById('guideDrawer').classList.toggle('open');
        }}

        function downloadSnapshot() {{
            network.fit({{ animation: false }});
            setTimeout(() => {{
                const canvas = container.getElementsByTagName('canvas')[0];
                if (!canvas) return;
                const link = document.createElement('a');
                link.download = 'Stage_3_2_Decoupled_Core_Topology.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            }}, 300);
        }}

        function searchNode(query) {{
            if (!query) return;
            const q = query.toLowerCase().trim();
            const found = allNodes.find(n => n.full_label.toLowerCase().includes(q));
            if (found) {{
                if (currentFilter !== 'all') {{
                    document.getElementById('compFilter').value = 'all';
                    filterComponent('all');
                }}
                network.focus(found.id, {{
                    scale: 1.8,
                    animation: {{ duration: 700, easingFunction: 'easeInOutQuad' }}
                }});
                network.selectNodes([found.id]);
                showInspector(found.id);
            }} else {{
                alert('Entity not found in decoupled core network: "' + query + '"');
            }}
        }}

        function showInspector(nodeId) {{
            const n = allNodes.find(item => item.id === nodeId);
            if (!n) return;
            document.getElementById('inspName').innerText = n.full_label;
            const isGiant = n.component === 1;
            const compDesc = isGiant ? 'Giant Core Component #1' : (n.comp_size > 1 ? `Sub-Network Component #${{n.component}} (${{n.comp_size}} nodes)` : `Isolated Singleton #${{n.component}}`);
            const html = `
                <div style="margin-bottom: 8px;">
                    <span class="metric-pill">Component: <b>#${{n.component}}</b></span>
                    <span class="metric-pill">Cluster Size: <b>${{n.comp_size}} node${{n.comp_size > 1 ? 's' : ''}}</b></span>
                    <span class="metric-pill">Core Degree: <b>${{n.degree}}</b></span>
                    <span class="metric-pill">In: <b>${{n.in_degree}}</b> | Out: <b>${{n.out_degree}}</b></span>
                </div>
                <div style="color: var(--text-sub); font-size: 11px; margin-bottom: 6px;">
                    <b>Classification:</b> ${{compDesc}}
                </div>
                <div style="color: var(--text-dim); font-size: 10px; line-height: 1.4;">
                    ${{isGiant 
                        ? 'Member of the primary connected core graph. Retains internal semantic relationships after global hub stripping.' 
                        : (n.comp_size > 1 
                            ? 'Decoupled autonomous sub-network. Connected internally, but was connected to the wider graph solely through global hubs.' 
                            : 'Peripheral leaf concept that was connected to the broader corpus exclusively through global hub routers.')}}
                </div>
            `;
            document.getElementById('inspContent').innerHTML = html;
            document.getElementById('inspectorCard').classList.add('visible');
        }}

        function closeInspector() {{
            document.getElementById('inspectorCard').classList.remove('visible');
        }}

        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                showInspector(params.nodes[0]);
            }} else {{
                closeInspector();
            }}
        }});
    </script>
</body>
</html>"""
        with open(output_path, "w") as f:
            f.write(template)


    def _generate_hub_backbone_network_graph(
        self, dg: nx.DiGraph, global_hubs: List[str],
        hub_partitions: List[HubPartition], node_metrics: Dict[str, NodeMetrics],
        output_path: Path
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hubs_set = set(global_hubs)
        hub_dg = dg.subgraph(global_hubs).copy() if dg else nx.DiGraph()
        hub_ug = hub_dg.to_undirected()

        hub_part_map = {}
        for hp in (hub_partitions or []):
            cid = getattr(hp, "hub_cluster_id", 0)
            for n in getattr(hp, "nodes", []):
                hub_part_map[n] = cid

        palette = [
            "#ef4444", "#f97316", "#eab308", "#10b981", "#06b6d4",
            "#3b82f6", "#8b5cf6", "#ec4899", "#f43f5e", "#14b8a6"
        ]

        all_betw = [getattr(m, "betweenness_centrality", 0.0) for m in (node_metrics or {}).values()]
        max_betw = max(all_betw) if all_betw and max(all_betw) > 0 else 1.0
        all_deg = [getattr(m, "degree_centrality", 0.0) for m in (node_metrics or {}).values()]
        max_deg = max(all_deg) if all_deg and max(all_deg) > 0 else 1.0

        leaderboard = []
        for h in global_hubs:
            m = (node_metrics or {}).get(h)
            p_i = getattr(m, "participation_coefficient", 0.0) if m else 0.0
            betw = getattr(m, "betweenness_centrality", 0.0) if m else 0.0
            deg_cent = getattr(m, "degree_centrality", 0.0) if m else 0.0
            raw_deg = dg.degree(h) if dg and h in dg else 0
            in_d = dg.in_degree(h) if dg and h in dg else 0
            out_d = dg.out_degree(h) if dg and h in dg else 0
            c_id = hub_part_map.get(h, 0)

            score = (0.50 * p_i) + (0.35 * (betw / max_betw)) + (0.15 * (deg_cent / max_deg))
            leaderboard.append({
                "id": h,
                "score": round(score * 100, 1),
                "cluster_id": c_id,
                "raw_degree": raw_deg,
                "in_degree": in_d,
                "out_degree": out_d,
                "p_i": round(p_i, 4),
                "betweenness": round(betw, 4),
            })

        leaderboard.sort(key=lambda x: x["score"], reverse=True)
        for idx, item in enumerate(leaderboard, start=1):
            item["rank"] = idx

        pos = nx.spring_layout(hub_ug, k=0.45, iterations=60, seed=42) if len(hub_ug) > 0 else {}

        vis_nodes = []
        for h in global_hubs:
            x, y = pos.get(h, (0.0, 0.0))
            c_id = hub_part_map.get(h, 0)
            color = palette[c_id % len(palette)]
            m = (node_metrics or {}).get(h)
            p_i = getattr(m, "participation_coefficient", 0.0) if m else 0.0
            betw = getattr(m, "betweenness_centrality", 0.0) if m else 0.0
            deg = dg.degree(h) if dg and h in dg else 0

            tooltip = (
                f"<b>{h}</b><br>"
                f"Hub Cluster: {c_id}<br>"
                f"Total Degree: {deg}<br>"
                f"P_i: {p_i:.4f}<br>"
                f"Betweenness: {betw:.4f}"
            )

            vis_nodes.append({
                "id": h,
                "label": h,
                "full_label": h,
                "x": round(float(x * 1200), 1),
                "y": round(float(y * 1200), 1),
                "size": 30 + min(25, deg * 0.8),
                "shape": "star",
                "color": {
                    "background": color,
                    "border": "#ffffff",
                    "highlight": {"background": "#ffffff", "border": color}
                },
                "borderWidth": 2,
                "font": {"color": "#ffffff", "size": 13, "face": "Inter, sans-serif"},
                "title": tooltip,
                "cluster_id": c_id,
                "p_i": round(p_i, 4),
                "betweenness": round(betw, 4),
                "degree": deg,
                "in_degree": dg.in_degree(h) if dg and h in dg else 0,
                "out_degree": dg.out_degree(h) if dg and h in dg else 0
            })

        vis_edges = []
        if dg:
            for idx, (u, v, data) in enumerate(hub_dg.edges(data=True)):
                vis_edges.append({
                    "id": f"e_{idx}",
                    "from": u,
                    "to": v,
                    "arrows": "to",
                    "title": data.get("predicate", ""),
                    "color": {"color": "rgba(239, 68, 68, 0.75)", "highlight": "#ffffff"},
                    "width": 2.5
                })

        num_hubs = len(global_hubs)
        num_edges = len(vis_edges)
        num_clusters = len(set(hub_part_map.values()))
        density = round(float(nx.density(hub_dg)), 4) if len(hub_dg) > 1 else 0.0

        nodes_json = json.dumps(vis_nodes)
        edges_json = json.dumps(vis_edges)
        leaderboard_json = json.dumps(leaderboard)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>[Stage 3.2: Decoupled Core Topology] Global Hubs Backbone Network</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <style>
        :root {{
            --bg-body: #0d0e12;
            --bg-header: rgba(20, 22, 31, 0.95);
            --border: #262c3e;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --accent-red: #ef4444;
            --accent-blue: #38bdf8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-body);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        header {{
            background: var(--bg-header);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }}
        .header-title h1 {{
            font-size: 16px;
            font-weight: 700;
        }}
        .header-sub {{
            font-size: 11px;
            color: var(--text-sub);
            margin-top: 2px;
            display: flex;
            gap: 10px;
        }}
        .badge {{
            background: rgba(239, 68, 68, 0.15);
            color: var(--accent-red);
            font-weight: 600;
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 10px;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}
        .header-actions {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        .btn {{
            background: #1e2230;
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .btn:hover {{ background: #282f44; border-color: #3b82f6; }}
        .btn-primary {{ background: #2563eb; border-color: #3b82f6; }}
        .search-input {{
            background: #181c28;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            width: 180px;
            outline: none;
        }}
        .main-layout {{
            flex: 1;
            display: flex;
            position: relative;
            height: calc(100vh - 58px);
        }}
        .canvas-area {{
            flex: 1;
            position: relative;
            height: 100%;
        }}
        #network {{
            width: 100%;
            height: 100%;
        }}
        .sidebar {{
            width: 380px;
            background: #14161f;
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            z-index: 5;
        }}
        .sidebar-header {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
            font-weight: 700;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .table-container {{
            flex: 1;
            overflow-y: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }}
        th {{
            background: #1c202d;
            padding: 8px 10px;
            text-align: left;
            color: var(--text-sub);
            position: sticky;
            top: 0;
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 8px 10px;
            border-bottom: 1px solid rgba(38, 44, 62, 0.5);
            cursor: pointer;
        }}
        tr:hover td {{ background: #222738; }}
        .rank-badge {{
            font-weight: 700;
            color: var(--accent-red);
        }}
        .inspector-card {{
            position: absolute;
            bottom: 14px;
            left: 14px;
            background: rgba(20, 22, 31, 0.94);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 11px;
            z-index: 5;
            max-width: 320px;
            display: none;
        }}
        .inspector-card.visible {{ display: block; }}
        .metric-pill {{
            display: inline-block;
            background: #252b3d;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            margin-right: 4px;
            margin-bottom: 4px;
        }}
        .metric-pill b {{ color: #ffffff; }}
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>[Stage 3.2: Decoupled Core Topology] Global Hubs Backbone Network</h1>
            <div class="header-sub">
                <span>Extracted Structural Router Grid</span>
                <span class="badge">Hubs: {num_hubs}</span>
                <span class="badge">Inter-Hub Edges: {num_edges}</span>
                <span class="badge">Partitions: {num_clusters}</span>
                <span class="badge">Density: {density}</span>
            </div>
        </div>
        <div class="header-actions">
            <input type="text" class="search-input" id="hubSearch" placeholder="Search hub node..." onkeydown="if(event.key==='Enter') searchHub(this.value)">
            <button class="btn" onclick="network.fit({{animation: {{duration: 500}}}})">Fit View</button>
            <button class="btn btn-primary" onclick="downloadSnapshot()">📷 Snapshot</button>
        </div>
    </header>

    <div class="main-layout">
        <div class="canvas-area">
            <div id="network"></div>
            <div class="inspector-card" id="inspectorCard">
                <button style="float:right;background:none;border:none;color:#94a3b8;cursor:pointer;" onclick="closeInspector()">×</button>
                <div style="font-size:14px;font-weight:700;margin-bottom:6px;" id="inspName">Hub</div>
                <div id="inspContent"></div>
            </div>
        </div>
        <div class="sidebar">
            <div class="sidebar-header">
                <span>Hub Leaderboard</span>
                <span style="font-size:10px;color:#94a3b8;">Top Routers</span>
            </div>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Hub Concept</th>
                            <th>Cluster</th>
                            <th>Deg</th>
                            <th>P_i</th>
                            <th>Betw</th>
                        </tr>
                    </thead>
                    <tbody id="leaderboardBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const nodesData = {nodes_json};
        const edgesData = {edges_json};
        const leaderboardData = {leaderboard_json};

        const nodes = new vis.DataSet(nodesData);
        const edges = new vis.DataSet(edgesData);
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};

        const options = {{
            physics: {{
                enabled: true,
                forceAtlas2Based: {{
                    gravitationalConstant: -120,
                    centralGravity: 0.01,
                    springLength: 160,
                    springConstant: 0.08
                }},
                solver: 'forceAtlas2Based',
                stabilization: {{ enabled: true, iterations: 150 }}
            }},
            interaction: {{ hover: true, tooltipDelay: 100, zoomView: true, dragView: true }}
        }};

        const network = new vis.Network(container, data, options);

        const tbody = document.getElementById('leaderboardBody');
        leaderboardData.forEach(item => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="rank-badge">${{item.rank}}</td>
                <td style="font-weight:600;color:#ffffff;">${{item.id}}</td>
                <td><span style="background:#252b3d;padding:1px 5px;border-radius:3px;font-size:10px;">C${{item.cluster_id}}</span></td>
                <td>${{item.raw_degree}}</td>
                <td>${{item.p_i}}</td>
                <td>${{item.betweenness}}</td>
            `;
            tr.onclick = () => focusHub(item.id);
            tbody.appendChild(tr);
        }});

        function focusHub(hubId) {{
            network.selectNodes([hubId]);
            network.focus(hubId, {{ scale: 1.5, animation: {{ duration: 500 }} }});
            const node = nodesData.find(n => n.id === hubId);
            if (node) showInspector(node);
        }}

        function searchHub(q) {{
            const val = q.toLowerCase().trim();
            if (!val) return;
            const found = nodesData.find(n => n.id.toLowerCase().includes(val));
            if (found) focusHub(found.id);
        }}

        function showInspector(node) {{
            document.getElementById('inspName').textContent = node.id;
            document.getElementById('inspContent').innerHTML = `
                <div>
                    <span class="metric-pill">Cluster: <b>${{node.cluster_id}}</b></span>
                    <span class="metric-pill">Total Degree: <b>${{node.degree}}</b></span>
                </div>
                <div style="margin-top:4px;">
                    <span class="metric-pill">In-Degree: <b>${{node.in_degree}}</b></span>
                    <span class="metric-pill">Out-Degree: <b>${{node.out_degree}}</b></span>
                </div>
                <div style="margin-top:4px;">
                    <span class="metric-pill">P_i: <b>${{node.p_i}}</b></span>
                    <span class="metric-pill">Betweenness: <b>${{node.betweenness}}</b></span>
                </div>
            `;
            document.getElementById('inspectorCard').classList.add('visible');
        }}

        function closeInspector() {{
            document.getElementById('inspectorCard').classList.remove('visible');
        }}

        network.on('selectNode', params => {{
            if (params.nodes && params.nodes.length > 0) {{
                const node = nodesData.find(n => n.id === params.nodes[0]);
                if (node) showInspector(node);
            }}
        }});

        network.on('deselectNode', closeInspector);

        function downloadSnapshot() {{
            const canvas = document.querySelector('#network canvas');
            if (!canvas) return;
            const link = document.createElement('a');
            link.download = 'hub_backbone_network.png';
            link.href = canvas.toDataURL('image/png', 1.0);
            link.click();
        }}
    </script>
</body>
</html>"""

        with open(output_path, "w") as f:
            f.write(html)

    def _generate_stage_3_3_community_visuals(self, dg: nx.DiGraph, result: TopologyResult, refined_triplets: List[dict], vis_dir: Path):
        vis_dir.mkdir(parents=True, exist_ok=True)
        self._generate_isolated_communities_graph(dg, result, vis_dir / "isolated_communities_only.html")
        self._generate_collapsed_modules_graph(dg, result, vis_dir, "Path 1: Community Workflow Path", "community")
        self._generate_llm_payload_gallery(dg, result, refined_triplets, vis_dir, "Path 1: Community Workflow Path", "community")

    def _generate_isolated_communities_graph(self, dg: nx.DiGraph, result: TopologyResult, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        css = self._get_shared_interactive_css()

        colors = [
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
            "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
            "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
            "#17becf", "#9edae5"
        ]
        comm_map = {}
        comm_counts = {}
        for comm in result.communities:
            for n in comm.nodes:
                comm_map[n] = comm.community_id
                comm_counts[comm.community_id] = comm_counts.get(comm.community_id, 0) + 1

        enriched_nodes = []
        for node in comm_map.keys():
            if node in dg:
                comm_id = comm_map[node]
                color = colors[comm_id % len(colors)]
                deg = dg.degree(node)

                role_desc = f"Community #{comm_id} Member — Intra-community degree: {deg}"
                tooltip = (
                    f"<b>{html.escape(str(node))}</b><br>"
                    f"Partition: Community #{comm_id}<br>"
                    f"Intra-Community Degree: {deg}<br>"
                    f"Status: 100% Isolated (Hubs Pruned)"
                )

                enriched_nodes.append({
                    "id": str(node),
                    "label": str(node) if deg >= 10 else "",
                    "full_label": str(node),
                    "shape": "dot",
                    "size": 15 + min(20, deg),
                    "color": color,
                    "borderWidth": 1.5,
                    "font": {"color": "#ffffff", "size": 10, "face": "Inter, sans-serif"},
                    "title": tooltip,
                    "comm_id": comm_id,
                    "deg": deg,
                    "role_desc": role_desc
                })

        enriched_edges = []
        for idx, (u, v, data) in enumerate(dg.edges(data=True)):
            if u in comm_map and v in comm_map and comm_map.get(u) == comm_map.get(v):
                enriched_edges.append({
                    "id": f"e_{idx}",
                    "from": str(u),
                    "to": str(v),
                    "arrows": "to",
                    "title": data.get("predicate", ""),
                    "color": {"color": "rgba(200, 200, 200, 0.3)", "highlight": "#38bdf8"},
                    "width": 1
                })

        total_nodes = len(enriched_nodes)
        total_edges = len(enriched_edges)
        num_communities = len(comm_counts)
        hubs_stripped = len(result.global_hubs)
        nodes_json = json.dumps(enriched_nodes)
        edges_json = json.dumps(enriched_edges)

        comm_options_html = f'<option value="all">All Communities ({num_communities})</option>'
        for c_id in sorted(comm_counts.keys()):
            comm_options_html += f'<option value="{c_id}">Community #{c_id} ({comm_counts[c_id]} entities)</option>'

        html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stage 3.3: Path 1 (Community Workflow) — Isolated Communities Topology</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <style>
{css}
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <div>
                <h1>Stage 3.3: Path 1 (Community Workflow) — Isolated Communities Topology</h1>
                <div class="header-sub">
                    <span>Intra-Community Cohesion: De-noised Thematic Neighborhoods (Hubs & Inter-Links Severed)</span>
                    <span class="stat-badge">Partitioned Entities: {total_nodes}</span>
                    <span class="stat-badge stat-badge-blue">Isolated Communities: {num_communities}</span>
                    <span class="stat-badge stat-badge-success">Intra-Community Edges: {total_edges}</span>
                    <span class="stat-badge stat-badge-danger">Hubs Stripped: {hubs_stripped} (100% Decoupled)</span>
                </div>
            </div>
        </div>
        <div class="header-actions">
            <div class="search-box">
                <input type="text" class="search-input" id="nodeSearch" placeholder="Search entity in community..." onkeydown="if(event.key==='Enter') searchNode(this.value)">
            </div>
            <select class="filter-select" id="commFilter" onchange="filterCommunity(this.value)">
                {comm_options_html}
            </select>
            <button class="btn" id="labelBtn" onclick="toggleAllLabels()">🏷️ Show All Labels</button>
            <button class="btn btn-paper" id="paperBtn" onclick="togglePaperMode()">📄 Paper Mode (White)</button>
            <button class="btn btn-guide" onclick="toggleGuide()">ℹ️ Guide & Reference</button>
            <button class="btn btn-primary" onclick="downloadSnapshot()">📷 Snapshot</button>
            <button class="btn" onclick="network.fit({{animation: {{duration: 500}}}})">Fit</button>
        </div>
    </header>

    <div class="canvas-container">
        <div id="loadingBar">
            <div class="loader-card">
                <div class="loader-title">Stabilizing ForceAtlas2 Physics Layout...</div>
                <div class="progress-track">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-label" id="progressText">Stabilizing 0%</div>
            </div>
        </div>

        <div class="legend-card" id="legendCard">
            <div class="legend-header">
                <span class="legend-title">Topological Legend</span>
                <span style="font-size: 10px; color: var(--text-dim);">Pure Cohesion</span>
            </div>
            <div class="legend-section">
                <div class="legend-section-title">Color Palette Mapping</div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: #38bdf8;">●</span>
                    <span><b>Categorical Palette:</b> Unique color per Community ID</span>
                </div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: #ffffff;">●</span>
                    <span><b>Node Size:</b> Proportional to local degree (15px → 35px)</span>
                </div>
            </div>
            <div class="legend-section">
                <div class="legend-section-title">Isolation & Filtering</div>
                <div class="legend-item">
                    <div class="legend-line" style="background: rgba(200, 200, 200, 0.4);"></div>
                    <span><b>Pure Intra-Community Edge:</b> Relationship within same cluster</span>
                </div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: #ef4444;">✕</span>
                    <span><b>Global Hubs & Cross-Links:</b> 100% Pruned</span>
                </div>
            </div>
        </div>

        <div class="inspector-card" id="inspectorCard">
            <button class="inspector-close" onclick="closeInspector()">×</button>
            <div class="inspector-name" id="inspName">Entity</div>
            <div id="inspContent"></div>
        </div>

        <div class="guide-drawer" id="guideDrawer">
            <div class="guide-header">
                <h2>📖 Isolated Communities Guide</h2>
                <button class="guide-close" onclick="toggleGuide()">×</button>
            </div>
            <div class="guide-content">
                <div class="guide-card">
                    <div class="guide-card-title">🎯 Visualization Objective</div>
                    <div>
                        Displays the knowledge graph partitioned into modular thematic communities after all global hubs and cross-community edges have been pruned. Reveals pure local semantic cohesion.
                    </div>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">🔍 Reading Geometric Constellations</div>
                    <ul class="takeaway-list">
                        <li><b>Spatial Islands:</b> Because inter-community edges are completely eliminated, physics stabilization organizes communities into independent geometric islands.</li>
                        <li><b>Local Centroids:</b> Larger nodes near the center of each constellation represent local semantic anchors with high within-module degree.</li>
                        <li><b>Absence of Cross-Talk:</b> Validates that the partition algorithm successfully extracted internally dense, externally isolated topic silos.</li>
                    </ul>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">💡 Community Isolation Filter</div>
                    <div>
                        Use the Community Filter dropdown in the header bar to isolate a single community (e.g. Community #0, Community #1) and inspect its internal sub-network.
                    </div>
                </div>
            </div>
        </div>

        <div id="network"></div>
    </div>

    <script>
        const allNodes = {nodes_json};
        const allEdges = {edges_json};

        const nodesDataSet = new vis.DataSet(allNodes);
        const edgesDataSet = new vis.DataSet(allEdges);

        const container = document.getElementById('network');
        const data = {{ nodes: nodesDataSet, edges: edgesDataSet }};

        const options = {{
            physics: {{
                enabled: true,
                forceAtlas2Based: {{
                    gravitationalConstant: -80,
                    centralGravity: 0.005,
                    springLength: 180,
                    springConstant: 0.08
                }},
                solver: 'forceAtlas2Based',
                stabilization: {{ enabled: true, iterations: 150, fit: true }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 80,
                hideEdgesOnDrag: true,
                zoomView: true,
                dragView: true
            }}
        }};

        const network = new vis.Network(container, data, options);

        network.on('stabilizationProgress', function(params) {{
            const pct = Math.round((params.iterations / params.total) * 100);
            document.getElementById('progressFill').style.width = pct + '%';
            document.getElementById('progressText').innerText = 'Stabilizing Layout: ' + pct + '%';
        }});

        network.on('stabilizationIterationsDone', function() {{
            document.getElementById('loadingBar').classList.add('hidden');
        }});

        setTimeout(() => {{
            const bar = document.getElementById('loadingBar');
            if (bar && !bar.classList.contains('hidden')) {{
                bar.classList.add('hidden');
            }}
        }}, 2500);

        let labelsVisible = false;
        function toggleAllLabels() {{
            labelsVisible = !labelsVisible;
            const updates = allNodes.map(n => ({{
                id: n.id,
                label: labelsVisible ? n.full_label : (n.deg >= 10 ? n.full_label : '')
            }}));
            nodesDataSet.update(updates);
            document.getElementById('labelBtn').innerText = labelsVisible ? '🏷️ Auto Labels' : '🏷️ Show All Labels';
        }}

        function filterCommunity(val) {{
            let filteredNodes;
            if (val === 'all') {{
                filteredNodes = allNodes.map(n => ({{ id: n.id, hidden: false }}));
            }} else {{
                const targetId = parseInt(val);
                filteredNodes = allNodes.map(n => ({{ id: n.id, hidden: n.comm_id !== targetId }}));
            }}
            nodesDataSet.update(filteredNodes);
            network.fit({{ animation: {{ duration: 500 }} }});
        }}

        let paperMode = false;
        function togglePaperMode() {{
            paperMode = !paperMode;
            document.body.classList.toggle('paper-white', paperMode);
            const edgeColor = paperMode ? 'rgba(71, 85, 105, 0.35)' : 'rgba(200, 200, 200, 0.3)';
            const fontColor = paperMode ? '#0f172a' : '#ffffff';
            const updates = allNodes.map(n => ({{
                id: n.id,
                font: {{ color: fontColor, size: 10, face: 'Inter, sans-serif' }}
            }}));
            nodesDataSet.update(updates);
            const edgeUpdates = allEdges.map(e => ({{
                id: e.id,
                color: {{ color: edgeColor, highlight: '#2563eb' }}
            }}));
            edgesDataSet.update(edgeUpdates);
            document.getElementById('paperBtn').innerText = paperMode ? '🌙 Dark Mode' : '📄 Paper Mode (White)';
        }}

        function toggleGuide() {{
            document.getElementById('guideDrawer').classList.toggle('open');
        }}

        function searchNode(query) {{
            if (!query) return;
            const q = query.toLowerCase().trim();
            const found = allNodes.find(n => n.full_label.toLowerCase().includes(q));
            if (found) {{
                network.focus(found.id, {{
                    scale: 1.8,
                    animation: {{ duration: 700, easingFunction: 'easeInOutQuad' }}
                }});
                network.selectNodes([found.id]);
                showInspector(found.id);
            }} else {{
                alert('Entity not found in isolated communities: "' + query + '"');
            }}
        }}

        function showInspector(nodeId) {{
            const n = allNodes.find(item => item.id === nodeId);
            if (!n) return;
            document.getElementById('inspName').innerText = n.full_label;
            const html = `
                <div style="margin-bottom: 8px;">
                    <span class="metric-pill">Community: <b>#${{n.comm_id}}</b></span>
                    <span class="metric-pill">Local Degree: <b>${{n.deg}}</b></span>
                </div>
                <div style="color: var(--text-sub); font-size: 11px; line-height: 1.4; margin-bottom: 6px;">
                    ${{n.role_desc}}
                </div>
                <div style="color: var(--text-dim); font-size: 10px;">
                    Entity completely decoupled from external hubs; retains dense relationships within Community #${{n.comm_id}}.
                </div>
            `;
            document.getElementById('inspContent').innerHTML = html;
            document.getElementById('inspectorCard').classList.add('visible');
        }}

        function closeInspector() {{
            document.getElementById('inspectorCard').classList.remove('visible');
        }}

        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                showInspector(params.nodes[0]);
            }} else {{
                closeInspector();
            }}
        }});

        function downloadSnapshot() {{
            network.fit({{ animation: false }});
            setTimeout(() => {{
                const canvas = container.getElementsByTagName('canvas')[0];
                if (!canvas) return;
                const link = document.createElement('a');
                link.download = 'Stage_3_3_Isolated_Communities.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            }}, 300);
        }}
    </script>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_out)


    def _generate_complete_topology_hairball(self, dg: nx.DiGraph, result: TopologyResult, output_path: Path, path_label: str, path_type: str):
        ug = dg.to_undirected()
        
        # 1. Precalculate 2D Force-Directed Layout deterministically in Python
        # Zero physics required in browser -> loads in <100ms with 0% CPU lag
        pos = nx.spring_layout(ug, k=0.10, iterations=35, seed=42)
        
        # 2. Partition Color Mapping
        comm_map = {}
        if result:
            for comm in result.communities:
                for n in comm.nodes:
                    comm_map[n] = comm.community_id
            hubs_set = set(result.global_hubs)
        else:
            hubs_set = set()
                    
        palette = [
            "#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f472b6",
            "#a78bfa", "#fb923c", "#4ade80", "#2dd4bf", "#e879f9",
            "#60a5fa", "#f87171"
        ]
        
        # 3. Build Nodes for Vis.js
        vis_nodes = []
        for node in dg.nodes():
            x, y = pos[node]
            deg = dg.degree(node)
            is_hub = node in hubs_set
            c_id = comm_map.get(node, -1)
            metrics = result.node_metrics.get(node) if result and hasattr(result, "node_metrics") else None
            cent = metrics.degree_centrality if metrics else 0.0
            p_i = metrics.participation_coefficient if metrics else 0.0
            betw = metrics.betweenness_centrality if metrics else 0.0

            if is_hub:
                node_color = "#ef4444"
                node_border = "#b91c1c"
                node_shape = "star"
                node_size = 28 + min(22, deg * 1.2)
            else:
                node_color = palette[c_id % len(palette)] if c_id >= 0 else "#94a3b8"
                node_border = "rgba(255, 255, 255, 0.4)"
                node_shape = "dot"
                node_size = 5 + min(12, deg * 0.7)

            label_text = node if (is_hub or deg >= 15) else ""
            cluster_name = f"Community {c_id}"

            tooltip = (
                f"<b>{node}</b><br>"
                f"Role: {'Global Hub' if is_hub else 'Entity'}<br>"
                f"Partition: {cluster_name if c_id >= 0 else 'Unassigned'}<br>"
                f"Total Degree: {deg} (In: {dg.in_degree(node)}, Out: {dg.out_degree(node)})<br>"
                f"P_i: {p_i:.4f}<br>"
                f"Betweenness: {betw:.4f}"
            )

            vis_nodes.append({
                "id": node,
                "label": label_text,
                "full_label": node,
                "x": round(float(x * 2200), 1),
                "y": round(float(y * 2200), 1),
                "size": round(node_size, 1),
                "shape": node_shape,
                "color": {
                    "background": node_color,
                    "border": node_border,
                    "highlight": {"background": "#ffffff", "border": node_color}
                },
                "borderWidth": 2 if is_hub else 1,
                "font": {"color": "#ffffff", "size": 13 if is_hub else 10, "face": "Inter, sans-serif"},
                "title": tooltip,
                "is_hub": is_hub,
                "degree": deg,
                "cluster_name": cluster_name if c_id >= 0 else "Unassigned",
                "p_i": round(p_i, 4),
                "betweenness": round(betw, 4),
                "in_degree": dg.in_degree(node),
                "out_degree": dg.out_degree(node)
            })

        # 4. Build Edges for Vis.js
        vis_edges = []
        for idx, (u, v, data) in enumerate(dg.edges(data=True)):
            is_inter_hub = (u in hubs_set and v in hubs_set)
            edge_color = "rgba(239, 68, 68, 0.65)" if is_inter_hub else "rgba(148, 163, 184, 0.16)"
            vis_edges.append({
                "id": f"e_{idx}",
                "from": u,
                "to": v,
                "arrows": "to",
                "title": data.get("predicate", ""),
                "color": {
                    "color": edge_color,
                    "highlight": "#ef4444" if is_inter_hub else "#38bdf8"
                },
                "width": 2.5 if is_inter_hub else 0.8,
                "smooth": False
            })

        total_nodes = len(dg.nodes())
        total_edges = len(dg.edges())
        total_hubs = len(hubs_set)
        total_clusters = len(result.communities) if result else 0
        graph_density = f"{nx.density(dg) * 100:.3f}%"

        nodes_json = json.dumps(vis_nodes)
        edges_json = json.dumps(vis_edges)

        template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>[__PATH_LABEL__] Complete Unstructured Topology Graph</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <style>
        :root {
            --bg-body: #0d0e12;
            --bg-header: rgba(20, 22, 31, 0.95);
            --bg-card: #1c202d;
            --border: #262c3e;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --text-dim: #64748b;
            --accent-blue: #38bdf8;
            --accent-red: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-body);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            transition: background-color 0.2s, color 0.2s;
        }
        header {
            background: var(--bg-header);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }
        .header-title h1 {
            font-size: 16px;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .header-sub {
            font-size: 11px;
            color: var(--text-sub);
            margin-top: 2px;
            display: flex;
            gap: 12px;
            align-items: center;
        }
        .stat-badge {
            background: rgba(56, 189, 248, 0.12);
            color: var(--accent-blue);
            font-weight: 600;
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 10px;
            border: 1px solid rgba(56, 189, 248, 0.25);
        }
        .header-actions {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .btn {
            background: #1e2230;
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        .btn:hover {
            background: #282f44;
            border-color: #3b82f6;
        }
        .btn-primary {
            background: #2563eb;
            border-color: #3b82f6;
        }
        .btn-primary:hover {
            background: #1d4ed8;
        }
        .btn-paper {
            background: #334155;
            border-color: #475569;
        }
        .btn-paper:hover {
            background: #475569;
        }
        .search-box {
            position: relative;
        }
        .search-input {
            background: #181c28;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            width: 190px;
            outline: none;
            transition: width 0.2s, border-color 0.2s;
        }
        .search-input:focus {
            width: 250px;
            border-color: #3b82f6;
        }
        .canvas-container {
            flex: 1;
            position: relative;
            width: 100%;
            height: calc(100vh - 58px);
        }
        #network {
            width: 100%;
            height: 100%;
        }
        .legend-card {
            position: absolute;
            top: 14px;
            right: 14px;
            background: rgba(20, 22, 31, 0.88);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 11px;
            color: var(--text-sub);
            z-index: 5;
            box-shadow: 0 6px 20px rgba(0,0,0,0.5);
            pointer-events: auto;
        }
        .legend-title {
            font-weight: 700;
            text-transform: uppercase;
            font-size: 9px;
            color: var(--text-dim);
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 7px;
            margin-bottom: 4px;
        }
        .legend-item:last-child { margin-bottom: 0; }
        .legend-icon {
            width: 12px;
            height: 12px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
        }
        .legend-line {
            width: 16px;
            height: 2px;
            border-radius: 1px;
        }
        .inspector-card {
            position: absolute;
            bottom: 14px;
            left: 14px;
            background: rgba(20, 22, 31, 0.92);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 11px;
            color: var(--text-sub);
            z-index: 5;
            box-shadow: 0 6px 24px rgba(0,0,0,0.5);
            max-width: 340px;
            display: none;
        }
        .inspector-card.visible { display: block; }
        .inspector-close {
            position: absolute;
            top: 8px;
            right: 8px;
            background: transparent;
            border: none;
            color: var(--text-dim);
            cursor: pointer;
            font-size: 14px;
        }
        .inspector-close:hover { color: var(--text-main); }
        .inspector-name {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 6px;
            padding-right: 18px;
        }
        .metric-pill {
            display: inline-block;
            background: #252b3d;
            color: #94a3b8;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            margin-right: 4px;
            margin-bottom: 4px;
        }
        .metric-pill b { color: #f8fafc; }

        /* PAPER PRINT MODE (WHITE BACKGROUND) */
        body.paper-white {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        body.paper-white header {
            background: #f8fafc !important;
            border-bottom: 1px solid #cbd5e1 !important;
            color: #0f172a !important;
        }
        body.paper-white .header-title h1 {
            color: #0f172a !important;
        }
        body.paper-white .header-sub {
            color: #475569 !important;
        }
        body.paper-white .btn {
            background: #f1f5f9;
            color: #0f172a;
            border-color: #cbd5e1;
        }
        body.paper-white .btn:hover {
            background: #e2e8f0;
        }
        body.paper-white .btn-primary {
            background: #2563eb;
            color: #ffffff;
        }
        body.paper-white .search-input {
            background: #f8fafc;
            border-color: #cbd5e1;
            color: #0f172a;
        }
        body.paper-white .legend-card {
            background: rgba(255, 255, 255, 0.92) !important;
            border-color: #cbd5e1 !important;
            color: #334155 !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1) !important;
        }
        body.paper-white .inspector-card {
            background: rgba(255, 255, 255, 0.95) !important;
            border-color: #cbd5e1 !important;
            color: #334155 !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.15) !important;
        }
        body.paper-white .inspector-name {
            color: #0f172a !important;
        }
        body.paper-white .metric-pill {
            background: #e2e8f0 !important;
            color: #475569 !important;
        }
        body.paper-white .metric-pill b {
            color: #0f172a !important;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <div>
                <h1>[__PATH_LABEL__] Complete Unstructured Topology Graph</h1>
                <div class="header-sub">
                    <span>Baseline Hairball (100% Unpruned)</span>
                    <span class="stat-badge">Entities: __TOTAL_NODES__</span>
                    <span class="stat-badge">Edges: __TOTAL_EDGES__</span>
                    <span class="stat-badge">Hubs: __TOTAL_HUBS__</span>
                    <span class="stat-badge">Partitions: __TOTAL_CLUSTERS__</span>
                    <span class="stat-badge">Density: __GRAPH_DENSITY__</span>
                </div>
            </div>
        </div>
        <div class="header-actions">
            <div class="search-box">
                <input type="text" class="search-input" id="nodeSearch" placeholder="Search entity in hairball..." onkeydown="if(event.key==='Enter') searchNode(this.value)">
            </div>
            <button class="btn" id="labelBtn" onclick="toggleAllLabels()">🏷️ Show All Labels</button>
            <button class="btn btn-paper" id="paperBtn" onclick="togglePaperMode()">📄 Paper Print Mode (White)</button>
            <button class="btn btn-primary" onclick="downloadSnapshot()">📷 Snapshot (PNG)</button>
            <button class="btn" onclick="network.fit({animation: {duration: 500}})">Fit</button>
        </div>
    </header>

    <div class="canvas-container">
        <div class="legend-card">
            <div class="legend-title">Graph Semantics</div>
            <div class="legend-item"><span class="legend-icon" style="color: #ef4444;">★</span> <span>Global Cross-Domain Hub</span></div>
            <div class="legend-item"><span class="legend-icon" style="color: #38bdf8;">●</span> <span>Modular Partition Entity</span></div>
            <div class="legend-item"><div class="legend-line" style="background: #ef4444;"></div> <span>Inter-Hub Backbone Edge</span></div>
            <div class="legend-item"><div class="legend-line" style="background: rgba(148, 163, 184, 0.25);"></div> <span>Volumetric Density Relationship</span></div>
        </div>

        <div class="inspector-card" id="inspectorCard">
            <button class="inspector-close" onclick="closeInspector()">×</button>
            <div class="inspector-name" id="inspName">Entity</div>
            <div id="inspContent"></div>
        </div>

        <div id="network"></div>
    </div>

    <script>
        const allNodes = __NODES_JSON__;
        const allEdges = __EDGES_JSON__;
        const pathType = "__PATH_TYPE__";

        const nodesDataSet = new vis.DataSet(allNodes);
        const edgesDataSet = new vis.DataSet(allEdges);
        const container = document.getElementById('network');
        const data = { nodes: nodesDataSet, edges: edgesDataSet };

        const options = {
            physics: { enabled: false }, // ZERO browser physics calculations (loads in <100ms)
            interaction: {
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true,
                navigationButtons: true
            },
            edges: {
                smooth: false // Straight lines for maximum rendering speed and clean density gradients
            }
        };

        const network = new vis.Network(container, data, options);

        let isWhiteMode = false;
        function togglePaperMode() {
            isWhiteMode = !isWhiteMode;
            document.body.classList.toggle('paper-white', isWhiteMode);
            document.getElementById('paperBtn').textContent = isWhiteMode ? '🌙 Presentation Dark Mode' : '📄 Paper Print Mode (White)';
            
            const fontColor = isWhiteMode ? '#0f172a' : '#ffffff';
            const edgeColor = isWhiteMode ? 'rgba(71, 85, 105, 0.25)' : 'rgba(148, 163, 184, 0.16)';
            
            const updatedEdges = allEdges.map(e => ({
                ...e,
                color: { color: e.color.color.includes('239, 68, 68') ? (isWhiteMode ? '#dc2626' : 'rgba(239, 68, 68, 0.65)') : edgeColor }
            }));
            edgesDataSet.clear();
            edgesDataSet.add(updatedEdges);

            const updatedNodes = allNodes.map(n => ({
                ...n,
                font: { ...n.font, color: fontColor }
            }));
            nodesDataSet.clear();
            nodesDataSet.add(updatedNodes);
        }

        let showAllLabels = false;
        function toggleAllLabels() {
            showAllLabels = !showAllLabels;
            document.getElementById('labelBtn').textContent = showAllLabels ? '🏷️ High-Degree Labels Only' : '🏷️ Show All Labels';
            const updatedNodes = allNodes.map(n => ({
                ...n,
                label: showAllLabels ? n.full_label : n.label
            }));
            nodesDataSet.clear();
            nodesDataSet.add(updatedNodes);
        }

        function downloadSnapshot() {
            const canvas = document.querySelector('#network canvas');
            if (!canvas) return;
            const link = document.createElement('a');
            link.download = `semantic_prism_unstructured_hairball_${pathType}.png`;
            link.href = canvas.toDataURL('image/png', 1.0);
            link.click();
        }

        function searchNode(query) {
            const q = query.toLowerCase().trim();
            if (!q) return;
            const found = allNodes.find(n => n.id.toLowerCase() === q || n.id.toLowerCase().includes(q));
            if (found) {
                selectAndFocus(found.id);
            }
        }

        function selectAndFocus(nodeId) {
            const node = allNodes.find(n => n.id === nodeId);
            if (!node) return;
            network.selectNodes([nodeId]);
            network.focus(nodeId, { scale: 1.4, animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
            showInspector(node);
        }

        function showInspector(node) {
            const card = document.getElementById('inspectorCard');
            document.getElementById('inspName').textContent = node.id;
            document.getElementById('inspContent').innerHTML = `
                <div style="margin-bottom: 6px;">
                    <span class="metric-pill">Role: <b>${node.is_hub ? 'Global Hub' : 'Entity'}</b></span>
                    <span class="metric-pill">Partition: <b>${node.cluster_name}</b></span>
                </div>
                <div>
                    <span class="metric-pill">Total Degree: <b>${node.degree}</b></span>
                    <span class="metric-pill">In: <b>${node.in_degree}</b></span>
                    <span class="metric-pill">Out: <b>${node.out_degree}</b></span>
                </div>
                <div style="margin-top: 4px;">
                    <span class="metric-pill">P_i: <b>${node.p_i}</b></span>
                    <span class="metric-pill">Betw: <b>${node.betweenness}</b></span>
                </div>
            `;
            card.classList.add('visible');
        }

        function closeInspector() {
            document.getElementById('inspectorCard').classList.remove('visible');
        }

        network.on('selectNode', function(params) {
            if (params.nodes && params.nodes.length > 0) {
                const node = allNodes.find(n => n.id === params.nodes[0]);
                if (node) showInspector(node);
            }
        });

        network.on('deselectNode', function() {
            closeInspector();
        });
    </script>
</body>
</html>"""

        rendered_html = (
            template
            .replace("__PATH_LABEL__", path_label)
            .replace("__PATH_TYPE__", path_type)
            .replace("__TOTAL_NODES__", f"{total_nodes:,}")
            .replace("__TOTAL_EDGES__", f"{total_edges:,}")
            .replace("__TOTAL_HUBS__", str(total_hubs))
            .replace("__TOTAL_CLUSTERS__", str(total_clusters))
            .replace("__GRAPH_DENSITY__", graph_density)
            .replace("__NODES_JSON__", nodes_json)
            .replace("__EDGES_JSON__", edges_json)
        )

        with open(output_path, "w") as f:
            f.write(rendered_html)


    @staticmethod
    def _extract_top_ngrams(raw_texts: List[str], top_n: int = 15) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extracts top unigrams and bigrams from raw text quotes, excluding standard stopwords."""
        import re
        stopwords = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
            "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
            "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
            "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
            "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
            "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
            "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", 
            "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", 
            "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", 
            "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
            "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", 
            "yourselves", "im", "ive", "id", "dont", "doesnt", "didnt", "cant", "couldnt", "wont", 
            "isnt", "arent", "wasnt", "werent", "really", "get", "got", "like", "feel", "even", 
            "much", "also", "one", "know", "going", "make", "time", "day", "days", "way", "well", "see",
            "back", "think", "still", "thing", "things", "something", "someone", "said", "say"
        }
        unigram_counts = Counter()
        bigram_counts = Counter()

        for text in raw_texts:
            if not text:
                continue
            words = re.findall(r'[a-zA-Z]{3,}', text.lower())
            filtered = [w for w in words if w not in stopwords]
            for w in filtered:
                unigram_counts[w] += 1
            for i in range(len(filtered) - 1):
                w1, w2 = filtered[i], filtered[i+1]
                if w1 != w2:
                    bigram_counts[f"{w1} {w2}"] += 1

        top_unigrams = [{"term": term, "count": count} for term, count in unigram_counts.most_common(top_n)]
        top_bigrams = [{"term": term, "count": count} for term, count in bigram_counts.most_common(top_n)]
        return top_unigrams, top_bigrams

    def _generate_llm_payload_gallery(self, dg: nx.DiGraph, result: TopologyResult, refined_triplets: List[dict], vis_dir: Path, path_label: str, path_type: str = "community"):
        agent_name = "leiden_schema_agent"
        
        synth_cfg = self.full_config.get('synthesis', {})
        min_cluster_size = synth_cfg.get('min_cluster_size', 12)
        max_triplets_cap = synth_cfg.get('max_triplets_per_target', 1800)
        
        # Resolve targets via TargetResolver (resolver.py)
        targets, _ = TargetResolver.qualify_and_resolve_targets(result.model_dump(), path_type, self.full_config)

        # Architectural tiers (Tier 1 Nucleus / Tier 2 Mechanics / Tier 3 Modifiers)
        tiers_enabled = self.config.get('architectural_tiers', {}).get('enabled', True)
        concept_embeddings = {}
        if tiers_enabled and targets:
            all_target_nodes = set()
            for target in targets:
                all_target_nodes.update(target.get("nodes", []))
            ref_cfg = getattr(self, 'full_config', {}).get('refinement', {}) if hasattr(self, 'full_config') else {}
            model_name = ref_cfg.get('embedding_model') or self.config.get('embedding_model')
            cache_dir = Path(ref_cfg.get('vector_cache_dir', 'outputs/02_refinement/cache'))
            concept_embeddings = TargetResolver._get_or_load_concept_embeddings(
                list(all_target_nodes),
                cache_dir=cache_dir,
                model_name=model_name
            )
            
        payload_data = []
        for i, target in enumerate(targets, start=1):
            target_type = target["type"]
            c_id = target["id"]
            target_nodes = set(target["nodes"])
            
            # Filter and cap triplets via TargetResolver (resolver.py)
            capped_triplets = TargetResolver.filter_and_cap_triplets(target_nodes, refined_triplets, result.model_dump(), max_triplets_cap)

            # Per-entity tier lookup: normalized name -> {tier, scores}
            tier_map = {}
            tier_counts = {}
            tier_rosters = {}
            if tiers_enabled:
                tiers_data = TargetResolver.compute_architectural_tiers(
                    target_nodes=target_nodes,
                    triplets=capped_triplets,
                    config=self.full_config,
                    concept_embeddings=concept_embeddings,
                    community_id=c_id
                )
                tier_counts = tiers_data.get("tier_counts", {})
                for tier_num, key in ((1, "tier_1_nucleus"), (2, "tier_2_mechanics"), (3, "tier_3_modifiers")):
                    roster = tiers_data.get(key, [])
                    tier_rosters[str(tier_num)] = [x["entity"] for x in roster]
                    for item in roster:
                        tier_map[str(item["entity"]).lower().strip()] = {
                            "tier": tier_num,
                            "S": item["composite_score"],
                            "P": item["prototypicality"],
                            "D": item["centrality"],
                            "G": item["fidelity"],
                            "deg_in": item["internal_degree"],
                            "deg_bound": item["boundary_degree"],
                            "quote": item["exemplar_quote"],
                            "demoted": bool(item.get("demoted_due_to_low_fidelity", False))
                        }
            
            associated_triplets = [
                {
                    "subject": str(t.get('subject', '')).strip(),
                    "predicate": str(t.get('predicate', '')).strip(),
                    "object": str(t.get('object', '')).strip(),
                    "rank_score": round(float(t.get('triplet_rank_score', 0.0)), 1),
                    "source_quote": str(t.get('source_quote', '')).strip()
                }
                for t in capped_triplets
            ]
            
            # Top 6 highest-scoring triplets
            top_triplets = associated_triplets[:6]

            # Extract raw quotes and compute unigrams & bigrams
            raw_quotes = [t.get('source_quote', '') for t in capped_triplets if t.get('source_quote')]
            top_unigrams, top_bigrams = self._extract_top_ngrams(raw_quotes, top_n=14)

            # Connected hubs (for graph node styling context)
            connected_hubs = [h for h in result.global_hubs if any(dg.has_edge(h, n) or dg.has_edge(n, h) for n in target["nodes"])]
                
            # Define output schema filename matching synthesizer.py naming style
            filename_suffix = f"community_{c_id}"
            filename = f"{i:02d}_{filename_suffix}.py"
            
            payload_data.append({
                "id": c_id,
                "type": target_type,
                "module_name": filename,
                "nodes": list(target_nodes),
                "triplet_count": len(associated_triplets),
                "triplets": associated_triplets,
                "top_triplets": top_triplets,
                "top_unigrams": top_unigrams,
                "top_bigrams": top_bigrams,
                "connected_hubs": connected_hubs,
                "tier_map": tier_map,
                "tier_counts": tier_counts,
                "tier_rosters": tier_rosters
            })
            
        payload_json = json.dumps(payload_data)
        hubs_json = json.dumps(result.global_hubs)
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>[{path_label}] Stage 4 LLM Payload Gallery</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <style>
        body {{ background-color: #121212; color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; }}
        h1 {{ margin-top: 0; color: #44aaff; font-size: 24px; }}
        .subtitle {{ color: #aaaaaa; margin-bottom: 20px; font-size: 14px; }}
        .container {{ display: flex; gap: 20px; height: 850px; }}
        .sidebar {{ width: 380px; background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 15px; overflow-y: auto; box-sizing: border-box; }}
        .canvas-container {{ flex: 1; background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; position: relative; }}
        #network {{ width: 100%; height: 100%; }}
        select {{ width: 100%; padding: 10px; background-color: #2a2a2a; color: white; border: 1px solid #444; border-radius: 4px; font-size: 15px; margin-bottom: 15px; cursor: pointer; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-right: 5px; background: #333; color: #44aaff; }}
        .card {{ background: #262626; padding: 12px; border-radius: 6px; margin-bottom: 12px; border-left: 4px solid #44aaff; }}
        .section-title {{ font-size: 12px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px; margin: 16px 0 8px 0; border-bottom: 1px solid #333; padding-bottom: 4px; }}
        .score-pill {{ display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 10px; font-weight: bold; background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; margin-right: 6px; }}
        .top-triplet-card {{ background: #22222a; border: 1px solid #333344; border-left: 3px solid #38bdf8; border-radius: 5px; padding: 7px 9px; margin-bottom: 6px; font-size: 12px; }}
        .tab-bar {{ display: flex; gap: 6px; margin: 8px 0 10px 0; }}
        .tab-btn {{ flex: 1; padding: 6px 10px; background: #262626; color: #888; border: 1px solid #444; border-radius: 4px; font-size: 12px; font-weight: 600; cursor: pointer; text-align: center; transition: all 0.2s; }}
        .tab-btn.active {{ background: #1e3a5f; color: #38bdf8; border-color: #38bdf8; }}
        .tab-content {{ max-height: 280px; overflow-y: auto; }}
        .pill-grid {{ display: flex; flex-wrap: wrap; gap: 5px; }}
        .ngram-pill {{ display: inline-flex; align-items: center; gap: 5px; background: #262626; border: 1px solid #3d3d3d; border-radius: 12px; padding: 3px 9px; font-size: 11.5px; color: #e2e8f0; }}
        .ngram-pill .count {{ font-size: 10px; color: #38bdf8; font-weight: bold; background: #1e293b; padding: 1px 5px; border-radius: 8px; }}
        code {{ color: #ff9933; font-family: monospace; }}
        .tier-legend {{ position: absolute; top: 12px; left: 12px; z-index: 10; background: rgba(15,23,42,0.88); border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; font-size: 12px; backdrop-filter: blur(6px); }}
        .tier-legend div {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; cursor: pointer; user-select: none; }}
        .tier-legend div.off {{ opacity: 0.35; }}
        .tier-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
        .tier-row {{ display: flex; gap: 6px; margin-bottom: 8px; }}
        .tier-stat {{ flex: 1; text-align: center; border-radius: 6px; padding: 6px 4px; font-size: 11px; background: #262626; border: 1px solid #333; }}
        .tier-stat b {{ display: block; font-size: 17px; }}
        .tier-chip {{ display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 10px; font-size: 11.5px; cursor: pointer; }}
        .tier-chip:hover {{ filter: brightness(1.25); }}
        .node-inspector {{ position: absolute; right: 12px; top: 12px; width: 300px; z-index: 10; background: rgba(15,23,42,0.94); border: 1px solid #334155; border-radius: 8px; padding: 12px; font-size: 12px; display: none; backdrop-filter: blur(6px); }}
        .node-inspector .metric {{ display: flex; justify-content: space-between; margin: 3px 0; }}
        .metric-bar {{ height: 4px; background: #1e293b; border-radius: 2px; margin-bottom: 5px; }}
        .metric-bar span {{ display: block; height: 100%; border-radius: 2px; background: #38bdf8; }}
    </style>
</head>
<body>
    <h1>[{path_label}] Stage 4 LLM Payload Gallery</h1>
    <div class="subtitle">
        Visualizing data sent to LLM for Stage 4 Pydantic schema synthesis.
        Current Configuration: <code>min_cluster_size: {min_cluster_size}</code> | <code>max_triplets_per_target: {max_triplets_cap}</code>.
    </div>
    
    <div class="container">
        <div class="sidebar">
            <label for="clusterSelect"><b>Select Partition Payload:</b></label>
            <select id="clusterSelect" onchange="loadCluster(this.value)"></select>
            
            <div id="payloadDetails"></div>
        </div>
        <div class="canvas-container">
            <div class="tier-legend" id="tierLegend">
                <div data-tier="1" onclick="toggleTier(1)"><span class="tier-dot" style="background:#f59e0b"></span>Tier 1 · Thematic Nucleus</div>
                <div data-tier="2" onclick="toggleTier(2)"><span class="tier-dot" style="background:#38bdf8"></span>Tier 2 · Relational Mechanics</div>
                <div data-tier="3" onclick="toggleTier(3)"><span class="tier-dot" style="background:#94a3b8"></span>Tier 3 · Contextual Modifiers</div>
                <div data-tier="0" onclick="toggleTier(0)"><span class="tier-dot" style="background:#475569"></span>Boundary Neighbor</div>
                <div data-tier="hub" onclick="toggleTier('hub')"><span class="tier-dot" style="background:#cc3333"></span>Global Hub Anchor</div>
            </div>
            <div class="node-inspector" id="nodeInspector"></div>
            <div id="network"></div>
        </div>
    </div>

    <script>
        const payloads = {payload_json};
        const globalHubs = new Set({hubs_json});
        const selectEl = document.getElementById('clusterSelect');
        const detailsEl = document.getElementById('payloadDetails');
        let network = null;
        let nodesDS = null;
        let currentPayload = null;
        const hiddenTiers = new Set();
        const TIER_STYLE = {{
            1: {{ color: '#f59e0b', size: 34, label: 'Tier 1 · Thematic Nucleus' }},
            2: {{ color: '#38bdf8', size: 22, label: 'Tier 2 · Relational Mechanics' }},
            3: {{ color: '#94a3b8', size: 14, label: 'Tier 3 · Contextual Modifiers' }}
        }};

        function escapeHtml(s) {{
            return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
        }}

        function toggleTier(t) {{
            const key = String(t);
            if (hiddenTiers.has(key)) hiddenTiers.delete(key); else hiddenTiers.add(key);
            document.querySelectorAll('#tierLegend div').forEach(d => d.classList.toggle('off', hiddenTiers.has(d.dataset.tier)));
            if (!nodesDS) return;
            nodesDS.update(nodesDS.get().map(n => ({{ id: n.id, hidden: hiddenTiers.has(String(n.tierKey)) }})));
        }}

        function metricRow(name, val) {{
            const pct = Math.max(0, Math.min(1, Number(val) || 0)) * 100;
            return `<div class="metric"><span>${{name}}</span><b>${{Number(val).toFixed(3)}}</b></div><div class="metric-bar"><span style="width:${{pct}}%"></span></div>`;
        }}

        function showInspector(nodeId) {{
            const el = document.getElementById('nodeInspector');
            if (!currentPayload || nodeId === undefined || nodeId === null) {{ el.style.display = 'none'; return; }}
            const info = (currentPayload.tier_map || {{}})[String(nodeId).toLowerCase().trim()];
            if (!info) {{
                el.innerHTML = `<b>${{escapeHtml(nodeId)}}</b><div style="color:#94a3b8;margin-top:4px;">Not a primary entity of this partition (boundary neighbor or hub). No tier score.</div>`;
                el.style.display = 'block';
                return;
            }}
            const st = TIER_STYLE[info.tier];
            el.innerHTML = `
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <b style="font-size:13px;">${{escapeHtml(nodeId)}}</b>
                    <span style="cursor:pointer;color:#64748b;" onclick="showInspector(null)">✕</span>
                </div>
                <span class="tier-chip" style="background:${{st.color}}22;border:1px solid ${{st.color}};color:${{st.color}};">${{st.label}}</span>
                ${{info.demoted ? '<span class="tier-chip" style="border:1px solid #ef4444;color:#fca5a5;">Demoted: low fidelity</span>' : ''}}
                <div style="margin-top:8px;">
                    ${{metricRow('Composite S<sub>u</sub>', info.S)}}
                    ${{metricRow('Prototypicality P<sub>u</sub>', info.P)}}
                    ${{metricRow('Centrality D<sub>u</sub>', info.D)}}
                    ${{metricRow('Fidelity Γ<sub>u</sub>', info.G)}}
                    <div class="metric"><span>Internal / Boundary degree</span><b>${{info.deg_in}} / ${{info.deg_bound}}</b></div>
                </div>
                ${{info.quote ? `<div style="margin-top:8px;color:#cbd5e1;font-style:italic;border-left:2px solid ${{st.color}};padding-left:6px;">"${{escapeHtml(info.quote)}}"</div>` : ''}}
            `;
            el.style.display = 'block';
        }}

        function focusNode(name) {{
            if (!network) return;
            const target = nodesDS.get().find(n => String(n.id).toLowerCase().trim() === String(name).toLowerCase().trim());
            if (!target) return;
            network.selectNodes([target.id]);
            network.focus(target.id, {{ scale: 1.2, animation: {{ duration: 500 }} }});
            showInspector(target.id);
        }}

        payloads.forEach((p, idx) => {{
            const opt = document.createElement('option');
            opt.value = idx;
            const label = `${{p.type === 'community' ? 'Community' : 'Structural Cluster'}} ${{p.id}} (${{p.nodes.length}} entities)`;
            opt.textContent = label;
            selectEl.appendChild(opt);
        }});

        function switchNgramTab(tab) {{
            const uniTab = document.getElementById('unigramsTab');
            const biTab = document.getElementById('bigramsTab');
            const uniBtn = document.getElementById('tabUnigramsBtn');
            const biBtn = document.getElementById('tabBigramsBtn');
            if (!uniTab || !biTab) return;
            if (tab === 'unigrams') {{
                uniTab.style.display = 'block';
                biTab.style.display = 'none';
                uniBtn.classList.add('active');
                biBtn.classList.remove('active');
            }} else {{
                uniTab.style.display = 'none';
                biTab.style.display = 'block';
                uniBtn.classList.remove('active');
                biBtn.classList.add('active');
            }}
        }}

        function loadCluster(idx) {{
            if (idx === "" || idx === null || idx === undefined) return;
            const payload = payloads[idx];
            
            // Format Top Triplets HTML
            const topTripletsHtml = payload.top_triplets && payload.top_triplets.length > 0
                ? payload.top_triplets.map(t => `
                    <div class="top-triplet-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                            <span class="score-pill">Score ${{t.rank_score}}</span>
                        </div>
                        <b>${{t.subject}}</b> <i style="color:#94a3b8;">--[${{t.predicate}}]--></i> <b>${{t.object}}</b>
                    </div>
                `).join('')
                : '<i style="color:#777; font-size:12px;">No top triplets available</i>';

            // Format Unigrams HTML
            const unigramsHtml = payload.top_unigrams && payload.top_unigrams.length > 0
                ? `<div class="pill-grid">${{payload.top_unigrams.map(u => `<span class="ngram-pill"><span>${{u.term}}</span><span class="count">${{u.count}}</span></span>`).join('')}}</div>`
                : '<i style="color:#777; font-size:12px;">No unigrams found</i>';

            // Format Bigrams HTML
            const bigramsHtml = payload.top_bigrams && payload.top_bigrams.length > 0
                ? `<div class="pill-grid">${{payload.top_bigrams.map(b => `<span class="ngram-pill"><span>${{b.term}}</span><span class="count">${{b.count}}</span></span>`).join('')}}</div>`
                : '<i style="color:#777; font-size:12px;">No bigrams found</i>';

            // Format Architectural Tier HTML
            currentPayload = payload;
            const tc = payload.tier_counts || {{}};
            const rosters = payload.tier_rosters || {{}};
            const chipList = (tier, limit) => (rosters[tier] || []).slice(0, limit).map(e =>
                `<span class="tier-chip" style="background:${{TIER_STYLE[tier].color}}22;border:1px solid ${{TIER_STYLE[tier].color}}66;color:${{TIER_STYLE[tier].color}};" onclick='focusNode(${{JSON.stringify(e).replace(/'/g, "&#39;")}})'>${{escapeHtml(e)}}</span>`
            ).join('');
            const tiersHtml = Object.keys(payload.tier_map || {{}}).length > 0 ? `
                <div class="section-title">Architectural Tiers:</div>
                <div class="tier-row">
                    <div class="tier-stat" style="border-color:#f59e0b66;"><b style="color:#f59e0b">${{tc.tier_1_nucleus || 0}}</b>Nucleus</div>
                    <div class="tier-stat" style="border-color:#38bdf866;"><b style="color:#38bdf8">${{tc.tier_2_mechanics || 0}}</b>Mechanics</div>
                    <div class="tier-stat" style="border-color:#94a3b866;"><b style="color:#94a3b8">${{tc.tier_3_modifiers || 0}}</b>Modifiers</div>
                </div>
                <div style="font-size:11px;color:#94a3b8;margin:4px 0;">Tier 1 · Thematic Nucleus</div>
                <div style="margin-bottom:6px;">${{chipList('1', 8)}}</div>
                <div style="font-size:11px;color:#94a3b8;margin:4px 0;">Tier 2 · Top Relational Mechanics</div>
                <div style="margin-bottom:6px;">${{chipList('2', 12)}}</div>
                <div style="font-size:11px;color:#94a3b8;margin:4px 0;">Tier 3 · Lowest-Scoring Modifiers</div>
                <div style="margin-bottom:6px;">${{(rosters['3'] || []).slice(-8).map(e => `<span class="tier-chip" style="background:#94a3b822;border:1px solid #94a3b866;color:#94a3b8;" onclick='focusNode(${{JSON.stringify(e).replace(/'/g, "&#39;")}})'>${{escapeHtml(e)}}</span>`).join('')}}</div>
            ` : '';

            // Render Details Panel
            detailsEl.innerHTML = `
                <div class="card" style="border-left: 4px solid #44aaff;">
                    <span class="badge" style="background: #44aaff; color: white;">Target Type</span> <b>${{payload.type.toUpperCase()}}</b><br>
                    <span class="badge">LLM Agent</span> <code>{agent_name}</code><br>
                    <span class="badge">Output Schema</span> <code>${{payload.module_name}}</code><br>
                    <span class="badge">Primary Entities</span> <b>${{payload.nodes.length}}</b><br>
                    <span class="badge">Total Triplets</span> <b>${{payload.triplet_count}} / {max_triplets_cap} (cap)</b>
                </div>

                ${{tiersHtml}}

                <div class="section-title">Top Ranked Triplets (Scored Backbone):</div>
                <div style="margin-bottom: 14px;">
                    ${{topTripletsHtml}}
                </div>

                <div class="section-title">Raw Text Vocabulary:</div>
                <div class="tab-bar">
                    <button class="tab-btn active" id="tabUnigramsBtn" onclick="switchNgramTab('unigrams')">Unigrams</button>
                    <button class="tab-btn" id="tabBigramsBtn" onclick="switchNgramTab('bigrams')">Bigrams</button>
                </div>
                <div id="unigramsTab" class="tab-content">${{unigramsHtml}}</div>
                <div id="bigramsTab" class="tab-content" style="display: none;">${{bigramsHtml}}</div>
            `;

            // Draw Subgraph
            const nodes = [];
            const edges = [];
            const addedNormalized = new Map();
            const primaryNodesSet = new Set(payload.nodes.map(x => String(x).toLowerCase().trim()));
            const hubsSet = new Set(Array.from(globalHubs).map(x => String(x).toLowerCase().trim()));

            // 1. Add all nodes involved in the triplets to guarantee everything is visualized
            payload.triplets.forEach(t => {{
                [t.subject, t.object].forEach(n => {{
                    const norm = String(n).toLowerCase().trim();
                    if (!addedNormalized.has(norm)) {{
                        addedNormalized.set(norm, n);
                        
                        const isPrimary = primaryNodesSet.has(norm);
                        const isHub = hubsSet.has(norm);
                        
                        // Set colors and shapes based on roles / architectural tier
                        const tierInfo = (payload.tier_map || {{}})[norm];
                        let color = '#44aaff'; // default: target spoke node
                        let shape = 'dot';
                        let size = 25;
                        let title = 'Entity';
                        let tierKey = 0;
                        
                        if (isHub) {{
                            shape = 'star';
                            size = 35;
                            color = '#cc3333'; // dark red for context hub anchor
                            title = 'Context Hub Anchor';
                            tierKey = 'hub';
                        }} else if (!isPrimary) {{
                            color = '#475569'; // muted slate for boundary/neighbor nodes
                            size = 12;
                            title = 'Neighbor Entity (Context)';
                        }} else if (tierInfo) {{
                            const st = TIER_STYLE[tierInfo.tier];
                            color = st.color;
                            size = st.size;
                            tierKey = tierInfo.tier;
                            title = `${{st.label}} | S=${{tierInfo.S}}`;
                            if (tierInfo.tier === 1) shape = 'diamond';
                        }} else {{
                            title = 'Primary Target Entity';
                        }}
                        
                        nodes.push({{ id: n, label: n, color: color, shape: shape, size: size, title: title, tierKey: tierKey, hidden: hiddenTiers.has(String(tierKey)),
                                      font: {{ color: '#e2e8f0', size: tierKey === 1 ? 18 : 12 }} }});
                    }}
                }});
            }});

            // 2. Fallback: Add genuine primary target nodes only if they had no associated triplets at all
            payload.nodes.forEach(n => {{
                const norm = String(n).toLowerCase().trim();
                if (!addedNormalized.has(norm)) {{
                    addedNormalized.set(norm, n);
                    const tierInfo = (payload.tier_map || {{}})[norm];
                    const st = tierInfo ? TIER_STYLE[tierInfo.tier] : null;
                    const tierKey = tierInfo ? tierInfo.tier : 0;
                    nodes.push({{ id: n, label: n, color: st ? st.color : '#44aaff', shape: tierKey === 1 ? 'diamond' : 'dot', size: st ? st.size : 25,
                                  title: st ? st.label : 'Primary Target Entity', tierKey: tierKey, hidden: hiddenTiers.has(String(tierKey)),
                                  font: {{ color: '#e2e8f0' }} }});
                }}
            }});

            // 3. Add edges from triplets using the exact added node IDs
            payload.triplets.forEach(t => {{
                const u = addedNormalized.get(String(t.subject).toLowerCase().trim()) || t.subject;
                const v = addedNormalized.get(String(t.object).toLowerCase().trim()) || t.object;
                edges.push({{ from: u, to: v, title: t.predicate, arrows: 'to', color: 'rgba(200,200,200,0.5)' }});
            }});

            const container = document.getElementById('network');
            nodesDS = new vis.DataSet(nodes);
            const data = {{ nodes: nodesDS, edges: new vis.DataSet(edges) }};
            const options = {{ physics: {{ enabled: true, solver: 'forceAtlas2Based' }} }};
            
            if (network) network.destroy();
            showInspector(null);
            network = new vis.Network(container, data, options);
            network.on('click', params => showInspector(params.nodes.length ? params.nodes[0] : null));
        }}

        if (payloads.length > 0) {{
            selectEl.value = 0;
            loadCluster(0);
        }} else {{
            const defaultOpt = document.createElement('option');
            defaultOpt.value = "";
            defaultOpt.textContent = "-- No Partitions Available --";
            selectEl.appendChild(defaultOpt);
            detailsEl.innerHTML = '<div style="color: #888888; font-style: italic; padding: 20px 0;">No partition payloads available.</div>';
        }}
    </script>
</body>
</html>"""
        with open(vis_dir / "stage_04_llm_payload_gallery.html", "w") as f:
            f.write(html_content)

    def _generate_collapsed_modules_graph(self, dg: nx.DiGraph, result: TopologyResult, vis_dir: Path, path_label: str = "Path 1: Community Workflow Path", path_type: str = "community"):
        clusters = result.communities
        cluster_label = "Community"
        css = self._get_shared_interactive_css()
        vis_dir.mkdir(parents=True, exist_ok=True)

        enriched_nodes = []
        hubs_set = set(result.global_hubs)
        hubs_count = 0
        modules_count = 0

        # Add Global Hubs as Star Nodes
        for h in hubs_set:
            metrics = result.node_metrics.get(h)
            cent = metrics.degree_centrality if metrics else 0.0
            hubs_count += 1
            role_desc = "Global Hub Router — Cross-cutting entity connecting across multiple discrete modules"
            tooltip = f"<b>{html.escape(str(h))}</b><br>Global Hub Router<br>Degree Centrality: {cent:.4f}"

            enriched_nodes.append({
                "id": str(h),
                "label": f"HUB: {h}",
                "full_label": str(h),
                "shape": "star",
                "size": 40,
                "color": "#ef4444",
                "borderWidth": 2,
                "font": {"color": "#ffffff", "size": 12, "face": "Inter, sans-serif"},
                "title": tooltip,
                "is_hub": True,
                "raw_title": f"Degree Centrality: {cent:.4f}",
                "role_desc": role_desc
            })

        # Add Clusters as Collapsed Box Nodes
        enriched_edges = []
        edge_idx = 0
        for cluster in clusters:
            c_id = getattr(cluster, 'community_id', getattr(cluster, 'cluster_id', 0))
            nodes = cluster.nodes
            mod_id = f"Module_{c_id}"
            label = f"{cluster_label} {c_id}\n({len(nodes)} entities)"
            raw_title = f"{cluster_label} {c_id} Members:\n" + "\n".join(nodes[:10]) + (f"\n... and {len(nodes)-10} more" if len(nodes) > 10 else "")
            tooltip = f"<b>{cluster_label} #{c_id} ({len(nodes)} entities)</b><br>" + "<br>".join(html.escape(n) for n in nodes[:10]) + (f"<br>... and {len(nodes)-10} more" if len(nodes) > 10 else "")
            modules_count += 1
            role_desc = f"Collapsed Module Block — Aggregates {len(nodes)} cohesive entity members"

            enriched_nodes.append({
                "id": mod_id,
                "label": label,
                "full_label": f"{cluster_label} {c_id} ({len(nodes)} entities)",
                "shape": "box",
                "size": 32,
                "color": "#38bdf8",
                "borderWidth": 2,
                "font": {"color": "#ffffff", "size": 12, "face": "Inter, sans-serif"},
                "title": tooltip,
                "is_hub": False,
                "raw_title": raw_title,
                "role_desc": role_desc
            })

            # Connect Hubs to Modules
            for h in hubs_set:
                if any(dg.has_edge(h, n) or dg.has_edge(n, h) for n in nodes):
                    enriched_edges.append({
                        "id": f"e_{edge_idx}",
                        "from": str(h),
                        "to": mod_id,
                        "color": "rgba(255, 100, 100, 0.7)",
                        "width": 2
                    })
                    edge_idx += 1

        total_nodes = len(enriched_nodes)
        total_edges = len(enriched_edges)
        nodes_json = json.dumps(enriched_nodes)
        edges_json = json.dumps(enriched_edges)

        html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stage 3.3: Macro Architecture — Collapsed Module Architecture Diagram</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
    <style>
{css}
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <div>
                <h1>Stage 3.3: Macro Architecture — Collapsed Module Architecture Diagram</h1>
                <div class="header-sub">
                    <span>Executive Modular Decomposition: High-Level Partition Blocks & Global Hub Routers</span>
                    <span class="stat-badge">Path: {path_label}</span>
                    <span class="stat-badge stat-badge-blue">Modules (■): {modules_count}</span>
                    <span class="stat-badge stat-badge-danger">Global Hubs (★): {hubs_count}</span>
                    <span class="stat-badge stat-badge-warning">Cross-Module Bridges: {total_edges}</span>
                </div>
            </div>
        </div>
        <div class="header-actions">
            <div class="search-box">
                <input type="text" class="search-input" id="nodeSearch" placeholder="Search module or hub..." onkeydown="if(event.key==='Enter') searchNode(this.value)">
            </div>
            <select class="filter-select" id="archFilter" onchange="filterArch(this.value)">
                <option value="all">All Components ({total_nodes})</option>
                <option value="modules">Modules Only ({modules_count})</option>
                <option value="hubs">Global Hubs Only ({hubs_count})</option>
            </select>
            <button class="btn btn-paper" id="paperBtn" onclick="togglePaperMode()">📄 Paper Mode (White)</button>
            <button class="btn btn-guide" onclick="toggleGuide()">ℹ️ Guide & Reference</button>
            <button class="btn btn-primary" onclick="downloadSnapshot()">📷 Snapshot</button>
            <button class="btn" onclick="network.fit({{animation: {{duration: 500}}}})">Fit</button>
        </div>
    </header>

    <div class="canvas-container">
        <div id="loadingBar">
            <div class="loader-card">
                <div class="loader-title">Stabilizing ForceAtlas2 Physics Layout...</div>
                <div class="progress-track">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-label" id="progressText">Stabilizing 0%</div>
            </div>
        </div>

        <div class="legend-card" id="legendCard">
            <div class="legend-header">
                <span class="legend-title">Topological Legend</span>
                <span style="font-size: 10px; color: var(--text-dim);">Macro Schema</span>
            </div>
            <div class="legend-section">
                <div class="legend-section-title">Architectural Components</div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: #38bdf8; font-size: 13px;">■</span>
                    <span><b>Collapsed Module</b> — Aggregated topical cluster with member count</span>
                </div>
                <div class="legend-item">
                    <span class="legend-icon" style="color: #ef4444;">★</span>
                    <span><b>Global Hub Router</b> — Connective entity linking multiple modules</span>
                </div>
            </div>
            <div class="legend-section">
                <div class="legend-section-title">Interfaces & Links</div>
                <div class="legend-item">
                    <div class="legend-line" style="background: rgba(239, 68, 68, 0.75);"></div>
                    <span><b>Cross-Module Bridge</b> — Direct interface from Hub to Module</span>
                </div>
            </div>
        </div>

        <div class="inspector-card" id="inspectorCard">
            <button class="inspector-close" onclick="closeInspector()">×</button>
            <div class="inspector-name" id="inspName">Component</div>
            <div id="inspContent"></div>
        </div>

        <div class="guide-drawer" id="guideDrawer">
            <div class="guide-header">
                <h2>📖 Collapsed Architecture Guide</h2>
                <button class="guide-close" onclick="toggleGuide()">×</button>
            </div>
            <div class="guide-content">
                <div class="guide-card">
                    <div class="guide-card-title">🎯 Visualization Objective</div>
                    <div>
                        Condenses thousands of granular entity triplets into an executive-level architectural blueprint. Shows how individual communities collapse into functional modules and how global hubs coordinate inter-module communication.
                    </div>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">🔍 Reading the Router Grid</div>
                    <ul class="takeaway-list">
                        <li><b>Central Core Hubs:</b> Ubiquitous entities link into almost every module, acting as system-wide syntactic and semantic routers.</li>
                        <li><b>Targeted Hubs:</b> Hubs that link to only 2–3 modules serve as specialized thematic bridges.</li>
                        <li><b>Insulated Modules:</b> Boxes on the periphery with few incident hub links represent strictly encapsulated, self-contained functional domains.</li>
                    </ul>
                </div>

                <div class="guide-card">
                    <div class="guide-card-title">💡 Interactive Member Inspection</div>
                    <div>
                        Click on any Module box to view its full title, entity count, and top constituent member entities in the Inspector panel.
                    </div>
                </div>
            </div>
        </div>

        <div id="network"></div>
    </div>

    <script>
        const allNodes = {nodes_json};
        const allEdges = {edges_json};

        const nodesDataSet = new vis.DataSet(allNodes);
        const edgesDataSet = new vis.DataSet(allEdges);

        const container = document.getElementById('network');
        const data = {{ nodes: nodesDataSet, edges: edgesDataSet }};

        const options = {{
            physics: {{
                enabled: true,
                forceAtlas2Based: {{
                    gravitationalConstant: -120,
                    centralGravity: 0.01,
                    springLength: 220,
                    springConstant: 0.08
                }},
                solver: 'forceAtlas2Based',
                stabilization: {{ enabled: true, iterations: 150, fit: true }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 80,
                hideEdgesOnDrag: true,
                zoomView: true,
                dragView: true
            }}
        }};

        const network = new vis.Network(container, data, options);

        network.on('stabilizationProgress', function(params) {{
            const pct = Math.round((params.iterations / params.total) * 100);
            document.getElementById('progressFill').style.width = pct + '%';
            document.getElementById('progressText').innerText = 'Stabilizing Layout: ' + pct + '%';
        }});

        network.on('stabilizationIterationsDone', function() {{
            document.getElementById('loadingBar').classList.add('hidden');
        }});

        setTimeout(() => {{
            const bar = document.getElementById('loadingBar');
            if (bar && !bar.classList.contains('hidden')) {{
                bar.classList.add('hidden');
            }}
        }}, 2500);

        function filterArch(val) {{
            let filteredNodes;
            if (val === 'modules') {{
                filteredNodes = allNodes.map(n => ({{ id: n.id, hidden: n.is_hub }}));
            }} else if (val === 'hubs') {{
                filteredNodes = allNodes.map(n => ({{ id: n.id, hidden: !n.is_hub }}));
            }} else {{
                filteredNodes = allNodes.map(n => ({{ id: n.id, hidden: false }}));
            }}
            nodesDataSet.update(filteredNodes);
            network.fit({{ animation: {{ duration: 500 }} }});
        }}

        let paperMode = false;
        function togglePaperMode() {{
            paperMode = !paperMode;
            document.body.classList.toggle('paper-white', paperMode);
            const edgeColor = paperMode ? 'rgba(239, 68, 68, 0.85)' : 'rgba(255, 100, 100, 0.7)';
            const fontColor = paperMode ? '#0f172a' : '#ffffff';
            const updates = allNodes.map(n => ({{
                id: n.id,
                font: {{ color: fontColor, size: 12, face: 'Inter, sans-serif' }}
            }}));
            nodesDataSet.update(updates);
            const edgeUpdates = allEdges.map(e => ({{
                id: e.id,
                color: edgeColor
            }}));
            edgesDataSet.update(edgeUpdates);
            document.getElementById('paperBtn').innerText = paperMode ? '🌙 Dark Mode' : '📄 Paper Mode (White)';
        }}

        function toggleGuide() {{
            document.getElementById('guideDrawer').classList.toggle('open');
        }}

        function searchNode(query) {{
            if (!query) return;
            const q = query.toLowerCase().trim();
            const found = allNodes.find(n => n.full_label.toLowerCase().includes(q) || (n.raw_title && n.raw_title.toLowerCase().includes(q)));
            if (found) {{
                network.focus(found.id, {{
                    scale: 1.6,
                    animation: {{ duration: 700, easingFunction: 'easeInOutQuad' }}
                }});
                network.selectNodes([found.id]);
                showInspector(found.id);
            }} else {{
                alert('Component not found: "' + query + '"');
            }}
        }}

        function showInspector(nodeId) {{
            const n = allNodes.find(item => item.id === nodeId);
            if (!n) return;
            document.getElementById('inspName').innerText = n.full_label;
            const lines = (n.raw_title || '').split('\\n');
            const detailsHtml = lines.map(l => `<div style="margin-bottom: 2px;">${{l}}</div>`).join('');
            const html = `
                <div style="margin-bottom: 8px;">
                    <span class="metric-pill">Type: <b>${{n.is_hub ? 'Global Hub Router' : 'Collapsed Module'}}</b></span>
                </div>
                <div style="color: var(--text-sub); font-size: 11px; line-height: 1.4; margin-bottom: 8px;">
                    ${{n.role_desc}}
                </div>
                <div style="background: rgba(0,0,0,0.25); border: 1px solid var(--border); border-radius: 4px; padding: 8px; font-size: 10px; color: var(--text-dim); max-height: 140px; overflow-y: auto;">
                    ${{detailsHtml}}
                </div>
            `;
            document.getElementById('inspContent').innerHTML = html;
            document.getElementById('inspectorCard').classList.add('visible');
        }}

        function closeInspector() {{
            document.getElementById('inspectorCard').classList.remove('visible');
        }}

        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                showInspector(params.nodes[0]);
            }} else {{
                closeInspector();
            }}
        }});

        function downloadSnapshot() {{
            network.fit({{ animation: false }});
            setTimeout(() => {{
                const canvas = container.getElementsByTagName('canvas')[0];
                if (!canvas) return;
                const link = document.createElement('a');
                link.download = 'Stage_3_3_Collapsed_Module_Architecture.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
            }}, 300);
        }}
    </script>
</body>
</html>"""

        with open(vis_dir / "collapsed_module_architecture.html", "w", encoding="utf-8") as f:
            f.write(html_out)

