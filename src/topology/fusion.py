import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from pyvis.network import Network
from src.topology.schemas import TopologyResult, UnifiedClusterAlignment, UnifiedTopologyResult


class TopologyFusionPipeline:
    """
    Dual-Path Jaccard Isomorphic Cluster Alignment & Fusion Engine.
    Executes pairwise Jaccard similarity evaluation J(W_i, K_j), classifies isomorphic fusion (J >= 0.70)
    and relational composition (0.20 <= J < 0.70), and exports unified partitions and HTML visualizations.
    """
    def __init__(self, config: Dict[str, Any]):
        self.full_config = config
        self.config = config.get('topology', {})
        self.fusion_cfg = self.config.get('fusion', self.config.get('alignment', {}))
        self.enable_unification = self.fusion_cfg.get('enable_unification', True)
        self.fusion_threshold = self.fusion_cfg.get('fusion_threshold', 0.70)
        self.composition_threshold = self.fusion_cfg.get('composition_threshold', 0.20)

    def align_and_unify_paths(self, comm_result: TopologyResult, emb_result: TopologyResult) -> UnifiedTopologyResult:
        print("   ==> Executing Dual-Path Jaccard Fusion & Unification (fusion.py)...")
        
        communities = comm_result.communities
        structural_clusters = emb_result.structural_clusters
        
        alignments = []
        fused_clusters = []
        compositional_links = []
        
        matched_comm_ids = set()
        matched_sc_ids = set()
        alignment_counter = 0
        
        for comm in communities:
            c_nodes = set(comm.nodes)
            c_id = comm.community_id
            
            for sc in structural_clusters:
                s_nodes = set(sc.nodes)
                s_id = sc.cluster_id
                
                if not c_nodes or not s_nodes:
                    continue
                    
                intersection = c_nodes.intersection(s_nodes)
                union = c_nodes.union(s_nodes)
                
                jaccard = len(intersection) / len(union) if union else 0.0
                
                if jaccard >= self.composition_threshold:
                    alignment_counter += 1
                    if jaccard >= self.fusion_threshold:
                        alignment_type = "isomorphic_fusion"
                        matched_comm_ids.add(c_id)
                        matched_sc_ids.add(s_id)
                        
                        fused_clusters.append({
                            "fused_cluster_id": len(fused_clusters),
                            "community_id": c_id,
                            "structural_cluster_id": s_id,
                            "jaccard_score": float(jaccard),
                            "nodes": sorted(list(union))
                        })
                    else:
                        alignment_type = "relational_composition"
                        compositional_links.append({
                            "composition_id": len(compositional_links),
                            "workflow_community_id": c_id,
                            "category_cluster_id": s_id,
                            "jaccard_score": float(jaccard),
                            "intersection_nodes": sorted(list(intersection))
                        })
                        
                    alignments.append(UnifiedClusterAlignment(
                        alignment_id=alignment_counter,
                        community_id=c_id,
                        structural_cluster_id=s_id,
                        jaccard_score=float(jaccard),
                        alignment_type=alignment_type,
                        intersection_nodes=sorted(list(intersection)),
                        union_nodes=sorted(list(union))
                    ))
                    
        all_comm_ids = set(c.community_id for c in communities)
        all_sc_ids = set(sc.cluster_id for sc in structural_clusters)
        
        unmatched_comm = sorted(list(all_comm_ids - matched_comm_ids))
        unmatched_sc = sorted(list(all_sc_ids - matched_sc_ids))
        
        combined_hubs = sorted(list(set(comm_result.global_hubs + emb_result.global_hubs)))
        combined_orphans = sorted(list(set(comm_result.orphans + emb_result.orphans)))
        
        unified_result = UnifiedTopologyResult(
            alignments=alignments,
            fused_clusters=fused_clusters,
            compositional_links=compositional_links,
            unmatched_communities=unmatched_comm,
            unmatched_structural_clusters=unmatched_sc,
            global_hubs=combined_hubs,
            orphans=combined_orphans,
            node_metrics=comm_result.node_metrics
        )
        
        # Save unified output
        out_dir = Path("outputs/03_topology/unified")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "topology_partitions.json", "w") as f:
            json.dump(unified_result.model_dump(), f, indent=2)
            
        print(f"      -> Discovered {len(fused_clusters)} Isomorphic Fused Clusters (J >= {self.fusion_threshold:.2f})")
        print(f"      -> Discovered {len(compositional_links)} Relational Composition Links ({self.composition_threshold:.2f} <= J < {self.fusion_threshold:.2f})")
        print(f"      -> Saved Unified Topology Partitions: {out_dir / 'topology_partitions.json'}")
        
        # Generate HTML Visualizations
        vis_dir = Path("outputs/visuals/unified")
        self._generate_unified_alignment_visuals(unified_result, comm_result, emb_result, vis_dir)
        
        return unified_result

    def _generate_unified_alignment_visuals(self, unified: UnifiedTopologyResult, comm_result: TopologyResult, emb_result: TopologyResult, vis_dir: Path):
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Interactive Bipartite Heatmap & Alignment Chart (Plotly)
        alignments_data = [
            {
                "comm_id": a.community_id,
                "sc_id": a.structural_cluster_id,
                "jaccard": round(a.jaccard_score, 4),
                "type": a.alignment_type,
                "overlap_count": len(a.intersection_nodes),
                "union_count": len(a.union_nodes)
            }
            for a in unified.alignments
        ]
        
        alignments_json = json.dumps(alignments_data)
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SemanticPrism: Dual-Path Jaccard Fusion & Alignment Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body {{ background-color: #1a1a2e; color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; }}
        h1 {{ color: #00d2ff; text-align: center; margin-bottom: 5px; }}
        p.subtitle {{ text-align: center; color: #aaa; margin-top: 0; margin-bottom: 25px; }}
        .stats-container {{ display: flex; justify-content: space-around; margin-bottom: 25px; gap: 15px; }}
        .stat-card {{ background: #16213e; border-radius: 8px; padding: 15px 25px; flex: 1; text-align: center; border: 1px solid #0f3460; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .stat-value {{ font-size: 28px; font-weight: bold; color: #e94560; margin-top: 5px; }}
        .stat-label {{ font-size: 13px; color: #8d99ae; text-transform: uppercase; letter-spacing: 1px; }}
        #heatmap {{ width: 100%; height: 600px; background: #16213e; border-radius: 8px; border: 1px solid #0f3460; margin-bottom: 25px; }}
    </style>
</head>
<body>
    <h1>SemanticPrism Dual-Path Jaccard Fusion Dashboard</h1>
    <p class="subtitle">Isomorphic Cluster Alignment & Relational Composition Synthesis</p>
    
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-label">Isomorphic Fused Clusters (J &ge; {self.fusion_threshold:.2f})</div>
            <div class="stat-value" style="color: #00f5d4;">{len(unified.fused_clusters)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Relational Compositions ({self.composition_threshold:.2f} &le; J &lt; {self.fusion_threshold:.2f})</div>
            <div class="stat-value" style="color: #fee440;">{len(unified.compositional_links)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Unmatched Workflows</div>
            <div class="stat-value" style="color: #ff0054;">{len(unified.unmatched_communities)}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Unmatched Categories</div>
            <div class="stat-value" style="color: #ff5400;">{len(unified.unmatched_structural_clusters)}</div>
        </div>
    </div>
    
    <div id="heatmap"></div>
    
    <script>
        const alignments = {alignments_json};
        
        if (alignments.length > 0) {{
            const commIds = [...new Set(alignments.map(a => 'W_' + a.comm_id))];
            const scIds = [...new Set(alignments.map(a => 'K_' + a.sc_id))];
            
            const zMatrix = commIds.map(cId => {{
                const cNum = parseInt(cId.replace('W_', ''));
                return scIds.map(sId => {{
                    const sNum = parseInt(sId.replace('K_', ''));
                    const match = alignments.find(a => a.comm_id === cNum && a.sc_id === sNum);
                    return match ? match.jaccard : 0.0;
                }});
            }});
            
            const plotData = [{{
                x: scIds,
                y: commIds,
                z: zMatrix,
                type: 'heatmap',
                colorscale: [
                    [0.0, '#16213e'],
                    [0.2, '#0f3460'],
                    [0.5, '#fee440'],
                    [0.7, '#00b4d8'],
                    [1.0, '#00f5d4']
                ],
                colorbar: {{ title: 'Jaccard Index' }}
            }}];
            
            const layout = {{
                title: {{ text: 'Path 1 (Leiden Workflows W) vs Path 2 (Node2Vec Categories K) Jaccard Matrix', font: {{ color: '#ffffff' }} }},
                paper_bgcolor: '#1a1a2e',
                plot_bgcolor: '#16213e',
                xaxis: {{ title: 'Node2Vec Category Clusters (K)', gridcolor: '#0f3460', color: '#ffffff' }},
                yaxis: {{ title: 'Leiden Workflow Communities (W)', gridcolor: '#0f3460', color: '#ffffff' }}
            }};
            
            Plotly.newPlot('heatmap', plotData, layout);
        }} else {{
            document.getElementById('heatmap').innerHTML = '<h3 style="text-align:center; padding-top: 250px; color:#8d99ae;">No alignments exceeded the composition threshold ({self.composition_threshold:.2f}).</h3>';
        }}
    </script>
</body>
</html>"""
        
        with open(vis_dir / "interactive_unified_alignment.html", "w") as f:
            f.write(html_content)

        # 2. Interactive Bipartite Alignment Network (PyVis)
        net_bipartite = Network(height="1000px", width="100%", directed=False, bgcolor="#1a1a2e", font_color="white", heading="[Unified] Bipartite Cluster Alignment Network")
        
        for comm in comm_result.communities:
            c_id = f"W_{comm.community_id}"
            label = f"Workflow {comm.community_id}\n({len(comm.nodes)} nodes)"
            title = f"Leiden Workflow {comm.community_id}\nMembers:\n" + "\n".join(comm.nodes[:10])
            net_bipartite.add_node(c_id, label=label, color="#00b4d8", shape="box", size=30, title=title, borderWidth=2)
            
        for sc in emb_result.structural_clusters:
            s_id = f"K_{sc.cluster_id}"
            label = f"Category {sc.cluster_id}\n({len(sc.nodes)} nodes)"
            title = f"Node2Vec Category {sc.cluster_id}\nMembers:\n" + "\n".join(sc.nodes[:10])
            net_bipartite.add_node(s_id, label=label, color="#00f5d4", shape="ellipse", size=30, title=title, borderWidth=2)
            
        for a in unified.alignments:
            c_id = f"W_{a.community_id}"
            s_id = f"K_{a.structural_cluster_id}"
            if a.alignment_type == "isomorphic_fusion":
                color = "rgba(0, 245, 212, 0.9)"
                width = 5
            else:
                color = "rgba(254, 228, 64, 0.7)"
                width = 3
            title = f"Type: {a.alignment_type}\nJaccard Index: {a.jaccard_score:.4f}\nIntersection: {len(a.intersection_nodes)} nodes"
            net_bipartite.add_edge(c_id, s_id, title=title, color=color, width=width)
            
        net_bipartite.set_options("""{"physics": {"forceAtlas2Based": {"gravitationalConstant": -100, "centralGravity": 0.01, "springLength": 200}, "solver": "forceAtlas2Based"}}""")
        net_bipartite.save_graph(str(vis_dir / "interactive_bipartite_alignment_network.html"))

        # 3. Interactive Fused Clusters Only Network (PyVis)
        net_fused = Network(height="1000px", width="100%", directed=False, bgcolor="#1a1a2e", font_color="white", heading="[Unified] Isomorphic Fused Clusters Network")
        colors = ["#00f5d4", "#7b2cbf", "#e94560", "#fee440", "#00b4d8", "#ff0054", "#9b5de5"]
        
        for fc in unified.fused_clusters:
            fc_id = f"Fused_{fc['fused_cluster_id']}"
            color = colors[fc['fused_cluster_id'] % len(colors)]
            label = f"Fused Cluster {fc['fused_cluster_id']}\n(W_{fc['community_id']} + K_{fc['structural_cluster_id']})\nJ={fc['jaccard_score']:.2f}"
            title = f"Fused Cluster {fc['fused_cluster_id']}\nNodes ({len(fc['nodes'])}):\n" + "\n".join(fc['nodes'][:15])
            net_fused.add_node(fc_id, label=label, color=color, shape="diamond", size=35, title=title, borderWidth=3)
            
        net_fused.set_options("""{"physics": {"forceAtlas2Based": {"gravitationalConstant": -80, "centralGravity": 0.01, "springLength": 180}, "solver": "forceAtlas2Based"}}""")
        net_fused.save_graph(str(vis_dir / "interactive_fused_clusters_only.html"))


class TargetResolver:
    """
    Target qualification and triplet filtering criteria used by Stage 4 for LLM synthesis.
    Provides methods to resolve qualified spoke targets, rank/cap global hubs, aggregate enum nodes,
    and filter/cap triplet payloads based on PageRank and intra-cluster edge priority.
    """
    
    @staticmethod
    def qualify_and_resolve_targets(topology: Dict[str, Any], path_type: str, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Resolves qualified spoke targets (>= min_cluster_size), global hub targets (top max_hub_targets),
        and aggregates low-density cluster nodes, orphans, and demoted hubs into all_enum_nodes.
        
        Returns:
            (targets, all_enum_nodes)
        """
        min_cluster_size = config.get('synthesis', {}).get('min_cluster_size', 32)
        max_hub_targets = config.get('synthesis', {}).get('max_hub_targets', 4)
        
        # 1. Resolve Hub Targets
        raw_hubs = topology.get("global_hubs", [])
        metrics = topology.get("node_metrics", {})
        
        def get_hub_sort_key(node_id):
            node_m = metrics.get(node_id, {})
            if isinstance(node_m, dict):
                return (node_m.get("betweenness_centrality", 0.0), node_m.get("degree_centrality", 0.0))
            return (getattr(node_m, "betweenness_centrality", 0.0), getattr(node_m, "degree_centrality", 0.0))
            
        sorted_hubs = sorted(raw_hubs, key=get_hub_sort_key, reverse=True)
        hubs = sorted_hubs[:max_hub_targets]
        discarded_hubs = sorted_hubs[max_hub_targets:]
        
        # 2. Phase 1: Aggregate Orphans + Low-Density Nodes + Discarded Hubs (Enums)
        orphans = list(topology.get("orphans", []))
        small_cluster_nodes = []
        
        if path_type == "embedding":
            for cluster in topology.get("structural_clusters", []):
                nodes = cluster.get("nodes", [])
                if len(nodes) < min_cluster_size:
                    small_cluster_nodes.extend(nodes)
        elif path_type == "unified":
            for fc in topology.get("fused_clusters", []):
                nodes = fc.get("nodes", [])
                if len(nodes) < min_cluster_size:
                    small_cluster_nodes.extend(nodes)
        else:
            for comm in topology.get("communities", []):
                nodes = comm.get("nodes", [])
                if len(nodes) < min_cluster_size:
                    small_cluster_nodes.extend(nodes)

        all_enum_nodes = sorted(list(set(orphans + small_cluster_nodes + discarded_hubs)))
        
        # 3. Phase 2: Consolidate Qualified Targets (>= min_cluster_size)
        targets = []
        if path_type == "embedding":
            for cluster in topology.get("structural_clusters", []):
                nodes_in_cluster = cluster.get("nodes", [])
                if len(nodes_in_cluster) >= min_cluster_size:
                    targets.append({"type": "structural_cluster", "id": cluster.get("cluster_id"), "nodes": nodes_in_cluster})
        elif path_type == "unified":
            for fc in topology.get("fused_clusters", []):
                nodes_in_fc = fc.get("nodes", [])
                if len(nodes_in_fc) >= min_cluster_size:
                    targets.append({"type": "fused_cluster", "id": fc.get("fused_cluster_id"), "nodes": nodes_in_fc})
        else:
            for comm in topology.get("communities", []):
                nodes_in_comm = comm.get("nodes", [])
                if len(nodes_in_comm) >= min_cluster_size:
                    targets.append({"type": "community", "id": comm.get("community_id"), "nodes": nodes_in_comm})

        targets.sort(key=lambda x: len(x["nodes"]), reverse=True)
        
        # Append Hub Targets
        for hub in hubs:
            targets.append({"type": "hub", "id": hub, "nodes": [hub]})
            
        return targets, all_enum_nodes

    @staticmethod
    def filter_and_cap_triplets(target_nodes: set, triplets: List[Dict[str, Any]], topology: Dict[str, Any], max_triplets_cap: int) -> List[Dict[str, Any]]:
        """
        Filters triplets incident to target_nodes, scores each statement using PageRank + Intra-Cluster edge bonus (+1.0),
        sorts descending by pagerank_score, and caps at max_triplets_cap.
        """
        metrics = topology.get("node_metrics", {})
        scored_triplets = []
        
        for idx, t in enumerate(triplets):
            subj = str(t.get('subject', '')).lower().strip()
            obj = str(t.get('object', '')).lower().strip()
            if subj in target_nodes or obj in target_nodes:
                subj_m = metrics.get(subj, {})
                obj_m = metrics.get(obj, {})
                
                subj_pr = subj_m.get("pagerank", 0.0) if isinstance(subj_m, dict) else getattr(subj_m, "pagerank", 0.0)
                obj_pr = obj_m.get("pagerank", 0.0) if isinstance(obj_m, dict) else getattr(obj_m, "pagerank", 0.0)
                
                base_score = subj_pr + obj_pr
                if (subj in target_nodes) and (obj in target_nodes):
                    base_score += 1.0
                    
                t_copy = dict(t)
                t_copy["pagerank_score"] = round(float(base_score), 6)
                scored_triplets.append((base_score, t_copy))
                
        scored_triplets.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored_triplets[:max_triplets_cap]]

    @staticmethod
    def export_resolved_targets_json(
        topology: Dict[str, Any],
        path_type: str,
        refined_triplets: List[Dict[str, Any]],
        config: Dict[str, Any],
        output_dir: Path
    ) -> Path:
        """
        Executes target qualification and triplet PageRank capping, and exports
        resolved_<path_type>_targets.json to output_dir with explicit rank scores.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        synth_cfg = config.get('synthesis', {})
        min_cluster_size = synth_cfg.get('min_cluster_size', 32)
        max_hub_targets = synth_cfg.get('max_hub_targets', 4)
        max_triplets_cap = synth_cfg.get('max_triplets_per_target', 1000)
        metrics = topology.get("node_metrics", {})
        
        # 1. Resolve target qualification & enum nodes
        targets, all_enum_nodes = TargetResolver.qualify_and_resolve_targets(topology, path_type, config)
        
        # 2. Build target payload records with sorted & capped triplets
        target_payloads = []
        for i, target in enumerate(targets, start=1):
            target_type = target["type"]
            c_id = target["id"]
            target_nodes = set(target["nodes"])
            
            # Filter, score, and cap triplets (top 1000 PageRank + Intra-cluster)
            capped_triplets = TargetResolver.filter_and_cap_triplets(
                target_nodes, refined_triplets, topology, max_triplets_cap
            )
            
            filename_suffix = f"{path_type}_{c_id}" if target_type != "hub" else f"hub_{c_id}"
            output_module_name = f"{i:02d}_{filename_suffix}.py"
            
            # Compute target-level ranking metrics
            if target_type == "hub":
                hub_m = metrics.get(c_id, {})
                betw_score = hub_m.get("betweenness_centrality", 0.0) if isinstance(hub_m, dict) else getattr(hub_m, "betweenness_centrality", 0.0)
                deg_score = hub_m.get("degree_centrality", 0.0) if isinstance(hub_m, dict) else getattr(hub_m, "degree_centrality", 0.0)
                rank_score = round(float(betw_score), 6)
                target_metrics = {
                    "rank_score": rank_score,
                    "betweenness_centrality": round(float(betw_score), 6),
                    "degree_centrality": round(float(deg_score), 6)
                }
            else:
                rank_score = float(len(target_nodes))
                target_metrics = {
                    "rank_score": rank_score
                }
            
            payload_item = {
                "target_index": i,
                "target_type": target_type,
                "target_id": c_id,
                "output_module_name": output_module_name,
                "rank_score": rank_score,
                "nodes_count": len(target_nodes),
                "nodes": sorted(list(target_nodes)),
                "triplets_count": len(capped_triplets),
                "triplets": capped_triplets
            }
            if target_type == "hub":
                payload_item["betweenness_centrality"] = target_metrics["betweenness_centrality"]
                payload_item["degree_centrality"] = target_metrics["degree_centrality"]
                
            target_payloads.append(payload_item)
            
        # 3. Construct master audit record
        audit_payload = {
            "path_type": path_type,
            "min_cluster_size": min_cluster_size,
            "max_hub_targets": max_hub_targets,
            "max_triplets_per_target": max_triplets_cap,
            "enum_nodes_count": len(all_enum_nodes),
            "enum_nodes": all_enum_nodes,
            "targets_count": len(target_payloads),
            "targets": target_payloads
        }
        
        if path_type == "community":
            target_filename = "resolved_community_targets.json"
        elif path_type == "embedding":
            target_filename = "resolved_embedded_targets.json"
        else:
            target_filename = f"resolved_{path_type}_targets.json"
            
        target_file = output_dir / target_filename
        with open(target_file, "w") as f:
            json.dump(audit_payload, f, indent=2)
            
        print(f"   [OK] Resolved Targets Saved: {target_file}")
        return target_file
