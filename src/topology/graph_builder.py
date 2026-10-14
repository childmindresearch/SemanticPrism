import json
import math
import shutil
import networkx as nx
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from src.topology.schemas import NodeMetrics, CommunityPartition, StructuralCluster, ThemeInheritance, TopologyResult, HubPartition
from src.topology.resolver import TargetResolver
from src.topology.visualizer import TopologyVisualizer
from src.topology.tabular_exporter import TopologyTabularExporter

import io
import contextlib

try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from cdlib import algorithms
except ImportError:
    algorithms = None


def _calc_distribution_stats(arr: List[Any]) -> Dict[str, Any]:
    """Computes summary statistics (min, max, mean, median, std_dev, skewness) for a numeric array."""
    if not arr:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std_dev": 0.0, "skewness": 0.0}
    arr_np = np.array(arr, dtype=float)
    skew = float(stats.skew(arr_np)) if len(arr_np) > 2 else 0.0
    if math.isnan(skew):
        skew = 0.0
    return {
        "min": int(np.min(arr_np)),
        "max": int(np.max(arr_np)),
        "mean": round(float(np.mean(arr_np)), 3),
        "median": round(float(np.median(arr_np)), 3),
        "std_dev": round(float(np.std(arr_np)), 3),
        "skewness": round(skew, 3)
    }


def _calculate_participation_coefficient(dg: nx.DiGraph, refined_triplets: List[dict]) -> Dict[str, float]:
    """
    Computes Participation Coefficient P_i for each node based on incident edge themes:
    P_i = 1 - sum( (k_{i,m} / k_i)^2 )
    where k_i is total degree of node i, and k_{i,m} is incident edge count associated with theme m.
    """
    node_edge_themes = defaultdict(list)
    for t in refined_triplets:
        subj = t.get('subject')
        obj = t.get('object')
        theme = t.get('theme_association', 'Other')
        if not theme or theme == 'Other':
            theme = 'Unassigned'
        if subj:
            node_edge_themes[str(subj).lower().strip()].append(theme)
        if obj:
            node_edge_themes[str(obj).lower().strip()].append(theme)
            
    p_scores = {}
    for node in dg.nodes():
        themes = node_edge_themes.get(node, [])
        k_i = len(themes)
        if k_i < 3:
            p_scores[node] = 0.0
            continue
        counts = defaultdict(int)
        for th in themes:
            counts[th] += 1
        sum_sq = sum((c / k_i) ** 2 for c in counts.values())
        p_scores[node] = float(max(0.0, 1.0 - sum_sq))
    return p_scores


def _calculate_betweenness(dg: nx.DiGraph) -> Dict[str, float]:
    """Computes shortest-path Betweenness Centrality for directed graph."""
    if len(dg) == 0:
        return {}
    if len(dg) > 500:
        k_sample = min(100, max(50, int(len(dg) * 0.1)))
        return nx.betweenness_centrality(dg, k=k_sample)
    return nx.betweenness_centrality(dg)


def _calculate_modularity_vitality(ug: nx.Graph, refined_triplets: List[dict]) -> Dict[str, float]:
    r"""
    Computes Delta Q(v) = Q(G) - Q(G \ v) for each node v in ug relative to primary theme partition.
    Negative value means removing node v INCREASES modularity Q (node blurs primary theme boundaries).
    """
    vitality = {}
    if len(ug) < 3:
        for n in ug.nodes(): vitality[n] = 0.0
        return vitality
        
    node_edge_themes = defaultdict(list)
    for t in refined_triplets:
        subj = t.get('subject')
        obj = t.get('object')
        theme = t.get('theme_association', 'Other')
        if not theme or theme == 'Other':
            theme = 'Unassigned'
        if subj:
            node_edge_themes[str(subj).lower().strip()].append(theme)
        if obj:
            node_edge_themes[str(obj).lower().strip()].append(theme)
            
    primary_theme = {}
    for node in ug.nodes():
        themes = node_edge_themes.get(node, [])
        if not themes:
            primary_theme[node] = 'Unassigned'
        else:
            counts = defaultdict(int)
            for th in themes: counts[th] += 1
            primary_theme[node] = max(counts.items(), key=lambda x: x[1])[0]
            
    theme_groups = defaultdict(list)
    for node, p_th in primary_theme.items():
        theme_groups[p_th].append(node)
        
    comms = [nodes for nodes in theme_groups.values() if len(nodes) > 0]
    if len(comms) < 2:
        for n in ug.nodes(): vitality[n] = 0.0
        return vitality
        
    try:
        base_q = nx.community.modularity(ug, comms)
        for n in ug.nodes():
            sub_g = ug.copy()
            sub_g.remove_node(n)
            sub_comms = [[x for x in comm if x != n] for comm in comms]
            sub_comms = [c for c in sub_comms if len(c) > 0]
            if len(sub_comms) >= 2 and len(sub_g.edges()) > 0:
                sub_q = nx.community.modularity(sub_g, sub_comms)
                vitality[n] = float(base_q - sub_q)
            else:
                vitality[n] = 0.0
    except Exception:
        for n in ug.nodes(): vitality[n] = 0.0
        
    return vitality


class TopologyPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.full_config = config
        self.config = config.get('topology', {})
        self.mode = self.config.get('execution_mode', 'community')
        self.inheritance_threshold = self.config.get('inheritance_overlap_threshold', 0.4)
        
        # Global Hub Pruning Config (Stages 3.1 & 3.2 - Universal Evaluation)
        self.hub_cfg = self.config.get('hub_pruning', {})
        self.hub_p_thresh = self.hub_cfg.get('participation_threshold', 0.80)
        self.min_hub_degree = self.hub_cfg.get('min_hub_degree', 40)
        self.hub_betw_percentile = self.hub_cfg.get('betweenness_percentile', 0.95)
        # Backward-compatibility aliases
        self.comm_p_thresh = self.hub_p_thresh
        self.comm_betw_percentile = self.hub_betw_percentile

        # Path 1 Config (Approach A: Community Path)
        self.comm_cfg = self.config.get('community_path', {})
        self.leiden_res = self.comm_cfg.get('leiden_resolution', 1.0)
        self.enable_mod_vitality = self.config.get('enable_modularity_vitality_pruning', False)

        # Dedicated Visualization Engine
        self.visualizer = TopologyVisualizer(config=config)

        # Tabular CSV Exporter & Top Hubs Export Target
        self.top_n_hubs_export = self.config.get('top_n_hubs_export', 30) or 30
        self.tabular_exporter = TopologyTabularExporter(config=self.config)

    def execute(self, refined_triplets: List[Dict[str, Any]]) -> Dict[str, TopologyResult]:
        print("=== SemanticPrism Stage 3: Topology Pipeline (Rebuilt 3-Stage Architecture) ===")
        print("[Topology] Initializing Stage 3 Graph Construction...")
        
        # 1. Normalize Triplets & Construct Raw Network Graphs
        dg = nx.DiGraph()
        normalized_triplets = []
        for t in refined_triplets:
            subj = t.get('subject')
            obj = t.get('object')
            pred = t.get('predicate', '')
            
            if not subj or not obj:
                continue
            
            subj_norm = str(subj).lower().strip()
            obj_norm = str(obj).lower().strip()
            
            t_norm = t.copy()
            t_norm['subject'] = subj_norm
            t_norm['object'] = obj_norm
            normalized_triplets.append(t_norm)
                
            if dg.has_edge(subj_norm, obj_norm):
                dg[subj_norm][obj_norm]['weight'] = dg[subj_norm][obj_norm].get('weight', 1) + 1
            else:
                dg.add_edge(subj_norm, obj_norm, predicate=pred, weight=1)
                
        ug = dg.to_undirected()

        # Build Theme Sets from refined triplets
        theme_sets = defaultdict(set)
        for t in refined_triplets:
            subj = t.get('subject')
            obj = t.get('object')
            theme = t.get('theme_association')
            if theme and theme != "Other":
                if subj: theme_sets[theme].add(str(subj).lower().strip())
                if obj: theme_sets[theme].add(str(obj).lower().strip())

        # ---------------------------------------------------------------------
        # STAGE 3.1: Raw Overall Graph Evaluation, Modularity & Hub Identification
        # ---------------------------------------------------------------------
        raw_diagnostics, global_hubs, global_hubs_registry, node_metrics = self._stage_3_1_raw_graph_evaluation(
            dg=dg,
            ug=ug,
            refined_triplets=refined_triplets,
            normalized_triplets=normalized_triplets,
            theme_sets=theme_sets
        )

        # ---------------------------------------------------------------------
        # STAGE 3.2: Hub Removal & Modular Separation Enhancement
        # ---------------------------------------------------------------------
        core_dg, core_ug, cluster_ug, orphans, hub_partitions, core_diagnostics, separation_report, attachable_leaves = self._stage_3_2_core_separation(
            dg=dg,
            ug=ug,
            global_hubs=global_hubs,
            raw_diagnostics=raw_diagnostics
        )

        # Update node_metrics with finalized segregated orphans from Stage 3.2
        orphans_set = set(orphans)
        for node, nm in node_metrics.items():
            nm.is_orphan = (node in orphans_set)

        # ---------------------------------------------------------------------
        # STAGE 3.3: Community Discovery (Leiden Modularity Optimization)
        # ---------------------------------------------------------------------
        results = {}
        results["community"] = self._stage_3_3_community_path(
            dg=dg,
            core_dg=core_dg,
            cluster_ug=cluster_ug,
            global_hubs=global_hubs,
            hub_partitions=hub_partitions,
            orphans=orphans,
            attachable_leaves=attachable_leaves,
            theme_sets=theme_sets,
            node_metrics=node_metrics,
            refined_triplets=refined_triplets
        )

        # ---------------------------------------------------------------------
        # VISUALIZATIONS GENERATION (Co-located directly in 03_topology directories)
        # ---------------------------------------------------------------------
        print("   ==> [Visuals] Generating Co-Located HTML Visuals in 03_topology...")
        # Remove legacy separate visuals directory if present
        legacy_vis_base = Path("outputs/visuals")
        if legacy_vis_base.exists():
            shutil.rmtree(legacy_vis_base, ignore_errors=True)

        # 1. 01_raw_graph_evaluation: Operations on entire raw graph
        self.visualizer.generate_stage_3_1_overview_visuals(
            dg=dg,
            ug=ug,
            global_hubs=global_hubs,
            node_metrics=node_metrics,
            raw_diagnostics=raw_diagnostics,
            vis_dir=Path("outputs/03_topology/01_raw_graph_evaluation"),
            community_result=results.get("community")
        )

        # 2. 02_core_separation: Reduced graph (hubs removed) & Hub Backbone Network
        self.visualizer.generate_stage_3_2_reduced_core_visuals(
            core_dg=core_dg,
            cluster_ug=cluster_ug,
            global_hubs=global_hubs,
            separation_report=separation_report,
            vis_dir=Path("outputs/03_topology/02_core_separation"),
            dg=dg,
            hub_partitions=hub_partitions,
            node_metrics=node_metrics
        )

        # 3. 03_paths/community: Dedicated community path visuals
        if "community" in results:
            self.visualizer.generate_stage_3_3_community_visuals(
                dg=dg,
                result=results["community"],
                refined_triplets=refined_triplets,
                vis_dir=Path("outputs/03_topology/03_paths/community")
            )

        # ---------------------------------------------------------------------
        # TABULAR DIAGNOSTICS EXPORT (CSV Reports for Spreadsheets & Diagnostics)
        # ---------------------------------------------------------------------
        if self.config.get("export_tabular_reports", True):
            print("   ==> [Tabular] Exporting Graph Diagnostics & Pruning Metrics to CSV...")
            top_out_dir = Path("outputs/03_topology")
            self.tabular_exporter.export_all_from_artifacts(top_out_dir)

        # ---------------------------------------------------------------------
        # CLUSTER EMBEDDING & FRÉCHET CENTROID DISPERSION DASHBOARD
        # ---------------------------------------------------------------------
        try:
            print("   ==> [Embedding] Generating Cluster Centroid Dispersion Dashboard...")
            from src.topology.embedding import ClusterEmbeddingVisualizer
            emb_vis = ClusterEmbeddingVisualizer(config=self.full_config)
            emb_vis.generate_centroid_dispersion_dashboard()
            print("   ==> [Embedding] Generating Interactive 3D Galaxy Cluster Visualization...")
            emb_vis.generate_3d_galaxy_dashboard()
        except Exception as e:
            print(f"   [Warning] Centroid dispersion dashboard generation skipped: {e}")

        print("[Topology] Stage 3 Pipeline execution complete.")
        return results

    # =========================================================================
    # STAGE 3.1: RAW GRAPH EVALUATION, MODULARITY & HUB IDENTIFICATION
    # =========================================================================
    def _stage_3_1_raw_graph_evaluation(
        self, dg: nx.DiGraph, ug: nx.Graph, refined_triplets: List[dict],
        normalized_triplets: List[dict], theme_sets: Dict[str, set]
    ) -> Tuple[Dict[str, Any], List[str], List[Dict[str, Any]], Dict[str, NodeMetrics]]:
        print("   ==> [Stage 3.1] Overall Graph Evaluation, Modularity & Hub Identification...")
        out_dir = Path("outputs/03_topology/01_raw_graph_evaluation")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save normalized triplets
        with open(out_dir / "normalized_triplets.json", "w") as f:
            json.dump(normalized_triplets, f, indent=2)

        # 2. Basic Scale & Density
        num_nodes = len(dg)
        num_dir_edges = dg.number_of_edges()
        num_undir_edges = ug.number_of_edges()
        density = round(float(nx.density(dg)), 6) if num_nodes > 1 else 0.0

        # 3. Degree Distribution Statistics
        deg_stats = _calc_distribution_stats([d for n, d in dg.degree()])
        in_deg_stats = _calc_distribution_stats([d for n, d in dg.in_degree()])
        out_deg_stats = _calc_distribution_stats([d for n, d in dg.out_degree()])

        # 4. Connected Components
        wcc = list(nx.weakly_connected_components(dg))
        scc = list(nx.strongly_connected_components(dg))
        ug_comps = list(nx.connected_components(ug))
        giant_comp = max(ug_comps, key=len) if ug_comps else set()
        giant_ratio = round(len(giant_comp) / max(1, num_nodes), 4)

        # 5. Macro Topological Properties
        reciprocity = round(float(nx.reciprocity(dg)), 4) if num_nodes > 0 else 0.0
        assortativity = round(float(nx.degree_assortativity_coefficient(ug)), 4) if num_nodes > 1 else 0.0
        transitivity = round(float(nx.transitivity(ug)), 4) if num_nodes > 0 else 0.0
        avg_clustering = round(float(nx.average_clustering(ug)), 4) if num_nodes > 0 else 0.0

        # 6. Baseline Raw Modularity (Leiden on raw graph)
        leiden_raw_q = 0.0
        leiden_raw_comms_count = 0
        if algorithms and len(ug.nodes()) > 0:
            try:
                node_list = list(ug.nodes())
                mapping = {n: idx for idx, n in enumerate(node_list)}
                rev_mapping = {idx: n for idx, n in enumerate(node_list)}
                g_int = nx.relabel_nodes(ug, mapping)
                raw_coms = algorithms.leiden(g_int)
                leiden_raw_comms = [[rev_mapping[x] for x in c] for c in raw_coms.communities]
                leiden_raw_q = round(float(nx.community.modularity(ug, leiden_raw_comms)), 4)
                leiden_raw_comms_count = len(leiden_raw_comms)
            except Exception:
                pass

        # 7. Semantic Theme Modularity
        theme_groups = defaultdict(list)
        for t in refined_triplets:
            th = t.get('theme_association', 'Unassigned')
            s, o = t.get('subject'), t.get('object')
            if s: theme_groups[th].append(str(s).lower().strip())
            if o: theme_groups[th].append(str(o).lower().strip())
        theme_comms = [list(set(nodes).intersection(set(ug.nodes()))) for nodes in theme_groups.values() if len(nodes) > 0]
        theme_comms = [c for c in theme_comms if len(c) > 0]
        theme_q = 0.0
        try:
            if len(theme_comms) >= 2:
                assigned = set()
                disjoint_theme_comms = []
                for tc in sorted(theme_comms, key=len, reverse=True):
                    curr = [n for n in tc if n not in assigned]
                    if curr:
                        disjoint_theme_comms.append(curr)
                        assigned.update(curr)
                unassigned = [n for n in ug.nodes() if n not in assigned]
                if unassigned:
                    disjoint_theme_comms.append(unassigned)
                if len(disjoint_theme_comms) >= 2:
                    theme_q = round(float(nx.community.modularity(ug, disjoint_theme_comms)), 4)
        except Exception:
            pass

        raw_diagnostics = {
            "graph_scale": {
                "num_nodes": num_nodes,
                "num_directed_edges": num_dir_edges,
                "num_undirected_edges": num_undir_edges,
                "density": density
            },
            "degree_distribution": {
                "total_degree": deg_stats,
                "in_degree": in_deg_stats,
                "out_degree": out_deg_stats
            },
            "component_structure": {
                "weakly_connected_components": len(wcc),
                "strongly_connected_components": len(scc),
                "undirected_connected_components": len(ug_comps),
                "giant_component_nodes": len(giant_comp),
                "giant_component_ratio": giant_ratio
            },
            "macro_topological_metrics": {
                "reciprocity": reciprocity,
                "degree_assortativity": assortativity,
                "transitivity": transitivity,
                "average_clustering_coefficient": avg_clustering
            },
            "baseline_modularity": {
                "leiden_raw_modularity": leiden_raw_q,
                "leiden_raw_community_count": leiden_raw_comms_count,
                "semantic_theme_modularity": theme_q
            }
        }

        with open(out_dir / "raw_graph_diagnostics.json", "w") as f:
            json.dump(raw_diagnostics, f, indent=2)

        # 8. Compute Node Centralities & Identify Hubs
        pagerank = nx.pagerank(dg) if len(dg) > 0 else {}
        degree_cent = nx.degree_centrality(dg) if len(dg) > 0 else {}
        betweenness = _calculate_betweenness(dg)
        participation = _calculate_participation_coefficient(dg, refined_triplets)
        if self.enable_mod_vitality:
            mod_vitality = _calculate_modularity_vitality(ug, refined_triplets)
        else:
            mod_vitality = {n: 0.0 for n in dg.nodes()}

        betweenness_sorted = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        betw_cutoff = max(1, int(len(betweenness_sorted) * (1.0 - self.hub_betw_percentile)))
        top_betweenness_nodes = set([n for n, score in betweenness_sorted[:betw_cutoff] if score > 0.02])

        if self.enable_mod_vitality:
            mod_v_sorted = sorted(mod_vitality.items(), key=lambda x: x[1])
            neg_mod_v_nodes = set([n for n, val in mod_v_sorted[:10] if val < -0.005 and dg.degree(n) >= 3])
        else:
            neg_mod_v_nodes = set()

        min_top_hubs = getattr(self, "top_n_hubs_export", 30) or 30

        global_hubs = []
        global_hubs_set = set()
        node_candidates = {}

        for node in dg.nodes():
            p_val = participation.get(node, 0.0)
            b_val = betweenness.get(node, 0.0)
            pr_val = pagerank.get(node, 0.0)
            deg_val = dg.degree(node)
            mv_val = mod_vitality.get(node, 0.0)

            is_thematic_bridge = (p_val >= self.hub_p_thresh) and (deg_val >= self.min_hub_degree)
            is_workflow_chokepoint = (node in top_betweenness_nodes)
            is_boundary_blurring = self.enable_mod_vitality and (node in neg_mod_v_nodes)

            reasons = []
            if is_thematic_bridge: reasons.append(f"Thematic Dispersion (P_i={p_val:.3f} >= {self.hub_p_thresh}, Deg={deg_val} >= {self.min_hub_degree})")
            if is_workflow_chokepoint: reasons.append(f"Workflow Chokepoint (Betweenness={b_val:.4f} > 0.02)")
            if is_boundary_blurring: reasons.append(f"Boundary Blurring (Delta Q={mv_val:.4f} < -0.005)")

            is_hub = (is_thematic_bridge or is_workflow_chokepoint or is_boundary_blurring)
            if is_hub:
                global_hubs.append(node)
                global_hubs_set.add(node)
            else:
                reasons = [f"Retained Core Concept (P_i={p_val:.3f} < {self.hub_p_thresh})"]

            node_candidates[node] = {
                "node_id": node,
                "removed": is_hub,
                "total_degree": deg_val,
                "in_degree": dg.in_degree(node),
                "out_degree": dg.out_degree(node),
                "pagerank": round(pr_val, 6),
                "betweenness_centrality": round(b_val, 6),
                "participation_coefficient": round(p_val, 4),
                "modularity_vitality": round(mv_val, 6),
                "qualification_reasons": reasons
            }

        # Select Top N by degree UNION all flagged hubs (Comprehensive Union)
        all_nodes_sorted_by_deg = sorted(
            dg.nodes(),
            key=lambda n: (dg.degree(n), betweenness.get(n, 0.0), pagerank.get(n, 0.0)),
            reverse=True
        )
        top_n_by_deg = set(all_nodes_sorted_by_deg[:min_top_hubs])
        registry_nodes_set = top_n_by_deg.union(global_hubs_set)

        global_hubs_registry = [node_candidates[n] for n in registry_nodes_set]
        # Sort registry primarily by total degree (descending), then betweenness, then pagerank
        global_hubs_registry.sort(
            key=lambda x: (x["total_degree"], x["betweenness_centrality"], x["pagerank"]),
            reverse=True
        )

        with open(out_dir / "global_hubs_registry.json", "w") as f:
            json.dump({
                "total_global_hubs": len(global_hubs),
                "total_global_hubs_removed": len(global_hubs),
                "total_registry_nodes": len(global_hubs_registry),
                "min_top_hubs_configured": min_top_hubs,
                "hub_criteria": {
                    "participation_threshold": self.hub_p_thresh,
                    "min_hub_degree": self.min_hub_degree,
                    "betweenness_percentile": self.hub_betw_percentile,
                    "betweenness_min_score": 0.02,
                    "enable_modularity_vitality": self.enable_mod_vitality
                },
                "hubs": global_hubs_registry
            }, f, indent=2)

        node_metrics = self._build_node_metrics(dg, ug, global_hubs, pagerank, degree_cent, betweenness, participation, mod_vitality)

        print(f"      -> Scale: {num_nodes} nodes, {num_dir_edges} edges, Density: {density:.6f}")
        print(f"      -> Baseline Modularity: Q_raw={leiden_raw_q:.4f} ({leiden_raw_comms_count} communities)")
        print(f"      -> Identified Global Hubs: {len(global_hubs)} nodes")

        return raw_diagnostics, global_hubs, global_hubs_registry, node_metrics

    # =========================================================================
    # STAGE 3.2: HUB REMOVAL & MODULAR SEPARATION ENHANCEMENT
    # =========================================================================
    def _stage_3_2_core_separation(
        self, dg: nx.DiGraph, ug: nx.Graph, global_hubs: List[str], raw_diagnostics: Dict[str, Any]
    ) -> Tuple[nx.DiGraph, nx.Graph, nx.Graph, List[str], List[HubPartition], Dict[str, Any], Dict[str, Any], Dict[str, str]]:
        print("   ==> [Stage 3.2] Removing Hubs & Quantifying Modular Separation Enhancement...")
        out_dir = Path("outputs/03_topology/02_core_separation")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Isolate Hub Backbone Network
        hub_dg = dg.subgraph(global_hubs).copy()
        hub_ug = ug.subgraph(global_hubs).copy()
        hub_partitions = self._partition_hub_network(dg, global_hubs)

        hub_backbone_data = {
            "num_hub_nodes": len(hub_ug),
            "num_inter_hub_edges": hub_ug.number_of_edges(),
            "inter_hub_density": round(float(nx.density(hub_dg)), 5) if len(hub_dg) > 1 else 0.0,
            "hub_partitions_count": len(hub_partitions),
            "hub_partitions": [hp.model_dump() for hp in hub_partitions]
        }
        with open(out_dir / "hub_backbone_network.json", "w") as f:
            json.dump(hub_backbone_data, f, indent=2)

        # 2. Extract Core Graph (G_core = G \ V_hubs)
        core_dg = dg.copy()
        core_dg.remove_nodes_from(global_hubs)
        core_ug = ug.copy()
        core_ug.remove_nodes_from(global_hubs)

        # 3. Segregate Core Nodes for Two-Pass Partitioning
        # True Isolates: degree == 0 in G_core (were connected ONLY to global hubs)
        true_isolates = [n for n in core_ug.nodes() if core_ug.degree(n) == 0]
        # Dense Core: degree > 1 in G_core (retained for Pass 1 clean community discovery)
        dense_core_nodes = set(n for n in core_ug.nodes() if core_ug.degree(n) > 1)
        # Pendant Leaves: degree == 1 in G_core
        d1_nodes = [n for n in core_ug.nodes() if core_ug.degree(n) == 1]

        # Map attachable leaves (whose sole neighbor is in dense core) vs isolated pairs/chains
        attachable_leaves = {}
        isolated_dyads = []
        for u in d1_nodes:
            nbrs = list(core_ug.neighbors(u))
            if len(nbrs) == 1 and nbrs[0] in dense_core_nodes:
                attachable_leaves[u] = nbrs[0]
            else:
                isolated_dyads.append(u)

        # True unattached orphans are degree-0 isolates and disconnected pairs
        orphans = sorted(true_isolates + isolated_dyads)

        with open(out_dir / "orphans_and_singletons.json", "w") as f:
            json.dump({
                "summary": {
                    "dense_core_nodes": len(dense_core_nodes),
                    "attachable_pendant_leaves": len(attachable_leaves),
                    "true_isolated_singletons": len(true_isolates),
                    "isolated_dyads": len(isolated_dyads),
                    "unattached_orphans_total": len(orphans)
                },
                "true_isolated_singletons": sorted(true_isolates),
                "attachable_pendant_leaves": [
                    {"node": u, "core_neighbor": v} for u, v in sorted(attachable_leaves.items())
                ],
                "isolated_dyads": sorted(isolated_dyads)
            }, f, indent=2)

        # 4. Target graph for cluster/community detection (G_cluster = dense core with degree > 1)
        cluster_ug = core_ug.subgraph(dense_core_nodes).copy()

        # 4b. Write Unified Removed Nodes Manifest
        removed_manifest = {
            "summary": {
                "total_raw_nodes": len(dg),
                "total_removed_or_segregated": len(global_hubs) + len(orphans) + len(attachable_leaves),
                "retained_dense_core_for_clustering": len(cluster_ug),
                "attachable_pendant_leaves": len(attachable_leaves),
                "unattached_orphans": len(orphans),
                "breakdown": {
                    "global_hubs_extracted": len(global_hubs),
                    "true_singletons": len(true_isolates),
                    "attachable_leaves": len(attachable_leaves),
                    "isolated_dyads": len(isolated_dyads)
                }
            },
            "global_hubs_removed": [
                {
                    "node_id": h,
                    "removal_role": "Global Hub",
                    "removal_reason": "High Thematic Dispersion / Shortest-Path Chokepoint",
                    "raw_total_degree": dg.degree(h),
                    "raw_in_degree": dg.in_degree(h),
                    "raw_out_degree": dg.out_degree(h)
                }
                for h in global_hubs
            ],
            "unattached_singletons_and_dyads": [
                {
                    "node_id": o,
                    "removal_role": "Singleton/Isolated Dyad",
                    "removal_reason": "No direct connection to dense core",
                    "core_degree": core_ug.degree(o),
                    "raw_degree": dg.degree(o)
                }
                for o in sorted(orphans)
            ],
            "attachable_pendant_leaves": [
                {
                    "node_id": u,
                    "core_neighbor": v,
                    "raw_degree": dg.degree(u)
                }
                for u, v in sorted(attachable_leaves.items())
            ],
            "retained_dense_core_nodes": sorted(list(cluster_ug.nodes()))
        }
        with open(out_dir / "removed_nodes_manifest.json", "w") as f:
            json.dump(removed_manifest, f, indent=2)

        # 5. Core Diagnostics
        core_nodes = len(core_ug)
        core_edges = core_ug.number_of_edges()
        core_density = round(float(nx.density(core_dg)), 6) if core_nodes > 1 else 0.0
        core_deg_stats = _calc_distribution_stats([d for n, d in core_ug.degree()])
        core_comps = list(nx.connected_components(core_ug))
        core_giant = max(core_comps, key=len) if core_comps else set()
        core_giant_ratio = round(len(core_giant) / max(1, core_nodes), 4)

        core_modularity_q = 0.0
        core_comm_count = 0
        if algorithms and len(cluster_ug.nodes()) > 0:
            try:
                node_list = list(cluster_ug.nodes())
                mapping = {n: idx for idx, n in enumerate(node_list)}
                rev_mapping = {idx: n for idx, n in enumerate(node_list)}
                g_int = nx.relabel_nodes(cluster_ug, mapping)
                coms = algorithms.leiden(g_int)
                comms_list = [[rev_mapping[x] for x in c] for c in coms.communities]
                core_modularity_q = round(float(nx.community.modularity(cluster_ug, comms_list)), 4)
                core_comm_count = len(comms_list)
            except Exception:
                pass

        cluster_deg_stats = _calc_distribution_stats([d for n, d in cluster_ug.degree()]) if len(cluster_ug) > 0 else {}
        cluster_comps = list(nx.connected_components(cluster_ug)) if len(cluster_ug) > 0 else []
        cluster_giant = max(cluster_comps, key=len) if cluster_comps else set()
        cluster_giant_ratio = round(len(cluster_giant) / max(1, len(cluster_ug)), 4) if len(cluster_ug) > 0 else 0.0

        # Macro Topological Metrics for Core and Cluster Subgraphs
        core_avg_clustering = round(float(nx.average_clustering(core_ug)), 4) if len(core_ug) > 0 else 0.0
        core_assortativity = round(float(nx.degree_assortativity_coefficient(core_ug)), 4) if len(core_ug) > 1 else 0.0
        core_transitivity = round(float(nx.transitivity(core_ug)), 4) if len(core_ug) > 0 else 0.0

        cluster_avg_clustering = round(float(nx.average_clustering(cluster_ug)), 4) if len(cluster_ug) > 0 else 0.0
        cluster_assortativity = round(float(nx.degree_assortativity_coefficient(cluster_ug)), 4) if len(cluster_ug) > 1 else 0.0
        cluster_transitivity = round(float(nx.transitivity(cluster_ug)), 4) if len(cluster_ug) > 0 else 0.0

        core_diagnostics = {
            "graph_scale": {
                "num_nodes": core_nodes,
                "num_edges": core_edges,
                "density": core_density,
                "cluster_subgraph_nodes": len(cluster_ug),
                "cluster_subgraph_edges": cluster_ug.number_of_edges()
            },
            "degree_distribution": core_deg_stats,
            "cluster_degree_distribution": cluster_deg_stats,
            "component_structure": {
                "connected_components_count": len(core_comps),
                "giant_component_nodes": len(core_giant),
                "giant_component_ratio": core_giant_ratio,
                "cluster_connected_components_count": len(cluster_comps),
                "cluster_giant_component_nodes": len(cluster_giant),
                "cluster_giant_component_ratio": cluster_giant_ratio
            },
            "modular_separation": {
                "core_leiden_modularity": core_modularity_q,
                "core_communities_count": core_comm_count
            },
            "macro_topological_metrics": {
                "core_average_clustering_coefficient": core_avg_clustering,
                "core_degree_assortativity": core_assortativity,
                "core_transitivity": core_transitivity,
                "cluster_average_clustering_coefficient": cluster_avg_clustering,
                "cluster_degree_assortativity": cluster_assortativity,
                "cluster_transitivity": cluster_transitivity
            }
        }
        with open(out_dir / "core_graph_diagnostics.json", "w") as f:
            json.dump(core_diagnostics, f, indent=2)

        # 6. Separation Impact Report
        raw_scale = raw_diagnostics["graph_scale"]
        raw_deg = raw_diagnostics["degree_distribution"]["total_degree"]
        raw_comp = raw_diagnostics["component_structure"]
        raw_mod = raw_diagnostics["baseline_modularity"]["leiden_raw_modularity"]

        modularity_gain = round(core_modularity_q - raw_mod, 4)
        separation_report = {
            "modularity_enhancement": {
                "raw_modularity_q": raw_mod,
                "core_modularity_q": core_modularity_q,
                "modularity_gain_delta_q": modularity_gain,
                "percent_modularity_increase": round((modularity_gain / max(0.001, raw_mod)) * 100, 2)
            },
            "network_fragmentation": {
                "raw_component_count": raw_comp["undirected_connected_components"],
                "core_component_count": len(core_comps),
                "raw_giant_component_ratio": raw_comp["giant_component_ratio"],
                "core_giant_component_ratio": core_giant_ratio
            },
            "degree_distribution_normalization": {
                "raw_max_degree": raw_deg.get("max", 0),
                "core_max_degree": core_deg_stats.get("max", 0),
                "raw_degree_std_dev": raw_deg.get("std_dev", 0),
                "core_degree_std_dev": core_deg_stats.get("std_dev", 0),
                "raw_degree_skewness": raw_deg.get("skewness", 0),
                "core_degree_skewness": core_deg_stats.get("skewness", 0),
                "skewness_reduction": round(raw_deg.get("skewness", 0) - core_deg_stats.get("skewness", 0), 3)
            }
        }
        with open(out_dir / "separation_impact_report.json", "w") as f:
            json.dump(separation_report, f, indent=2)

        print(f"      -> Core Graph: {core_nodes} nodes, {core_edges} edges, {len(orphans)} orphans segregated")
        print(f"      -> Modularity Jump: Q_raw={raw_mod:.4f} -> Q_core={core_modularity_q:.4f} (ΔQ={modularity_gain:+.4f})")
        print(f"      -> Component Fragmentation: {raw_comp['undirected_connected_components']} -> {len(core_comps)} components")

        return core_dg, core_ug, cluster_ug, orphans, hub_partitions, core_diagnostics, separation_report, attachable_leaves

    # =========================================================================
    # STAGE 3.3 - PATH 1: COMMUNITY PATH (Approach A - Leiden Modularity)
    # =========================================================================
    def _stage_3_3_community_path(
        self, dg: nx.DiGraph, core_dg: nx.DiGraph, cluster_ug: nx.Graph,
        global_hubs: List[str], hub_partitions: List[HubPartition], orphans: List[str],
        attachable_leaves: Dict[str, str],
        theme_sets: Dict[str, set], node_metrics: Dict[str, NodeMetrics], refined_triplets: List[dict]
    ) -> TopologyResult:
        print("   ==> [Stage 3.3 - Path 1] Executing Community Path (Leiden Modularity)...")
        path_out_dir = Path("outputs/03_topology/03_paths/community")
        path_out_dir.mkdir(parents=True, exist_ok=True)

        communities = []
        if algorithms and len(cluster_ug.nodes()) > 0:
            try:
                node_list = list(cluster_ug.nodes())
                mapping = {n: idx for idx, n in enumerate(node_list)}
                rev_mapping = {idx: n for idx, n in enumerate(node_list)}
                g_int = nx.relabel_nodes(cluster_ug, mapping)
                coms = algorithms.leiden(g_int)
                for i, c_nodes in enumerate(coms.communities):
                    orig_nodes = [rev_mapping[n] for n in c_nodes]
                    communities.append(CommunityPartition(community_id=i, nodes=orig_nodes))
            except Exception as e:
                print(f"[Topology Community] Error in Leiden: {e}. Falling back to connected components...")
                for i, comp in enumerate(nx.connected_components(cluster_ug)):
                    communities.append(CommunityPartition(community_id=i, nodes=list(comp)))
        else:
            for i, comp in enumerate(nx.connected_components(cluster_ug)):
                communities.append(CommunityPartition(community_id=i, nodes=list(comp)))

        # Initial Core Community Map
        comm_map = {}
        for c in communities:
            for n in c.nodes: comm_map[n] = c.community_id

        # Baseline Core Modularity Q (computed on pure dense core)
        final_q = 0.0
        try:
            if len(communities) >= 2:
                final_q = round(float(nx.community.modularity(cluster_ug, [c.nodes for c in communities])), 4)
        except Exception:
            pass

        # Pass 2: Two-Pass Leaf Reattachment
        # Re-attach degree-1 pendant leaves to their unique core neighbor's community
        comm_lookup = {c.community_id: c for c in communities}
        reattached_leaf_count = 0
        for u, v in attachable_leaves.items():
            if v in comm_map:
                target_cid = comm_map[v]
                comm_lookup[target_cid].nodes.append(u)
                comm_map[u] = target_cid
                reattached_leaf_count += 1

        # Post-Reattachment Community Diagnostics
        comm_stats = []
        for c in communities:
            c.nodes = sorted(c.nodes)
            c_nodes = c.nodes
            n_nodes = len(c_nodes)
            # Internal edge density in directed graph
            c_sub = dg.subgraph(c_nodes)
            n_edges = c_sub.number_of_edges()
            internal_density = round(n_edges / (n_nodes * (n_nodes - 1)), 4) if n_nodes > 1 else 1.0
            comm_stats.append({
                "community_id": c.community_id,
                "node_count": n_nodes,
                "internal_edge_count": n_edges,
                "internal_density": internal_density,
                "exemplar_nodes": c_nodes[:5]
            })

        path_diagnostics = {
            "algorithm": "Leiden Modularity Optimization (Two-Pass Leaf Reattachment)",
            "resolution": self.leiden_res,
            "dense_core_nodes": len(cluster_ug),
            "pendant_leaves_reattached": reattached_leaf_count,
            "total_partitioned_nodes": sum(len(c.nodes) for c in communities),
            "unattached_orphans": len(orphans),
            "total_communities": len(communities),
            "final_core_modularity_q": final_q,
            "community_size_stats": _calc_distribution_stats([len(c.nodes) for c in communities]),
            "communities": comm_stats
        }
        with open(path_out_dir / "community_path_diagnostics.json", "w") as f:
            json.dump(path_diagnostics, f, indent=2)

        # Inter-Community Highway Matrix
        highways = defaultdict(lambda: {"edge_count": 0, "predicates": defaultdict(int)})
        for u, v, data in dg.edges(data=True):
            if u in comm_map and v in comm_map:
                cu, cv = comm_map[u], comm_map[v]
                if cu != cv:
                    highways[(cu, cv)]["edge_count"] += 1
                    pred = data.get("predicate", "related_to")
                    highways[(cu, cv)]["predicates"][pred] += 1

        highways_list = []
        for (cu, cv), h_data in sorted(highways.items(), key=lambda x: x[1]["edge_count"], reverse=True):
            top_preds = sorted(h_data["predicates"].items(), key=lambda x: x[1], reverse=True)[:3]
            highways_list.append({
                "source_community": cu,
                "target_community": cv,
                "directed_edge_count": h_data["edge_count"],
                "top_predicates": [{"predicate": p, "count": cnt} for p, cnt in top_preds]
            })
        with open(path_out_dir / "community_highways.json", "w") as f:
            json.dump({
                "total_highways": len(highways_list),
                "highways": highways_list
            }, f, indent=2)

        theme_inheritance = self._calculate_theme_inheritance(theme_sets)
        result = TopologyResult(
            global_hubs=global_hubs,
            hub_partitions=hub_partitions,
            communities=communities,
            structural_clusters=[],
            orphans=orphans,
            node_metrics=node_metrics,
            theme_inheritance=theme_inheritance
        )

        with open(path_out_dir / "topology_partitions.json", "w") as f:
            json.dump(result.model_dump(), f, indent=2)

        TargetResolver.export_resolved_targets_json(
            topology=result.model_dump(),
            path_type="community",
            refined_triplets=refined_triplets,
            config=self.full_config,
            output_dir=path_out_dir
        )

        print(f"      -> Discovered {len(communities)} communities (Modularity Q={final_q:.4f})")
        print(f"      -> Mapped {len(highways_list)} inter-community workflow highways")

        return result


    # =========================================================================
    # CORE METRIC HELPERS
    # =========================================================================
    def _build_node_metrics(
        self,
        dg: nx.DiGraph,
        ug: nx.Graph,
        global_hubs: List[str],
        pagerank: dict,
        degree_cent: dict,
        betweenness: dict,
        participation: dict,
        mod_vitality: dict,
        orphans: Optional[List[str]] = None
    ) -> Dict[str, NodeMetrics]:
        if orphans is None:
            orphans_set = set(node for node in ug.nodes() if ug.degree(node) <= 1 and node not in global_hubs)
        else:
            orphans_set = set(orphans)
        node_metrics = {}
        for node in dg.nodes():
            in_edges_list = [{"node": u, "predicate": data.get("predicate", "")} for u, v, data in dg.in_edges(node, data=True)]
            out_edges_list = [{"node": v, "predicate": data.get("predicate", "")} for u, v, data in dg.out_edges(node, data=True)]
            
            node_metrics[node] = NodeMetrics(
                node_id=node,
                degree_centrality=degree_cent.get(node, 0.0),
                pagerank=pagerank.get(node, 0.0),
                betweenness_centrality=betweenness.get(node, 0.0),
                participation_coefficient=participation.get(node, 0.0),
                modularity_vitality=mod_vitality.get(node, 0.0),
                is_hub=(node in global_hubs),
                is_orphan=(node in orphans_set),
                in_edges=in_edges_list,
                out_edges=out_edges_list
            )
        return node_metrics

    def _calculate_theme_inheritance(self, theme_sets: Dict[str, set]) -> List[ThemeInheritance]:
        theme_inheritance = []
        for child_theme, child_nodes in theme_sets.items():
            for parent_theme, parent_nodes in theme_sets.items():
                if child_theme == parent_theme or not child_nodes:
                    continue
                overlap = len(child_nodes.intersection(parent_nodes)) / len(child_nodes)
                if overlap >= self.inheritance_threshold:
                    theme_inheritance.append(ThemeInheritance(
                        parent_theme=parent_theme,
                        child_theme=child_theme,
                        overlap_score=overlap
                    ))
        return theme_inheritance

    def _partition_hub_network(self, dg: nx.DiGraph, global_hubs: List[str]) -> List[HubPartition]:
        if not global_hubs:
            return []
            
        if len(global_hubs) <= 2:
            return [HubPartition(hub_cluster_id=0, nodes=list(global_hubs))]

        hubs_set = set(global_hubs)
        hub_subgraph = nx.DiGraph()
        for h in hubs_set:
            hub_subgraph.add_node(h)
            
        for u, v, data in dg.edges(data=True):
            if u in hubs_set and v in hubs_set:
                if hub_subgraph.has_edge(u, v):
                    hub_subgraph[u][v]['weight'] = hub_subgraph[u][v].get('weight', 1) + 1
                else:
                    hub_subgraph.add_edge(u, v, weight=1)

        ug_hubs = hub_subgraph.to_undirected()
        if len(ug_hubs.edges()) == 0:
            return [HubPartition(hub_cluster_id=i, nodes=[h]) for i, h in enumerate(global_hubs)]

        hub_partitions = []
        if algorithms:
            try:
                node_list = list(ug_hubs.nodes())
                mapping = {n: idx for idx, n in enumerate(node_list)}
                rev_mapping = {idx: n for idx, n in enumerate(node_list)}
                int_ug_hubs = nx.relabel_nodes(ug_hubs, mapping)
                coms = algorithms.leiden(int_ug_hubs)
                for i, c_nodes in enumerate(coms.communities):
                    orig_nodes = [rev_mapping[n] for n in c_nodes]
                    hub_partitions.append(HubPartition(hub_cluster_id=i, nodes=orig_nodes))
            except Exception:
                for i, comp in enumerate(nx.connected_components(ug_hubs)):
                    hub_partitions.append(HubPartition(hub_cluster_id=i, nodes=list(comp)))
        else:
            for i, comp in enumerate(nx.connected_components(ug_hubs)):
                hub_partitions.append(HubPartition(hub_cluster_id=i, nodes=list(comp)))
                
        return hub_partitions

    # =========================================================================
    # BACKWARD-COMPATIBLE VISUALIZATION DELEGATES
    # Delegated directly to TopologyVisualizer in src.topology.visualizer
    # =========================================================================

    def _generate_stage_3_1_overview_visuals(self, *args, **kwargs):
        return self.visualizer.generate_stage_3_1_overview_visuals(*args, **kwargs)

    def _generate_stage_3_2_reduced_core_visuals(self, *args, **kwargs):
        return self.visualizer.generate_stage_3_2_reduced_core_visuals(*args, **kwargs)

    def _generate_stage_3_3_community_visuals(self, *args, **kwargs):
        return self.visualizer.generate_stage_3_3_community_visuals(*args, **kwargs)
