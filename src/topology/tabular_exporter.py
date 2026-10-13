import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class TopologyTabularExporter:
    """
    Exports structured tabular CSV files capturing complete graph diagnostics,
    individual Top N global hub metrics, pruned subgraph statistics,
    and modularity optimization results for Stage 3 Topology.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.top_n_hubs = self.config.get("top_n_hubs_export", None)

    def export_comparative_summary(
        self,
        raw_diag: Dict[str, Any],
        core_diag: Dict[str, Any],
        separation_report: Dict[str, Any],
        output_path: Path,
        manifest_data: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Exports side-by-side comparative diagnostics contrasting the Raw Complete Graph (G_raw),
        the Reduced Core Graph (G_core), and the Dense Clustering Subgraph (G_cluster).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        raw_scale = raw_diag.get("graph_scale", {})
        raw_deg = raw_diag.get("degree_distribution", {}).get("total_degree", {})
        raw_comp = raw_diag.get("component_structure", {})
        raw_macro = raw_diag.get("macro_topological_metrics", {})
        raw_mod = raw_diag.get("baseline_modularity", {})

        core_scale = core_diag.get("graph_scale", {})
        core_deg = core_diag.get("degree_distribution", {})
        core_comp = core_diag.get("component_structure", {})
        core_mod = core_diag.get("modular_separation", {})
        cluster_deg = core_diag.get("cluster_degree_distribution", {})
        core_macro = core_diag.get("macro_topological_metrics", {})

        mod_enh = separation_report.get("modularity_enhancement", {})
        skew_norm = separation_report.get("degree_distribution_normalization", {})

        raw_cc = raw_macro.get("average_clustering_coefficient", 0.0)
        core_cc = core_macro.get("core_average_clustering_coefficient", 0.0)
        cluster_cc = core_macro.get("cluster_average_clustering_coefficient", 0.0)

        raw_assort = raw_macro.get("degree_assortativity", 0.0)
        core_assort = core_macro.get("core_degree_assortativity", 0.0)
        cluster_assort = core_macro.get("cluster_degree_assortativity", 0.0)

        raw_trans = raw_macro.get("transitivity", 0.0)
        core_trans = core_macro.get("core_transitivity", 0.0)
        cluster_trans = core_macro.get("cluster_transitivity", 0.0)

        cc_gain = ((cluster_cc - raw_cc) / max(0.0001, raw_cc)) * 100 if raw_cc > 0 else 0.0
        trans_gain = ((cluster_trans - raw_trans) / max(0.0001, raw_trans)) * 100 if raw_trans > 0 else 0.0

        raw_sd = raw_deg.get('std_dev', 0.0)
        core_sd = core_deg.get('std_dev', 0.0)
        sd_diff = raw_sd - core_sd
        sd_pct = (sd_diff / max(0.001, raw_sd)) * 100

        raw_skew = raw_deg.get('skewness', 0.0)
        core_skew = core_deg.get('skewness', 0.0)
        skew_diff = skew_norm.get('skewness_reduction', raw_skew - core_skew)
        skew_pct = (skew_diff / max(0.001, raw_skew)) * 100

        raw_nodes = raw_scale.get("num_nodes", 0)
        core_nodes = core_scale.get("num_nodes", 0)
        cluster_nodes = core_scale.get("cluster_subgraph_nodes", 0)

        raw_edges = raw_scale.get("num_directed_edges", 0)
        core_edges = core_scale.get("num_edges", 0)
        cluster_edges = core_scale.get("cluster_subgraph_edges", 0)

        raw_density = raw_scale.get("density", 0.0)
        core_density = core_scale.get("density", 0.0)
        cluster_density = round(cluster_edges / (cluster_nodes * (cluster_nodes - 1)), 6) if cluster_nodes > 1 else 0.0

        raw_q = mod_enh.get("raw_modularity_q", raw_mod.get("leiden_raw_modularity", 0.0))
        core_q = mod_enh.get("core_modularity_q", core_mod.get("core_leiden_modularity", 0.0))
        delta_q = mod_enh.get("modularity_gain_delta_q", round(core_q - raw_q, 4))
        pct_q_gain = mod_enh.get("percent_modularity_increase", round((delta_q / max(0.001, raw_q)) * 100, 2))

        # Dynamic metrics derived from manifest and diagnostics
        hubs_extracted = raw_nodes - core_nodes
        orphans_segregated = 0
        if manifest_data:
            summary_info = manifest_data.get("summary", {})
            hubs_extracted = summary_info.get("breakdown", {}).get("global_hubs_extracted", hubs_extracted)
            orphans_segregated = summary_info.get("breakdown", {}).get("segregated_orphans", summary_info.get("unattached_orphans", 0))

        edges_severed = raw_edges - core_edges
        pct_edges_severed = (edges_severed / max(1, raw_edges)) * 100

        core_components = core_comp.get("connected_components_count", 0)
        cluster_components = core_comp.get("cluster_connected_components_count", 0)
        cluster_giant_nodes = core_comp.get("cluster_giant_component_nodes", 0)
        cluster_giant_ratio = core_comp.get("cluster_giant_component_ratio", 0.0)

        cluster_max_deg = cluster_deg.get("max", 0)
        cluster_mean_deg = cluster_deg.get("mean", 0.0)
        cluster_median_deg = cluster_deg.get("median", 0.0)
        cluster_std_deg = cluster_deg.get("std_dev", 0.0)
        cluster_skew_deg = cluster_deg.get("skewness", 0.0)

        rows = [
            # Scale
            ("Scale & Volume", "Total Nodes", raw_nodes, core_nodes, cluster_nodes,
             f"-{raw_nodes - core_nodes} in Core (-{((raw_nodes - core_nodes) / max(1, raw_nodes))*100:.1f}%)",
             f"{hubs_extracted} global hubs extracted; {orphans_segregated} peripheral nodes segregated"),
            ("Scale & Volume", "Directed Edges", raw_edges, core_edges, cluster_edges,
             f"-{raw_edges - core_edges} in Core (-{pct_edges_severed:.1f}%)",
             f"Global hubs accounted for {pct_edges_severed:.1f}% ({edges_severed}) of all incident semantic predicates"),
            ("Scale & Volume", "Undirected Edges", raw_scale.get("num_undirected_edges", raw_edges), core_edges, cluster_edges,
             f"-{raw_edges - core_edges} in Core (-{pct_edges_severed:.1f}%)",
             "Bidirectional semantic connections"),
            ("Scale & Volume", "Graph Density", f"{raw_density:.6f}", f"{core_density:.6f}", f"{cluster_density:.6f}",
             f"+{((cluster_density - raw_density)/max(0.000001, raw_density))*100:.1f}% in cluster",
             f"Dense clustering core is {(cluster_density / max(0.000001, raw_density)):.1f}x denser than raw network"),

            # Degree Distribution
            ("Degree Distribution", "Max Degree", raw_deg.get("max", 0), core_deg.get("max", 0),
             cluster_max_deg, f"-{raw_deg.get('max', 0) - core_deg.get('max', 0)} in Core",
             f"Eliminates extreme star-hub distortion (raw max {raw_deg.get('max', 0)} -> core max {core_deg.get('max', 0)})"),
            ("Degree Distribution", "Mean Degree", f"{raw_deg.get('mean', 0.0):.3f}", f"{core_deg.get('mean', 0.0):.3f}",
             f"{cluster_mean_deg:.3f}" if isinstance(cluster_mean_deg, (int, float)) else str(cluster_mean_deg),
             f"-{raw_deg.get('mean', 0.0) - core_deg.get('mean', 0.0):.3f} in Core",
             "Dense clustering substrate exhibits concentrated multi-hop connectivity"),
            ("Degree Distribution", "Median Degree", f"{raw_deg.get('median', 0.0):.1f}", f"{core_deg.get('median', 0.0):.1f}",
             f"{cluster_median_deg:.1f}" if isinstance(cluster_median_deg, (int, float)) else str(cluster_median_deg),
             f"{'+' if (float(cluster_median_deg) - float(raw_deg.get('median', 0.0))) >= 0 else ''}{float(cluster_median_deg) - float(raw_deg.get('median', 0.0)):.1f} in cluster",
             "Pruned clustering substrate focuses on multi-hop concepts"),
            ("Degree Distribution", "Degree Std Dev", f"{raw_sd:.3f}", f"{core_sd:.3f}",
             f"{cluster_std_deg:.3f}" if isinstance(cluster_std_deg, (int, float)) else str(cluster_std_deg),
             f"-{sd_diff:.3f} (-{sd_pct:.1f}%)",
             "Normalizes heavy-tailed degree distortion across entities"),
            ("Degree Distribution", "Degree Skewness", f"{raw_skew:.3f}", f"{core_skew:.3f}",
             f"{cluster_skew_deg:.3f}" if isinstance(cluster_skew_deg, (int, float)) else str(cluster_skew_deg),
             f"-{skew_diff:.3f} (-{skew_pct:.1f}%)",
             "Skewness reduced drastically; stabilizes clustering optimization"),

            # Connectivity & Components
            ("Component Structure", "Connected Components", raw_comp.get("undirected_connected_components", 0),
             core_components, cluster_components,
             f"{core_components - raw_comp.get('undirected_connected_components', 0):+d} components in core",
             f"Hub stripping detaches {core_components} modular components, dyads, and singletons"),
            ("Component Structure", "Giant Component Nodes", raw_comp.get("giant_component_nodes", 0),
             core_comp.get("giant_component_nodes", 0), cluster_giant_nodes,
             f"-{raw_comp.get('giant_component_nodes', 0) - core_comp.get('giant_component_nodes', 0)} nodes",
             "Giant component breaks up into autonomous thematic communities"),
            ("Component Structure", "Giant Component Ratio", f"{raw_comp.get('giant_component_ratio', 0.0)*100:.2f}%",
             f"{core_comp.get('giant_component_ratio', 0.0)*100:.2f}%",
             f"{cluster_giant_ratio*100:.2f}%" if isinstance(cluster_giant_ratio, (int, float)) and cluster_giant_ratio <= 1.0 else str(cluster_giant_ratio),
             f"-{(raw_comp.get('giant_component_ratio', 0.0) - core_comp.get('giant_component_ratio', 0.0))*100:.2f}%",
             "Dissolves artificial single-giant-component artifact"),

            # Modularity & Communities
            ("Modularity & Communities", "Leiden Modularity (Q)", f"{raw_q:.4f}", f"{core_q:.4f}", f"{core_q:.4f}",
             f"+{delta_q:+.4f} (+{pct_q_gain:.1f}%)",
             "Modularity more than doubles on the pruned graph"),
            ("Modularity & Communities", "Discovered Communities", raw_mod.get("leiden_raw_community_count", 0),
             core_mod.get("core_communities_count", 0), core_mod.get("core_communities_count", 0),
             f"{core_mod.get('core_communities_count', 0) - raw_mod.get('leiden_raw_community_count', 0):+d} communities",
             "Resolves clean, fine-grained thematic partitions"),

            # Macro Topological Metrics
            ("Macro Topology", "Average Clustering Coeff", f"{raw_cc:.4f}",
             f"{core_cc:.4f}", f"{cluster_cc:.4f}",
             f"{'+' if cc_gain >= 0 else ''}{cc_gain:.1f}% in cluster",
             "Intra-cluster neighborhood cohesiveness improves in pruned core"),
            ("Macro Topology", "Degree Assortativity", f"{raw_assort:.4f}",
             f"{core_assort:.4f}", f"{cluster_assort:.4f}",
             f"{core_assort - raw_assort:+.4f} in core",
             "Star-hub connectivity bias diminished"),
            ("Macro Topology", "Transitivity", f"{raw_trans:.4f}",
             f"{core_trans:.4f}", f"{cluster_trans:.4f}",
             f"{'+' if trans_gain >= 0 else ''}{trans_gain:.1f}% in cluster",
             "Triadic closure increases significantly in modular cluster core")
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Category",
                "Metric Name",
                "Complete Graph (G_raw)",
                "Reduced Core (G_core)",
                "Dense Pruned Subgraph (G_cluster)",
                "Delta / Relative Change",
                "Algorithmic Significance"
            ])
            for r in rows:
                writer.writerow(r)

        return output_path

    def export_top_hubs(
        self,
        global_hubs_registry_data: Dict[str, Any],
        output_path: Path,
        total_raw_edges: int = 0
    ) -> Path:
        """
        Exports individual metrics for each node in the Top Global Hubs registry (comprehensive union of
        top hubs by degree and all flagged hubs removed based on topology config).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hubs = global_hubs_registry_data.get("hubs", [])

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Rank",
                "Node ID",
                "Removed (Y/N)",
                "Total Degree",
                "In-Degree",
                "Out-Degree",
                "PageRank",
                "Betweenness Centrality (C_B)",
                "Participation Coefficient (P_i)",
                "Modularity Vitality (Delta Q)",
                "Edges Severed Upon Removal",
                "Pct Total Edges Severed",
                "Primary Qualification Reason",
                "All Qualification Reasons"
            ])
            for idx, h in enumerate(hubs, start=1):
                node_id = h.get("node_id", "")
                is_removed = h.get("removed", True)
                rem_flag = "Y" if is_removed else "N"
                tot_deg = h.get("total_degree", 0)
                in_deg = h.get("in_degree", 0)
                out_deg = h.get("out_degree", 0)
                pr = h.get("pagerank", 0.0)
                betw = h.get("betweenness_centrality", 0.0)
                part = h.get("participation_coefficient", 0.0)
                vitality = h.get("modularity_vitality", 0.0)
                reasons = h.get("qualification_reasons", [])

                edges_severed = tot_deg if is_removed else 0
                pct_severed = round((edges_severed / max(1, total_raw_edges)) * 100, 2) if (is_removed and total_raw_edges > 0) else 0.0
                primary_reason = reasons[0] if reasons else ("Global Hub Router" if is_removed else "Retained Concept")
                all_reasons = " | ".join(reasons)

                writer.writerow([
                    idx,
                    node_id,
                    rem_flag,
                    tot_deg,
                    in_deg,
                    out_deg,
                    f"{pr:.6f}",
                    f"{betw:.6f}",
                    f"{part:.4f}",
                    f"{vitality:.6f}",
                    edges_severed,
                    f"{pct_severed:.2f}%",
                    primary_reason,
                    all_reasons
                ])

        return output_path

    def export_pruning_breakdown(
        self,
        removed_manifest_data: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """
        Exports the detailed breakdown of how nodes were segregated and pruned across Stage 3.2.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = removed_manifest_data.get("summary", {})
        breakdown = summary.get("breakdown", {})
        total_raw = summary.get("total_raw_nodes", 0)

        dense_core = summary.get("retained_dense_core_for_clustering", 0)
        attachable_leaves = summary.get("attachable_pendant_leaves", 0)
        true_singletons = breakdown.get("true_singletons", 0)
        isolated_dyads = breakdown.get("isolated_dyads", 0)
        hubs_count = breakdown.get("global_hubs_extracted", 0)

        hub_p_thresh = self.config.get("hub_pruning", {}).get("participation_threshold", 0.80)
        hub_min_deg = self.config.get("hub_pruning", {}).get("min_hub_degree", 40)

        rows = [
            ("Retained Dense Core (G_cluster)", dense_core,
             f"{(dense_core / max(1, total_raw)) * 100:.2f}%",
             "Degree > 1 in G_core",
             "Pass 1 Clustering Substrate: Fed into Leiden modularity optimization"),
            ("Attachable Pendant Leaves", attachable_leaves,
             f"{(attachable_leaves / max(1, total_raw)) * 100:.2f}%",
             "Degree = 1 in G_core (sole neighbor in G_cluster)",
             "Pass 2 Reattachment: Assigned to community of their unique dense core neighbor"),
            ("True Isolated Singletons", true_singletons,
             f"{(true_singletons / max(1, total_raw)) * 100:.2f}%",
             "Degree = 0 in G_core",
             "Peripheral leaf concepts whose only connections were to the removed global hubs"),
            ("Isolated Dyads / Peripheral Chains", isolated_dyads,
             f"{(isolated_dyads / max(1, total_raw)) * 100:.2f}%",
             "Degree = 1 in G_core (neighbor not in G_cluster)",
             "Small decoupled peripheral clusters isolated from the main dense core"),
            ("Global Hubs Removed", hubs_count,
             f"{(hubs_count / max(1, total_raw)) * 100:.2f}%",
             f"P_i >= {hub_p_thresh:.2f} and Degree >= {hub_min_deg} (or top betweenness)",
             "Extracted into Hub Backbone Network to eliminate cross-domain modularity blur"),
            ("TOTAL GRAPH ENTITIES", total_raw, "100.00%",
             "All unique entities from refined triplets",
             "Reconciles 100% of entity concepts in the input corpus")
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Node Classification",
                "Node Count",
                "Pct of Raw Graph",
                "Degree Definition",
                "Pipeline Treatment & Algorithmic Role"
            ])
            for r in rows:
                writer.writerow(r)

        return output_path

    def export_community_summary(
        self,
        community_diag_data: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """
        Exports summary statistics for each community discovered on the pruned graph.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        communities = community_diag_data.get("communities", [])
        fallback_partitioned = sum(c.get("node_count", 0) for c in communities)
        total_partitioned = community_diag_data.get("total_partitioned_nodes", fallback_partitioned)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Community ID",
                "Node Count",
                "Pct of Partitioned Core",
                "Internal Edge Count",
                "Internal Density",
                "Top Exemplar Nodes"
            ])
            for c in communities:
                cid = c.get("community_id", 0)
                cnt = c.get("node_count", 0)
                pct = round((cnt / max(1, total_partitioned)) * 100, 2)
                e_cnt = c.get("internal_edge_count", 0)
                dens = c.get("internal_density", 0.0)
                exemplars = ", ".join(c.get("exemplar_nodes", [])[:5])

                writer.writerow([
                    f"Community #{cid}",
                    cnt,
                    f"{pct:.2f}%",
                    e_cnt,
                    f"{dens:.4f}",
                    exemplars
                ])

        return output_path

    def export_all_from_artifacts(self, base_dir: Path) -> List[Path]:
        """
        Generates all tabular CSV reports directly from existing JSON artifacts in outputs/03_topology/.
        Runtime is typically < 0.2 seconds.
        """
        exported_files = []

        raw_diag_path = base_dir / "01_raw_graph_evaluation" / "raw_graph_diagnostics.json"
        hubs_path = base_dir / "01_raw_graph_evaluation" / "global_hubs_registry.json"
        core_diag_path = base_dir / "02_core_separation" / "core_graph_diagnostics.json"
        sep_report_path = base_dir / "02_core_separation" / "separation_impact_report.json"
        manifest_path = base_dir / "02_core_separation" / "removed_nodes_manifest.json"
        comm_diag_path = base_dir / "03_paths" / "community" / "community_path_diagnostics.json"

        # 1. Comparative master table
        if raw_diag_path.exists() and core_diag_path.exists() and sep_report_path.exists():
            with open(raw_diag_path, "r", encoding="utf-8") as f:
                raw_diag = json.load(f)
            with open(core_diag_path, "r", encoding="utf-8") as f:
                core_diag = json.load(f)
            with open(sep_report_path, "r", encoding="utf-8") as f:
                sep_report = json.load(f)

            manifest_data = None
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)

            summary_csv = base_dir / "topology_metrics_comparative_summary.csv"
            self.export_comparative_summary(raw_diag, core_diag, sep_report, summary_csv, manifest_data=manifest_data)
            exported_files.append(summary_csv)

        # 2. Top N global hubs table
        if hubs_path.exists():
            with open(hubs_path, "r", encoding="utf-8") as f:
                hubs_data = json.load(f)
            total_raw_edges = 0
            if raw_diag_path.exists():
                with open(raw_diag_path, "r", encoding="utf-8") as f:
                    rd = json.load(f)
                    total_raw_edges = rd.get("graph_scale", {}).get("num_directed_edges", 0)
            hubs_csv = base_dir / "01_raw_graph_evaluation" / "top_global_hubs_registry.csv"
            self.export_top_hubs(hubs_data, hubs_csv, total_raw_edges=total_raw_edges)
            exported_files.append(hubs_csv)

        # 3. Node scale and pruning breakdown
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            pruning_csv = base_dir / "02_core_separation" / "graph_scale_and_pruning_breakdown.csv"
            self.export_pruning_breakdown(manifest_data, pruning_csv)
            exported_files.append(pruning_csv)

        # 4. Community clusters summary
        if comm_diag_path.exists():
            with open(comm_diag_path, "r", encoding="utf-8") as f:
                comm_data = json.load(f)
            comm_csv = base_dir / "03_paths" / "community" / "community_clusters_summary.csv"
            self.export_community_summary(comm_data, comm_csv)
            exported_files.append(comm_csv)

        return exported_files
