import argparse
import json
from pathlib import Path
import sys

from src.topology.tabular_exporter import TopologyTabularExporter


def run_tabular_export(base_dir: Path, top_hubs: int | None = None) -> None:
    """Fast regeneration of tabular CSV reports directly from existing Stage 3 JSON diagnostics."""
    if not base_dir.exists():
        print(f"CRITICAL ERROR: Directory '{base_dir}' does not exist.")
        sys.exit(1)

    print(f"=== SemanticPrism: Generating Stage 3 Tabular Reports ({base_dir}) ===")
    config = {}
    if top_hubs is not None:
        config["top_n_hubs_export"] = top_hubs

    exporter = TopologyTabularExporter(config=config)
    exported = exporter.export_all_from_artifacts(base_dir)

    if not exported:
        print("WARNING: No tabular files were generated. Ensure Stage 3 JSON diagnostics exist.")
        sys.exit(1)

    print("\n✅ Successfully Generated Tabular CSV Artifacts:")
    for p in exported:
        rel_path = p.relative_to(Path.cwd()) if p.is_relative_to(Path.cwd()) else p
        size_kb = p.stat().st_size / 1024.0
        with open(p, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f) - 1  # exclude header
        print(f"   • {rel_path} ({line_count} rows, {size_kb:.1f} KB)")

    print("\n=== Tabular Export Complete ===")


def run_resolver_only(base_dir: Path, config: dict) -> None:
    """Fast recomputation of TargetResolver community architectural tiers directly from existing partitions."""
    from src.topology.resolver import TargetResolver
    comm_partitions_path = base_dir / "03_paths" / "community" / "topology_partitions.json"
    refined_path = Path("outputs/02_refinement/refined_triplets.json")
    if not comm_partitions_path.exists():
        print(f"CRITICAL ERROR: '{comm_partitions_path}' not found. Please run Stage 3 first.")
        sys.exit(1)
    if not refined_path.exists():
        print(f"CRITICAL ERROR: '{refined_path}' not found.")
        sys.exit(1)

    print(f"=== SemanticPrism: Running Fast TargetResolver Tier Update ({base_dir}) ===")
    with open(comm_partitions_path, "r", encoding="utf-8") as f:
        topology = json.load(f)
    with open(refined_path, "r", encoding="utf-8") as f:
        refined_triplets = json.load(f)

    out_dir = base_dir / "03_paths" / "community"
    TargetResolver.export_resolved_targets_json(
        topology=topology,
        path_type="community",
        refined_triplets=refined_triplets,
        config=config,
        output_dir=out_dir
    )

    # Rebuild the tier-aware Stage 4 payload gallery from the same inputs
    import networkx as nx
    from src.topology.schemas import TopologyResult
    from src.topology.visualizer import TopologyVisualizer

    dg = nx.DiGraph()
    normalized_triplets = []
    for t in refined_triplets:
        subj, obj = t.get("subject"), t.get("object")
        if not subj or not obj:
            continue
        s_norm, o_norm = str(subj).lower().strip(), str(obj).lower().strip()
        t_norm = dict(t)
        t_norm["subject"], t_norm["object"] = s_norm, o_norm
        normalized_triplets.append(t_norm)
        dg.add_edge(s_norm, o_norm)

    result = TopologyResult(**topology)
    TopologyVisualizer(config=config)._generate_llm_payload_gallery(
        dg, result, normalized_triplets, out_dir, "Path 1: Community Workflow Path", "community"
    )
    print(f"   [OK] Tier-Aware Payload Gallery Saved: {out_dir / 'stage_04_llm_payload_gallery.html'}")
    print("✅ TargetResolver Fast Update Complete.")


def main():
    parser = argparse.ArgumentParser(
        description="SemanticPrism Stage 3: Topology Pipeline (Core-Separation & Community Architecture)"
    )
    parser.add_argument(
        "--tabular-only",
        action="store_true",
        help="Fast regeneration of tabular CSV reports from existing Stage 3 diagnostic artifacts (< 0.5s)",
    )
    parser.add_argument(
        "--resolver-only",
        action="store_true",
        help="Fast recomputation of TargetResolver community architectural tiers from existing partitions (< 2s)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="outputs/03_topology",
        help="Path to Stage 3 topology outputs directory (default: outputs/03_topology)",
    )
    parser.add_argument(
        "--top-hubs",
        type=int,
        default=None,
        help="Limit number of Top N Hubs exported in CSV (default: all qualified hubs)",
    )
    args = parser.parse_args()

    if args.tabular_only:
        run_tabular_export(Path(args.dir), args.top_hubs)
        return

    from src.config import settings

    if args.resolver_only:
        run_resolver_only(Path(args.dir), settings)
        return

    # Lazy-load heavy pipeline dependencies for full run
    from src.config import settings
    from src.topology.graph_builder import TopologyPipeline

    print("=== SemanticPrism Stage 3: Topology Pipeline (Core-Separation & Community Architecture) ===")
    
    input_path = Path("outputs/02_refinement/refined_triplets.json")
    if not input_path.exists():
        print(f"CRITICAL ERROR: Required input file '{input_path}' not found.")
        print("Please run Stage 2 first ('python3 run_stage_2_refinement.py' or 'python3 run_pipeline.py').")
        sys.exit(1)
        
    try:
        with open(input_path, "r") as f:
            refined_triplets = json.load(f)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load refined triplets: {e}")
        sys.exit(1)
        
    pipeline = TopologyPipeline(config=settings)
    
    try:
        pipeline.execute(refined_triplets)
            
        # Verify outputs across all 3 stages
        s1_diag = Path("outputs/03_topology/01_raw_graph_evaluation/raw_graph_diagnostics.json")
        s1_hubs = Path("outputs/03_topology/01_raw_graph_evaluation/global_hubs_registry.json")
        s2_diag = Path("outputs/03_topology/02_core_separation/core_graph_diagnostics.json")
        s2_impact = Path("outputs/03_topology/02_core_separation/separation_impact_report.json")
        s3_comm = Path("outputs/03_topology/03_paths/community/community_path_diagnostics.json")
        comm_out = Path("outputs/03_topology/03_paths/community/topology_partitions.json")
        comm_targets = Path("outputs/03_topology/03_paths/community/resolved_community_targets.json")
        
        print("\n--- Output Artifacts Verification ---")
        if s1_diag.exists() and s1_hubs.exists():
            print("   [OK] Stage 3.1 Raw Graph Evaluation & Hub Registry Saved (01_raw_graph_evaluation/)")
        if s2_diag.exists() and s2_impact.exists():
            print("   [OK] Stage 3.2 Core Separation & Impact Report Saved (02_core_separation/)")
        if s3_comm.exists():
            print("   [OK] Stage 3.3 Community Diagnostics & Highways Saved (03_paths/community/)")
        if comm_out.exists():
            print(f"   [OK] Community Canonical Partitions: {comm_out}")
        if comm_targets.exists():
            print(f"   [OK] Community Resolved Targets: {comm_targets}")

        v1 = Path("outputs/03_topology/01_raw_graph_evaluation/complete_topology_graph.html")
        v2 = Path("outputs/03_topology/02_core_separation/reduced_core_graph.html")
        v2_hub = Path("outputs/03_topology/02_core_separation/hub_backbone_network.html")
        v3_comm = Path("outputs/03_topology/03_paths/community/isolated_communities_only.html")

        print("\n--- Visuals Verification (Co-Located in 03_topology) ---")
        if v1.exists():
            print(f"   [OK] Stage 3.1 Visuals: {v1.parent}")
        if v2.exists() and v2_hub.exists():
            print(f"   [OK] Stage 3.2 Visuals (Reduced Graph & Hub Backbone): {v2.parent}")
        elif v2.exists():
            print(f"   [OK] Stage 3.2 Visuals (Reduced Graph): {v2.parent}")
        if v3_comm.exists():
            print(f"   [OK] Stage 3.3 Community Visuals: {v3_comm.parent}")

        # Tabular CSV Verification
        csv_sum = Path("outputs/03_topology/topology_metrics_comparative_summary.csv")
        csv_hubs = Path("outputs/03_topology/01_raw_graph_evaluation/top_global_hubs_registry.csv")
        csv_prune = Path("outputs/03_topology/02_core_separation/graph_scale_and_pruning_breakdown.csv")
        csv_comm = Path("outputs/03_topology/03_paths/community/community_clusters_summary.csv")

        print("\n--- Tabular CSV Reports Verification ---")
        if csv_sum.exists():
            print(f"   [OK] Comparative Summary Table: {csv_sum}")
        if csv_hubs.exists():
            print(f"   [OK] Top Global Hubs Registry Table: {csv_hubs}")
        if csv_prune.exists():
            print(f"   [OK] Graph Scale & Pruning Breakdown: {csv_prune}")
        if csv_comm.exists():
            print(f"   [OK] Community Clusters Summary: {csv_comm}")
            
        print("=== Stage 3 Topology Completed Successfully ===")
    except Exception as e:
        print(f"Error during Topology execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
