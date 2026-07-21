import json
from pathlib import Path
from src.config import settings
from src.topology.graph_builder import TopologyPipeline
from src.topology.fusion import TopologyFusionPipeline

def main():
    print("=== SemanticPrism Stage 3: Topology Pipeline (Dual-Path Architecture) ===")
    
    input_path = Path("outputs/02_refinement/refined_triplets.json")
    if not input_path.exists():
        print(f"Error: Required input file '{input_path}' not found.")
        print("Please run Stage 2 first.")
        return
        
    try:
        with open(input_path, "r") as f:
            refined_triplets = json.load(f)
    except Exception as e:
        print(f"Failed to load refined triplets: {e}")
        return
        
    pipeline = TopologyPipeline(config=settings)
    fusion_pipeline = TopologyFusionPipeline(config=settings)
    
    try:
        results = pipeline.execute(refined_triplets)
        
        # 1. Export Resolved Target JSON files for Path 1 and Path 2
        from src.topology.fusion import TargetResolver
        comm_targets_data, emb_targets_data = None, None
        
        if "community" in results:
            comm_target_file = TargetResolver.export_resolved_targets_json(
                results["community"].model_dump(), "community", refined_triplets, settings, Path("outputs/03_topology/community")
            )
            with open(comm_target_file, "r") as f:
                comm_targets_data = json.load(f)

        if "embedding" in results:
            emb_target_file = TargetResolver.export_resolved_targets_json(
                results["embedding"].model_dump(), "embedding", refined_triplets, settings, Path("outputs/03_topology/embedding")
            )
            with open(emb_target_file, "r") as f:
                emb_targets_data = json.load(f)
        
        # 2. Dual-Path Fusion: Evaluate Jaccard Alignment DIRECTLY on the RESOLVED TARGET PAYLOADS
        if comm_targets_data and emb_targets_data and fusion_pipeline.enable_unification:
            unified_res = fusion_pipeline.align_and_unify_paths(
                comm_targets_data, emb_targets_data, results.get("community"), results.get("embedding")
            )
            TargetResolver.export_resolved_targets_json(
                unified_res.model_dump(), "unified", refined_triplets, settings, Path("outputs/03_topology/unified")
            )
            
        # Verify outputs
        norm_triplets_out = Path("outputs/03_topology/normalized_triplets.json")
        comm_out = Path("outputs/03_topology/community/topology_partitions.json")
        emb_out = Path("outputs/03_topology/embedding/topology_partitions.json")
        unified_out = Path("outputs/03_topology/unified/topology_partitions.json")
        
        if norm_triplets_out.exists():
            print(f"   [OK] Normalized Triplets Saved: {norm_triplets_out}")
        if comm_out.exists():
            print(f"   [OK] Path 1 (Community) Partitions: {comm_out}")
        if emb_out.exists():
            print(f"   [OK] Path 2 (Embedding) Partitions: {emb_out}")
        if unified_out.exists():
            print(f"   [OK] Dual-Path Unified Partitions: {unified_out}")
            
        print("=== Stage 3 Topology Completed Successfully ===")
    except Exception as e:
        print(f"Error during Topology execution: {e}")

if __name__ == "__main__":
    main()
