import json
from pathlib import Path
from src.config import settings
from src.topology.graph_builder import TopologyPipeline

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
    
    try:
        results = pipeline.execute(refined_triplets)
        
        # Verify outputs
        norm_triplets_out = Path("outputs/03_topology/normalized_triplets.json")
        comm_out = Path("outputs/03_topology/community/topology_partitions.json")
        emb_out = Path("outputs/03_topology/embedding/topology_partitions.json")
        
        if norm_triplets_out.exists():
            print(f"   [OK] Normalized Triplets Saved: {norm_triplets_out}")
        if comm_out.exists():
            print(f"   [OK] Path 1 (Community) Partitions: {comm_out}")
        if emb_out.exists():
            print(f"   [OK] Path 2 (Embedding) Partitions: {emb_out}")
            
        print("=== Stage 3 Topology Completed Successfully ===")
    except Exception as e:
        print(f"Error during Topology execution: {e}")

if __name__ == "__main__":
    main()
