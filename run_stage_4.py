import sys
import json
from pathlib import Path
from src.config import settings
from src.synthesis.synthesizer import SynthesisPipeline

from src.topology.resolver import TargetResolver

def load_json(filepath, required=True):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        if required:
            print(f"CRITICAL ERROR: Required file {filepath} not found.")
        return None

def main():
    print("=== SemanticPrism Stage 4: Synthesis Pipeline ===")
    
    config = settings

    refined_triplets = load_json("outputs/02_refinement/refined_triplets.json")
    original_triplets = load_json("outputs/01_extraction/original_triplets.json")
    taxonomic_map = load_json("outputs/02_refinement/taxonomic_map.json", required=False) or {}
    master_themes_raw = load_json("outputs/01_extraction/master_themes.json", required=False) or {}

    # Check if required preceding stage outputs exist
    if not refined_triplets or not original_triplets:
        print("CRITICAL ERROR: Cannot run Stage 4. Missing required preceding stage outputs (Stages 1 & 2).")
        print("Please run preceding stages first ('python3 run_pipeline.py').")
        sys.exit(1)

    # Check synthesis execution mode
    synth_mode = config.get('synthesis', {}).get('execution_mode', 'community')
    print(f"   -> Synthesis Execution Mode: '{synth_mode}'")

    # Phase 0: Dynamic Target Resolution using CURRENT synthesis.yaml config
    print("   -> Phase 0: Resolving Synthesis Targets from Stage 3 Graph Partitions...")
    comm_targets = None
    emb_targets = None

    if synth_mode in ("both", "community"):
        comm_topology = load_json("outputs/03_topology/community/topology_partitions.json", required=False)
        if comm_topology:
            comm_target_file = TargetResolver.export_resolved_targets_json(
                comm_topology, "community", refined_triplets, config, Path("outputs/04_synthesis/community")
            )
            comm_targets = load_json(comm_target_file)
        else:
            print("   [Warning] Path 1 (Community) topology_partitions.json not found.")

    if synth_mode in ("both", "embedding"):
        emb_topology = load_json("outputs/03_topology/embedding/topology_partitions.json", required=False)
        if emb_topology:
            emb_target_file = TargetResolver.export_resolved_targets_json(
                emb_topology, "embedding", refined_triplets, config, Path("outputs/04_synthesis/embedding")
            )
            emb_targets = load_json(emb_target_file)
        else:
            print("   [Warning] Path 2 (Embedding) topology_partitions.json not found.")

    if not (comm_targets or emb_targets):
        print("Error: Cannot run Stage 4. Missing required Stage 3 topology partition outputs.")
        sys.exit(1)

    master_themes = []
    if isinstance(master_themes_raw, dict):
        master_themes = master_themes_raw.get("master_themes", [])
    elif isinstance(master_themes_raw, list):
        master_themes = master_themes_raw

    pipeline = SynthesisPipeline(config)
    pipeline.execute(comm_targets=comm_targets, emb_targets=emb_targets, refined_triplets=refined_triplets, original_triplets=original_triplets, taxonomic_map=taxonomic_map, master_themes=master_themes)
    
    print("=== Stage 4 Synthesis Completed Successfully ===")

if __name__ == "__main__":
    main()
